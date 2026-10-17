"""The setup checklist of a routine (#161): filled in from the plan, confirmed and stored, a missing source handled the
way the user chose (skip quietly, tell me, try again later), "ask me first", and the Test run's report."""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest
import respx

from sentient.integrations.plugins import web
from sentient.tasks import checklist
from sentient.tasks.checklist import probe_page as real_probe_page
from sentient.tasks.scripts import normalize_script
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import json_calls

PAGE = "http://studio.example/events.html"
PROMPT = f"every morning at 8, email me a summary of new events on {PAGE}"
REFINE = {"name": "Studio events", "description": PROMPT, "priority": 1,
          "schedule": {"type": "recurring", "frequency": "daily", "time": "08:00"}}
PLAN = {"name": "Studio events digest", "description": "Email a summary of new studio events",
        "plan": [{"tool": "web", "description": f"Read {PAGE}"},
                 {"tool": "mailer", "description": "Email Maya a short summary of the new events"}],
        "clarifying_questions": []}
SEND = [tool_call("mailer_send_email", to="maya@example.com", subject="New events", body="Pottery night on Oct 22.")]


def _mail(log: list[str]) -> ToolPlugin:
    @tool("mailer_list_inbox", risk=Risk.read)
    async def mailer_list_inbox(ctx: ToolContext) -> dict:
        """List the inbox."""
        return {"emails": []}

    @tool("mailer_send_email", risk=Risk.send)
    async def mailer_send_email(ctx: ToolContext, to: str, subject: str = "", body: str = "") -> dict:
        """Send an email."""
        log.append(f"send:{to}")
        return {"sent": True}

    class Mailer(ToolPlugin):
        id = "mailer"
        display_name = "Mailer"
        tools = [mailer_list_inbox, mailer_send_email]

    return Mailer()


@pytest.fixture
def page(monkeypatch):
    """Whether the studio page answers: ``page["down"] = "reason"`` takes it down."""
    state: dict[str, str | None] = {"down": None}
    probed: list[str] = []

    async def probe(url: str) -> str | None:
        probed.append(url)
        return state["down"]

    monkeypatch.setattr(checklist, "probe_page", probe)
    state["probed"] = probed  # type: ignore[assignment]
    return state


async def _routine(app, *, if_missing: str = "notify", approval: str = "plan", source: dict | None = None,
                   schedule: dict | None = None) -> str:
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Studio events", "description": PROMPT, "original_prompt": PROMPT, "status": "approval_pending",
        # 20:00 in Kolkata: the frozen clock (07:30 there) never reaches a scheduled run in these tests
        "schedule": schedule or {"type": "recurring", "frequency": "daily", "time": "20:00", "timezone": "Asia/Kolkata"},
        "plan": PLAN["plan"], "created_at": now, "updated_at": now,
    })
    answers: dict = {"if_missing": if_missing, "approval": approval, "confirm": True}
    if source is not None:
        answers["source"] = source
    await app.tasks.update(task_id, {"setup": answers})
    await app.tasks.approve(task_id)
    return task_id


def _events(app, event: str) -> list:
    async def get():
        return [n for n in await app.notifications.list() if (n.get("payload") or {}).get("event") == event]
    return get()


# ---------------------------------------------------------------------------- filled in from the plan, confirmed
async def test_planning_a_routine_fills_in_its_checklist_for_the_user_to_confirm(make_app):
    llm = FakeProvider(json_replies=[REFINE, PLAN])
    app = await make_app(llm)
    app.registry.register(_mail([]))
    created = await app.tasks.create_task(PROMPT)
    await app.tasks.drain()
    task = await app.tasks.get(created["task_id"])
    assert task["status"] == "approval_pending"
    setup = task["setup"]
    assert setup["source"] == {"kind": "page", "target": PAGE, "label": "studio.example/events.html"}
    assert setup["output"] == "Email Maya a short summary of the new events"
    assert setup["approval"] == "plan" and setup["acts"] == ["Mailer"]
    assert setup["if_missing"] == "notify" and setup["confirmed_at"] is None
    assert len(json_calls(llm, "planner")) == 2  # filled in by code, no extra model call

    # the user changes two answers, the schedule's time zone, and confirms
    sched = {**task["schedule"], "timezone": "Europe/London"}
    task = await app.tasks.update(task["task_id"], {"schedule": sched, "setup": {
        "if_missing": "retry", "approval": "ask", "output": "  A short email with the new events  ", "confirm": True,
    }})
    assert task["schedule"]["timezone"] == "Europe/London"
    assert task["setup"]["if_missing"] == "retry" and task["setup"]["approval"] == "ask"
    assert task["setup"]["output"] == "A short email with the new events"
    assert task["setup"]["confirmed_at"]
    assert task["setup"]["source"]["target"] == PAGE  # kept


async def test_one_off_tasks_have_no_checklist_and_bad_answers_are_refused(make_app):
    app = await make_app(FakeProvider())
    now = app.tasks.now_iso()
    once = await app.tasks.repo.insert_task({"name": "x", "status": "approval_pending", "plan": PLAN["plan"],
                                             "schedule": {"type": "once", "run_at": None},
                                             "created_at": now, "updated_at": now})
    assert (await app.tasks.get(once))["setup"] is None
    with pytest.raises(ValueError, match="Only recurring and triggered"):
        await app.tasks.update(once, {"setup": {"confirm": True}})
    task_id = await _routine(app)
    for bad in ({"if_missing": "sometimes"}, {"approval": "never"}, {"source": {"kind": "page", "target": "ftp://x"}},
                {"source": {"kind": "app", "target": "nope"}}, {"source": {"kind": "planet"}}, {"source": "x"}):
        with pytest.raises(ValueError):
            await app.tasks.update(task_id, {"setup": bad})
    # turning a one-off task into a routine fills its checklist in
    task = await app.tasks.update(once, {"schedule": {"type": "recurring", "frequency": "interval", "interval_minutes": 60}})
    assert task["setup"]["source"]["target"] == PAGE and task["setup"]["confirmed_at"] is None  # from its plan
    task = await app.tasks.update(once, {"schedule": {"type": "once", "run_at": None}})
    assert task["setup"] is None  # no longer a routine: no checklist


def test_prefill_guesses_the_source_from_the_trigger_a_folder_or_the_plan(make_app):
    class Registry:
        class _P:
            def __init__(self, pid, name, auth="oauth", tools=()):
                self.id, self.display_name, self.auth_type, self.tools = pid, name, auth, list(tools)

        def __init__(self):
            read = type("T", (), {"risk": Risk.read, "internal": False})()
            self.plugins = {"gmail": self._P("gmail", "Gmail", tools=[read]), "web": self._P("web", "Web Pages", "builtin")}

        def plugin(self, pid):
            return self.plugins.get(pid)

    reg = Registry()
    triggered = {"schedule": {"type": "triggered", "source": "gmail", "event": "new_email"}, "plan": []}
    assert checklist.guess_source(triggered, reg) == {"kind": "app", "target": "gmail", "label": "Gmail"}
    folder = {"original_prompt": r"every evening tidy C:\Users\maya\Downloads please", "plan": []}
    assert checklist.guess_source(folder, reg)["target"] == r"C:\Users\maya\Downloads"
    # a file it writes isn't there before the first run: the folder it goes in is what must exist
    output = {"original_prompt": r"every friday save a digest to C:\Users\maya\Documents\digest.txt", "plan": []}
    assert checklist.guess_source(output, reg)["target"] == r"C:\Users\maya\Documents"
    assert checklist.guess_source({"original_prompt": "tidy ~/notes/today.md", "plan": []}, reg)["target"] == "~/notes"
    planned = {"plan": [{"tool": "web", "description": "search"}, {"tool": "gmail", "description": "read new mail"}]}
    assert checklist.guess_source(planned, reg)["target"] == "gmail"
    assert checklist.guess_source({"plan": [{"tool": "web", "description": "search the news"}]}, reg)["kind"] == "none"


# ---------------------------------------------------------------------------- a missing source, each choice
async def test_skip_quietly_records_a_skipped_run_and_tells_nobody(make_app, page):
    log: list[str] = []
    app = await make_app(FakeProvider(replies=[SEND, "Sent."]))
    app.registry.register(_mail(log))
    task_id = await _routine(app, if_missing="skip")
    page["down"] = "the page studio.example/events.html couldn't be reached"

    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert run["status"] == "skipped" and run["retry_at"] is None
    assert run["error"] == "Skipped: the page studio.example/events.html couldn't be reached."
    assert task["status"] == "active" and task["next_execution_at"]  # waits for its next morning
    assert log == [] and await _events(app, "source_missing") == []
    assert await _events(app, "run_failed") == []
    assert [u["message"]["content"] for u in run["progress_updates"]] == [run["error"]]  # one plain line, no model


async def test_notify_me_skips_and_sends_one_notice(make_app, page):
    log: list[str] = []
    app = await make_app(FakeProvider())
    app.registry.register(_mail(log))
    task_id = await _routine(app, if_missing="notify")
    page["down"] = "the page studio.example/events.html answered with an error (404)"

    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "skipped" and log == []
    [note] = await _events(app, "source_missing")
    assert note["title"] == "Skipped: Studio events"
    assert note["message"] == ("'Studio events' didn't run. The page studio.example/events.html answered with an "
                               "error (404).")
    assert note["payload"]["if_missing"] == "notify" and note["payload"]["run_id"] == run["run_id"]


async def test_retry_later_tries_again_then_gives_up_and_tells(make_app, page, clock):
    log: list[str] = []
    app = await make_app(FakeProvider(replies=[SEND, "Sent the summary."]), clock=clock)
    app.registry.register(_mail(log))
    app.config.tasks.source_retry_minutes = 15
    app.config.tasks.source_retries = 2
    task_id = await _routine(app, if_missing="retry")
    page["down"] = "the page studio.example/events.html couldn't be reached"

    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    first = (await app.tasks.get(task_id))["runs"][-1]
    assert first["status"] == "skipped" and first["retry_at"] == app.tasks.now_iso().replace("02:00:00", "02:15:00")
    assert "Trying again at 07:45 (try 1 of 2)" in first["error"]  # 02:15 UTC in Asia/Kolkata
    assert await _events(app, "source_missing") == []

    assert await app.tasks.tick() == []  # not yet
    clock.dt += timedelta(minutes=15)
    [second_id] = await app.tasks.tick()
    await app.tasks.drain()
    runs = (await app.tasks.get(task_id))["runs"]
    second = runs[-1]
    assert second["run_id"] == second_id and second["retry_of"] == first["run_id"] and second["status"] == "skipped"
    assert "(try 2 of 2)" in second["error"] and runs[0]["retry_at"] is None  # a retry is taken once

    clock.dt += timedelta(minutes=15)
    assert len(await app.tasks.tick()) == 1
    await app.tasks.drain()
    third = (await app.tasks.get(task_id))["runs"][-1]
    assert third["status"] == "skipped" and third["retry_at"] is None
    assert third["error"].endswith("It tried again 2 times, so it waits for its next run.")
    [note] = await _events(app, "source_missing")
    assert note["payload"]["if_missing"] == "retry"
    assert log == []
    clock.dt += timedelta(minutes=15)
    assert await app.tasks.tick() == []  # nothing more until its next morning


async def test_a_retry_runs_for_real_once_the_page_is_back(make_app, page, clock):
    log: list[str] = []
    app = await make_app(FakeProvider(replies=[SEND, "Sent the summary."]), clock=clock)
    app.registry.register(_mail(log))
    task_id = await _routine(app, if_missing="retry")
    page["down"] = "the page studio.example/events.html couldn't be reached"
    await app.tasks.run_now(task_id)
    await app.tasks.drain()

    page["down"] = None
    clock.dt += timedelta(minutes=app.config.tasks.source_retry_minutes)
    await app.tasks.tick()
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert run["status"] == "completed" and log == ["send:maya@example.com"]
    assert run["progress_updates"][0]["message"]["content"] == "Trying again now: what it reads wasn't there last time."
    assert task["status"] == "active"


async def test_a_retry_is_dropped_when_the_task_is_paused_or_a_newer_run_came(make_app, page, clock):
    app = await make_app(FakeProvider(replies=["Nothing new.", "Nothing new."]), clock=clock)
    app.registry.register(_mail([]))
    task_id = await _routine(app, if_missing="retry")
    page["down"] = "the page studio.example/events.html couldn't be reached"
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    await app.tasks.update(task_id, {"enabled": False})
    clock.dt += timedelta(minutes=20)
    assert await app.tasks.tick() == []

    await app.tasks.update(task_id, {"enabled": True})
    await app.tasks.run_now(task_id)  # skipped again, with a retry of its own
    await app.tasks.drain()
    page["down"] = None
    await app.tasks.run_now(task_id)  # a newer run: the page is back and it ran
    await app.tasks.drain()
    clock.dt += timedelta(minutes=20)
    assert await app.tasks.tick() == []
    assert [r["status"] for r in (await app.tasks.get(task_id))["runs"]] == ["skipped", "skipped", "completed"]


async def test_an_app_that_is_not_connected_is_a_missing_source_but_its_own_event_is_not(make_app):
    app = await make_app(FakeProvider(replies=["Filed it.", "Filed it."]))
    app.registry.register(_mail([]))
    connected = {"mailer": False}

    async def is_connected(pid):
        return connected.get(pid, True)

    app.integrations.is_connected = is_connected  # type: ignore[method-assign]
    sched = {"type": "triggered", "source": "mailer", "event": "new_email", "filter": {}}
    task_id = await _routine(app, if_missing="skip", schedule=sched, source={"kind": "app", "target": "mailer"})
    assert (await app.tasks.get(task_id))["setup"]["source"] == {"kind": "app", "target": "mailer", "label": "Mailer"}
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "skipped" and run["error"] == "Skipped: Mailer isn't connected."
    # an email from the Mailer itself started this one: the app is plainly there
    await app.tasks.handle_event("mailer", "new_email", {"id": "m1", "subject": "Hi"})
    await app.tasks.drain()
    assert (await app.tasks.get(task_id))["runs"][-1]["status"] == "completed"


async def test_a_folder_that_is_gone_is_a_missing_source(make_app, tmp_path):
    app = await make_app(FakeProvider(replies=["Tidied."]))
    task_id = await _routine(app, if_missing="skip", source={"kind": "folder", "target": str(tmp_path / "gone")})
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "skipped" and "isn't there" in run["error"]
    (tmp_path / "gone").mkdir()
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    assert (await app.tasks.get(task_id))["runs"][-1]["status"] == "completed"


async def test_a_check_script_follows_the_same_choice(make_app, page, clock):
    app = await make_app(FakeProvider(), clock=clock)
    now = app.tasks.now_iso()
    code = f"from sentient_tools import tools, result\nresult(tools.web_fetch(url='{PAGE}')['content'])\n"
    task_id = await app.tasks.repo.insert_task({
        "name": "Studio page watch", "description": f"Watch {PAGE}", "original_prompt": f"watch {PAGE} every hour",
        "status": "approval_pending", "task_type": "script",
        "script": normalize_script({"code": code, "condition": "changed", "then": "notify"}),
        "schedule": {"type": "recurring", "frequency": "interval", "interval_minutes": 60, "timezone": "UTC"},
        "plan": [{"tool": "", "description": "Check the page"}], "created_at": now, "updated_at": now,
    })
    await app.tasks.update(task_id, {"setup": {"if_missing": "retry", "confirm": True}})
    assert (await app.tasks.get(task_id))["setup"]["source"]["target"] == PAGE  # found in the script
    await app.tasks.approve(task_id)
    ran: list[str] = []

    async def sandbox(code, trigger_data=None):
        ran.append(code)
        return {"ok": True, "result": "events"}

    app.tasks._sandbox_run = sandbox  # type: ignore[method-assign]
    page["down"] = "the page studio.example/events.html couldn't be reached"
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert ran == []  # the script never ran
    check = task["script"]["checks"][-1]
    assert check["outcome"] == "skipped" and "(try 1 of 3)" in check["note"]
    assert task["next_execution_at"] == (clock.dt + timedelta(minutes=15)).isoformat(timespec="seconds")
    assert await _events(app, "script_failed") == []


# ---------------------------------------------------------------------------- what it may do without asking
async def test_ask_me_first_holds_a_send_and_asks_with_the_tasks_reason(make_app, page):
    log: list[str] = []
    llm = FakeProvider(replies=[SEND])
    app = await make_app(llm)
    app.registry.register(_mail(log))
    task_id = await _routine(app, approval="ask")
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert run["status"] == "waiting_for_user" and log == []
    question = run["pending_question"]
    assert question["kind"] == "approval" and "You set this task to ask you first." in question["question"]
    assert "Allow for this task" in question["options"]

    # "Allow for this task" runs the held call, and lets later runs go ahead without asking
    llm.replies = ["Sent.", SEND, "Sent again."]
    await app.tasks.answer_question(task_id, run["run_id"], "Allow for this task")
    await app.tasks.drain()
    assert log == ["send:maya@example.com"]
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    assert (await app.tasks.get(task_id))["runs"][-1]["status"] == "completed"
    assert log == ["send:maya@example.com", "send:maya@example.com"]


async def test_go_ahead_with_the_plan_sends_without_asking(make_app, page):
    log: list[str] = []
    app = await make_app(FakeProvider(replies=[SEND, "Sent."]))
    app.registry.register(_mail(log))
    task_id = await _routine(app, approval="plan")
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    assert (await app.tasks.get(task_id))["runs"][-1]["status"] == "completed" and log == ["send:maya@example.com"]


async def test_a_lasting_allow_rule_wins_over_ask_me_first(make_app, page):
    log: list[str] = []
    app = await make_app(FakeProvider(replies=[SEND, "Sent."]))
    app.registry.register(_mail(log))
    app.config.tools.approvals.rules = {"mailer_send_email": "allow"}
    task_id = await _routine(app, approval="ask")
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    assert log == ["send:maya@example.com"]


# ---------------------------------------------------------------------------- Test run
async def test_test_run_sends_nothing_and_reports_what_it_would_have_sent(make_app, page):
    log: list[str] = []
    app = await make_app(FakeProvider(replies=[SEND, "Done."]))
    app.registry.register(_mail(log))
    task_id = await _routine(app, approval="ask")
    await app.tasks.dry_run(task_id)
    await app.tasks.drain()
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["dry_run"] is True and run["status"] == "completed" and log == []
    [action] = run["would_do"]
    assert action["summary"].startswith("Send an email in Mailer: To maya@example.com")
    assert action["asks"] is True  # a real run would ask first: the checklist says so


async def test_test_run_fails_plainly_when_the_source_is_missing(make_app, page):
    log: list[str] = []
    app = await make_app(FakeProvider(replies=[SEND, "Done."]))
    app.registry.register(_mail(log))
    task_id = await _routine(app, if_missing="retry")
    page["down"] = "the page studio.example/events.html couldn't be reached"
    await app.tasks.dry_run(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert run["dry_run"] is True and run["status"] == "error" and log == []
    assert run["error"] == ("The page studio.example/events.html couldn't be reached, so a real run would skip that run "
                            "and try again a little later. Nothing was sent, changed or run.")
    assert task["status"] == "active" and run["retry_at"] is None  # a test run never schedules anything
    assert len(await _events(app, "run_failed")) == 1  # "Try run failed"


# ---------------------------------------------------------------------------- looking at the page itself
@respx.mock
async def test_probe_page_says_why_a_page_is_missing(monkeypatch):
    monkeypatch.setattr(web, "_check_url", lambda url: url)  # no DNS in tests
    respx.get("https://studio.example/ok").respond(200, text="<p>events</p>")
    respx.get("https://studio.example/gone").respond(404)
    respx.get("https://studio.example/down").mock(side_effect=httpx.ConnectError("refused"))
    assert await real_probe_page("https://studio.example/ok") is None
    assert await real_probe_page("https://studio.example/gone") == (
        "the page studio.example/gone answered with an error (404)")
    assert await real_probe_page("https://studio.example/down") == "the page studio.example/down couldn't be reached"


@respx.mock
async def test_probe_page_checks_every_address_it_is_sent_on_to(monkeypatch):
    def check(url):
        if "127.0.0.1" in url:
            raise web.IntegrationError("Fetching addresses on this computer isn't allowed.")
        return url

    monkeypatch.setattr(web, "_check_url", check)
    respx.get("https://studio.example/moved").respond(301, headers={"Location": "/events"})
    respx.get("https://studio.example/events").respond(200)
    sneaky = respx.get("http://127.0.0.1:8080/admin").respond(200)
    respx.get("https://studio.example/sneaky").respond(302, headers={"Location": "http://127.0.0.1:8080/admin"})
    respx.get("https://studio.example/loop").respond(302, headers={"Location": "/loop"})
    assert await real_probe_page("https://studio.example/moved") is None
    assert await real_probe_page("https://studio.example/sneaky") is None  # not judged, and never requested
    assert not sneaky.called
    assert await real_probe_page("https://studio.example/loop") == "the page studio.example/loop sends visitors around in circles"


async def test_probe_page_leaves_addresses_sentient_never_opens_alone():
    # this computer: reading the page would be refused anyway, so there is nothing to judge (and no request is made)
    assert await real_probe_page("http://127.0.0.1:9/events") is None
