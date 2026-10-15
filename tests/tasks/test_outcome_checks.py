"""Check the outcome, not the agent's opinion of it (#165): code checks what a run's calls really did."""

from __future__ import annotations

import httpx

from sentient import paths
from sentient.agent import outcomes
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.builtin import files_tool
from tests.conftest import FakeProvider
from tests.tasks.conftest import RESULT


def _mailer(failing: set[str], log: list[str]) -> ToolPlugin:
    """An MCP server's mail tool (checked by its name): sends to each address, except those in ``failing``."""

    @tool("mcp_mailer_send_email", risk=Risk.write)
    async def send_email(ctx: ToolContext, to: str, subject: str = "", body: str = "") -> dict:
        """Send an email to one or more comma-separated addresses."""
        sent, failed = [], []
        for addr in [a.strip() for a in to.split(",") if a.strip()]:
            log.append(addr)
            if addr in failing:
                failed.append({"to": addr, "error": "mailbox not found"})
            else:
                sent.append(addr)
        return {"content": "", "structured": {"sent": sent, "failed": failed}}

    class Mailer(ToolPlugin):
        id = "mcp_mailer"
        display_name = "mailer"
        auth = "mcp"
        tools = [send_email]

    return Mailer()


def _call(cid: str, tool_name: str, /, **arguments) -> dict:
    return {"id": cid, "name": tool_name, "arguments": arguments}


WRITE = _call("c1", "file_write", name="notes.md", content="Agenda:\n- budget\n")
SEND_BOTH = _call("c2", "mcp_mailer_send_email", to="asha@example.com, bad@example.com", subject="Notes", body="Hi")


async def _task(app) -> str:
    now = app.tasks.now_iso()
    return await app.tasks.repo.insert_task({
        "name": "Share notes", "description": "Write the notes and email them", "status": "pending",
        "schedule": {"type": "once", "run_at": None},
        "plan": [{"tool": "files", "description": "Write notes.md"}, {"tool": "mcp_mailer", "description": "Email it"}],
        "created_at": now, "updated_at": now,
    })


async def _run(app, task_id: str) -> dict:
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    return await app.tasks.get(task_id)


async def test_a_send_that_failed_for_one_recipient_needs_attention_whatever_the_answer_says(make_app):
    log: list[str] = []
    llm = FakeProvider(replies=[[WRITE], [SEND_BOTH], "I've saved the notes and sent the email to everyone."],
                       json_replies=[dict(RESULT)])
    app = await make_app(llm)
    app.registry.register(_mailer({"bad@example.com"}, log))
    task = await _run(app, await _task(app))
    run = task["runs"][-1]

    assert run["status"] == "completed_with_errors" and task["status"] == "completed_with_errors"
    assert run["error"] == (
        "Sentient said it sent the email, but mailer reported a problem: it couldn't do it for "
        "bad@example.com (mailbox not found)."
    )
    assert task["error"] == run["error"]
    file_check, mail_check = run["checks"]
    assert file_check["ok"] is True and file_check["how"] == "read_back" and file_check["action"] == "file written"
    assert mail_check["ok"] is False and mail_check["missed"] == ["bad@example.com"]
    assert mail_check["message"] == run["error"]
    # the run log says it plainly, and the tool result carries its check
    events = [e["message"] for e in await app.tasks.run_events(task["task_id"], run["run_id"])]
    assert {"type": "error", "content": run["error"]} in events
    assert [e["check"]["ok"] for e in events if e.get("type") == "tool_result"] == [True, False]
    notes = [n for n in await app.notifications.list() if n.get("task_id") == task["task_id"]]
    assert notes[0]["title"] == "Task needs attention" and run["error"] in notes[0]["message"]
    assert notes[0]["payload"]["needs_attention"] is True
    # the report writer is told what really happened
    report = [c for c in llm.calls if c.get("json") and c["role"] == "fast"][-1]
    assert "checked_outcome" in report["messages"][1]["content"]


async def test_a_clean_run_says_what_was_checked(make_app):
    log: list[str] = []
    send = _call("c2", "mcp_mailer_send_email", to="asha@example.com", subject="Notes", body="Hi")
    llm = FakeProvider(replies=[[WRITE], [send], "Saved and sent."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    app.registry.register(_mailer(set(), log))
    task = await _run(app, await _task(app))
    run = task["runs"][-1]

    assert run["status"] == "completed" and run["error"] is None
    assert [c["ok"] for c in run["checks"]] == [True, True]
    events = [e["message"] for e in await app.tasks.run_events(task["task_id"], run["run_id"])]
    assert {"type": "info", "content": "Checked: file written, email sent."} in events
    assert (paths.files_dir() / "notes.md").exists()


async def test_a_later_call_that_covers_the_failure_settles_it(make_app):
    log: list[str] = []
    again = _call("c3", "mcp_mailer_send_email", to="bad@example.org", subject="Notes", body="Hi")
    llm = FakeProvider(replies=[[SEND_BOTH], [again], "Sent to both, after fixing an address."],
                       json_replies=[dict(RESULT)])
    app = await make_app(llm)
    # only the first address bounces; the fixed address goes through, but it is not the one that failed
    app.registry.register(_mailer({"bad@example.com"}, log))
    task = await _run(app, await _task(app))
    assert task["runs"][-1]["status"] == "completed_with_errors"  # a different address doesn't settle it

    log.clear()
    retry = _call("c4", "mcp_mailer_send_email", to="bad@example.com", subject="Notes", body="Hi")
    llm2 = FakeProvider(replies=[[SEND_BOTH], [retry], "Sent to both."], json_replies=[dict(RESULT)])
    app2 = await make_app(llm2, db_name="tasks2.db")
    failing = {"bad@example.com"}
    plugin = _mailer(failing, log)
    app2.registry.register(plugin)

    original = plugin.tools[0].fn

    async def flaky(ctx, **kwargs):
        result = await original(ctx, **kwargs)
        failing.clear()  # the second try goes through
        return result

    plugin.tools[0].fn = flaky
    task2 = await _run(app2, await _task(app2))
    run2 = task2["runs"][-1]
    assert run2["status"] == "completed" and run2["checks"][0]["settled"] is True


async def test_an_error_result_is_a_failure_even_when_the_model_carries_on(make_app):
    @tool("postbox_send", risk=Risk.send, outcome="email")
    async def postbox_send(ctx: ToolContext, to: str, subject: str = "", body: str = "") -> dict:
        """Send an email."""
        return {"error": "Postbox request failed (403 insufficient permissions)."}

    class Postbox(ToolPlugin):
        id = "postbox"
        display_name = "Postbox"
        tools = [postbox_send]

    llm = FakeProvider(replies=[[_call("c1", "postbox_send", to="asha@example.com")], "Done."],
                       json_replies=[dict(RESULT)])
    app = await make_app(llm)
    app.registry.register(Postbox())
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Email Asha", "description": "Email Asha", "status": "pending", "schedule": {"type": "once"},
        "plan": [{"tool": "postbox", "description": "Send the email"}], "created_at": now, "updated_at": now,
    })
    run = (await _run(app, task_id))["runs"][-1]
    assert run["status"] == "completed_with_errors"
    assert run["error"] == (
        "Sentient tried to send the email, but Postbox reported a problem: Postbox request failed (403 insufficient "
        "permissions)."
    )


async def test_reading_back_a_file_that_doesnt_hold_what_was_written(make_app, monkeypatch):
    app = await make_app()
    ctx = app.agent.tool_context(None, "task")
    tool_ = app.registry.get("file_write")
    res = await tool_.call(ctx, {"name": "a.txt", "content": "one\ntwo\n"})
    good = await app.agent.check_outcome(tool_, {"name": "a.txt", "content": "one\ntwo\n"}, res, False, ctx)
    assert good["ok"] is True and good["how"] == "read_back"
    (paths.files_dir() / "a.txt").write_text("something else", encoding="utf-8")
    assert files_tool._wrote({"name": "a.txt", "content": "one\ntwo\n"}, res, ctx) == {
        "ok": False, "detail": "the file a.txt doesn't hold what was written",
    }
    (paths.files_dir() / "a.txt").unlink()
    bad = await app.agent.check_outcome(tool_, {"name": "a.txt", "content": "one\ntwo\n"}, res, False, ctx)
    assert bad["ok"] is False and bad["detail"] == "the file a.txt isn't in the files folder"


async def test_gmail_and_calendar_read_their_effect_back(monkeypatch):
    from sentient.integrations.plugins import gcalendar, gmail

    async def fake_gapi(ctx, pid, method, url, **kwargs):
        if url.endswith("/messages/m1"):
            return {"id": "m1", "labelIds": ["SENT"]}
        if url.endswith("/messages/m2"):
            return {"id": "m2", "labelIds": ["DRAFT"]}
        request = httpx.Request(method, url)
        raise httpx.HTTPStatusError("gone", request=request, response=httpx.Response(404, request=request))

    monkeypatch.setattr(gmail, "gapi", fake_gapi)
    monkeypatch.setattr(gcalendar, "gapi", fake_gapi)
    assert await gmail.in_sent({}, {"sent": True, "id": "m1"}, None) == {"ok": True}
    assert (await gmail.in_sent({}, {"sent": True, "id": "m2"}, None))["ok"] is False
    assert (await gmail.in_sent({}, {"sent": True, "id": "m3"}, None))["detail"] == (
        "Gmail doesn't have the email it said it sent"
    )
    assert await gmail.in_sent({}, {"sent": True}, None) is None  # nothing to read back: the result stands
    gone = await gcalendar.event_exists({}, {"created": True, "event": {"id": "e1"}}, None)
    assert gone == {"ok": False, "detail": "the event isn't on your Google Calendar"}


def test_result_problem_reads_the_usual_shapes():
    rp = outcomes.result_problem
    assert rp({"sent": True, "id": "x"}) is None
    assert rp({"error": "boom"}) == ("boom", [])
    assert rp({"ok": False, "message": "quota exceeded"}) == ("quota exceeded", [])
    assert rp({"status": "failed", "to": "a@x.com", "reason": "blocked"}) == ("blocked (a@x.com)", ["a@x.com"])
    assert rp({"content": '{"sent": [], "failed": ["b@x.com"]}'}) == ("it couldn't do it for b@x.com", ["b@x.com"])
    assert rp({"content": "Error: SMTP 550"}) == ("Error: SMTP 550", [])
    assert rp({"content": "Message sent"}) is None
    assert rp({"results": [{"to": "a", "status": "sent"}, {"to": "b", "status": "error", "error": "x"}]}) == (
        "x (b)", ["b"]
    )
    assert rp({"failed": []}) is None


def test_finalize_words_the_mismatch_from_what_the_answer_claimed():
    check = {"tool": "t", "kind": "event", "ok": False, "detail": "the event isn't on your Google Calendar",
             "targets": ["lunch"], "missed": []}
    said = outcomes.finalize([check], "I've created the event.")
    assert said[0]["message"] == "Sentient said it created the event, but the event isn't on your Google Calendar."
    tried = outcomes.finalize([check], "Here is the plan.")
    assert tried[0]["message"].startswith("Sentient tried to create the event, but")
    ok = {**check, "ok": True, "detail": ""}
    assert outcomes.failures(outcomes.finalize([check, ok], "")) == []  # the same event, made again
    assert outcomes.summary(outcomes.finalize([ok], "")) == "Checked: event created."


async def test_a_chat_reply_carries_its_checks_and_keeps_them(make_app):
    from sentient.llm.events import Done, ToolResultEvent

    log: list[str] = []
    llm = FakeProvider(replies=[[WRITE, SEND_BOTH], "Saved the notes and emailed both of them."])
    app = await make_app(llm)
    app.registry.register(_mailer({"bad@example.com"}, log))
    sid = await app.store.create_session(channel="desktop")
    events = [ev async for ev in app.agent.run_turn(sid, "save my notes and email them", channel="desktop")]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [r.check["ok"] for r in results] == [True, False]
    [done] = [e for e in events if isinstance(e, Done)]
    assert [c["ok"] for c in done.checks] == [True, False]
    assert done.checks[1]["message"] == (
        "Sentient said it sent the email, but mailer reported a problem: it couldn't do it for "
        "bad@example.com (mailbox not found)."
    )
    final = [r for r in await app.store.recent_messages(sid, 20) if r["role"] == "assistant"][-1]
    assert final["checks"] == done.checks

    llm.replies.append("Hello!")
    events = [ev async for ev in app.agent.run_turn(sid, "thanks", channel="desktop")]
    assert next(e for e in events if isinstance(e, Done)).checks == []


def test_a_failure_with_no_target_is_settled_only_by_the_next_call_of_that_tool():
    bad = {"tool": "gcal_quick_add", "kind": "event", "ok": False, "detail": "x", "targets": [], "missed": []}
    other = {"tool": "file_write", "kind": "file", "ok": True, "targets": ["a.md"], "missed": []}
    retry = {"tool": "gcal_quick_add", "kind": "event", "ok": True, "targets": ["lunch"], "missed": []}
    assert outcomes.failures(outcomes.finalize([bad, other, retry], "")) == []  # the retry right after it
    again = {**bad, "detail": "y"}
    assert len(outcomes.failures(outcomes.finalize([bad, again, retry], ""))) == 1  # the next call failed too


async def test_a_stopped_reply_keeps_its_checks(make_app):
    from sentient.agent.sources import AnswerSources
    from sentient.memory.sources import MemorySources

    app = await make_app()
    sid = await app.store.create_session(channel="desktop")
    checks = outcomes.finalize([{"tool": "file_write", "kind": "file", "ok": True, "action": "file written"}], "")
    kept = await app.agent._persist_stopped(sid, "Saving", MemorySources(), AnswerSources(), checks)
    assert kept["checks"] == checks
    final = [r for r in await app.store.recent_messages(sid, 5) if r["role"] == "assistant"][-1]
    assert final["checks"] == checks
