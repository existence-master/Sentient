"""Monitors that only speak when something changed (#157): no model for an unchanged source, a judge for changes,
a quiet history of checks, and fuzzy time windows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sentient.tasks import schedule as sched
from sentient.tasks import scripts
from sentient.tasks.schedule import calculate_next_run, normalize_schedule
from sentient.tasks.scripts import normalize_script
from tests.conftest import FakeProvider
from tests.tasks.test_scripts_triggers_retry import FakeSandbox, _events, _script_task

PAGE_V1 = "Pune Library News\n- Reading room reopens on Monday\n- Book sale on 20 October"
PAGE_V2 = PAGE_V1 + "\n- Author talk on 25 October"


def _page(text: str) -> dict:
    return {"ok": True, "result": text}


async def _check(app, task_id, times=1):
    for _ in range(times):
        await app.tasks.run_now(task_id)
        await app.tasks.drain()
    return await app.tasks.get(task_id)


async def test_unchanged_source_costs_no_model_call_and_stays_quiet(make_app):
    llm = FakeProvider()
    app = await make_app(llm)
    app.sandbox.run = FakeSandbox([_page(PAGE_V1)] * 4)
    task_id = await _script_task(app, condition="noteworthy", status="active", name="Library news")
    task = await _check(app, task_id, times=4)
    assert llm.calls == []  # the first check is the baseline, the next three are the same: no judge
    assert _events(await app.notifications.list(), task_id) == []
    checks = task["script"]["checks"]
    assert [c["outcome"] for c in checks] == ["baseline", "nothing_new"]
    assert checks[1]["count"] == 3 and checks[1]["last_at"]  # three quiet checks fold into one entry


async def test_a_change_that_matters_sends_one_notification_naming_it(make_app):
    verdict = {"noteworthy": True, "change": "A new item: Author talk on 25 October."}
    llm = FakeProvider(json_replies=[verdict])
    app = await make_app(llm)
    app.sandbox.run = FakeSandbox([_page(PAGE_V1), _page(PAGE_V2), _page(PAGE_V2)])
    task_id = await _script_task(app, condition="noteworthy", status="active", name="Library news")
    task = await _check(app, task_id, times=3)
    judge = [c for c in llm.calls if c.get("json")]
    assert len(judge) == 1 and judge[0]["role"] == "fast"
    assert "+ - Author talk on 25 October" in judge[0]["messages"][1]["content"]  # only the changed line
    assert "Reading room" not in judge[0]["messages"][1]["content"]
    notes = [n for n in await app.notifications.list() if n.get("task_id") == task_id]
    assert [n["payload"]["event"] for n in notes] == ["script_alert"]
    assert notes[0]["message"] == "A new item: Author talk on 25 October." and notes[0]["title"] == "Library news"
    assert [c["outcome"] for c in task["script"]["checks"]] == ["baseline", "changed", "nothing_new"]
    assert task["script"]["checks"][1]["note"] == "A new item: Author talk on 25 October."


async def test_a_change_the_judge_calls_minor_is_recorded_but_not_sent(make_app):
    llm = FakeProvider(json_replies=[{"noteworthy": False, "change": "Only the visitor counter went up."}])
    app = await make_app(llm)
    app.sandbox.run = FakeSandbox([_page(PAGE_V1 + "\nVisitors: 10"), _page(PAGE_V1 + "\nVisitors: 11")])
    task_id = await _script_task(app, condition="noteworthy", status="active")
    task = await _check(app, task_id, times=2)
    assert _events(await app.notifications.list(), task_id) == []
    assert task["script"]["checks"][-1] == {
        "at": task["script"]["last_run_at"], "outcome": "minor", "note": "Only the visitor counter went up.",
    }
    assert task["script"]["last_result"].endswith("Visitors: 11")  # the new value is the next baseline


async def test_a_judge_without_an_answer_still_reports_the_change(make_app):
    llm = FakeProvider(json_replies=[{"something": "else"}] * 3)
    app = await make_app(llm)
    app.sandbox.run = FakeSandbox([_page(PAGE_V1), _page(PAGE_V2)])
    task_id = await _script_task(app, condition="noteworthy", status="active", name="Library news")
    await _check(app, task_id, times=2)
    notes = [n for n in await app.notifications.list() if n.get("task_id") == task_id]
    assert len(notes) == 1 and notes[0]["message"].startswith("'Library news' noticed a change:")


async def test_noteworthy_then_run_starts_a_run_only_when_it_matters(make_app):
    llm = FakeProvider(
        replies=["Wrote up the author talk."],
        json_replies=[{"noteworthy": True, "change": "Author talk added."},
                      {"summary": "Done.", "links_created": [], "links_found": [], "files_created": [], "tools_used": []}],
    )
    app = await make_app(llm)
    app.sandbox.run = FakeSandbox([_page(PAGE_V1), _page(PAGE_V1), _page(PAGE_V2)])
    task_id = await _script_task(app, condition="noteworthy", then="run", status="active")
    task = await _check(app, task_id, times=3)
    assert len(task["runs"]) == 1
    assert task["runs"][0]["trigger_event_data"]["message"] == "Author talk added."


async def test_other_conditions_keep_a_history_too(make_app):
    app = await make_app(FakeProvider())
    app.sandbox.run = FakeSandbox([
        {"ok": True, "result": {"alert": False}}, {"ok": True, "result": {"alert": False}},
        {"ok": True, "result": {"alert": True, "message": "Price dropped to $90"}},
        {"ok": False, "error": "NameError: x"},
    ])
    task_id = await _script_task(app, status="active")
    task = await _check(app, task_id, times=4)
    assert [(c["outcome"], c.get("count", 1)) for c in task["script"]["checks"]] == [
        ("quiet", 2), ("alert", 1), ("failed", 1),
    ]
    assert task["script"]["checks"][1]["note"] == "Price dropped to $90"


def test_history_is_bounded_and_switching_condition_keeps_baseline():
    checks: list = []
    for i in range(scripts.MAX_CHECKS + 10):
        checks = scripts.add_check(checks, "changed", f"t{i}", "x")
    assert len(checks) == scripts.MAX_CHECKS and checks[-1]["at"] == f"t{scripts.MAX_CHECKS + 9}"
    previous = {**normalize_script({"code": "print(1)", "condition": "changed"}), "last_result": "v", "checks": checks}
    switched = normalize_script({"condition": "noteworthy"}, previous)
    assert switched["last_result"] == "v" and switched["checks"] == checks
    other = normalize_script({"condition": "every_run"}, previous)
    assert other["last_result"] is None and other["checks"] == checks  # a new kind of check starts over
    assert scripts.describe_for_approval(switched).startswith("Runs a small check script with no AI calls. When")


def test_diff_and_verdict_helpers():
    assert scripts.diff_text({"price": 100, "title": "Desk"}, {"price": 90, "title": "Desk"}) == (
        '- "price": 100,\n+ "price": 90,'
    )
    assert scripts.read_verdict({"noteworthy": "yes", "change": " New  item "}) == (True, "New item")
    assert scripts.read_verdict("nonsense") == (None, "")


# ---------------------------------------------------------------------------- fuzzy windows
def test_fuzzy_windows_schedule_within_their_range():
    tz = "Asia/Kolkata"
    s = normalize_schedule({"type": "recurring", "frequency": "daily", "window": "Morning"}, tz)
    assert s["window"] == "morning"
    now = datetime(2026, 9, 15, 2, 0, tzinfo=UTC)  # 07:30 in Pune: today's morning window has started
    seen = set()
    for _ in range(200):
        local = calculate_next_run(s, now).astimezone(sched.get_tz(tz))
        assert local.date().isoformat() == "2026-09-16" and 7 <= local.hour < 10
        seen.add((local.hour, local.minute))
    assert len(seen) > 20  # it really varies inside the window
    early = datetime(2026, 9, 15, 0, 0, tzinfo=UTC)  # 05:30 in Pune: this morning still counts
    assert calculate_next_run(s, early).astimezone(sched.get_tz(tz)).date().isoformat() == "2026-09-15"
    weekly = normalize_schedule(
        {"type": "recurring", "frequency": "weekly", "days": ["Friday"], "window": "evening"}, tz,
    )
    nxt = calculate_next_run(weekly, now).astimezone(sched.get_tz(tz))
    assert nxt.strftime("%A") == "Friday" and 17 <= nxt.hour < 20
    assert nxt - now.astimezone(sched.get_tz(tz)) < timedelta(days=7)
    # an unknown window is dropped and the exact time is used
    plain = normalize_schedule({"type": "recurring", "frequency": "daily", "time": "08:15", "window": "brunch"}, tz)
    assert "window" not in plain
    assert calculate_next_run(plain, now).astimezone(sched.get_tz(tz)).strftime("%H:%M") == "08:15"
