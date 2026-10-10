"""Task automation: source.items triggers, webhooks, task.run_finished, script jobs and run retry."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.tasks import scripts
from sentient.tasks.prompts import RETRY_NOTE
from sentient.tasks.schedule import calculate_next_run, normalize_schedule
from sentient.tasks.scripts import ScriptInvalid, normalize_script
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import REFINE_ONCE, RESULT, json_calls, stream_calls

ALERT_CODE = "from sentient_tools import tools, result\nresult({'alert': False})\n"
TIME_PLAN = [{"tool": "time", "description": "Get the current date"}]


class FakeSandbox:
    """Stands in for ``app.sandbox.run``: returns scripted SandboxResults and records calls."""

    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.calls: list[dict] = []

    async def __call__(self, code, **kwargs):
        self.calls.append({"code": code, **kwargs})
        out = self.outcomes.pop(0) if self.outcomes else {"ok": True, "result": {"alert": False}}
        return {"backend": "process", "stdout": "", "stderr": "", "files_created": [], "tool_calls": 0,
                "duration_ms": 3, "error": None, "result": None, **out}


async def _settle(app):
    await asyncio.sleep(0.05)  # bus -> tasks consumer hop
    await asyncio.wait_for(app.tasks._items.join(), 5)
    await app.tasks.drain()


async def _insert(app, **fields):
    now = app.tasks.now_iso()
    base = {"name": "Test task", "description": "Test task", "status": "active", "plan": TIME_PLAN,
            "created_at": now, "updated_at": now}
    return await app.tasks.repo.insert_task({**base, **fields})


async def _triggered(app, source, event="", filt=None, **fields):
    schedule = {"type": "triggered", "source": source, "event": event, "filter": filt or {}}
    return await _insert(app, schedule=schedule, **fields)


async def _script_task(app, *, condition="alert", then="notify", code=ALERT_CODE, schedule=None, **fields):
    schedule = schedule or {"type": "recurring", "frequency": "interval", "interval_minutes": 60, "timezone": "UTC"}
    script = normalize_script({"code": code, "condition": condition, "then": then})
    return await _insert(app, task_type="script", script=script, schedule=schedule,
                         **{"status": "approval_pending", "name": "Price watch", **fields})


def _events(notes, task_id):
    return [n["payload"].get("event") for n in notes if n.get("task_id") == task_id]


# ---------------------------------------------------------------------------- source.items
async def test_source_items_fire_triggered_tasks_for_every_origin_once(make_app):
    llm = FakeProvider(replies=["done"] * 6, json_replies=[dict(RESULT) for _ in range(6)])
    app = await make_app(llm)
    invoices = await _triggered(app, "gmail", "new_email", {"subject": {"$contains": "invoice"}})
    all_mail = await _triggered(app, "gmail", "new_email")
    calendar = await _triggered(app, "gcalendar", "new_event")
    email = {"id": "m1", "subject": "Invoice 7", "sender_email": "a@b.com"}
    other = {"id": "m2", "subject": "Hello"}
    for origin in ("poll", "feed"):  # the same item on two paths
        app.bus.publish("source.items", {"source": "gmail", "event": "new_email", "origin": origin, "items": [email, other]})
    app.bus.publish("source.items", {"source": "gcalendar", "event": "new_event", "origin": "feed",
                                     "items": [{"id": "e1", "summary": "Standup"}]})
    await _settle(app)

    inv = await app.tasks.get(invoices)
    assert [r["trigger_event_data"] for r in inv["runs"]] == [email]
    assert sorted(r["trigger_event_data"]["id"] for r in (await app.tasks.get(all_mail))["runs"]) == ["m1", "m2"]
    assert len((await app.tasks.get(calendar))["runs"]) == 1
    # a direct call (older proactivity path) for an item a task already handled is a no-op
    assert await app.tasks.handle_event("gmail", "new_email", email) == []
    # a task enabled later still gets new items
    assert len(await app.tasks.handle_source_items(
        {"source": "gmail", "event": "new_email", "origin": "poll", "items": [{"id": "m3", "subject": "invoice 8"}]}
    )) == 2
    await app.tasks.drain()


async def test_webhook_trigger_matches_hook_id_and_body_fields(make_app):
    llm = FakeProvider(replies=["Build failure noted."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    hook = await _triggered(app, "webhook", "hook123", {"status": "failed"})
    other_hook = await _triggered(app, "webhook", "hook999")
    items = [
        {"id": "w1", "name": "CI", "body": {"status": "failed", "build": 9041}, "received_at": "2026-09-15T10:00:00+00:00"},
        {"id": "w2", "name": "CI", "body": {"status": "passed", "build": 9042}, "received_at": "2026-09-15T10:01:00+00:00"},
    ]
    app.bus.publish("source.items", {"source": "webhook", "event": "hook123", "origin": "webhook", "items": items})
    await _settle(app)
    task = await app.tasks.get(hook)
    assert [r["trigger_event_data"]["id"] for r in task["runs"]] == ["w1"]
    assert task["runs"][0]["status"] == "completed" and task["status"] == "active"
    assert "9041" in stream_calls(llm)[0]["messages"][0]["content"]
    assert (await app.tasks.get(other_hook))["runs"] == []


# ---------------------------------------------------------------------------- task.run_finished
async def test_run_finished_payload_counts_tool_errors_and_skills(make_app):
    llm = FakeProvider(
        replies=[[tool_call("skill_view", name="haiku-writing")], [tool_call("no_such_tool")], "All done."],
        json_replies=[dict(RESULT)],
    )
    app = await make_app(llm)
    task_id = await _insert(app, status="approval_pending", schedule={"type": "once", "run_at": None})
    async with app.bus.subscribe() as q:
        await app.tasks.approve(task_id)
        await app.tasks.drain()
        events = []
        while not q.empty():
            events.append(q.get_nowait())
    finished = [e["data"] for e in events if e["type"] == "task.run_finished"]
    task = await app.tasks.get(task_id)
    run = task["runs"][0]
    errors = sum(1 for u in run["progress_updates"] if u["message"].get("type") == "tool_result" and u["message"].get("is_error"))
    assert errors >= 1
    assert finished == [{
        "task_id": task_id, "run_id": run["run_id"], "status": "completed",
        "tool_errors": errors, "skills_viewed": ["haiku-writing"],
    }]


# ---------------------------------------------------------------------------- script jobs
async def test_script_job_lifecycle_alert_errors_and_recovery(make_app, clock):
    llm = FakeProvider()
    app = await make_app(llm, clock=clock)
    sandbox = FakeSandbox([
        {"ok": True, "result": {"alert": False}},
        {"ok": True, "result": {"alert": True, "message": "Price dropped to $90"}},
        {"ok": False, "error": "NameError: name 'x' is not defined"},
        {"ok": False, "error": "NameError again"},
        {"ok": True, "result": {"alert": False}},
    ])
    app.sandbox.run = sandbox
    task_id = await _script_task(app)

    task = await app.tasks.approve(task_id)
    assert task["task_type"] == "script" and task["status"] == "active"
    assert task["next_execution_at"] == "2026-09-15T03:00:00+00:00"  # 02:00 + 60 minutes
    assert task["script"]["code"] == ALERT_CODE and task["script"]["last_result"] is None

    clock.dt = datetime(2026, 9, 15, 3, 1, tzinfo=UTC)
    assert await app.tasks.tick() == []  # script checks create no runs
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "active" and task["runs"] == []
    assert task["script"]["last_result"] == {"alert": False}
    assert task["script"]["last_run_at"] == "2026-09-15T03:01:00+00:00"
    assert task["next_execution_at"] == "2026-09-15T04:01:00+00:00"
    call = sandbox.calls[0]
    assert call["code"] == ALERT_CODE and call["channel"] == "task"
    assert "current_datetime" in call["allowed_tools"] and "file_write" not in call["allowed_tools"]
    assert _events(await app.notifications.list(), task_id) == []

    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    notes = await app.notifications.list()
    alert = next(n for n in notes if n["payload"].get("event") == "script_alert")
    assert alert["message"] == "Price dropped to $90" and alert["kind"] == "task" and alert["task_id"] == task_id

    for _ in range(2):  # two failures in a row: one notification
        await app.tasks.run_now(task_id)
        await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "active" and "NameError again" in task["script"]["last_error"]
    assert task["script"]["last_result"] == {"alert": True, "message": "Price dropped to $90"}
    assert _events(await app.notifications.list(), task_id).count("script_failed") == 1

    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["script"]["last_error"] is None
    events = _events(await app.notifications.list(), task_id)
    assert events.count("script_recovered") == 1 and events.count("script_alert") == 1
    assert llm.calls == []  # no model calls for script-only checks


async def test_script_job_changed_then_run(make_app):
    llm = FakeProvider(replies=["Logged the new price."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    app.sandbox.run = FakeSandbox([
        {"ok": True, "result": {"price": 100}},  # baseline
        {"ok": True, "result": {"price": 100}},  # same
        {"ok": True, "result": None, "stdout": "  "},  # empty: not a change, keeps the baseline
        {"ok": True, "result": {"price": 90}},  # changed
    ])
    task_id = await _script_task(app, condition="changed", then="run", status="active")
    for _ in range(3):
        await app.tasks.run_now(task_id)
        await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["runs"] == [] and task["script"]["last_result"] == {"price": 100}

    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "active" and len(task["runs"]) == 1
    assert task["script"]["last_result"] == {"price": 90}
    run = task["runs"][0]
    assert run["status"] == "completed"
    assert run["trigger_event_data"]["script_result"] == {"price": 90}
    assert "price" in stream_calls(llm)[0]["messages"][0]["content"]


async def test_every_run_reports_identical_output_and_changed_does_not(make_app):
    app = await make_app(FakeProvider())
    same = [{"ok": True, "result": None, "stdout": "3 new posts drafted\n"} for _ in range(3)]
    app.sandbox.run = FakeSandbox([*same, {"ok": True, "result": None, "stdout": ""}, *same])
    every = await _script_task(app, condition="every_run", status="active", name="Posting report")
    for _ in range(4):  # the fourth check prints nothing: no report
        await app.tasks.run_now(every)
        await app.tasks.drain()
    changed = await _script_task(app, condition="changed", status="active", name="Posting watch")
    for _ in range(3):
        await app.tasks.run_now(changed)
        await app.tasks.drain()
    notes = await app.notifications.list()
    reports = [n for n in notes if n.get("task_id") == every and n["payload"].get("event") == "script_alert"]
    assert len(reports) == 3 and {n["message"] for n in reports} == {"3 new posts drafted"}
    assert "script_alert" not in _events(notes, changed)  # the same output three times is never a change
    task = await app.tasks.get(every)
    assert task["status"] == "active" and task["script"]["condition"] == "every_run"
    assert scripts.describe_for_approval(task["script"]).startswith("Runs a small check script with no AI calls; after every")
    # switching an existing job between the two is a normal script edit
    switched = await app.tasks.update(changed, {"script": {"condition": "every_run"}})
    assert switched["script"]["condition"] == "every_run"
    with pytest.raises(ScriptInvalid):
        await app.tasks.update(changed, {"script": {"condition": "sometimes"}})


async def test_triggered_script_job_gets_the_event(make_app):
    app = await make_app(FakeProvider())
    sandbox = FakeSandbox([{"ok": True, "result": {"alert": True, "message": "Deploy failed"}}])
    app.sandbox.run = sandbox
    task_id = await _script_task(
        app, status="active", schedule={"type": "triggered", "source": "webhook", "event": "h1", "filter": {}},
    )
    assert await app.tasks.handle_event("webhook", "h1", {"id": "w9", "body": {"ok": False}}) == []
    await app.tasks.drain()
    code = sandbox.calls[0]["code"]
    assert code.startswith("import json as _sentient_json") and "w9" in code and code.endswith(ALERT_CODE)
    assert "script_alert" in _events(await app.notifications.list(), task_id)
    assert (await app.tasks.get(task_id))["status"] == "active"


async def test_script_job_without_sandbox_records_a_readable_error(make_app):
    app = await make_app(FakeProvider())
    task_id = await _script_task(app, status="active")
    app.sandbox.run = None
    await app.tasks.run_now(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert "not available" in task["script"]["last_error"] and task["status"] == "active"


PLAN_SCRIPT = {
    "name": "Bitcoin price alert",
    "description": "Alert when bitcoin drops below 50k",
    "plan": [{"tool": "internet_search", "description": "Check the bitcoin price"}],
    "script": {"code": "from sentient_tools import tools, result\nresult({'alert': False})", "condition": "alert",
               "then": "notify"},
    "schedule": {"type": "recurring", "frequency": "interval", "interval_minutes": 30},
    "clarifying_questions": [],
}


async def test_planner_produces_script_job(make_app):
    llm = FakeProvider(json_replies=[REFINE_ONCE, PLAN_SCRIPT])
    app = await make_app(llm)
    task = await app.tasks.create_task("Tell me when bitcoin drops below 50k")
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "approval_pending" and task["task_type"] == "script"
    assert task["script"]["code"] == PLAN_SCRIPT["script"]["code"]
    assert task["script"]["condition"] == "alert" and task["script"]["then"] == "notify"
    assert task["schedule"]["frequency"] == "interval" and task["schedule"]["interval_minutes"] == 30
    assert "sentient_tools" in json_calls(llm, "planner")[1]["messages"][0]["content"]
    assert "approval_needed" in _events(await app.notifications.list(), task["task_id"])


async def test_planner_script_with_invalid_code_is_fixed_or_rejected(make_app):
    bad = {**PLAN_SCRIPT, "script": {**PLAN_SCRIPT["script"], "code": "def broken(:\n"}}
    llm = FakeProvider(json_replies=[REFINE_ONCE, bad, PLAN_SCRIPT, REFINE_ONCE, bad, bad])
    app = await make_app(llm)
    fixed = await app.tasks.create_task("Tell me when bitcoin drops below 50k")
    await app.tasks.drain()
    fixed = await app.tasks.get(fixed["task_id"])
    assert fixed["status"] == "approval_pending" and fixed["script"]["code"] == PLAN_SCRIPT["script"]["code"]
    assert "cannot be used" in json_calls(llm, "planner")[2]["messages"][1]["content"]

    rejected = await app.tasks.create_task("Tell me when ether drops below 2k")
    await app.tasks.drain()
    rejected = await app.tasks.get(rejected["task_id"])
    assert rejected["status"] == "error" and "not valid Python" in rejected["error"]
    assert rejected["script"] is None


async def test_script_validation_and_update(make_app):
    app = await make_app(FakeProvider())
    with pytest.raises(ScriptInvalid):
        normalize_script({"code": "x = ("})
    with pytest.raises(ScriptInvalid):
        normalize_script({"code": "x = 1", "condition": "sometimes"})
    task_id = await _script_task(app, status="active")
    await app.tasks.repo.update_task(task_id, {"script": {**normalize_script({"code": ALERT_CODE}), "last_result": 5}})
    same = await app.tasks.update(task_id, {"script": {"then": "run"}})
    assert same["script"]["then"] == "run" and same["script"]["last_result"] == 5
    edited = await app.tasks.update(task_id, {"script": {"code": "x = 2"}})
    assert edited["script"]["last_result"] is None and edited["status"] == "active"  # user edits need no re-approval
    with pytest.raises(ValueError):
        await app.tasks.update(task_id, {"script": {"code": "x = ("}})
    plain = await app.tasks.update(task_id, {"script": None})
    assert plain["task_type"] == "single" and plain["script"] is None

    # chat tool edits of script code go back for approval
    script_id = await _script_task(app, status="active")
    ctx = app.agent.tool_context("s1", "desktop")
    res = await app.registry.get("update_task").call(ctx, {"task_id": script_id, "script_code": "x = 3"})
    assert res["status"] == "success" and res["task"]["status"] == "approval_pending"
    assert res["task"]["script"]["code"] == "x = 3"
    bad = await app.registry.get("update_task").call(ctx, {"task_id": script_id, "script_code": "x = ("})
    assert bad["status"] == "failure" and "not valid Python" in bad["error"]
    renamed = await app.registry.get("update_task").call(ctx, {"task_id": script_id, "name": "BTC watch"})
    assert renamed["task"]["name"] == "BTC watch"


def test_script_helpers_and_interval_schedule():
    assert scripts.outcome_value({"ok": True, "result": None, "stdout": "42\n"}) == "42"
    assert scripts.outcome_error({"ok": False, "stderr": "Traceback\nValueError: boom"}) == "The check script failed: ValueError: boom"
    assert scripts.should_act("changed", 2, None) is False
    assert scripts.should_act("changed", {"a": 1, "b": 2}, {"b": 2, "a": 1}) is False
    assert scripts.should_act("alert", {"alert": 1}, None) is True
    sched = normalize_schedule({"type": "recurring", "frequency": "hourly", "time": "09:00"}, "Asia/Kolkata")
    assert sched == {"type": "recurring", "frequency": "interval", "interval_minutes": 60, "timezone": "Asia/Kolkata"}
    assert normalize_schedule({"frequency": "interval", "interval_minutes": 1}, "UTC")["interval_minutes"] == 5
    now = datetime(2026, 9, 15, 10, 0, 30, tzinfo=UTC)
    assert calculate_next_run({"frequency": "interval", "interval_minutes": 30}, now) == datetime(2026, 9, 15, 10, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------- retry
async def test_retry_failed_run_continues_from_checkpoint(make_app):
    llm = FakeProvider(
        replies=[[tool_call("file_write", name="part1.txt", content="one")], "", "", "Finished part1.txt."],  # empty twice: the loop nudges once
        json_replies=[dict(RESULT)],
    )
    app = await make_app(llm)
    task_id = await _insert(app, status="approval_pending", schedule={"type": "once", "run_at": None},
                            plan=[{"tool": "files", "description": "Write part1.txt"}])
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    failed = task["runs"][0]
    assert task["status"] == "error" and failed["status"] == "error"
    failure = next(n for n in await app.notifications.list() if n["payload"].get("event") == "run_failed")
    assert "Retry" in failure["message"] and failure["payload"]["run_id"] == failed["run_id"]
    assert (await app.tasks.get(task_id))["runs"][0]["retry_of"] is None

    task = await app.tasks.retry_run(task_id, failed["run_id"])
    assert task["status"] == "processing"
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    retry = task["runs"][1]
    assert task["status"] == "completed" and retry["status"] == "completed" and retry["retry_of"] == failed["run_id"]
    messages = stream_calls(llm)[-1]["messages"]
    assert messages[-1]["content"].startswith(RETRY_NOTE.split("{error}")[0])
    assert any(m["role"] == "tool" and m.get("name") == "file_write" for m in messages)
    assert len(stream_calls(llm)) == 4  # file_write was not repeated (write, empty, empty after nudge, retry)
    contents = [u["message"].get("content") for u in retry["progress_updates"]]
    assert "Retrying the failed run, continuing from where it stopped." in contents

    from sentient.tasks.service import TaskConflict

    with pytest.raises(TaskConflict):
        await app.tasks.retry_run(task_id, retry["run_id"])


# ---------------------------------------------------------------------------- routes
@pytest.fixture
def script_client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider(json_replies=[REFINE_ONCE, PLAN_SCRIPT])
    core = SentientApp(config, llm=llm, db_path=isolated_home / "script-routes.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c, core


def _wait(client, task_id, statuses, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in statuses:
            return task
        time.sleep(0.05)
    raise AssertionError(f"task stuck in {task['status']}")


def test_script_routes(script_client):
    client, core = script_client
    sandbox = FakeSandbox([{"ok": True, "result": {"alert": True, "message": "Test alert"}}, {"ok": False, "error": "boom"}])
    core.sandbox.run = sandbox
    task_id = client.post("/api/tasks", json={"prompt": "Tell me when bitcoin drops below 50k"}).json()["task_id"]
    task = _wait(client, task_id, {"approval_pending"})
    assert task["task_type"] == "script" and task["script"]["last_result"] is None

    tested = client.post(f"/api/tasks/{task_id}/script/test")
    assert tested.status_code == 200
    body = tested.json()
    assert body["ok"] is True and body["result"] == {"alert": True, "message": "Test alert"}
    assert {"ok", "backend", "stdout", "stderr", "result", "files_created", "tool_calls", "duration_ms", "error"} <= set(body)
    assert client.get(f"/api/tasks/{task_id}").json()["script"]["last_result"] is None
    assert client.post(f"/api/tasks/{task_id}/script/test", json={"code": "print(1)"}).json()["ok"] is False
    assert sandbox.calls[-1]["code"] == "print(1)"
    assert client.post(f"/api/tasks/{task_id}/script/test", json={"code": "def (:"}).status_code == 400

    assert client.patch(f"/api/tasks/{task_id}", json={"script": {"code": "def (:"}}).status_code == 400
    patched = client.patch(f"/api/tasks/{task_id}", json={"script": {"condition": "changed"}}).json()
    assert patched["script"]["condition"] == "changed" and patched["script"]["code"] == PLAN_SCRIPT["script"]["code"]
    approved = client.post(f"/api/tasks/{task_id}/approve").json()
    assert approved["status"] == "active" and approved["next_execution_at"]

    plain = client.post("/api/tasks/nope/script/test")
    assert plain.status_code == 404
    runs_retry = client.post(f"/api/tasks/{task_id}/runs/nope/retry")
    assert runs_retry.status_code == 404


def test_interval_next_run_is_from_now():
    now = datetime(2026, 9, 15, 2, 0, tzinfo=UTC)
    assert calculate_next_run({"type": "recurring", "frequency": "interval", "interval_minutes": 60}, now) == now + timedelta(hours=1)
