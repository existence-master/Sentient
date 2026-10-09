"""Stop everything and tasks: running runs are cancelled, questions keep waiting, nothing starts until resume."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from sentient.tasks.service import STOPPED_NOTE
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import PLAN_TIME, REFINE_DAILY, RESULT, GatedProvider

ASK = [tool_call("ask_user", question="Which flight should I book?", options=["Morning", "Evening"])]


async def _insert(app, name: str, **fields) -> str:
    now = app.tasks.now_iso()
    return await app.tasks.repo.insert_task({
        "name": name, "description": name, "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None},
        "plan": [{"tool": "time", "description": "Check today's date"}],
        "created_at": now, "updated_at": now, **fields,
    })


async def test_stop_cancels_running_runs_and_keeps_questions_waiting(make_app):
    # call 1: the first task asks a question; call 2: the second task calls a tool; call 3 blocks (a long run)
    llm = GatedProvider(replies=[ASK, [tool_call("current_datetime")], "never"], block_on_call=3)
    app = await make_app(llm)
    waiting_id = await _insert(app, "Book a flight")
    await app.tasks.approve(waiting_id)
    await app.tasks.drain()
    assert (await app.tasks.get(waiting_id))["status"] == "waiting_for_user"

    long_id = await _insert(app, "Long report")
    await app.tasks.approve(long_id)
    await asyncio.wait_for(llm.blocked.wait(), 5)

    result = await asyncio.wait_for(app.stop_all(), 3)
    assert result["stopped"] and result["cancelled"] >= 1
    long_task = await app.tasks.get(long_id)
    run = long_task["runs"][-1]
    assert long_task["status"] == "cancelled" and run["status"] == "cancelled"
    assert STOPPED_NOTE in [u["message"].get("content") for u in run["progress_updates"]]
    assert not [t for t in app.tasks._runs.values() if not t.done()]

    waiting = await app.tasks.get(waiting_id)
    assert waiting["status"] == "waiting_for_user" and waiting["runs"][-1]["status"] == "waiting_for_user"
    assert [q["task_id"] for q in await app.tasks.waiting_questions()] == [waiting_id]

    # a cancelled run can be retried from where it stopped, once the user wants it
    llm.replies = ["Done."]
    llm.json_replies = [dict(RESULT)]
    await app.tasks.retry_run(long_id, run["run_id"])
    await app.tasks.drain()
    assert (await app.tasks.get(long_id))["status"] == "completed"


async def test_scheduled_task_waits_while_stopped_and_starts_after_resume(make_app, clock):
    llm = FakeProvider(replies=["Today is 2026-09-15."], json_replies=[REFINE_DAILY, PLAN_TIME, RESULT])
    app = await make_app(llm, clock=clock)
    created = await app.tasks.create_task("Every day at 9am tell me the date")
    await app.tasks.drain()
    task = await app.tasks.approve(created["task_id"])
    assert task["next_execution_at"] == "2026-09-15T03:30:00+00:00"

    await app.stop_all()
    clock.dt = datetime(2026, 9, 15, 3, 31, tzinfo=UTC)
    assert await app.tasks.tick() == []
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "active" and task["runs"] == []

    await app.resume()
    run_ids = await app.tasks.tick()
    assert len(run_ids) == 1  # the overdue run starts on the first tick after resume
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert [r["status"] for r in task["runs"]] == ["completed"]
    assert task["next_execution_at"] == "2026-09-16T03:30:00+00:00"


async def test_triggered_task_does_not_fire_while_stopped(make_app):
    llm = FakeProvider(replies=["Sorted."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id = await _insert(
        app, "Sort new orders", status="active",
        schedule={"type": "triggered", "source": "webhook", "event": "orders", "filter": {}},
    )
    await app.stop_all()
    assert await app.tasks.handle_event("webhook", "orders", {"id": "1"}) == []
    assert (await app.tasks.get(task_id))["runs"] == []
    await app.resume()
    assert len(await app.tasks.handle_event("webhook", "orders", {"id": "1"})) == 1  # not marked seen while stopped
    await app.tasks.drain()


async def test_stopped_engine_restarts_paused(make_app, clock):
    llm = FakeProvider(json_replies=[REFINE_DAILY, PLAN_TIME])
    app = await make_app(llm, db_name="paused.db", clock=clock)
    created = await app.tasks.create_task("Every day at 9am tell me the date")
    await app.tasks.drain()
    await app.tasks.approve(created["task_id"])
    await app.stop_all()
    await app.stop()

    clock.dt = datetime(2026, 9, 15, 3, 31, tzinfo=UTC)
    again = await make_app(FakeProvider(replies=["Today."], json_replies=[RESULT]), db_name="paused.db", clock=clock)
    assert again.stopped
    assert await again.tasks.tick() == []
    await again.resume()
    assert len(await again.tasks.tick()) == 1
    await again.tasks.drain()
