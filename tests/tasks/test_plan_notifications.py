"""Plan-ready cards retire when the plan is decided anywhere; recovered runs still notify."""

from datetime import timedelta

from tests.conftest import FakeProvider
from tests.tasks.conftest import RESULT


async def _pending_task(app, *, run_at=None):
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task(
        {
            "name": "Write haiku",
            "description": "Write a haiku and save it",
            "status": "approval_pending",
            "schedule": {"type": "once", "run_at": run_at},
            "plan": [{"tool": "files", "description": "Save the haiku"}],
            "created_at": now,
            "updated_at": now,
        }
    )
    note = await app.notify(
        "task", "I've created a new plan for you: 'Write haiku'", title="Plan ready for approval",
        payload={"task_id": task_id, "event": "approval_needed"},
    )
    return task_id, note["id"]


async def test_decline_marks_plan_card_declined(make_app):
    app = await make_app(FakeProvider())
    task_id, note_id = await _pending_task(app)
    await app.tasks.decline(task_id)
    note = await app.notifications.get(note_id)
    assert note["payload"]["status"] == "declined"
    assert note["read"]


async def test_approve_marks_plan_card_approved(make_app):
    app = await make_app(FakeProvider())
    later = (app.tasks.now() + timedelta(days=1)).isoformat()
    task_id, note_id = await _pending_task(app, run_at=later)
    task = await app.tasks.approve(task_id)
    assert task["status"] == "pending"
    note = await app.notifications.get(note_id)
    assert note["payload"]["status"] == "approved"


async def test_startup_sweep_resolves_stale_plan_cards(make_app):
    app = await make_app(FakeProvider())
    task_id, note_id = await _pending_task(app)
    _still_waiting, waiting_note = await _pending_task(app)
    await app.tasks.repo.update_task(task_id, {"status": "completed"})
    await app.tasks.recover_interrupted()
    assert (await app.notifications.get(note_id))["payload"]["status"] == "approved"
    assert "status" not in (await app.notifications.get(waiting_note))["payload"]


async def test_recovered_run_sends_completion_notification(make_app):
    app = await make_app(FakeProvider(json_replies=[dict(RESULT)]))
    repo = app.tasks.repo
    now = app.tasks.now_iso()
    task_id = await repo.insert_task(
        {
            "name": "Write haiku",
            "status": "processing",
            "schedule": {"type": "once", "run_at": None},
            "plan": [{"tool": "files", "description": "Save the haiku"}],
            "created_at": now,
            "updated_at": now,
        }
    )
    run_id = await repo.insert_run(task_id, now=now)
    await repo.finish_run(run_id, "completed", error=None, now=now)
    await app.tasks.recover_interrupted()
    await app.tasks.drain()
    events = [n["payload"].get("event") for n in await app.notifications.list() if n.get("task_id") == task_id]
    assert "run_completed" in events
