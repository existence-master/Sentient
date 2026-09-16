"""A task stopped between 'run completed' and 'task settled' must recover as completed, not error."""

from tests.conftest import FakeProvider
from tests.tasks.conftest import RESULT


async def test_recovery_keeps_completed_status_and_regenerates_result(make_app):
    llm = FakeProvider(json_replies=[dict(RESULT)])
    app = await make_app(llm)
    repo = app.tasks.repo
    now = app.tasks.now_iso()
    task_id = await repo.insert_task(
        {
            "name": "Write haiku",
            "description": "Write a haiku and save it",
            "status": "processing",
            "schedule": {"type": "once", "run_at": None},
            "plan": [{"tool": "files", "description": "Save the haiku"}],
            "created_at": now,
            "updated_at": now,
        }
    )
    run_id = await repo.insert_run(task_id, now=now)
    assert await repo.finish_run(run_id, "completed", error=None, now=now)

    report = await app.tasks.recover_interrupted()
    await app.tasks.drain()

    task = await app.tasks.get(task_id)
    assert task["status"] == "completed", report
    run = task["runs"][-1]
    assert run["status"] == "completed"
    assert run["result"] and run["result"]["summary"] == RESULT["summary"]


async def test_recovery_marks_claimed_task_without_runs_as_error(make_app):
    app = await make_app(FakeProvider())
    repo = app.tasks.repo
    now = app.tasks.now_iso()
    task_id = await repo.insert_task(
        {
            "name": "Claimed then crashed",
            "status": "processing",
            "schedule": {"type": "once", "run_at": None},
            "created_at": now,
            "updated_at": now,
        }
    )
    await app.tasks.recover_interrupted()
    task = await app.tasks.get(task_id)
    assert task["status"] == "error"
