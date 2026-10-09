"""An "ask" rule stops a task run, says which rule did it, and sends the usual failure notification (ADR 0016)."""

from __future__ import annotations

from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import RESULT, stream_calls

STOPPED = "Files is set to Ask, and tasks can't ask yet. Change it in Settings > Approvals & safety."


async def test_ask_rule_stops_a_task_run_with_a_plain_reason(make_app, config, isolated_home):
    config.tools.approvals.rules = {"files": "ask"}
    llm = FakeProvider(
        replies=[[tool_call("file_write", name="note.txt", content="hi")], "Saved note.txt."],
        json_replies=[dict(RESULT)],
    )
    app = await make_app(llm)
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Save a note", "description": "Save a note", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None}, "plan": [{"tool": "files", "description": "Save note.txt"}],
        "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()

    task = await app.tasks.get(task_id)
    run = task["runs"][0]
    assert task["status"] == "error" and run["status"] == "error"
    assert run["error"] == STOPPED
    assert not (isolated_home / "files" / "note.txt").exists()
    assert len(stream_calls(llm)) == 1  # the run stopped at the refused call
    failure = next(n for n in await app.notifications.list() if n["payload"].get("event") == "run_failed")
    assert failure["title"] == "Task failed" and STOPPED in failure["message"]
