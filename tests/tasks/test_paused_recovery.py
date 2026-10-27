"""Pausing a task holds across a restart (#304).

On start the engine resumes interrupted runs and plans tasks left planning. A paused task's work is held instead: its
interrupted run and a run waiting on its approval end ``cancelled`` with a plain note, a task left planning waits with
one, and nothing calls the model for it. Resuming the task carries the held run on from its checkpoint (or plans the
task), instead of starting a fresh run.
"""

from __future__ import annotations

import asyncio

from sentient.tasks import ask
from sentient.tasks.prompts import PAUSE_RESUMED_NOTE
from sentient.tasks.service import PAUSED_NOTE, PLAN_PAUSED_NOTE
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import PLAN_FILES, REFINE_ONCE, RESULT, GatedProvider, stream_calls

WRITE_A = [tool_call("file_write", name="a.txt", content="one")]


def _contents(run: dict) -> list[str]:
    return [u["message"].get("content") for u in run["progress_updates"]]


async def _paused_mid_run(make_app, db_name: str, *, schedule: dict | None = None) -> tuple[str, str]:
    """A task paused while its run is working, then Sentient quits mid-run."""
    llm = GatedProvider(replies=[WRITE_A, "unused"], block_on_call=2)
    app = await make_app(llm, db_name=db_name)
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Save notes", "description": "Save the notes", "status": "approval_pending",
        "schedule": schedule or {"type": "once", "run_at": None},
        "plan": [{"tool": "files", "description": "Save the notes to a.txt"}], "created_at": now, "updated_at": now,
    })
    if schedule:
        await app.tasks.run_now(task_id)
    else:
        await app.tasks.approve(task_id)
    await asyncio.wait_for(llm.blocked.wait(), 5)
    task = await app.tasks.update(task_id, {"enabled": False})
    assert task["status"] == "processing" and not task["enabled"]
    await app.stop()
    return task_id, task["runs"][-1]["run_id"]


async def test_a_paused_tasks_interrupted_run_is_held_then_carries_on_when_resumed(make_app, isolated_home):
    task_id, run_id = await _paused_mid_run(make_app, "held.db")
    llm = FakeProvider(replies=["Saved the notes to a.txt."], json_replies=[dict(RESULT)])
    app = await make_app(llm, db_name="held.db")

    report = await app.tasks.recover_interrupted()
    await app.tasks.drain()
    assert report["held"] == [run_id] and report["resumed"] == []
    assert llm.calls == []  # no model call for the paused task
    task = await app.tasks.get(task_id)
    assert task["status"] == "pending" and not task["enabled"] and task["error"] == PAUSED_NOTE
    run = task["runs"][-1]
    assert run["status"] == "cancelled" and run["error"] == PAUSED_NOTE and PAUSED_NOTE in _contents(run)
    assert await app.tasks.tick() == [] and llm.calls == []  # the scheduler leaves it alone too

    task = await app.tasks.update(task_id, {"enabled": True})
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "completed" and task["enabled"] and task["error"] is None
    assert len(task["runs"]) == 2  # the held run carried on; no fresh run was started besides it
    carried = task["runs"][-1]
    assert carried["retry_of"] == run_id and carried["status"] == "completed"
    assert "The task was resumed. Carrying on from where it stopped." in _contents(carried)
    messages = stream_calls(llm)[0]["messages"]
    assert messages[-1] == {"role": "user", "content": PAUSE_RESUMED_NOTE}
    assert any(m["role"] == "tool" and m.get("name") == "file_write" for m in messages)  # finished steps kept
    assert (isolated_home / "files" / "a.txt").read_text() == "one"

    await app.tasks.update(task_id, {"enabled": False})
    await app.tasks.update(task_id, {"enabled": True})  # a second Resume carries nothing on again
    await app.tasks.drain()
    assert len((await app.tasks.get(task_id))["runs"]) == 2


async def test_a_paused_routine_is_held_and_keeps_its_schedule(make_app):
    schedule = {"type": "recurring", "frequency": "daily", "time": "09:00"}
    task_id, run_id = await _paused_mid_run(make_app, "routine.db", schedule=schedule)
    llm = FakeProvider()
    app = await make_app(llm, db_name="routine.db")
    report = await app.tasks.recover_interrupted()
    await app.tasks.drain()
    assert report["held"] == [run_id] and llm.calls == []
    task = await app.tasks.get(task_id)
    assert task["status"] == "active" and task["next_execution_at"] and task["error"] == PAUSED_NOTE


async def test_an_approval_a_paused_task_waits_on_is_held_and_asked_again_on_resume(make_app, config, isolated_home):
    config.tools.approvals.rules = {"files": "ask"}
    app = await make_app(FakeProvider(replies=[WRITE_A]), db_name="approval.db")
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Save notes", "description": "Save the notes", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None},
        "plan": [{"tool": "files", "description": "Save the notes to a.txt"}], "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    run_id = task["runs"][-1]["run_id"]
    assert task["status"] == "waiting_for_user" and task["runs"][-1]["pending_question"]["kind"] == "approval"
    # paused by a version before #408 (which holds the approval at once): the approval was left waiting
    await app.tasks.repo.update_task(task_id, {"enabled": False})
    await app.stop()

    llm = FakeProvider(replies=[WRITE_A, "Saved the notes."], json_replies=[dict(RESULT)])
    app = await make_app(llm, db_name="approval.db")
    report = await app.tasks.recover_interrupted()
    await app.tasks.drain()
    assert report["held"] == [run_id] and llm.calls == []
    task = await app.tasks.get(task_id)
    assert task["status"] == "pending" and task["error"] == PAUSED_NOTE
    run = await app.tasks.repo.get_run(run_id)
    assert run["status"] == "cancelled" and run["pending_question"] is None
    assert any(m.get("role") == "tool" and ask.PAUSED_CALL_NOTE in m.get("content", "") for m in run["messages"])
    note = next(n for n in await app.notifications.list() if (n.get("payload") or {}).get("event") == "question")
    assert note["payload"]["status"] == "cancelled"

    await app.tasks.update(task_id, {"enabled": True})
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    carried = task["runs"][-1]
    assert carried["retry_of"] == run_id and carried["status"] == "waiting_for_user"  # asks again, nothing written
    assert carried["pending_question"]["kind"] == "approval"
    assert not (isolated_home / "files" / "a.txt").exists()
    await app.tasks.answer_question(task_id, carried["run_id"], "Allow once")
    await app.tasks.drain()
    assert (await app.tasks.get(task_id))["status"] == "completed"
    assert (isolated_home / "files" / "a.txt").read_text() == "one"


async def test_a_question_a_paused_task_asked_keeps_waiting(make_app):
    app = await make_app(FakeProvider(replies=[[tool_call("ask_user", question="Which file?")]]), db_name="ask.db")
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Save notes", "description": "Save the notes", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None},
        "plan": [{"tool": "files", "description": "Save the notes"}], "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    await app.tasks.update(task_id, {"enabled": False})
    report = await app.tasks.recover_interrupted()
    assert report["held"] == []
    assert (await app.tasks.get(task_id))["status"] == "waiting_for_user"


async def test_a_paused_task_left_planning_is_planned_only_when_resumed(make_app):
    llm = FakeProvider(json_replies=[REFINE_ONCE, PLAN_FILES])
    app = await make_app(llm)
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Write haiku", "description": "Write a haiku and save it", "status": "planning", "enabled": False,
        "original_context": {"source": "manual_creation"}, "created_at": now, "updated_at": now,
    })
    report = await app.tasks.recover_interrupted()
    await app.tasks.drain()
    assert report["held"] == [task_id] and report["replanned"] == [] and llm.calls == []
    task = await app.tasks.get(task_id)
    assert task["status"] == "planning" and task["error"] == PLAN_PAUSED_NOTE

    await app.tasks.update(task_id, {"enabled": True})
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "approval_pending" and task["error"] is None and task["plan"]


async def test_a_run_started_since_replaces_the_held_one(make_app):
    task_id, run_id = await _paused_mid_run(make_app, "replaced.db")
    llm = FakeProvider(replies=["Saved."], json_replies=[dict(RESULT)])
    app = await make_app(llm, db_name="replaced.db")
    await app.tasks.recover_interrupted()
    await app.tasks.run_now(task_id)  # the user starts it afresh instead of resuming
    await app.tasks.drain()
    assert len((await app.tasks.get(task_id))["runs"]) == 2

    await app.tasks.update(task_id, {"enabled": True})  # Resume later: the old held run is not carried on
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert len(task["runs"]) == 2 and not any(r["retry_of"] == run_id for r in task["runs"])


async def test_an_approved_call_that_already_started_is_never_held_for_a_repeat(make_app):
    """A paused one-call task (a follow-up's Send) whose call had started when Sentient stopped ends as before: it is
    not held, because carrying it on would make the call again."""
    llm = FakeProvider()
    app = await make_app(llm)
    now = app.tasks.now_iso()
    call = {"tool": "file_write", "arguments": {"name": "sent.txt", "content": "hi"}, "done_text": "Saved."}
    task_id = await app.tasks.repo.insert_task({
        "name": "Save it", "description": "Save it", "status": "processing", "enabled": False,
        "schedule": {"type": "once", "run_at": None}, "plan": [{"tool": "files", "description": "Save it"}],
        "original_context": {"source": "chat", "fixed_call": call}, "created_at": now, "updated_at": now,
    })
    run_id = await app.tasks.repo.insert_run(task_id, now=now)
    await app.tasks.repo.add_event(run_id, {"type": "tool_call", "tool_name": "file_write"}, now)

    report = await app.tasks.recover_interrupted()
    await app.tasks.drain()
    assert report["held"] == [] and report["resumed"] == [run_id] and llm.calls == []
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "error" and "did not do it again" in run["error"]
