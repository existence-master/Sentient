"""Pausing a task stops the run that is working right now (#408).

The run stops at its next safe point: before its next model call or tool call, never in the middle of a tool call. It
is held like the work a restart interrupts while the task is paused (#304): the run ends ``cancelled`` with
``Paused before it finished. Resume the task to continue.``, and resuming the task carries it on from its transcript
as a linked retry, without repeating finished steps. A run waiting on an approval is held at once. A one-call run whose
call already started (and a swarm's run) can't stop part-way: it finishes, and its log says so.
"""

from __future__ import annotations

import asyncio

from sentient.tasks import ask
from sentient.tasks.prompts import PAUSE_RESUMED_NOTE
from sentient.tasks.service import CALL_FINISHES_NOTE, PAUSED_NOTE
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider
from tests.tasks.conftest import RESULT, GatedProvider, stream_calls


def write(name: str, n: int) -> list[dict]:
    return [{"id": f"call_{n}", "name": "file_write", "arguments": {"name": name, "content": name.split(".")[0]}}]


def _contents(run: dict) -> list[str]:
    return [u["message"].get("content") for u in run["progress_updates"]]


def _calls(run: dict) -> list[str]:
    return [str((u["message"].get("parameters") or {}).get("name")) for u in run["progress_updates"]
            if u["message"].get("type") == "tool_call"]


async def _approved(app, *, name: str = "Save notes", **fields) -> str:
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": name, "description": "Save the notes to a.txt and b.txt", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None},
        "plan": [{"tool": "files", "description": "Save a.txt, then b.txt"}], "created_at": now, "updated_at": now,
        **fields,
    })
    await app.tasks.approve(task_id)
    return task_id


async def test_pausing_stops_the_working_run_before_its_next_step_and_resume_carries_it_on(make_app, isolated_home):
    # the second model call is in flight when the user pauses: it may answer, but its call to write b.txt never runs
    llm = GatedProvider(replies=[write("a.txt", 1), write("b.txt", 2), write("b.txt", 3), "Saved a.txt and b.txt."],
                        json_replies=[dict(RESULT)], block_on_call=2)
    app = await make_app(llm)
    task_id = await _approved(app)
    await asyncio.wait_for(llm.blocked.wait(), 5)

    task = await app.tasks.update(task_id, {"enabled": False})
    assert task["status"] == "processing" and not task["enabled"]  # it stops at the next safe point
    llm.gate.set()
    await app.tasks.drain()

    assert len(stream_calls(llm)) == 2  # no model call after the pause
    assert not (isolated_home / "files" / "b.txt").exists()  # and no tool call
    task = await app.tasks.get(task_id)
    assert task["status"] == "pending" and not task["enabled"] and task["error"] == PAUSED_NOTE
    assert task["next_execution_at"] is None and await app.tasks.tick() == []
    held = task["runs"][-1]
    assert held["status"] == "cancelled" and held["error"] == PAUSED_NOTE and PAUSED_NOTE in _contents(held)
    assert _calls(held) == ["a.txt"]

    await app.tasks.update(task_id, {"enabled": True})
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "completed" and task["error"] is None and len(task["runs"]) == 2
    carried = task["runs"][-1]
    assert carried["retry_of"] == held["run_id"] and carried["status"] == "completed"
    assert "The task was resumed. Carrying on from where it stopped." in _contents(carried)
    assert _calls(carried) == ["b.txt"]  # a.txt is not written again
    messages = stream_calls(llm)[2]["messages"]  # the carried-on run's transcript (the loop adds to it as it goes)
    before = messages[: messages.index({"role": "user", "content": PAUSE_RESUMED_NOTE})]
    # what it carries on from: a.txt's call and result; the b.txt call that never ran was not kept
    assert [m.get("tool_call_id") for m in before if m["role"] == "tool"] == ["call_1"]
    assert not any(m.get("tool_calls") and m["tool_calls"][0]["id"] == "call_2" for m in before)
    assert (isolated_home / "files" / "a.txt").read_text() == "a"
    assert (isolated_home / "files" / "b.txt").read_text() == "b"


async def test_resuming_before_the_safe_point_lets_the_run_go_on(make_app):
    llm = GatedProvider(replies=[write("a.txt", 1), "Saved a.txt."], json_replies=[dict(RESULT)], block_on_call=2)
    app = await make_app(llm)
    task_id = await _approved(app)
    await asyncio.wait_for(llm.blocked.wait(), 5)
    await app.tasks.update(task_id, {"enabled": False})
    await app.tasks.update(task_id, {"enabled": True})  # changed their mind while the model was still answering
    llm.gate.set()
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "completed" and len(task["runs"]) == 1 and task["error"] is None


def slow_plugin() -> tuple[ToolPlugin, asyncio.Event, asyncio.Event]:
    """A tool that waits in the middle of its call until ``release`` is set; ``started`` says it is in it."""
    started, release = asyncio.Event(), asyncio.Event()

    @tool("slow_save", risk=Risk.write)
    async def slow_save(ctx: ToolContext, name: str) -> dict:
        """Save something slowly."""
        started.set()
        await release.wait()
        return {"saved": name}

    class Slow(ToolPlugin):
        id = "slow"
        display_name = "Slow"
        tools = [slow_save]

    return Slow(), started, release


async def test_a_tool_call_in_progress_finishes_and_keeps_its_result(make_app):
    llm = FakeProvider(replies=[[{"id": "call_1", "name": "slow_save", "arguments": {"name": "a"}}], "Saved."],
                       json_replies=[dict(RESULT)])
    app = await make_app(llm)
    plugin, started, release = slow_plugin()
    app.registry.register(plugin)
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Save slowly", "description": "Save a", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None}, "plan": [{"tool": "slow", "description": "Save a"}],
        "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await asyncio.wait_for(started.wait(), 5)
    await app.tasks.update(task_id, {"enabled": False})  # in the middle of the call: it is never cut off
    release.set()
    await app.tasks.drain()

    assert len(stream_calls(llm)) == 1
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "cancelled" and run["error"] == PAUSED_NOTE
    results = [u["message"] for u in run["progress_updates"] if u["message"].get("type") == "tool_result"]
    assert results and results[0]["result"] == {"saved": "a"} and not results[0]["is_error"]
    saved = await app.tasks.repo.get_run(run["run_id"])  # the checkpoint it carries on from keeps the result
    assert any(m.get("role") == "tool" and "saved" in m.get("content", "") for m in saved["messages"])


async def test_a_one_step_action_that_already_started_finishes_and_says_so(make_app):
    llm = FakeProvider()
    app = await make_app(llm)
    plugin, started, release = slow_plugin()
    app.registry.register(plugin)
    now = app.tasks.now_iso()
    call = {"tool": "slow_save", "arguments": {"name": "a"}, "done_text": "Saved."}
    task_id = await app.tasks.repo.insert_task({
        "name": "Save it", "description": "Save it", "status": "pending",
        "schedule": {"type": "once", "run_at": None}, "plan": [{"tool": "slow", "description": "Save it"}],
        "original_context": {"source": "chat", "fixed_call": call}, "created_at": now, "updated_at": now,
    })
    await app.tasks.run_now(task_id)
    await asyncio.wait_for(started.wait(), 5)
    await app.tasks.update(task_id, {"enabled": False})
    release.set()
    await app.tasks.drain()

    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert run["status"] == "completed" and not task["enabled"]
    assert CALL_FINISHES_NOTE in _contents(run) and PAUSED_NOTE not in _contents(run)
    assert await app.tasks.tick() == []


async def test_a_run_waiting_on_an_approval_is_held_at_once_and_asks_again_on_resume(make_app, config, isolated_home):
    config.tools.approvals.rules = {"files": "ask"}
    llm = FakeProvider(replies=[write("a.txt", 1), write("a.txt", 2), "Saved a.txt."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id = await _approved(app)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    run_id = task["runs"][-1]["run_id"]
    assert task["status"] == "waiting_for_user" and task["runs"][-1]["pending_question"]["kind"] == "approval"

    await app.tasks.update(task_id, {"enabled": False})
    await app.tasks.drain()
    assert len(stream_calls(llm)) == 1
    task = await app.tasks.get(task_id)
    assert task["status"] == "pending" and task["error"] == PAUSED_NOTE
    run = await app.tasks.repo.get_run(run_id)
    assert run["status"] == "cancelled" and run["error"] == PAUSED_NOTE and run["pending_question"] is None
    assert any(m.get("role") == "tool" and ask.PAUSED_CALL_NOTE in m.get("content", "") for m in run["messages"])
    note = next(n for n in await app.notifications.list() if (n.get("payload") or {}).get("event") == "question")
    assert note["payload"]["status"] == "cancelled"

    await app.tasks.update(task_id, {"enabled": True})
    await app.tasks.drain()
    carried = (await app.tasks.get(task_id))["runs"][-1]
    assert carried["retry_of"] == run_id and carried["status"] == "waiting_for_user"
    assert carried["pending_question"]["kind"] == "approval" and not (isolated_home / "files" / "a.txt").exists()
    await app.tasks.answer_question(task_id, carried["run_id"], "Allow once")
    await app.tasks.drain()
    assert (await app.tasks.get(task_id))["status"] == "completed"
    assert (isolated_home / "files" / "a.txt").read_text() == "a"


async def test_a_question_to_the_user_keeps_waiting_when_paused(make_app):
    llm = FakeProvider(replies=[[{"id": "call_1", "name": "ask_user", "arguments": {"question": "Which file?"}}]])
    app = await make_app(llm)
    task_id = await _approved(app)
    await app.tasks.drain()
    await app.tasks.update(task_id, {"enabled": False})
    task = await app.tasks.get(task_id)
    assert task["status"] == "waiting_for_user" and task["runs"][-1]["status"] == "waiting_for_user"
