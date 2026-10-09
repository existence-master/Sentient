"""Stuck runs (issue #134): no progress, the same error again and again, or a step only the user can do.

A stuck run pauses as ``waiting_for_user`` with a plain reason and the options "Try again", "Skip this step" and
"Cancel", and the user gets a "<task> is stuck" notification. Nothing ends silently.
"""

from __future__ import annotations

import asyncio

from sentient.llm.provider import StreamChunk, ToolCall
from sentient.tasks import executor, stuck
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider
from tests.tasks.conftest import RESULT, GatedProvider, stream_calls


@tool("slow_lookup", risk=Risk.read)
async def slow_lookup(ctx: ToolContext) -> str:
    """Look something up (it hangs)."""
    await asyncio.sleep(30)
    return "late"


@tool("book_table", risk=Risk.write)
async def book_table(ctx: ToolContext, time: str) -> dict:
    """Book a table at the restaurant."""
    return {"error": "The booking server said no."}


@tool("open_site", risk=Risk.read)
async def open_site(ctx: ToolContext) -> dict:
    """Open the restaurant's site."""
    return {"error": "This looks like a password field.", "needs_user": "the page asks for your password"}


@tool("long_search", risk=Risk.read)
async def long_search(ctx: ToolContext) -> str:
    """Search every table (slow, but it reports progress)."""
    for i in range(8):
        await asyncio.sleep(0.2)
        await ctx.progress({"kind": "status", "text": f"Checked {i + 1} of 8"})
    return "found a table at 8pm"


class Restaurant(ToolPlugin):
    id = "restaurant"
    display_name = "Restaurant"
    tools = [slow_lookup, book_table, open_site, long_search]


class SlowStreamingProvider(FakeProvider):
    """A slow CPU model: thinks, then writes, one small chunk every 0.2 s, well past the no-activity limit."""

    async def stream(self, role, messages, tools=None, *, model=None):
        self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
        reply = self.replies.pop(0) if self.replies else "ok"
        for _ in range(5):
            await asyncio.sleep(0.2)
            yield StreamChunk(thinking="hmm ", model="fake")
        if isinstance(reply, list):
            yield StreamChunk(done=True, tool_calls=[ToolCall(**tc) for tc in reply], model="fake")
            return
        for word in reply.split():
            await asyncio.sleep(0.2)
            yield StreamChunk(text=word + " ", model="fake")
        yield StreamChunk(done=True, model="fake")


def call(name: str, n: int, **arguments) -> list[dict]:
    return [{"id": f"call_{n}", "name": name, "arguments": arguments}]


async def start_task(app) -> str:
    app.registry.register(Restaurant())
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Book the table", "description": "Book a table for two at 8pm", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None},
        "plan": [{"tool": "restaurant", "description": "Book the table"}],
        "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    return task_id


async def stuck_run(app, task_id: str, reason: str) -> str:
    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert task["status"] == "waiting_for_user" and run["status"] == "waiting_for_user", run["error"]
    question = run["pending_question"]
    assert question["kind"] == "stuck" and question["reason"] == reason
    assert question["question"] == f"I'm stuck: {reason}. What should I do?"
    assert question["options"] == ["Try again", "Skip this step", "Cancel"]
    [note] = [n for n in await app.notifications.list() if n["payload"].get("run_id") == run["run_id"]]
    assert note["title"] == "Book the table is stuck"
    assert note["message"] == f"Sentient is stuck on 'Book the table': {reason}. Open it to help or cancel."
    assert note["payload"]["event"] == "question" and note["payload"]["stuck"] is True
    return run["run_id"]


async def answer(app, task_id: str, run_id: str, text: str) -> dict:
    await app.tasks.answer_question(task_id, run_id, text)
    await app.tasks.drain()
    return await app.tasks.get(task_id)


async def test_a_hung_step_gets_stuck_and_try_again_carries_on(make_app, config):
    config.tasks.stuck_after_minutes = 0.01  # 0.6 s without any activity
    llm = FakeProvider(replies=[call("slow_lookup", 1), "Booked a table for 8pm."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id = await start_task(app)
    run_id = await stuck_run(app, task_id, "Restaurant hasn't responded for 0.01 minutes")
    assert len(stream_calls(llm)) == 1

    task = await answer(app, task_id, run_id, "Try again")
    assert task["status"] == "completed", task["runs"][-1]["error"]
    resumed = stream_calls(llm)[1]["messages"]
    assert not any(m.get("tool_calls") for m in resumed)  # the hung call was dropped, so it is simply made again
    assert resumed[-1]["role"] == "user" and "asked you to try again" in resumed[-1]["content"]
    assert "Restaurant hasn't responded" in resumed[-1]["content"]


async def test_a_slow_model_that_keeps_thinking_and_writing_is_never_stuck(make_app, config):
    config.tasks.stuck_after_minutes = 0.01  # 0.6 s; each model reply and the tool take 1.6 s or more
    llm = SlowStreamingProvider(
        replies=[call("long_search", 1), "Booked a table for two at 8pm tonight."], json_replies=[dict(RESULT)]
    )
    app = await make_app(llm)
    task_id = await start_task(app)
    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert task["status"] == "completed" and run["status"] == "completed", run["error"]
    assert not [n for n in await app.notifications.list() if n["payload"].get("stuck")]
    results = [u["message"] for u in run["progress_updates"] if u["message"]["type"] == "tool_result"]
    assert results and results[0]["result"] == "found a table at 8pm"  # the slow tool finished, it wasn't cut off


async def test_a_silent_model_gets_stuck_and_cancel_ends_the_run(make_app, config):
    config.tasks.stuck_after_minutes = 0.01
    llm = GatedProvider(block_on_call=1)
    app = await make_app(llm)
    task_id = await start_task(app)
    run_id = await stuck_run(app, task_id, "the AI model hasn't answered for 0.01 minutes")

    task = await answer(app, task_id, run_id, "cancel")
    run = task["runs"][-1]
    assert task["status"] == "cancelled" and run["status"] == "cancelled" and run["pending_question"] is None
    assert "Cancelled after getting stuck." in [u["message"].get("content") for u in run["progress_updates"]]
    assert llm.stream_calls == 1  # the model was asked once and never again


async def test_the_same_error_again_and_again_gets_stuck_and_skip_moves_on(make_app, config):
    config.tasks.stuck_after_repeated_errors = 3
    llm = FakeProvider(
        replies=[*(call("book_table", i, time=t) for i, t in enumerate(["20:00", "20:15", "20:30"])), "Skipped booking."],
        json_replies=[dict(RESULT)],
    )
    app = await make_app(llm)
    task_id = await start_task(app)
    run_id = await stuck_run(
        app, task_id, "Restaurant keeps failing with the same error: The booking server said no"
    )
    assert len(stream_calls(llm)) == 3

    task = await answer(app, task_id, run_id, "Skip this step")
    assert task["status"] == "completed", task["runs"][-1]["error"]
    last = stream_calls(llm)[3]["messages"][-1]
    assert last["role"] == "user" and "skip that step" in last["content"]


async def test_a_step_only_the_user_can_do_gets_stuck_at_once(make_app):
    llm = FakeProvider(replies=[call("open_site", 1), "Signed in and booked."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id = await start_task(app)
    run_id = await stuck_run(app, task_id, "the page asks for your password")
    assert len(stream_calls(llm)) == 1

    task = await answer(app, task_id, run_id, "I signed in myself, carry on")
    assert task["status"] == "completed", task["runs"][-1]["error"]
    last = stream_calls(llm)[1]["messages"][-1]
    assert "The user replied: I signed in myself, carry on" in last["content"]


async def test_a_crash_inside_sentient_still_tells_the_user(make_app, monkeypatch):
    def boom(run):
        raise RuntimeError("broken")

    monkeypatch.setattr(executor, "is_first_retry_attempt", boom)
    app = await make_app(FakeProvider())
    task_id = await start_task(app)
    task = await app.tasks.get(task_id)
    assert task["status"] == "error" and task["runs"][-1]["status"] == "error"
    [failed] = [n for n in await app.notifications.list() if n["payload"].get("event") == "run_failed"]
    assert "Something went wrong inside Sentient" in failed["message"]


async def test_a_working_run_shows_its_last_activity(make_app, monkeypatch):
    monkeypatch.setattr(executor, "HEARTBEAT_S", 0.0)
    beats: list[str] = []
    llm = FakeProvider(replies=["All booked for 8pm."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    original = app.tasks.heartbeat

    async def spy(task_id: str, run_id: str) -> None:
        beats.append(run_id)
        await original(task_id, run_id)

    app.tasks.heartbeat = spy  # type: ignore[method-assign]
    task_id = await start_task(app)
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert beats and set(beats) == {run["run_id"]}  # the model's streamed reply kept the run alive
    assert run["last_activity_at"] == run["progress_updates"][-1]["timestamp"]


def test_a_person_check_on_a_page_is_a_step_only_the_user_can_do():
    page = {"url": "https://example.com", "title": "Just a moment", "text": "Verify you are human by completing this."}
    assert stuck.blocked_reason("browser_snapshot", page) == "the site wants proof that you're a person (a CAPTCHA)"
    assert stuck.blocked_reason("web_fetch", page) is None  # only the browser's own pages
    assert stuck.blocked_reason("browser_snapshot", {"title": "Menu", "text": "Book a table"}) is None
