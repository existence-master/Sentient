"""Stuck runs (issue #134): no progress, the same error again and again, or a step only the user can do.

A stuck run pauses as ``waiting_for_user`` with a plain reason and the options "Try again", "Skip this step" and
"Cancel", and the user gets a "<task> is stuck" notification. Nothing ends silently.
"""

from __future__ import annotations

import asyncio

from sentient.llm.provider import StreamChunk, ToolCall
from sentient.tasks import executor, stuck
from sentient.tools.base import Risk, Tool, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider
from tests.tasks.conftest import RESULT, GatedProvider, SkipClock, stream_calls


@tool("book_table", risk=Risk.write)
async def book_table(ctx: ToolContext, time: str) -> dict:
    """Book a table at the restaurant."""
    return {"error": "The booking server said no."}


@tool("open_site", risk=Risk.read)
async def open_site(ctx: ToolContext) -> dict:
    """Open the restaurant's site."""
    return {"error": "This looks like a password field.", "needs_user": "the page asks for your password"}


class Restaurant(ToolPlugin):
    id = "restaurant"
    display_name = "Restaurant"
    tools = [book_table, open_site]


# The no-activity tests skip time (``skip_clock``) instead of sleeping: 30 s without activity is stuck, and the slow
# model and tool report something every 10 s, so a busy machine would have to stall the loop for 20 s of real time
# to change the outcome.
STALL_MINUTES = 0.5
STEP_S = 10.0
# drain() bounds its wait on the loop's clock, which the slow-model test moves forward by about 250 s
DRAIN_S = 600.0


class SlowStreamingProvider(FakeProvider):
    """A slow CPU model: thinks, then writes, one small chunk every ``STEP_S``, well past the no-activity limit."""

    def __init__(self, clock: SkipClock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clock = clock

    async def stream(self, role, messages, tools=None, *, model=None):
        self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
        reply = self.replies.pop(0) if self.replies else "ok"
        for _ in range(5):
            await self.clock.sleep(STEP_S)
            yield StreamChunk(thinking="hmm ", model="fake")
        if isinstance(reply, list):
            yield StreamChunk(done=True, tool_calls=[ToolCall(**tc) for tc in reply], model="fake")
            return
        for word in reply.split():
            await self.clock.sleep(STEP_S)
            yield StreamChunk(text=word + " ", model="fake")
        yield StreamChunk(done=True, model="fake")


def call(name: str, n: int, **arguments) -> list[dict]:
    return [{"id": f"call_{n}", "name": name, "arguments": arguments}]


async def start_task(app, *tools: Tool) -> str:
    restaurant = Restaurant()
    restaurant.tools = [*restaurant.tools, *tools]
    app.registry.register(restaurant)
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Book the table", "description": "Book a table for two at 8pm", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None},
        "plan": [{"tool": "restaurant", "description": "Book the table"}],
        "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain(DRAIN_S)
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
    await app.tasks.drain(DRAIN_S)
    return await app.tasks.get(task_id)


async def test_a_hung_step_gets_stuck_and_try_again_carries_on(make_app, config, skip_clock):
    config.tasks.stuck_after_minutes = STALL_MINUTES

    @tool("slow_lookup", risk=Risk.read)
    async def slow_lookup(ctx: ToolContext) -> str:
        """Look something up (it hangs)."""
        await skip_clock.sleep(STALL_MINUTES * 60 + 1)  # no sign of life for longer than the limit
        await asyncio.Event().wait()
        return "late"

    llm = FakeProvider(replies=[call("slow_lookup", 1), "Booked a table for 8pm."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id = await start_task(app, slow_lookup)
    run_id = await stuck_run(app, task_id, "Restaurant hasn't responded for 0.5 minutes")
    assert len(stream_calls(llm)) == 1

    task = await answer(app, task_id, run_id, "Try again")
    assert task["status"] == "completed", task["runs"][-1]["error"]
    resumed = stream_calls(llm)[1]["messages"]
    assert not any(m.get("tool_calls") for m in resumed)  # the hung call was dropped, so it is simply made again
    assert resumed[-1]["role"] == "user" and "asked you to try again" in resumed[-1]["content"]
    assert "Restaurant hasn't responded" in resumed[-1]["content"]


async def test_a_slow_model_that_keeps_thinking_and_writing_is_never_stuck(make_app, config, skip_clock, monkeypatch):
    config.tasks.stuck_after_minutes = STALL_MINUTES  # 30 s; each model reply and the tool take 50 s or more
    seen = asyncio.Event()  # the run's stuck check saw an event
    see = stuck.Watch.see

    def watch_see(self, event) -> None:
        see(self, event)
        seen.set()

    monkeypatch.setattr(stuck.Watch, "see", watch_see)

    @tool("long_search", risk=Risk.read)
    async def long_search(ctx: ToolContext) -> str:
        """Search every table (slow, but it reports progress)."""
        for i in range(8):
            await skip_clock.sleep(STEP_S)
            seen.clear()
            await ctx.progress({"kind": "status", "text": f"Checked {i + 1} of 8"})
            # progress reaches the run long before 10 real seconds pass; with skipped time, wait until it has
            await seen.wait()
        return "found a table at 8pm"

    llm = SlowStreamingProvider(
        skip_clock, replies=[call("long_search", 1), "Booked a table for two at 8pm tonight."],
        json_replies=[dict(RESULT)],
    )
    app = await make_app(llm)
    task_id = await start_task(app, long_search)
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
