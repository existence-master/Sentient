"""Loop breaker and budgets in ``Agent.run_loop`` (issue #133).

An 8B model once made 25 identical malformed calls in a row. The same tool with the same arguments getting the
same result ``tools.repeated_call_limit`` times stops the loop; chat nudges the model once first. Token and cost
budgets stop unattended loops before the next model call. Everything is deterministic: no model decides.
"""

from __future__ import annotations

from sentient.agent.loop import REPEAT_NUDGE, Budget, LoopResult
from sentient.app import SentientApp
from sentient.llm.events import Done
from sentient.llm.provider import StreamChunk, ToolCall
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider

NUDGE_START = REPEAT_NUDGE.split("{name}")[0]


def read(name: str, n: int = 0) -> list[dict]:
    return [{"id": f"call_{n}", "name": "file_read", "arguments": {"name": name}}]


class RecordingProvider(FakeProvider):
    """Keeps each call's message contents; optionally reports usage and a price on every reply."""

    def __init__(self, *args, model: str = "fake", tokens: int = 0, cost: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen: list[list[str]] = []
        self.model, self.tokens, self.cost = model, tokens, cost

    async def stream(self, role, messages, tools=None, *, model=None):
        self.seen.append([str(m.get("content")) for m in messages])
        reply = self.replies.pop(0) if self.replies else "ok"
        usage = {"prompt_tokens": self.tokens, "completion_tokens": 0} if self.tokens else {}
        if isinstance(reply, list):
            calls = [ToolCall(**tc) for tc in reply]
            yield StreamChunk(done=True, tool_calls=calls, usage=usage, model=self.model, cost=self.cost)
            return
        yield StreamChunk(text=reply, model=self.model)
        yield StreamChunk(done=True, usage=usage, model=self.model, cost=self.cost)


def nudges(llm: RecordingProvider) -> int:
    return sum(1 for contents in llm.seen if any(c.startswith(NUDGE_START) for c in contents))


async def chat(config, replies, *, plugin: ToolPlugin | None = None):
    llm = RecordingProvider(replies=replies)
    app = await SentientApp(config, llm=llm, enable_background=False).start()
    try:
        if plugin is not None:
            app.registry.register(plugin)
        sid = await app.store.create_session(channel="desktop")
        done = None
        async for ev in app.agent.run_turn(sid, "Read notes.txt for me"):
            if isinstance(ev, Done):
                done = ev
        rows = await app.store.recent_messages(sid, 50)
        return llm, done, rows
    finally:
        await app.stop()


# ---------------------------------------------------------------------- chat
async def test_chat_nudges_once_then_stops_with_a_plain_reply(config):
    llm, done, rows = await chat(config, [read("notes.txt", i) for i in range(5)])
    assert len(llm.seen) == 4  # three identical calls, the nudge, one more repeat, then the turn ends
    assert nudges(llm) == 1 and "file_read" in llm.seen[3][-1]
    assert done is not None and "kept repeating the same step (file_read" in done.content
    assert rows[-1]["role"] == "assistant" and rows[-1]["content"] == done.content


async def test_chat_nudge_lets_the_model_recover(config):
    replies = [*(read("notes.txt", i) for i in range(3)), "I couldn't find notes.txt in your files."]
    llm, done, _ = await chat(config, replies)
    assert nudges(llm) == 1
    assert done is not None and done.content == "I couldn't find notes.txt in your files."


async def test_different_arguments_do_not_trip(config):
    replies = [*(read(f"notes-{i}.txt", i) for i in range(5)), "None of those files exist."]
    llm, done, _ = await chat(config, replies)
    assert nudges(llm) == 0 and done is not None and done.content == "None of those files exist."


async def test_same_call_with_a_changing_result_does_not_trip(config):
    state = {"n": 0}

    @tool("poll_status", risk=Risk.read)
    async def poll_status(ctx: ToolContext) -> dict:
        """Check the status."""
        state["n"] += 1
        return {"status": f"step {state['n']}"}

    class Polls(ToolPlugin):
        id = "polls"
        display_name = "Polls"
        tools = [poll_status]

    replies = [*([{"id": f"c{i}", "name": "poll_status", "arguments": {}}] for i in range(5)), "It finished."]
    llm, done, _ = await chat(config, replies, plugin=Polls())
    assert state["n"] == 5 and nudges(llm) == 0 and done is not None and done.content == "It finished."


async def test_limit_zero_turns_the_breaker_off(config):
    config.tools.repeated_call_limit = 0
    llm, done, _ = await chat(config, [*(read("notes.txt", i) for i in range(5)), "Not found."])
    assert nudges(llm) == 0 and done is not None and done.content == "Not found."


# ---------------------------------------------------------------------- unattended loops
async def loop(config, llm, *, budget: Budget | None = None, max_rounds: int = 10) -> LoopResult:
    app = await SentientApp(config, llm=llm, enable_background=False).start()
    try:
        result = LoopResult()
        messages = [{"role": "system", "content": "Work."}, {"role": "user", "content": "Go."}]
        ctx = app.agent.tool_context(None, "task")
        async for _ in app.agent.run_loop(
            messages, ctx, result=result, role="executor", max_rounds=max_rounds, use_approvals=False,
            source="task", budget=budget,
        ):
            pass
        return result
    finally:
        await app.stop()


async def test_unattended_loop_stops_without_a_nudge(config):
    llm = RecordingProvider(replies=[read("notes.txt", i) for i in range(5)])
    result = await loop(config, llm)
    assert len(llm.seen) == 3 and nudges(llm) == 0
    assert result.stopped_by_repeat == result.error
    assert "file_read ran 3 times with the same details" in result.error


async def test_token_budget_stops_before_the_next_model_call(config):
    llm = RecordingProvider(replies=[read(f"n{i}.txt", i) for i in range(5)], tokens=400)
    result = await loop(config, llm, budget=Budget(max_tokens=1000))
    assert len(llm.seen) == 3  # 400, 800, 1200 tokens: no fourth call
    assert result.stopped_by_budget == result.error
    assert result.error.startswith("Stopped after using 1,200 tokens without finishing.")


async def test_cost_budget_counts_only_known_prices(config):
    llm = RecordingProvider(replies=[read(f"n{i}.txt", i) for i in range(5)], tokens=10, cost=0.4)
    result = await loop(config, llm, budget=Budget(max_cost_usd=1.0))
    assert len(llm.seen) == 3 and "about $1.20" in (result.error or "")

    unpriced = RecordingProvider(replies=[*(read(f"n{i}.txt", i) for i in range(3)), "Done."], tokens=10)
    result = await loop(config, unpriced, budget=Budget(max_cost_usd=0.01))
    assert result.error is None and result.text == "Done."


async def test_local_models_are_not_counted(config):
    llm = RecordingProvider(
        replies=[*(read(f"n{i}.txt", i) for i in range(3)), "Done."], model="ollama_chat/qwen3:8b", tokens=5000,
    )
    budget = Budget(max_tokens=100)
    result = await loop(config, llm, budget=budget)
    assert result.error is None and result.text == "Done." and budget.tokens == 0


async def test_step_limit_still_applies(config):
    llm = RecordingProvider(replies=[read(f"n{i}.txt", i) for i in range(5)])
    result = await loop(config, llm, max_rounds=2)
    assert result.hit_step_limit and result.error is None and len(llm.seen) == 2
