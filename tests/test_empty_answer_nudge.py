"""Real qwen3:8b sometimes ended a round with only hidden thinking (empty reply) after a tool call.
The loop asks once for a reply instead of showing the user nothing."""

from sentient.agent.loop import EMPTY_ANSWER_NUDGE
from sentient.app import SentientApp
from sentient.llm.events import Done
from tests.conftest import FakeProvider


def tc(tool_name: str, **arguments):
    return {"id": f"call_{tool_name}", "name": tool_name, "arguments": arguments}


class RecordingProvider(FakeProvider):
    """Keeps a copy of each call's message contents (the loop later removes the nudge from its list)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen: list[list[str]] = []

    async def stream(self, role, messages, tools=None, *, model=None):
        self.seen.append([str(m.get("content")) for m in messages])
        async for chunk in super().stream(role, messages, tools, model=model):
            yield chunk


async def _turn(config, replies):
    llm = RecordingProvider(replies=replies)
    app = await SentientApp(config, llm=llm, enable_background=False).start()
    try:
        sid = await app.store.create_session(channel="desktop")
        done = None
        async for ev in app.agent.run_turn(sid, "What time is it?"):
            if isinstance(ev, Done):
                done = ev
        rows = await app.store.recent_messages(sid, 50)
        return llm, done, rows
    finally:
        await app.stop()


def _nudged_calls(llm: RecordingProvider) -> int:
    return sum(1 for contents in llm.seen if EMPTY_ANSWER_NUDGE in contents)


async def test_empty_reply_after_tool_is_nudged_once(config):
    llm, done, rows = await _turn(config, [[tc("current_datetime")], "", "It is 9:41."])
    assert done is not None and done.content == "It is 9:41."
    assert _nudged_calls(llm) == 1
    assert all(r.get("content") != EMPTY_ANSWER_NUDGE for r in rows)  # never persisted


async def test_nudge_happens_only_once(config):
    llm, done, _ = await _turn(config, [[tc("current_datetime")], "", ""])
    assert done is not None and done.content == ""
    assert _nudged_calls(llm) == 1
    assert len(llm.seen) == 3
