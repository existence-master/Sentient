"""A reply that fails in a messaging app ends cleanly: the error is said once and no status line is left (#334)."""

from __future__ import annotations

from sentient.llm.provider import ProviderError
from tests.conftest import FakeProvider, tool_call


class FailsAfterTool(FakeProvider):
    """Asks for one tool, then the model fails."""

    async def stream(self, role, messages, tools=None, *, model=None):
        if not self.calls:
            async for chunk in super().stream(role, messages, tools, model=model):
                yield chunk
            return
        self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
        raise ProviderError("The model stopped answering.")


async def test_a_model_error_after_a_tool_clears_the_status_line(tg):
    tg.app.agent.llm = FailsAfterTool([[tool_call("current_datetime")]])
    chat = await tg.pair(42)
    await tg.say(42, "what time is it")
    assert "<i>Checking the time...</i>" in [p["text"] for p in tg.api.sent("sendMessage")]
    screen = tg.api.screen()
    assert "<i>Checking the time...</i>" not in screen  # the done after the error cleared it
    assert sum("something went wrong" in t for t in screen) == 1
    assert "The model stopped answering." in screen[-1]
    assert not tg.ch.runtime("42").running
    rows = await tg.app.store.recent_messages(chat["session_id"], 10)
    assert rows[-1]["role"] != "assistant"  # nothing of the failed reply is saved
