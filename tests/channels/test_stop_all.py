"""`/stopall` and `/resume` from a paired chat (docs/API.md section 17)."""

from __future__ import annotations

import asyncio

import pytest

from tests.channels.conftest import GatedProvider, update


@pytest.fixture
def gated() -> GatedProvider:
    return GatedProvider(["first reply"], block_on_call=1)


async def test_stopall_from_a_paired_chat_stops_everything(tg, gated):
    tg.app.agent.llm = gated
    chat = await tg.pair(42)
    tg.ch.dispatch(update(42, "tell me a long story"))
    await asyncio.wait_for(gated.blocked.wait(), 5)

    await tg.say(42, "/stopall")
    assert tg.api.screen()[-1].startswith("Stopped everything (1 running job cancelled).")
    assert "/resume" in tg.api.screen()[-1]
    assert tg.app.stopped and tg.app.stop_state["source"] == "telegram"
    assert not tg.ch.runtime("42").running
    history = await tg.app.store.recent_messages(chat["session_id"], 10)
    assert history[-1]["role"] == "assistant" and "stopped" in history[-1]["content"]

    await tg.say(42, "/stop all")  # the spelled-out form works too, and stopping again is harmless
    assert tg.api.screen()[-1].startswith("Stopped everything.")

    await tg.say(42, "/resume")
    assert tg.api.screen()[-1].startswith("Resumed.") and not tg.app.stopped
    await tg.say(42, "/resume")
    assert "isn't stopped" in tg.api.screen()[-1]


async def test_stopall_from_an_unpaired_chat_is_ignored(tg):
    await tg.pair(42)
    await tg.say(99, "/stopall")
    assert not tg.app.stopped
    assert "isn't paired" in tg.api.screen()[-1]
    await tg.app.stop_all()
    await tg.say(99, "/resume")
    assert tg.app.stopped  # an unpaired chat cannot resume either


async def test_help_lists_stopall(tg):
    await tg.pair(42)
    await tg.say(42, "/help")
    assert "/stopall" in tg.api.screen()[-1] and "/resume" in tg.api.screen()[-1]
