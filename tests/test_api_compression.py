"""Compression keeps API provenance after a user trusts the chat again."""

from __future__ import annotations

import pytest

from sentient.app import SentientApp
from sentient.tools.rules import API_SOURCE
from tests.conftest import FakeProvider

FACT = "Maya's sister Anya lives in Pune"


@pytest.mark.parametrize("channel", ["api", "desktop"])
async def test_compression_flush_preserves_the_source_of_trusted_chat_words(config, isolated_home, channel):
    config.memory.extract_after_turn = False
    config.memory.facts_top_k = 0
    config.chat.auto_title = False
    config.chat.tool_selection = "all"
    config.chat.compress_after_messages = 1000
    llm = FakeProvider(replies=["Received."] * 7)
    core = await SentientApp(
        config, llm=llm, db_path=isolated_home / "compression.db", enable_background=False,
    ).start()
    try:
        sid = await core.store.create_session(channel=channel)
        for index in range(7):
            async for _ in core.agent.run_turn(
                sid, f"My sister Anya lives in Pune now. Note number {index}.", channel=channel,
            ):
                pass
        # Finish the ordinary post-turn jobs before lowering the threshold, so only the explicit call compresses.
        await core.agent.drain()
        trusted = await core.agent.trust_chat(sid)
        assert trusted.trusted == (channel == "api")
        assert (await core.store.get_session(sid))["untrusted"] == ""
        user_rows = [row for row in await core.store.recent_messages(sid, 30) if row["role"] == "user"]
        assert len(user_rows) == 7
        assert all(row["name"] == (API_SOURCE if channel == "api" else None) for row in user_rows)

        config.chat.history_window = 2
        config.chat.compress_after_messages = 12
        config.memory.flush_enabled = True
        analysis = {"topics": ["Relationships"], "memory_type": "long-term", "duration": None}
        llm.json_replies[:] = [
            {"facts": [FACT]},
            {"action": "ADD", "fact_id": None, "content": FACT, "analysis": analysis, **analysis},
        ]
        llm.text_replies[:] = [FACT + "."]

        await core.agent._maybe_compress(sid)

        assert (await core.store.get_session(sid))["context_summary"]
        active = await core.memory.list_facts()
        pending = await core.memory.pending_facts()
        if channel == "api":
            assert active == []
            [held] = pending
            assert held["content"] == FACT and held["status"] == "pending"
            assert held["review"]["from"] == API_SOURCE
            assert held["review"]["session_id"] == sid
            assert API_SOURCE in llm.text_calls[-1]["messages"][1]["content"]
        else:
            assert pending == []
            [kept] = active
            assert kept["content"] == FACT and kept["status"] == "active"
    finally:
        await core.stop()


async def test_compression_flush_keeps_the_end_of_api_messages_after_assistant_turns(config, isolated_home):
    config.chat.history_window = 2
    config.chat.compress_after_messages = 12
    config.memory.flush_enabled = True
    config.memory.flush_max_chars = 500
    llm = FakeProvider(json_replies=[{"facts": []}])
    llm.text_replies[:] = ["Compressed notes from another app."]
    core = await SentientApp(
        config, llm=llm, db_path=isolated_home / "compression-boundary.db", enable_background=False,
    ).start()
    try:
        sid = await core.store.create_session(channel="api")
        for index in range(6):
            await core.store.add_message(sid, "assistant", "Acknowledged the preceding note.")
            await core.store.add_message(
                sid, "user", "The app supplied these details. " + "orchard " * 45 + f"API_TAIL_{index}", name=API_SOURCE,
            )
        # These recent rows stay outside the compressed transcript.
        await core.store.add_message(sid, "assistant", "Recent reply.")
        await core.store.add_message(sid, "user", "Recent API note.", name=API_SOURCE)
        await core.agent._mark_chat(sid, API_SOURCE)
        assert (await core.agent.trust_chat(sid)).trusted

        await core.agent._maybe_compress(sid)

        extraction = next(call for call in llm.calls if call.get("json"))
        # Check the actual prompt after FactMemory.flush_transcript applies its assistant-first shortening.
        prompt = extraction["messages"][1]["content"]
        assert "API_TAIL_5" in prompt
        assert API_SOURCE in prompt
    finally:
        await core.stop()
