"""Summaries retain API message provenance after their source chat is trusted again."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sentient.app import SentientApp
from sentient.tools.rules import API_SOURCE
from tests.conftest import FakeProvider

TEXT = "The orchard contact list is ready."
SUMMARY = "I received a note that the orchard contact list is ready."


@pytest.mark.parametrize("channel", ["api", "desktop"])
async def test_pending_summaries_keep_the_origin_of_their_contributing_messages(config, isolated_home, monkeypatch, channel):
    llm = FakeProvider(replies=["Received the note."])
    llm.text_replies[:] = [SUMMARY]
    app = await SentientApp(
        config, llm=llm, db_path=isolated_home / "api-summary.db", enable_background=False,
    ).start()
    try:
        # Even an API-labelled chat can contain genuine desktop messages. Its channel alone must not taint them.
        source = await app.store.create_session(channel="api")
        async for _ in app.agent.run_turn(source, TEXT, channel=channel):
            pass
        if channel == "api":
            assert (await app.agent.trust_chat(source)).trusted
        assert (await app.store.get_session(source))["untrusted"] == ""

        now = datetime.now(UTC)
        await app.store.execute("UPDATE messages SET created_at = ? WHERE session_id = ?", (
            (now - timedelta(hours=2)).isoformat(), source,
        ))
        episodic = app.memory.episodic
        [created] = await episodic.summarize_pending(now=now, user_name="Maya")
        assert await episodic.pending_chunks(now) == []
        [listed] = await episodic.list()
        expected_source = API_SOURCE if channel == "api" else None
        assert created["untrusted"] == listed["untrusted"] == expected_source
        transcript = llm.text_calls[0]["messages"][1]["content"]
        if channel == "api":
            assert f"user (from {API_SOURCE}): {TEXT}" in transcript
        else:
            assert f"user: {TEXT}" in transcript and API_SOURCE not in transcript

        destination = await app.store.create_session(channel="desktop")
        ctx = app.agent.tool_context(destination, "desktop")
        lookup = app.registry.get("history_semantic_search")
        semantic = await lookup.call(ctx, {"query": "orchard contact list"})
        if channel == "api":
            assert SUMMARY not in str(semantic)
        else:
            assert semantic["summaries"][0]["summary"] == SUMMARY
        assert ctx.untrusted == ""

        async def no_embeddings(texts):
            raise RuntimeError("Exercise the keyword fallback.")

        monkeypatch.setattr(episodic, "_embed", no_embeddings)
        keyword = await episodic.search("orchard contact list")
        assert [item["content"] for item in keyword] == ([] if channel == "api" else [SUMMARY])
    finally:
        await app.stop()
