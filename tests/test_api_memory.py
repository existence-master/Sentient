"""API request text is outside content, even when it fully supports an extracted fact (#211)."""

from __future__ import annotations

import pytest

from sentient.app import SentientApp
from sentient.llm.events import ApprovalRequest, ToolResultEvent
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.rules import API_SOURCE
from tests.conftest import FakeProvider, tool_call

FACT = "Maya's sister Anya lives in Pune"
TEXT = "My sister Anya lives in Pune now."


def _save_decision() -> dict:
    analysis = {"topics": ["Relationships"], "memory_type": "long-term", "duration": None}
    # Both the held-fact analysis and active-fact ADD paths get valid scripted metadata.
    return {**analysis, "action": "ADD", "fact_id": None, "content": FACT, "analysis": analysis}


@pytest.fixture
async def memory_chat(config, isolated_home):
    config.memory.extract_after_turn = True
    llm = FakeProvider(replies=["Noted."], json_replies=[{"facts": [FACT]}, _save_decision()])
    app = await SentientApp(
        config, llm=llm, db_path=isolated_home / "api-memory.db", enable_background=False,
    ).start()
    app.fake = llm
    try:
        yield app
    finally:
        await app.stop()


@pytest.mark.parametrize("channel", ["api", "desktop"])
async def test_grounded_fact_requires_review_when_the_words_came_from_an_api(memory_chat, channel):
    app = memory_chat
    sid = await app.store.create_session(channel=channel)
    source = "another app (API)" if channel == "api" else "Mail"
    # An existing mark must not be bypassed by treating API text as the owner's own words.
    # A genuine desktop statement still qualifies for the existing grounded-fact exception.
    await app.store.execute("UPDATE sessions SET untrusted = ? WHERE id = ?", (source, sid))

    async for _ in app.agent.run_turn(sid, TEXT, channel=channel):
        pass
    await app.agent.drain()

    active = await app.memory.list_facts()
    pending = await app.memory.pending_facts()
    if channel == "api":
        assert active == []
        [held] = pending
        assert held["content"] == FACT and held["status"] == "pending"
        assert held["review"]["from"] == source
        assert held["review"]["session_id"] == sid
        assert FACT not in await app.agent.system_prompt("Where does Anya live?", "desktop")
    else:
        assert pending == []
        [kept] = active
        assert kept["content"] == FACT and kept["status"] == "active"


async def test_memory_tool_cannot_treat_api_text_as_the_owners_own_words(memory_chat):
    app = memory_chat
    app.config.memory.extract_after_turn = False
    app.fake.replies[:] = [[tool_call("memory_remember", fact=FACT)], "Saved for review."]
    app.fake.json_replies[:] = [_save_decision()]
    sid = await app.store.create_session(channel="api")

    events = [event async for event in app.agent.run_turn(sid, TEXT, channel="api")]

    saved = next(event for event in events if isinstance(event, ToolResultEvent) and event.name == "memory_remember")
    assert saved.result["status"] == "pending"
    assert await app.memory.list_facts() == []
    [held] = await app.memory.pending_facts()
    assert held["content"] == FACT and held["review"]["from"] == API_SOURCE


async def test_api_turn_memory_keeps_genuine_desktop_interjections(memory_chat):
    from sentient.tools.builtin.memory_tool import _own_words

    app = memory_chat
    sid = await app.store.create_session(channel="api")
    await app.store.add_message(sid, "user", "This belongs to an earlier turn.")
    await app.store.add_message(sid, "user", TEXT, name=API_SOURCE)
    await app.store.add_message(sid, "user", "I prefer green tea.", interjection=True)

    assert await _own_words(app.agent.tool_context(sid, "api")) == "I prefer green tea."


async def test_recalled_api_text_still_requires_approval_after_its_chat_is_trusted(config, isolated_home):
    calls: list[str] = []

    @tool("history_probe_send", risk=Risk.send, untrusted_output=False)
    async def history_probe_send(ctx: ToolContext, body: str) -> dict:
        """Send a note in the history probe."""
        calls.append(body)
        return {"sent": True}

    class Probe(ToolPlugin):
        id = "history_probe"
        display_name = "History probe"
        tools = [history_probe_send]

    config.tools.approvals.rules = {"history_probe_send": "allow"}
    llm = FakeProvider(replies=[
        "Received the note.",
        [tool_call("memory_search_history", query="orchard")],
        [tool_call("history_probe_send", body="orchard note")],
        "I did not send it.",
    ])
    app = await SentientApp(
        config, llm=llm, db_path=isolated_home / "api-history.db", enable_background=False,
    ).start()
    app.registry.register(Probe())
    try:
        source = await app.store.create_session(channel="api")
        async for _ in app.agent.run_turn(source, "The orchard contact list is ready.", channel="api"):
            pass
        assert (await app.agent.trust_chat(source)).trusted
        assert (await app.store.get_session(source))["untrusted"] == ""

        destination = await app.store.create_session(channel="desktop")
        asked, hits = [], []
        async for event in app.agent.run_turn(destination, "Find the earlier orchard note and send it."):
            if isinstance(event, ToolResultEvent) and event.name == "memory_search_history":
                hits = event.result
            elif isinstance(event, ApprovalRequest):
                assert calls == []
                asked.append(event)
                app.approvals.resolve(event.approval_id, "deny")

        assert [event.name for event in asked] == ["history_probe_send"]
        assert API_SOURCE in asked[0].untrusted
        assert calls == []
        [hit] = [hit for hit in hits if hit["session_id"] == source]
        assert hit["role"] == "user" and hit["from"] == API_SOURCE
        assert (await app.store.get_session(destination))["untrusted"] == API_SOURCE
    finally:
        await app.stop()
