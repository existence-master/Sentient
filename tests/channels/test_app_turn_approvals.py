"""Turns another app steers, an API turn (#211) and an MCP `ask_sentient` turn (#443), share one approval path: a
call that needs the user's OK raises exactly one ``kind: "approval"`` notification, the paired chat gets exactly one
Allow and Deny message for it, and either answer resolves the waiting turn."""

from __future__ import annotations

import asyncio

import pytest

from sentient.gateway.routes import openai_api
from tests.channels.conftest import callback, until
from tests.conftest import FakeProvider, tool_call

WRITE = [tool_call("file_write", name="a.txt", content="one")]


@pytest.fixture
def llm() -> FakeProvider:
    return FakeProvider(replies=[WRITE, "Done."])


def _approval_messages(tg, chat_id: int) -> list[dict]:
    return [p for p in tg.api.button_messages() if str(p["chat_id"]) == str(chat_id) and "Approval needed" in p["text"]]


async def _start(tg, source: str) -> asyncio.Task:
    """Start a turn from ``source`` whose one call needs the user's OK; the task ends when the turn does."""
    if source == "api":
        session_id = await openai_api._session(tg.app, "Notes app")
        out, _ = openai_api._start(tg.app, session_id, "save a note")

        async def drain() -> None:
            while await asyncio.wait_for(out.get(), 30) is not None:
                pass

        return asyncio.create_task(drain())
    # what POST /api/chat runs for the MCP server's ask_sentient tool (channel "mcp", the client's name)
    session_id = await tg.app.store.create_session(channel="mcp")

    async def turn() -> None:
        async for _ in tg.app.agent.run_turn(session_id, "save a note", channel="mcp", app_name="Test client"):
            pass

    return asyncio.create_task(turn())


@pytest.mark.parametrize("source", ["api", "mcp"])
@pytest.mark.parametrize("answer", ["allow", "deny"])
async def test_an_app_turn_asks_once_and_either_answer_resolves_it(tg, isolated_home, source, answer):
    await tg.pair(42)
    tg.app.config.tools.approvals.rules = {"files": "ask"}
    tg.app.approvals.timeout_s = 20
    async with tg.app.bus.subscribe() as events:
        turn = await _start(tg, source)
        await until(lambda: bool(_approval_messages(tg, 42)))
        msg = _approval_messages(tg, 42)[-1]
        buttons = {b["text"]: b["callback_data"] for row in msg["reply_markup"]["inline_keyboard"] for b in row}
        tg.ch.dispatch(callback(42, msg["_id"], buttons["Allow" if answer == "allow" else "Deny"]))
        await tg.ch.wait_idle()
        await asyncio.wait_for(turn, 30)
        published = [events.get_nowait() for _ in range(events.qsize())]

    raised = [e for e in published if e["type"] == "notification.new" and e["data"]["kind"] == "approval"]
    assert len(raised) == 1  # one notification, not one from the agent and another from the API route
    assert len(_approval_messages(tg, 42)) == 1  # and one Allow and Deny message on the phone
    [note] = [n for n in await tg.app.notifications.list(20) if n["kind"] == "approval"]
    assert note["payload"]["channel"] == source
    assert note["payload"]["status"] == answer
    assert tg.app.approvals.pending_ids() == []
    assert (isolated_home / "files" / "a.txt").exists() == (answer == "allow")
