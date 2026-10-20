"""#211: an approval an API turn waits on reaches the paired chat while nobody is in the app, like a task's, and its
buttons there answer it."""

from __future__ import annotations

import asyncio

import pytest

from sentient.gateway.routes import openai_api
from tests.channels.conftest import callback, until
from tests.conftest import FakeProvider, tool_call

WRITE = [tool_call("file_write", name="a.txt", content="one")]


@pytest.fixture
def llm() -> FakeProvider:
    return FakeProvider(replies=[WRITE, "Saved it."])


def _approval_messages(tg, chat_id: int) -> list[dict]:
    return [p for p in tg.api.button_messages() if str(p["chat_id"]) == str(chat_id) and "Approval needed" in p["text"]]


async def _api_turn(tg) -> asyncio.Queue:
    tg.app.config.tools.approvals.rules = {"files": "ask"}
    tg.app.approvals.timeout_s = 20
    session_id = await openai_api._session(tg.app, "Notes app")
    out, _ = openai_api._start(tg.app, session_id, "save a note")
    await until(lambda: bool(_approval_messages(tg, 42)))
    return out


async def _finish(out: asyncio.Queue) -> list:
    events = []
    while (item := await asyncio.wait_for(out.get(), 30)) is not None:
        events.append(item)
    return events


async def _status(app) -> str | None:
    [note] = [n for n in await app.notifications.list(20) if n["kind"] == "approval"]
    return note["payload"].get("status")


async def test_allow_on_the_phone_lets_the_api_turn_carry_on(tg, isolated_home):
    await tg.pair(42)
    out = await _api_turn(tg)
    msg = _approval_messages(tg, 42)[-1]
    buttons = {b["text"]: b["callback_data"] for row in msg["reply_markup"]["inline_keyboard"] for b in row}
    assert "Allow" in buttons and "Deny" in buttons
    tg.ch.dispatch(callback(42, msg["_id"], buttons["Allow"]))
    await tg.ch.wait_idle()
    await _finish(out)
    assert (isolated_home / "files" / "a.txt").read_text() == "one"
    assert await _status(tg.app) == "allow"


async def test_deny_on_the_phone_means_not_done(tg, isolated_home):
    await tg.pair(42)
    out = await _api_turn(tg)
    msg = _approval_messages(tg, 42)[-1]
    deny = next(b["callback_data"] for row in msg["reply_markup"]["inline_keyboard"] for b in row if b["text"] == "Deny")
    tg.ch.dispatch(callback(42, msg["_id"], deny))
    await tg.ch.wait_idle()
    await _finish(out)
    assert not (isolated_home / "files" / "a.txt").exists()
    assert await _status(tg.app) == "deny"
