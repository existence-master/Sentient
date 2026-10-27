"""An MCP turn's approval (kind "approval", #443): held back while the user is in the app, delivered
to the phone when they leave if it is still waiting, and answerable only from a chat it was
delivered to (the ownership rule every other button here follows, #139/#337)."""

from __future__ import annotations

import asyncio

from tests.channels.conftest import callback, until

APPROVAL_ID = "a1" * 16  # broker ids are uuid hex; the button data must stay within Telegram's limit


def _na_messages(tg, chat_id: int) -> list[dict]:
    return [p for p in tg.api.button_messages()
            if str(p["chat_id"]) == str(chat_id) and "Approval needed" in p["text"]]


async def _raise_mcp_approval(app):
    fut = app.approvals.create(APPROVAL_ID, ["allow", "deny"])
    await app.notify(
        "approval", "Run code: print(1)", title="Approval needed",
        payload={"approval_id": APPROVAL_ID, "name": "run_snippet", "risk": "exec",
                 "arguments": {"code": "print(1)"}},
    )
    return fut


async def test_a_held_mcp_approval_reaches_the_phone_when_the_user_leaves(tg):
    await tg.pair(42)
    await tg.pair(43)
    tg.app.presence.set("desktop", True)
    fut = await _raise_mcp_approval(tg.app)
    await until(lambda: bool(tg.app.channels._held))  # held back while the user is looking at the app
    assert _na_messages(tg, 42) == []

    tg.app.presence.set("desktop", False)
    await until(lambda: bool(_na_messages(tg, 42)))
    msg = _na_messages(tg, 42)[-1]
    buttons = [b for row in msg["reply_markup"]["inline_keyboard"] for b in row]
    assert [b["callback_data"] for b in buttons] == [f"na:a:{APPROVAL_ID}", f"na:d:{APPROVAL_ID}"]
    assert all(len(b["callback_data"].encode()) <= 64 for b in buttons)

    tg.ch.dispatch(callback(43, msg["_id"], buttons[0]["callback_data"]))  # a chat it wasn't sent to
    await tg.ch.wait_idle()
    assert not fut.done()

    tg.ch.dispatch(callback(42, msg["_id"], buttons[0]["callback_data"]))
    await tg.ch.wait_idle()
    assert fut.done() and fut.result() == "allow"


async def test_a_held_mcp_approval_answered_in_the_app_is_not_delivered_later(tg):
    await tg.pair(42)
    tg.app.presence.set("desktop", True)
    fut = await _raise_mcp_approval(tg.app)
    await until(lambda: bool(tg.app.channels._held))
    assert tg.app.approvals.resolve(APPROVAL_ID, "deny")  # answered in the app while it was held
    tg.app.presence.set("desktop", False)
    await asyncio.sleep(0.3)  # give the held delivery its chance to (not) happen
    assert _na_messages(tg, 42) == []
    assert fut.result() == "deny"
