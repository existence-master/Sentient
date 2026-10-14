"""Undo and Send now buttons in a messaging app, and readable previews on its approval messages (#142)."""

from __future__ import annotations

from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.channels.conftest import callback, until, update
from tests.conftest import tool_call

SENT: list[str] = []


@tool("mail_send_email", risk=Risk.send)
async def mail_send_email(ctx: ToolContext, to: str, subject: str = "", body: str = "") -> dict:
    """Send an email."""
    SENT.append(to)
    return {"sent": True}


class Mail(ToolPlugin):
    id = "mail"
    display_name = "Mail"
    tools = [mail_send_email]


SEND = tool_call("mail_send_email", to="sam@example.com", subject="Lunch", body="Friday at 1?")


async def test_undo_button_stops_the_send(tg, llm):
    SENT.clear()
    tg.app.registry.register(Mail())
    tg.app.config.tools.approvals.undo_seconds = 30
    await tg.pair(42)
    llm.replies = [[SEND], "Okay, I didn't send it."]
    tg.ch.dispatch(update(42, "email sam@example.com about lunch"))
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    keyboard = msg["reply_markup"]["inline_keyboard"]
    assert [b["text"] for row in keyboard for b in row] == ["Undo", "Send now"]
    assert "sending in 30 s" in msg["text"] and "To: sam@example.com" in msg["text"] and "Friday at 1?" in msg["text"]
    tg.ch.dispatch(callback(42, msg["_id"], keyboard[0][0]["callback_data"]))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Undone. Nothing was sent."
    assert SENT == []
    assert tg.api.screen()[-1] == "Okay, I didn't send it."
    # pressing again: it is no longer waiting
    tg.ch.dispatch(callback(42, msg["_id"], keyboard[0][0]["callback_data"]))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "This is no longer waiting."


async def test_send_now_and_another_chat_cannot_press(tg, llm):
    SENT.clear()
    tg.app.registry.register(Mail())
    tg.app.config.tools.approvals.undo_seconds = 30
    await tg.pair(42)
    await tg.pair(43)
    llm.replies = [[SEND], "Sent."]
    tg.ch.dispatch(update(42, "email sam@example.com about lunch"))
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    send_now = msg["reply_markup"]["inline_keyboard"][0][1]["callback_data"]
    tg.ch.dispatch(callback(43, msg["_id"], send_now))
    await until(lambda: bool(tg.api.sent("answerCallbackQuery")))
    assert "wasn't sent to this chat" in tg.api.sent("answerCallbackQuery")[-1]["text"] and SENT == []
    tg.ch.dispatch(callback(42, msg["_id"], send_now))
    await tg.ch.wait_idle()
    assert SENT == ["sam@example.com"]
    assert tg.api.screen()[-1] == "Sent."


async def test_approval_message_shows_the_preview_not_raw_json(tg, llm):
    SENT.clear()
    tg.app.registry.register(Mail())
    tg.app.config.tools.approvals.mode = "always"
    await tg.pair(42)
    llm.replies = [[SEND], "Okay."]
    tg.ch.dispatch(update(42, "email sam@example.com about lunch"))
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    assert "Approval needed" in msg["text"] and "Subject: Lunch" in msg["text"] and '"subject"' not in msg["text"]
    deny = msg["reply_markup"]["inline_keyboard"][-1][0]["callback_data"]
    tg.ch.dispatch(callback(42, msg["_id"], deny))
    await tg.ch.wait_idle()
    assert SENT == []


async def test_a_send_that_fails_after_the_window_is_not_called_sent(tg, llm):
    SENT.clear()

    @tool("mail_send_email", risk=Risk.send)
    async def broken(ctx: ToolContext, to: str, subject: str = "", body: str = "") -> dict:
        """Send an email."""
        return {"error": "The mail server said no."}

    class Broken(ToolPlugin):
        id = "mail"
        display_name = "Mail"
        tools = [broken]

    tg.app.registry.register(Broken(), replace=True)
    tg.app.config.tools.approvals.undo_seconds = 1  # the window runs out on its own
    await tg.pair(42)
    llm.replies = [[SEND], "It failed."]
    await tg.say(42, "email sam@example.com about lunch")
    msg = tg.api.button_messages()[-1]
    edits = [p for p in tg.api.sent("editMessageText") if p["message_id"] == msg["_id"]]
    assert any("reported an error" in p["text"] for p in edits)
    assert not any("<b>Sent</b>" in p["text"] for p in edits)
