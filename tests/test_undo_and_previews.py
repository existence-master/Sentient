"""Previews, undo windows and reversible defaults (#142)."""

from __future__ import annotations

import asyncio

from sentient.agent import preview, reversible
from sentient.agent import undo as undo_rules
from sentient.app import SentientApp
from sentient.llm.events import ApprovalRequest, ToolResultEvent, UndoWindow
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call


def _apps(log: list[str]) -> list[ToolPlugin]:
    @tool("mail_send_email", risk=Risk.send)
    async def mail_send_email(ctx: ToolContext, to: str, subject: str = "", body: str = "") -> dict:
        """Send an email."""
        log.append(f"send:{to}")
        return {"sent": True}

    @tool("shop_buy", risk=Risk.send, describe_fn=lambda a, c: {"risk_label": "Purchase"})
    async def shop_buy(ctx: ToolContext, item: str, amount: str = "") -> dict:
        """Buy something (a purchase that is also posted)."""
        log.append(f"buy:{item}")
        return {"ok": True}

    @tool("chat_post_message", risk=Risk.send)
    async def chat_post_message(ctx: ToolContext, channel: str, text: str) -> dict:
        """Post a message in a channel."""
        log.append(f"post:{channel}")
        return {"ok": True}

    @tool("notes_list_notes", risk=Risk.read)
    async def notes_list_notes(ctx: ToolContext) -> dict:
        """List notes."""
        return {"notes": ["3"]}

    @tool("notes_delete_note", risk=Risk.send)
    async def notes_delete_note(ctx: ToolContext, note_id: str) -> dict:
        """Delete a note for good."""
        log.append(f"delete:{note_id}")
        return {"deleted": True}

    @tool("notes_trash_note", risk=Risk.send)
    async def notes_trash_note(ctx: ToolContext, note_id: str) -> dict:
        """Move a note to the trash."""
        log.append(f"trash:{note_id}")
        return {"trashed": True}

    @tool("notes_restore_note", risk=Risk.write)
    async def notes_restore_note(ctx: ToolContext, note_id: str) -> dict:
        """Bring a note back from the trash."""
        log.append(f"restore:{note_id}")
        return {"restored": True}

    class Mail(ToolPlugin):
        id = "mail"
        display_name = "Mail"
        tools = [mail_send_email, chat_post_message, shop_buy]

    class Notes(ToolPlugin):
        id = "notes"
        display_name = "Notes"
        tools = [notes_list_notes, notes_delete_note, notes_trash_note, notes_restore_note]

    return [Mail(), Notes()]


async def _start(config, isolated_home, llm, log: list[str], *, seconds: int = 1) -> SentientApp:
    config.tools.approvals.undo_seconds = seconds
    config.chat.tool_selection = "all"
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "undo.db", enable_background=False).start()
    for plugin in _apps(log):
        s.registry.register(plugin)
    return s


SEND = tool_call("mail_send_email", to="sam@example.com", subject="Lunch", body="Friday at 1?")


# ----------------------------------------------------------------------------- previews
def test_preview_of_an_email_names_the_recipients_subject_and_text():
    p = preview.build("gmail_send", {"to": "sam@example.com", "cc": "priya@example.com", "subject": "Lunch",
                                     "body": "Friday at 1?"})
    assert p == {
        "kind": "email", "title": "Email", "body": "Friday at 1?",
        "fields": [{"label": "To", "value": "sam@example.com"}, {"label": "Cc", "value": "priya@example.com"},
                   {"label": "Subject", "value": "Lunch"}],
    }
    assert preview.as_text(p) == "To: sam@example.com\nCc: priya@example.com\nSubject: Lunch\n\nFriday at 1?"


def test_preview_of_messages_posts_deletes_and_nested_calls():
    msg = preview.build("slack_post_message", {"channel": "#team", "text": "Shipped!"})
    assert msg["kind"] == "post" and msg["fields"] == [{"label": "Where", "value": "#team"}] and msg["body"] == "Shipped!"
    assert preview.build("whatsapp_send_message", {"to": "+91 98765 43210", "text": "hi"})["kind"] == "message"
    gone = preview.build("notes_delete_note", {"note_id": "3"})
    assert gone["title"] == "Delete for good" and gone["fields"] == [{"label": "Item", "value": "3"}]
    nested = {"tools": [{"tool_slug": "GMAIL_SEND_EMAIL", "arguments": {"recipient_email": "a@b.co", "body": "x"}}]}
    p = preview.build("COMPOSIO_MULTI_EXECUTE_TOOL", nested)
    assert p["kind"] == "email" and p["fields"] == [{"label": "To", "value": "a@b.co"}] and p["body"] == "x"
    assert preview.build("weather_now", {"city_id": 7}) is None  # a look-up has nothing to preview
    assert preview.build("shop_buy", {"item": "kettle", "amount": "$40"}, purchase=True)["kind"] == "purchase"


def test_only_emails_messages_and_posts_wait_never_purchases_or_code():
    log: list[str] = []
    mail, post, buy = _apps(log)[0].tools
    assert undo_rules.delayable(mail, mail.name, {}, Risk.send)
    assert undo_rules.delayable(post, post.name, {}, Risk.send)
    assert not undo_rules.delayable(buy, buy.name, {}, Risk.send, purchase=True)
    assert not undo_rules.delayable(mail, mail.name, {}, Risk.exec)
    assert not undo_rules.delayable(mail, mail.name, {}, Risk.read)
    assert undo_rules.attended("desktop", "chat", False) and undo_rules.attended("telegram", "chat", False)
    assert not undo_rules.attended("cli", "chat", False)
    assert not undo_rules.attended("desktop", "task", False) and not undo_rules.attended("desktop", "chat", True)


# ----------------------------------------------------------------------------- the undo window
async def _turn(s: SentientApp, sid: str, text: str, on_window=None, decision: str = "allow") -> list:
    events: list = []
    async for ev in s.agent.run_turn(sid, text, channel="desktop"):
        events.append(ev)
        if isinstance(ev, ApprovalRequest):
            s.approvals.resolve(ev.approval_id, decision)
        if isinstance(ev, UndoWindow) and on_window is not None:
            on_window(ev)
    return events


async def test_undo_within_the_window_sends_nothing(config, isolated_home):
    log: list[str] = []
    llm = FakeProvider(replies=[[SEND], "Okay, not sent."])
    s = await _start(config, isolated_home, llm, log, seconds=30)
    try:
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "email sam@example.com about lunch",
                             on_window=lambda ev: s.undo.respond(ev.undo_id, "undo"))
        window = next(e for e in events if isinstance(e, UndoWindow))
        assert window.seconds == 30 and window.preview["kind"] == "email"
        assert window.preview["fields"][0] == {"label": "To", "value": "sam@example.com"}
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result.result["undone"] is True and result.is_error
        assert log == []  # nothing reached the mailer
        assert s.undo.pending_ids() == []
        # the model reads that it was not sent
        assert "NOT SENT" in llm.calls[-1]["messages"][-1]["content"]
    finally:
        await s.stop()


async def test_a_send_goes_out_when_the_window_ends(config, isolated_home):
    log: list[str] = []
    s = await _start(config, isolated_home, FakeProvider(replies=[[SEND], "Sent."]), log, seconds=1)
    try:
        sid = await s.store.create_session(channel="desktop")
        seen: list[str] = []
        loop = asyncio.get_running_loop()
        started = loop.time()
        events = await _turn(s, sid, "email sam@example.com about lunch", on_window=lambda ev: seen.append(str(log)))
        assert seen == ["[]"]  # still waiting when the window opened
        assert log == ["send:sam@example.com"] and loop.time() - started >= 0.9
        kinds = [type(e).__name__ for e in events]
        assert kinds.index("UndoWindow") < kinds.index("ToolResultEvent")
    finally:
        await s.stop()


async def test_send_now_ends_the_window_early(config, isolated_home):
    log: list[str] = []
    s = await _start(config, isolated_home, FakeProvider(replies=[[SEND], "Sent."]), log, seconds=30)
    try:
        sid = await s.store.create_session(channel="desktop")
        loop = asyncio.get_running_loop()
        started = loop.time()
        await _turn(s, sid, "email sam@example.com", on_window=lambda ev: s.undo.respond(ev.undo_id, "send_now"))
        assert log == ["send:sam@example.com"] and loop.time() - started < 10
        assert not s.undo.respond("nope", "undo")  # nothing is waiting
    finally:
        await s.stop()


async def test_approval_comes_first_then_the_window_with_a_preview(config, isolated_home):
    log: list[str] = []
    s = await _start(config, isolated_home, FakeProvider(replies=[[SEND], "Okay."]), log, seconds=30)
    s.config.tools.approvals.mode = "ask"
    try:
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "email sam@example.com about lunch",
                             on_window=lambda ev: s.undo.respond(ev.undo_id, "undo"))
        kinds = [type(e).__name__ for e in events]
        assert kinds.index("ApprovalRequest") < kinds.index("UndoWindow")
        asked = next(e for e in events if isinstance(e, ApprovalRequest))
        assert asked.preview["body"] == "Friday at 1?" and asked.preview["title"] == "Email"
        assert log == []
        # a "no" never opens a window
        s.agent.llm.replies = [[SEND], "Okay."]
        events = await _turn(s, sid, "email sam@example.com again", decision="deny")
        assert not any(isinstance(e, UndoWindow) for e in events) and log == []
    finally:
        await s.stop()


async def test_no_window_for_purchases_other_channels_or_when_turned_off(config, isolated_home):
    log: list[str] = []
    buy = tool_call("shop_buy", item="kettle")
    s = await _start(config, isolated_home, FakeProvider(replies=[[buy], "Bought.", [SEND], "Sent.", [SEND], "Sent."]),
                     log, seconds=30)
    try:
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "buy the kettle")
        assert not any(isinstance(e, UndoWindow) for e in events) and log == ["buy:kettle"]
        cli = [e async for e in s.agent.run_turn(sid, "email sam@example.com", channel="cli")]
        assert not any(isinstance(e, UndoWindow) for e in cli) and log[-1] == "send:sam@example.com"
        s.config.tools.approvals.undo_seconds = 0
        events = await _turn(s, sid, "email sam@example.com once more")
        assert not any(isinstance(e, UndoWindow) for e in events) and len(log) == 3
    finally:
        await s.stop()


async def test_stopping_the_reply_during_the_window_sends_nothing(config, isolated_home):
    log: list[str] = []
    s = await _start(config, isolated_home, FakeProvider(replies=[[SEND], "Sent."]), log, seconds=30)
    try:
        sid = await s.store.create_session(channel="desktop")
        opened = asyncio.Event()

        async def run() -> None:
            async for ev in s.agent.run_turn(sid, "email sam@example.com", channel="desktop"):
                if isinstance(ev, UndoWindow):
                    opened.set()

        task = asyncio.create_task(run())
        await asyncio.wait_for(opened.wait(), 5)
        await s.agent.cancel_session(sid)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0.05)
        assert log == [] and s.undo.pending_ids() == []
    finally:
        await s.stop()


# ----------------------------------------------------------------------------- trash instead of delete
def test_trash_and_restore_twins_are_found_by_name():
    log: list[str] = []
    notes = _apps(log)[1]

    class Registry:
        def tools(self, include_hidden=False):
            return notes.tools

    by_name = {t.name: t for t in notes.tools}
    assert reversible.trash_twin(by_name["notes_delete_note"], Registry()) is by_name["notes_trash_note"]
    assert reversible.restore_twin(by_name["notes_trash_note"], Registry()) is by_name["notes_restore_note"]
    assert reversible.trash_twin(by_name["notes_trash_note"], Registry()) is None
    assert reversible.fit_arguments(by_name["notes_trash_note"], {"note_id": "3", "extra": 1}) == {"note_id": "3"}
    assert reversible.fit_arguments(by_name["notes_trash_note"], {}) is None
    assert reversible.asks_for_good("delete note 3 permanently") and reversible.asks_for_good("remove it for good")
    assert not reversible.asks_for_good("delete note 3")


async def test_a_delete_goes_to_the_trash_unless_the_user_says_for_good(config, isolated_home):
    log: list[str] = []
    gone = tool_call("notes_delete_note", note_id="3")
    llm = FakeProvider(replies=[[gone], "Moved it.", [gone], "Deleted."])
    s = await _start(config, isolated_home, llm, log)
    try:
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "delete note 3")
        call = next(e for e in events if type(e).__name__ == "ToolCallEvent")
        assert call.name == "notes_trash_note" and log == ["trash:3"]
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result.result["note"] == reversible.TRASH_NOTE
        rows = await s.store.recent_messages(sid, 10)
        assert any(r["role"] == "tool" and r["name"] == "notes_trash_note" for r in rows)
        await _turn(s, sid, "now delete note 3 permanently")
        assert log == ["trash:3", "delete:3"]
        # with prefer_trash off, a delete is a delete
        s.config.tools.approvals.prefer_trash = False
        s.agent.llm.replies = [[gone], "Deleted."]
        await _turn(s, sid, "delete note 3")
        assert log[-1] == "delete:3"
    finally:
        await s.stop()


async def test_the_activity_timeline_offers_restore_for_the_trash(config, isolated_home):
    log: list[str] = []
    gone = tool_call("notes_delete_note", note_id="3")
    s = await _start(config, isolated_home, FakeProvider(replies=[[gone], "Moved it.", [SEND], "Okay."]), log,
                     seconds=30)
    try:
        sid = await s.store.create_session(channel="desktop")
        await _turn(s, sid, "delete note 3")
        await _turn(s, sid, "email sam@example.com", on_window=lambda ev: s.undo.respond(ev.undo_id, "undo"))
        items = (await s.timeline.past(kinds=["chat"]))["items"]
        trashed = next(i for i in items if "trash" in i["title"].lower())
        assert trashed["undo"] == {"action": "restore", "label": "Restore"}
        held = next(i for i in items if i["status"] == "stopped")
        assert held["detail"] == "You pressed Undo, so nothing was sent." and held["undo"] is None
        assert await s.timeline.restore(trashed["id"]) == {"ok": True, "result": {"restored": True}}
        assert log[-1] == "restore:3"
        assert (await s.timeline.restore(held["id"]))["ok"] is False
        assert (await s.timeline.restore("run:nope"))["ok"] is False
    finally:
        await s.stop()


# ----------------------------------------------------------------------------- a rule changed during the window (#365)
async def _turn_changing_rule(s: SentientApp, sid: str, text: str, rule: str, decision: str = "allow") -> list:
    """A turn whose send waits in its undo window while the user sets the mailer's rule to ``rule`` in Settings."""
    events: list = []
    async for ev in s.agent.run_turn(sid, text, channel="desktop"):
        events.append(ev)
        if isinstance(ev, UndoWindow):
            s.approvals.config.rules = {"mail_send_email": rule}
        if isinstance(ev, ApprovalRequest):
            s.approvals.resolve(ev.approval_id, decision)
    return events


async def _start_unasked(config, isolated_home, llm, log):
    config.tools.approvals.mode = "off"  # the send runs without a question when it is planned
    config.tools.approvals.second_look_model = False
    return await _start(config, isolated_home, llm, log, seconds=1)


async def test_a_rule_set_to_ask_during_the_window_asks_before_sending(config, isolated_home):
    log: list[str] = []
    s = await _start_unasked(config, isolated_home, FakeProvider(replies=[[SEND], "Not sent.", [SEND], "Sent."]), log)
    try:
        sid = await s.store.create_session(channel="desktop")
        events = await _turn_changing_rule(s, sid, "email sam@example.com about lunch", "ask", decision="deny")
        kinds = [type(e).__name__ for e in events]
        assert kinds.index("UndoWindow") < kinds.index("ApprovalRequest") < kinds.index("ToolResultEvent")
        assert log == []
        assert next(e for e in events if isinstance(e, ToolResultEvent)).result["declined"] is True
        # a yes then sends it, once
        s.approvals.config.rules = {}
        events = await _turn_changing_rule(s, sid, "email sam@example.com about lunch", "ask", decision="allow")
        assert any(isinstance(e, ApprovalRequest) for e in events) and log == ["send:sam@example.com"]
    finally:
        await s.stop()


async def test_only_if_i_asked_set_during_the_window(config, isolated_home):
    log: list[str] = []
    llm = FakeProvider(replies=[[SEND], "Sent.", [SEND], "Not sent."])
    s = await _start_unasked(config, isolated_home, llm, log)
    try:
        sid = await s.store.create_session(channel="desktop")
        # the user's own words asked for this email: it goes out without a question
        events = await _turn_changing_rule(s, sid, "Email sam@example.com about lunch", "if_asked")
        assert not any(isinstance(e, ApprovalRequest) for e in events) and log == ["send:sam@example.com"]
        # they asked for something else: it asks first
        s.approvals.config.rules = {}
        events = await _turn_changing_rule(s, sid, "What's for lunch on Friday?", "if_asked", decision="deny")
        assert [e.name for e in events if isinstance(e, ApprovalRequest)] == ["mail_send_email"]
        assert log == ["send:sam@example.com"]
    finally:
        await s.stop()


async def test_a_yes_already_given_is_not_asked_again_and_an_unchanged_rule_does_not_ask(config, isolated_home):
    log: list[str] = []
    llm = FakeProvider(replies=[[SEND], "Sent.", [SEND], "Sent."])
    s = await _start(config, isolated_home, llm, log, seconds=1)
    s.config.tools.approvals.mode = "ask"
    s.config.tools.approvals.second_look_model = False
    try:
        sid = await s.store.create_session(channel="desktop")
        events = await _turn_changing_rule(s, sid, "email sam@example.com about lunch", "ask")
        assert sum(isinstance(e, ApprovalRequest) for e in events) == 1 and log == ["send:sam@example.com"]
        s.config.tools.approvals.mode = "off"
        s.approvals.config.rules = {"mail_send_email": "allow"}
        events = await _turn_changing_rule(s, sid, "email sam@example.com again", "allow")
        assert not any(isinstance(e, ApprovalRequest) for e in events) and len(log) == 2
    finally:
        await s.stop()
