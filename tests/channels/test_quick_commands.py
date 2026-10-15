"""Quick commands from a paired chat (#150): done without the model, and the timer goes off in that same chat."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from tests.channels.conftest import until


async def test_timer_from_telegram_rings_in_that_chat(tg, llm):
    await tg.pair(42)
    before = len(tg.api.sent("sendMessage"))
    await tg.say(42, "set a timer for 1 minute please")
    await until(lambda: len(tg.api.sent("sendMessage")) > before)
    assert "Timer set for 1 minute." in tg.api.sent("sendMessage")[-1]["text"]
    assert llm.calls == []  # no model call
    item = (await tg.app.reminders.list())[0]
    assert item["channel"] == "telegram"

    sent = len(tg.api.sent("sendMessage"))
    assert await tg.app.reminders.fire_due(datetime.now(UTC) + timedelta(minutes=2)) == 1
    await until(lambda: len(tg.api.sent("sendMessage")) > sent)
    out = tg.api.sent("sendMessage")[-1]
    assert str(out["chat_id"]) == "42" and "Timer done" in out["text"] and "1 minute timer is done" in out["text"]


async def test_reminder_set_on_the_desktop_stays_on_the_desktop(tg, monkeypatch):
    await tg.pair(42)
    handled: list[int] = []
    deliver = tg.app.channels.deliver_notification

    async def spy(note: dict) -> int:
        sent = await deliver(note)
        handled.append(sent)
        return sent

    monkeypatch.setattr(tg.app.channels, "deliver_notification", spy)
    sid = await tg.app.store.create_session(channel="desktop")
    async for _ in tg.app.agent.run_turn(sid, "remind me in 5 minutes to stretch", channel="desktop"):
        pass
    before = len(tg.api.sent("sendMessage"))
    await tg.app.reminders.fire_due(datetime.now(UTC) + timedelta(minutes=6))
    await until(lambda: handled)
    assert handled == [0] and len(tg.api.sent("sendMessage")) == before


async def test_with_the_setting_a_desktop_reminder_and_alarm_also_go_to_the_delivery_chats(tg):
    """#359: channels.deliver_reminders sends reminders and alarms set on the desktop to the delivery chats too;
    a timer still stays on the desktop."""
    chat = await tg.pair(42)
    assert chat["deliver"]
    tg.app.config.channels.deliver_reminders = True
    sid = await tg.app.store.create_session(channel="desktop")
    for text in ("remind me in 5 minutes to stretch", "set an alarm in 5 minutes", "set a timer for 5 minutes"):
        async for _ in tg.app.agent.run_turn(sid, text, channel="desktop"):
            pass
    before = len(tg.api.sent("sendMessage"))
    assert await tg.app.reminders.fire_due(datetime.now(UTC) + timedelta(minutes=6)) == 3
    await until(lambda: len(tg.api.sent("sendMessage")) >= before + 2)
    texts = [m["text"] for m in tg.api.sent("sendMessage")[before:]]
    assert len(texts) == 2 and all(m["chat_id"] in (42, "42") for m in tg.api.sent("sendMessage")[before:])
    assert any("Reminder" in t and "Stretch" in t for t in texts)
    assert any("Alarm" in t and "It's" in t for t in texts)
    assert not any("Timer done" in t for t in texts)
    rows: list = []

    async def recorded() -> bool:
        rows[:] = await tg.app.store.fetchall("SELECT reason, what FROM channel_deliveries WHERE reason LIKE 'reminder%'")
        return len(rows) >= 2

    for _ in range(300):
        if await recorded():
            break
        await asyncio.sleep(0.01)
    assert len(rows) == 2
    assert {r["reason"] for r in rows} == {"reminder_copy"} and {r["what"] for r in rows} == {"reminder"}


async def test_a_reminder_from_a_chat_that_is_no_longer_paired_goes_nowhere_else(tg):
    """#359 review: the copy to the delivery chats is only for reminders set outside a messaging chat."""
    await tg.pair(42)
    tg.app.config.channels.deliver_reminders = True
    note = {"kind": "reminder", "title": "Reminder", "message": "Stretch",
            "payload": {"event": "reminder", "channel": "telegram", "session_id": "gone"}}
    assert await tg.app.channels._reminder_delivery(note) == "desktop"
    desktop = {**note, "payload": {"event": "reminder", "channel": "desktop", "session_id": "a-desktop-chat"}}
    assert await tg.app.channels._reminder_delivery(desktop) == "default"
    timer = {**note, "payload": {"event": "timer", "channel": "desktop"}}
    assert await tg.app.channels._reminder_delivery(timer) == "desktop"


async def test_without_the_setting_a_desktop_alarm_stays_on_the_desktop(tg):
    await tg.pair(42)
    assert tg.app.config.channels.deliver_reminders is False  # off unless you turn it on
    sid = await tg.app.store.create_session(channel="desktop")
    async for _ in tg.app.agent.run_turn(sid, "set an alarm in 5 minutes", channel="desktop"):
        pass
    before = len(tg.api.sent("sendMessage"))
    assert await tg.app.reminders.fire_due(datetime.now(UTC) + timedelta(minutes=6)) == 1
    note = (await tg.app.notifications.list())[0]
    assert await tg.app.channels.deliver_notification(note) == 0
    assert len(tg.api.sent("sendMessage")) == before
