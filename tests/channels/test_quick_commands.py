"""Quick commands from a paired chat (#150): done without the model, and the timer goes off in that same chat."""

from __future__ import annotations

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
