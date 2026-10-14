"""Notifications sent to paired chats are recorded and shown on the Activity timeline under Messaging apps (#345)."""

from __future__ import annotations

from tests.channels.conftest import tg_error, until

BRIEF = {"day": "2026-10-12", "title": "Your Daily Brief for Monday",
         "sections": [{"id": "calendar", "label": "Calendar", "feedback": None}],
         "items": [{"id": "calendar-1", "section": "calendar", "text": "09:30 Design review", "link": None,
                    "why": "On your calendar today", "feedback": None}],
         "skipped": [], "expires_at": "2026-10-13T00:00:00+00:00"}


async def _sent(s, **kw) -> list[dict]:
    return [i for i in (await s.timeline.past(kinds=["channels"], **kw))["items"] if i["id"].startswith("sent:")]


async def test_a_delivered_brief_shows_under_messaging_apps(tg):
    chat = await tg.pair(42)
    events: list[dict] = []
    async with tg.app.bus.subscribe() as queue:
        await tg.app.notify("brief", "- 09:30 Design review", title=BRIEF["title"],
                            payload={"brief": BRIEF, "status": "active", "task_id": "brief-task"})
        while not events:
            event = await queue.get()
            if event["type"] == "channel.delivery":
                events.append(event["data"])
    assert events[0]["status"] == "delivered" and events[0]["title"] == "Your Daily Brief for Monday"

    [item] = await _sent(tg.app)
    assert item["kind"] == "channels" and item["status"] == "done"
    assert item["title"] == "Sent to Telegram: Your Daily Brief for Monday"
    assert item["why"] == f'To your chat "{chat["label"]}". That chat gets Sentient\'s notifications'
    assert item["detail"] == "Your brief"
    assert (item["app"], item["app_label"]) == ("telegram", "Telegram")
    assert item["ref"]["type"] == "task" and item["ref"]["id"] == "brief-task"

    assert await _sent(tg.app, app="telegram", q="daily brief")
    assert await _sent(tg.app, app="discord") == []
    assert await _sent(tg.app, q="nothing like this") == []
    assert {"id": "telegram", "label": "Telegram"} in await tg.app.timeline.apps()


async def test_a_failed_delivery_says_why(tg):
    await tg.pair(42)
    tg.api.queue["sendMessage"] = [tg_error(403, "Forbidden: bot was blocked by the user")]
    await tg.app.notify("task", "It stopped.", title="Task failed",
                        payload={"task_id": "t1", "event": "run_failed", "run_id": "r1"})
    await until(lambda: bool(tg.api.sent("sendMessage")))
    for _ in range(50):
        if await _sent(tg.app):
            break
        await tg.ch.wait_idle()
    [item] = await _sent(tg.app)
    assert item["status"] == "failed" and item["title"] == "Couldn't send to Telegram: Task failed"
    assert "blocked" in item["detail"] and "AAAAAAAAAA" not in item["detail"]  # says why, never the token
    assert item["ref"] == {"type": "task", "id": "t1", "run_id": "r1", "notification_id": item["ref"]["notification_id"]}


async def test_a_helper_summary_sent_to_its_chat_is_recorded(tg):
    chat = await tg.pair(42)
    tg.app.bus.publish("subagent.updated", {
        "subagent_id": "sa1", "session_id": chat["session_id"], "status": "completed", "background": True,
        "goal": "Research flights to Goa", "summary": "Cheapest is IndiGo on Friday.",
    })
    await until(lambda: any("Background work finished" in t for t in tg.api.screen()))
    for _ in range(50):
        if await _sent(tg.app):
            break
        await tg.ch.wait_idle()
    [item] = await _sent(tg.app)
    assert item["title"] == "Sent to Telegram: Background work finished"
    assert item["why"].endswith("A helper you started in that chat finished")
    assert item["ref"] == {"type": "chat", "id": chat["session_id"]}
