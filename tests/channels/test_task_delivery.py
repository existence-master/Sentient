"""Where a task's notifications go (issue #227): the default paired chats, the desktop only, or chosen chats."""

from __future__ import annotations

import asyncio

import pytest

from tests.channels.conftest import until
from tests.channels.test_task_questions import ASK_FLIGHT, FLIGHTS, RESULT
from tests.conftest import FakeProvider


@pytest.fixture
def llm() -> FakeProvider:
    return FakeProvider(replies=[ASK_FLIGHT, "Done with the task."], json_replies=[dict(RESULT)])


async def _task(app, deliver_to=None, **fields) -> str:
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Morning brief", "description": "Morning brief", "status": "active", "plan": [],
        "created_at": now, "updated_at": now, **fields,
    })
    if deliver_to is not None:
        await app.tasks.update(task_id, {"deliver_to": deliver_to})
    return task_id


async def _finished(app, task_id: str, name: str = "Morning brief") -> None:
    await app.notify("task", f"Task '{name}' has finished with status: completed.", title="Task completed",
                     payload={"task_id": task_id, "event": "run_completed"})


def _chats(tg, before: int) -> list[str]:
    return [str(p["chat_id"]) for p in tg.api.sent("sendMessage")[before:]]


async def test_default_goes_to_every_chat_with_delivery_on(tg):
    await tg.pair(42)
    await tg.pair(43)
    await tg.app.channels.set_deliver("telegram", "43", False)
    task_id = await _task(tg.app)
    assert (await tg.app.tasks.get(task_id))["deliver_to"] == "default"
    before = len(tg.api.sent("sendMessage"))
    await _finished(tg.app, task_id)
    await until(lambda: len(tg.api.sent("sendMessage")) > before)
    await asyncio.sleep(0.2)
    assert _chats(tg, before) == ["42"]


async def test_desktop_only_sends_nothing_to_chats(tg):
    await tg.pair(42)
    desktop = await _task(tg.app, "desktop")
    other = await _task(tg.app, name="Other task")
    before = len(tg.api.sent("sendMessage"))
    await _finished(tg.app, desktop)
    await tg.app.notify("task", "It failed.", title="Task failed", payload={"task_id": desktop, "event": "run_failed"})
    await _finished(tg.app, other, "Other task")  # delivered in order after the two above
    await until(lambda: len(tg.api.sent("sendMessage")) > before)
    await asyncio.sleep(0.2)
    texts = [p["text"] for p in tg.api.sent("sendMessage")[before:]]
    assert len(texts) == 1 and "Other task" in texts[0]
    assert any(n["task_id"] == desktop for n in await tg.app.notifications.list())  # the app still has it


async def test_chosen_chat_gets_it_whatever_the_switches_say(tg):
    await tg.pair(42)
    await tg.pair(43)
    await tg.app.channels.set_deliver("telegram", "43", False)
    tg.app.config.channels.deliver_task_results = False
    task_id = await _task(tg.app, [{"channel": "telegram", "chat_id": "43"}, {"channel": "discord", "chat_id": "9"}])
    before = len(tg.api.sent("sendMessage"))
    await _finished(tg.app, task_id)
    await until(lambda: len(tg.api.sent("sendMessage")) > before)
    await asyncio.sleep(0.2)
    assert _chats(tg, before) == ["43"]  # Discord isn't connected, chat 42 wasn't picked


async def test_task_question_goes_to_the_chosen_chat_and_a_reply_answers_it(tg):
    await tg.pair(42)
    await tg.pair(43)
    task_id = await _task(tg.app, [{"channel": "telegram", "chat_id": "43"}], status="approval_pending",
                          schedule={"type": "once", "run_at": None},
                          plan=[{"tool": "time", "description": "Check the date"}])
    before = len(tg.api.sent("sendMessage"))
    await tg.app.tasks.approve(task_id)
    await tg.app.tasks.drain()
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    assert str(msg["chat_id"]) == "43" and "Which flight should I book?" in msg["text"]
    assert "42" not in _chats(tg, before)
    await tg.say(43, FLIGHTS[0], reply_to_message={"message_id": msg["_id"]})
    await tg.app.tasks.drain()
    task = await tg.app.tasks.get(task_id)
    assert task["runs"][-1]["status"] == "completed"


async def test_a_brief_follows_its_task(tg):
    await tg.pair(42)
    await tg.pair(43)
    brief = {"day": "2026-10-12", "title": "Your Daily Brief for Monday",
             "sections": [{"id": "calendar", "label": "Calendar", "feedback": None}],
             "items": [{"id": "calendar-1", "section": "calendar", "text": "09:30 Design review", "link": None,
                        "why": "On your calendar today", "feedback": None}],
             "skipped": [], "expires_at": "2026-10-13T00:00:00+00:00"}
    task_id = await _task(tg.app, [{"channel": "telegram", "chat_id": "42"}], name="Daily Brief")
    before = len(tg.api.sent("sendMessage"))
    await tg.app.notify("brief", "- 09:30 Design review", title=brief["title"],
                        payload={"brief": brief, "status": "active", "task_id": task_id})
    await until(lambda: len(tg.api.sent("sendMessage")) > before)
    await asyncio.sleep(0.2)
    assert _chats(tg, before) == ["42"]


async def test_deliver_to_is_checked_and_kept(tg):
    task_id = await _task(tg.app)
    chats = [{"channel": "WhatsApp", "chat_id": "self"}, {"channel": "whatsapp", "chat_id": "self"}]
    saved = await tg.app.tasks.update(task_id, {"deliver_to": chats})
    assert saved["deliver_to"] == [{"channel": "whatsapp", "chat_id": "self"}]
    assert (await tg.app.tasks.update(task_id, {"deliver_to": []}))["deliver_to"] == "desktop"
    assert (await tg.app.tasks.update(task_id, {"deliver_to": "default"}))["deliver_to"] == "default"
    for bad in ("everywhere", [{"channel": "sms", "chat_id": "1"}], [{"channel": "telegram"}], {"channel": "telegram"}):
        with pytest.raises(ValueError):
            await tg.app.tasks.update(task_id, {"deliver_to": bad})
    await tg.app.tasks.update(task_id, {"deliver_to": "desktop"})
    copy = await tg.app.tasks.rerun(task_id)
    assert copy["deliver_to"] == "desktop"
