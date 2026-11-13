"""A task's messages in a paired chat read as one thread: later ones reply to its first, and a reply to any of
them answers the question the task is waiting on (issue #182)."""

from __future__ import annotations

import pytest

from tests.channels.conftest import eventually, until
from tests.channels.test_task_questions import (
    ASK_FLIGHT,
    RESULT,
    _executor_answer,
    _question_message,
    _waiting_task,
)
from tests.conftest import FakeProvider


@pytest.fixture
def llm() -> FakeProvider:
    return FakeProvider(replies=[ASK_FLIGHT, "Done with the task."], json_replies=[dict(RESULT)])


async def _task_messages(tg, task_id: str) -> list[dict]:
    """The messages delivered for ``task_id`` (not the chat's replies about it)."""
    store = tg.app.channels.store
    return [p for p in tg.api.sent("sendMessage") if await store.task_for_message("telegram", "42", str(p["_id"])) == task_id]


async def test_a_tasks_later_messages_reply_to_its_first(tg, llm):
    await tg.pair(42)
    task_id, _ = await _waiting_task(tg.app, "Book a flight to Goa")
    question = await _question_message(tg, "Which flight should I book?")
    first, *_ = await eventually(lambda: _task_messages(tg, task_id))
    assert first["_id"] == question["_id"] and "reply_parameters" not in first

    await tg.say(42, "The early one please", reply_to_message={"message_id": question["_id"]})
    await tg.app.tasks.drain()
    await until(lambda: any("Task completed" in t for t in tg.api.screen()))
    done = next(p for p in tg.api.sent("sendMessage") if "Task completed" in p["text"])
    await eventually(lambda: tg.app.channels.store.task_for_message("telegram", "42", str(done["_id"])))
    later = (await _task_messages(tg, task_id))[1:]
    assert later and all(
        p["reply_parameters"] == {"message_id": first["_id"], "allow_sending_without_reply": True} for p in later
    )

    # another task starts its own thread
    llm.replies[:] = [ASK_FLIGHT]
    other_id, _ = await _waiting_task(tg.app, "Find a hotel in Goa")
    await until(lambda: any("Find a hotel in Goa" in t for t in tg.api.screen()))
    other_first, *_ = await eventually(lambda: _task_messages(tg, other_id))
    assert "reply_parameters" not in other_first


async def test_a_reply_anywhere_in_the_thread_answers_the_waiting_question(tg, llm):
    await tg.pair(42)
    task_id, _ = await _waiting_task(tg.app, "Book a flight to Goa")
    await _question_message(tg, "Which flight should I book?")
    # an earlier message of the task (a progress note, say), which is not the question
    await tg.app.channels.store.add_task_messages("telegram", "42", ["7001"], task_id=task_id)

    await tg.say(42, "The early one please", reply_to_message={"message_id": 7001})
    await tg.app.tasks.drain()
    assert any("passed your answer to" in t for t in tg.api.screen())
    assert (await tg.app.tasks.get(task_id))["status"] == "completed"
    assert _executor_answer(llm, 1) == "The early one please"

    # with nothing waiting, a reply in the thread is normal chat
    llm.replies[:] = ["Glad to help."]
    await tg.say(42, "Thanks", reply_to_message={"message_id": 7001})
    assert "Glad to help." in tg.api.screen()[-1]


async def test_every_chunk_of_a_long_task_message_stays_in_its_thread(tg):
    await tg.pair(42)
    before = len(tg.api.sent("sendMessage"))
    ids = await tg.ch.send_markdown("42", "word " * 2000, reply_to="1001")  # over Telegram's 4096 characters
    sent = tg.api.sent("sendMessage")[before:]
    assert len(ids) > 1 and len(sent) == len(ids)
    assert all(p["reply_parameters"]["message_id"] == 1001 for p in sent)
