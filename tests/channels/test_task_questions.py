"""Questions from running tasks reach paired chats with quick-reply buttons, and a plain reply answers
the one waiting question (issue #108)."""

from __future__ import annotations

import json

import pytest

from tests.channels.conftest import callback, until
from tests.conftest import FakeProvider, tool_call

FLIGHTS = ["IndiGo 6E 123", "Air India AI 456"]
RESULT = {"summary": "Booked.", "links_created": [], "links_found": [], "files_created": [], "tools_used": []}
ASK_FLIGHT = [tool_call("ask_user", question="Which flight should I book?", options=FLIGHTS)]
ASK_HOTEL = [tool_call("ask_user", question="Which hotel area do you prefer?")]


@pytest.fixture
def llm() -> FakeProvider:
    return FakeProvider(replies=[ASK_FLIGHT, "Done with the task."], json_replies=[dict(RESULT)])


async def _waiting_task(app, name: str) -> tuple[str, str]:
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": name, "description": name, "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None},
        "plan": [{"tool": "time", "description": "Check the date"}],
        "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "waiting_for_user"
    return task_id, task["runs"][-1]["run_id"]


def _executor_answer(llm: FakeProvider, call_index: int) -> str:
    calls = [c for c in llm.calls if c.get("role") == "executor" and not c.get("json")]
    return json.loads(calls[call_index]["messages"][-1]["content"])["answer"]


async def test_question_arrives_with_buttons_and_a_tap_answers_it(tg, llm):
    await tg.pair(42)
    task_id, run_id = await _waiting_task(tg.app, "Book a flight to Goa")
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    assert msg["text"].startswith("<b>Book a flight to Goa needs your answer</b>")
    assert "Which flight should I book?" in msg["text"]
    rows = msg["reply_markup"]["inline_keyboard"]
    buttons = [b for row in rows for b in row]
    assert [b["text"] for b in buttons] == FLIGHTS
    assert buttons[1]["callback_data"] == f"tq:1:{run_id}"

    tg.ch.dispatch(callback(42, msg["_id"], buttons[1]["callback_data"]))
    await tg.ch.wait_idle()
    await tg.app.tasks.drain()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == f"Answered: {FLIGHTS[1]}"
    assert any(
        p["message_id"] == msg["_id"] and f"<b>Answered: {FLIGHTS[1]}</b>" in p["text"]
        for p in tg.api.sent("editMessageText")
    )
    task = await tg.app.tasks.get(task_id)
    assert task["status"] == "completed"
    assert _executor_answer(llm, 1) == FLIGHTS[1]

    # pressing again after it was answered
    tg.ch.dispatch(callback(42, msg["_id"], buttons[0]["callback_data"]))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "This question is no longer waiting."


async def test_answered_in_the_app_settles_the_buttons(tg):
    await tg.pair(42)
    task_id, run_id = await _waiting_task(tg.app, "Book a flight to Goa")
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    await tg.app.tasks.answer_question(task_id, run_id, FLIGHTS[0])
    await until(lambda: any(p["message_id"] == msg["_id"] for p in tg.api.sent("editMessageText")))
    assert f"Answered: {FLIGHTS[0]}" in next(
        p["text"] for p in tg.api.sent("editMessageText") if p["message_id"] == msg["_id"]
    )
    await tg.app.tasks.drain()


async def test_free_text_answers_the_single_waiting_question(tg, llm):
    await tg.pair(42)
    task_id, _ = await _waiting_task(tg.app, "Book a flight to Goa")
    await until(lambda: bool(tg.api.button_messages()))
    executor_calls = len(llm.calls)

    await tg.say(42, "The early one please")
    await tg.app.tasks.drain()
    assert any("passed your answer to" in t and "Book a flight to Goa" in t for t in tg.api.screen())
    task = await tg.app.tasks.get(task_id)
    assert task["status"] == "completed"
    assert _executor_answer(llm, 1) == "The early one please"
    # no chat turn was started for that message: only the resumed executor called the model
    assert {c["role"] for c in llm.calls[executor_calls:] if not c.get("json")} == {"executor"}


async def test_free_text_with_two_waiting_questions_asks_to_use_the_app(tg, llm):
    llm.replies[:] = [ASK_FLIGHT, ASK_HOTEL]
    await tg.pair(42)
    await _waiting_task(tg.app, "Book a flight to Goa")
    await _waiting_task(tg.app, "Find a hotel in Goa")
    calls = len(llm.calls)

    await tg.say(42, "Yes")
    reply = tg.api.screen()[-1]
    assert "2 tasks are waiting for your answer" in reply and "Sentient app" in reply
    assert len(llm.calls) == calls  # nothing answered, no chat turn
    assert len(await tg.app.tasks.waiting_questions()) == 2


async def test_chat_without_delivery_is_a_normal_chat(tg, llm):
    await tg.pair(42)
    await tg.app.channels.set_deliver("telegram", "42", False)
    task_id, _ = await _waiting_task(tg.app, "Book a flight to Goa")
    await tg.say(42, "Hello there")
    assert (await tg.app.tasks.get(task_id))["status"] == "waiting_for_user"
    assert any(c.get("role") == "primary" for c in llm.calls)
