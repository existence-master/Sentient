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


async def _question_message(tg, question: str) -> dict:
    """The sendMessage call that delivered ``question`` (waits for delivery)."""
    def find():
        return next((p for p in tg.api.sent("sendMessage") if question in p["text"]), None)

    await until(lambda: find() is not None)
    return find()


def _reply(message: dict) -> dict:
    return {"reply_to_message": {"message_id": message["_id"]}}


async def test_question_text_tells_how_to_answer(tg, llm):
    await tg.pair(42)
    llm.replies[:] = [ASK_HOTEL]
    await _waiting_task(tg.app, "Find a hotel in Goa")
    msg = await _question_message(tg, "Which hotel area do you prefer?")
    assert "reply_markup" not in msg and msg["text"].endswith("<i>Reply to this message with your answer.</i>")


async def test_reply_to_the_question_answers_it(tg, llm):
    await tg.pair(42)
    task_id, run_id = await _waiting_task(tg.app, "Book a flight to Goa")
    msg = await _question_message(tg, "Which flight should I book?")
    assert msg["text"].endswith("<i>Tap an option, or reply to this message with your answer.</i>")
    # remembered in SQLite, so a reply still matches after a restart
    assert await tg.app.channels.store.question_for("telegram", "42", str(msg["_id"])) == {"task_id": task_id, "run_id": run_id}
    calls = len(llm.calls)

    await tg.say(42, "The early one please", **_reply(msg))
    await tg.app.tasks.drain()
    assert any("passed your answer to" in t and "Book a flight to Goa" in t for t in tg.api.screen())
    assert (await tg.app.tasks.get(task_id))["status"] == "completed"
    assert _executor_answer(llm, 1) == "The early one please"
    # no chat turn was started for that message: only the resumed executor called the model
    assert {c["role"] for c in llm.calls[calls:] if not c.get("json")} == {"executor"}


async def test_a_message_that_is_not_a_reply_is_normal_chat(tg, llm):
    llm.replies[:] = [ASK_FLIGHT, "It looks sunny in Bengaluru today."]
    await tg.pair(42)
    task_id, _ = await _waiting_task(tg.app, "Book a flight to Goa")
    other = await _question_message(tg, "Which flight should I book?")

    await tg.say(42, "What's the weather like?")
    assert any(c.get("role") == "primary" for c in llm.calls)
    assert "sunny" in tg.api.screen()[-1]
    assert (await tg.app.tasks.get(task_id))["status"] == "waiting_for_user"
    # a reply to some other message (not a question) is normal chat too
    llm.replies[:] = ["Glad to help."]
    paired = next(p for p in tg.api.sent("sendMessage") if "Paired!" in p["text"])
    assert paired["_id"] != other["_id"]
    await tg.say(42, "Thanks", **_reply(paired))
    assert "Glad to help." in tg.api.screen()[-1]
    assert (await tg.app.tasks.get(task_id))["status"] == "waiting_for_user"


async def test_two_waiting_questions_each_reply_answers_its_own(tg, llm):
    llm.replies[:] = [ASK_FLIGHT, ASK_HOTEL, "Hotel noted.", "Flight booked."]
    llm.json_replies[:] = [dict(RESULT), dict(RESULT)]
    await tg.pair(42)
    flight_id, _ = await _waiting_task(tg.app, "Book a flight to Goa")
    hotel_id, _ = await _waiting_task(tg.app, "Find a hotel in Goa")
    flight_msg = await _question_message(tg, "Which flight should I book?")
    hotel_msg = await _question_message(tg, "Which hotel area do you prefer?")

    await tg.say(42, "Near the beach", **_reply(hotel_msg))
    await tg.app.tasks.drain()
    assert (await tg.app.tasks.get(hotel_id))["status"] == "completed"
    assert (await tg.app.tasks.get(flight_id))["status"] == "waiting_for_user"
    assert _executor_answer(llm, 2) == "Near the beach"

    await tg.say(42, "The IndiGo one", **_reply(flight_msg))
    await tg.app.tasks.drain()
    assert (await tg.app.tasks.get(flight_id))["status"] == "completed"
    assert _executor_answer(llm, 3) == "The IndiGo one"


async def test_reply_to_an_already_answered_question(tg, llm):
    await tg.pair(42)
    task_id, run_id = await _waiting_task(tg.app, "Book a flight to Goa")
    msg = await _question_message(tg, "Which flight should I book?")
    await tg.app.tasks.answer_question(task_id, run_id, FLIGHTS[0])
    await tg.app.tasks.drain()
    calls = len(llm.calls)

    await tg.say(42, "Actually the later one", **_reply(msg))
    assert "already been handled" in tg.api.screen()[-1]
    assert len(llm.calls) == calls  # neither an answer nor a chat turn
