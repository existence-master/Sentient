"""A task waiting for approval reaches the paired chat while nobody is in the app, with scoped answers, and only
the chat it was sent to can answer it (#139)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sentient.channels.service import OLDER_REQUEST, OLDER_REQUEST_REPLY, request_tag
from sentient.tasks import ask
from tests.channels.conftest import callback, until, update
from tests.conftest import FakeProvider, tool_call

RESULT = {"summary": "Saved.", "links_created": [], "links_found": [], "files_created": [], "tools_used": ["files"]}
OPTIONS = ["Allow once", "Allow for this task", "Allow for 1 hour", "Always allow", "Deny"]
WRITE_A = [tool_call("file_write", name="a.txt", content="one")]
WRITE_B = [tool_call("file_write", name="b.txt", content="two")]


@pytest.fixture
def llm() -> FakeProvider:
    return FakeProvider(replies=[WRITE_A, WRITE_B, "Saved both."], json_replies=[dict(RESULT)])


async def _waiting_task(app) -> tuple[str, str]:
    app.config.tools.approvals.rules = {"files": "ask"}
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Save notes", "description": "Save two notes", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None}, "plan": [{"tool": "files", "description": "Save the notes"}],
        "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "waiting_for_user"
    return task_id, task["runs"][-1]["run_id"]


def _approval_messages(tg, chat_id: int) -> list[dict]:
    return [p for p in tg.api.button_messages() if str(p["chat_id"]) == str(chat_id) and "Approval needed" in p["text"]]


async def test_task_approval_reaches_the_phone_and_allow_for_this_task_carries_on(tg, isolated_home):
    await tg.pair(42)
    task_id, run_id = await _waiting_task(tg.app)
    await until(lambda: bool(_approval_messages(tg, 42)))
    msg = _approval_messages(tg, 42)[-1]
    assert msg["text"].startswith("<b>Approval needed: Save notes needs your OK</b>")
    assert "reply to this message with once, task, hour, always or no" in msg["text"]
    buttons = [b for row in msg["reply_markup"]["inline_keyboard"] for b in row]
    assert [b["text"] for b in buttons] == OPTIONS
    pending = (await tg.app.tasks.repo.get_run(run_id))["pending_question"]
    assert buttons[1]["callback_data"] == f"ta:1:{run_id}:{request_tag(pending['expires_at'])}"
    assert all(len(b["callback_data"].encode()) <= 64 for b in buttons)  # Telegram's limit

    tg.ch.dispatch(callback(42, msg["_id"], buttons[1]["callback_data"]))
    await tg.ch.wait_idle()
    await tg.app.tasks.drain()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Answered: Allow for this task"
    task = await tg.app.tasks.get(task_id)
    assert task["status"] == "completed"
    assert (isolated_home / "files" / "b.txt").exists()  # the second write went through without asking
    assert len(_approval_messages(tg, 42)) == 1
    [grant] = await tg.app.approvals.grants.list()
    assert (grant["scope"], grant["task_id"], grant["via"]) == ("task", task_id, "telegram")


async def test_a_reply_word_answers_the_approval(tg, llm):
    await tg.pair(42)
    llm.replies[:] = [WRITE_A, "Saved one."]
    task_id, _ = await _waiting_task(tg.app)
    await until(lambda: bool(_approval_messages(tg, 42)))
    msg = _approval_messages(tg, 42)[-1]
    await tg.say(42, "hour", reply_to_message={"message_id": msg["_id"]})
    await tg.app.tasks.drain()
    assert (await tg.app.tasks.get(task_id))["status"] == "completed"
    [grant] = await tg.app.approvals.grants.list()
    assert grant["scope"] == "timed" and grant["via"] == "telegram"


async def test_only_the_chat_it_was_sent_to_can_answer(tg, isolated_home):
    await tg.pair(42)
    task_id, run_id = await _waiting_task(tg.app)
    await until(lambda: bool(_approval_messages(tg, 42)))
    msg = _approval_messages(tg, 42)[-1]
    allow = f"ta:0:{run_id}"

    # a chat that was never paired
    tg.ch.dispatch(callback(77, msg["_id"], allow))
    await tg.ch.wait_idle()
    assert "isn't paired" in tg.api.sent("answerCallbackQuery")[-1]["text"]
    await tg.say(77, "yes", reply_to_message={"message_id": msg["_id"]})

    # another paired chat pressing a button of a message it never got
    await tg.pair(43)
    tg.ch.dispatch(callback(43, msg["_id"], allow))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == (
        "This request wasn't sent to this chat, so it can't be answered here."
    )
    # nor can an ordinary question button answer it
    tg.ch.dispatch(callback(42, msg["_id"], f"tq:0:{run_id}"))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Unknown button."

    assert (await tg.app.tasks.get(task_id))["status"] == "waiting_for_user"
    assert not (isolated_home / "files" / "a.txt").exists()


async def test_held_while_in_the_app_and_sent_when_the_person_leaves(tg):
    await tg.pair(42)
    tg.app.presence.set("window", True)
    _, run_id = await _waiting_task(tg.app)
    for _ in range(20):  # give delivery a chance: nothing goes out while the window is in front
        await tg.ch.wait_idle()
    assert _approval_messages(tg, 42) == []

    tg.app.presence.set("window", False)
    await until(lambda: bool(_approval_messages(tg, 42)))
    assert f"ta:0:{run_id}" in str(_approval_messages(tg, 42)[-1]["reply_markup"])


async def test_answered_in_the_app_before_leaving_is_never_sent(tg):
    await tg.pair(42)
    tg.app.presence.set("window", True)
    task_id, run_id = await _waiting_task(tg.app)
    await tg.app.tasks.answer_question(task_id, run_id, "Allow once")
    await tg.app.tasks.drain()
    tg.app.presence.drop("window")
    for _ in range(20):
        await tg.ch.wait_idle()
    assert _approval_messages(tg, 42) == []


async def test_an_expired_request_cant_be_approved_from_the_phone(tg, isolated_home, monkeypatch):
    await tg.pair(42)
    task_id, _ = await _waiting_task(tg.app)
    await until(lambda: bool(_approval_messages(tg, 42)))
    msg = _approval_messages(tg, 42)[-1]
    later = datetime.now(UTC) + timedelta(days=1)  # the request's time has passed
    real_expired = ask.expired
    monkeypatch.setattr(ask, "expired", lambda pending, now=None: real_expired(pending, now or later))

    allow = msg["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    tg.ch.dispatch(callback(42, msg["_id"], allow))
    await tg.ch.wait_idle()
    assert "expired" in tg.api.sent("answerCallbackQuery")[-1]["text"]
    run = (await tg.app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "error" and not (isolated_home / "files" / "a.txt").exists()


async def test_chat_approval_answers_only_from_its_chat_and_can_be_timed(tg, llm):
    tg.app.config.tools.approvals.mode = "always"
    await tg.pair(42)
    await tg.pair(43)
    llm.replies[:] = [[tool_call("current_datetime")], "It is noon."]
    tg.ch.dispatch(update(42, "what time is it"))
    await until(lambda: any("Approval needed" in p["text"] for p in tg.api.button_messages()))
    msg = next(p for p in tg.api.button_messages() if "Approval needed" in p["text"])
    buttons = [b for row in msg["reply_markup"]["inline_keyboard"] for b in row]
    assert [b["text"] for b in buttons] == ["Allow", "Allow for this chat", "Allow for 1 hour", "Always allow", "Deny"]
    timed = buttons[2]["callback_data"]
    assert timed.startswith("ap:h:")

    tg.ch.dispatch(callback(43, msg["_id"], timed))  # another paired chat
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == (
        "This request wasn't sent to this chat, so it can't be answered here."
    )
    tg.ch.dispatch(callback(42, msg["_id"], timed))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Allowed for a while"
    [grant] = await tg.app.approvals.grants.list()
    assert (grant["tool"], grant["scope"], grant["via"]) == ("current_datetime", "timed", "telegram")


SECOND_WRITE = [{**WRITE_B[0], "id": "call_write_b"}]  # its own call id, as a real model gives


async def _second_request(tg, task_id: str, run_id: str, first: dict) -> dict:
    """Answer the first request with its own button ("Allow once") and wait for the run's second one."""
    tg.ch.dispatch(callback(42, first["_id"], first["reply_markup"]["inline_keyboard"][0][0]["callback_data"]))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Answered: Allow once"
    await tg.app.tasks.drain()
    await until(lambda: len(_approval_messages(tg, 42)) == 2)
    second = _approval_messages(tg, 42)[-1]
    assert (await tg.app.tasks.get(task_id))["status"] == "waiting_for_user"
    assert (await tg.app.tasks.repo.get_run(run_id))["pending_question"]["approval"]["tool"] == "file_write"
    return second


async def test_an_older_button_of_the_same_run_cant_answer_a_newer_request(tg, llm, isolated_home):
    """#337: the first request's button, pressed again once the run asks about its second write, is refused."""
    llm.replies[1] = SECOND_WRITE
    await tg.pair(42)
    task_id, run_id = await _waiting_task(tg.app)
    await until(lambda: bool(_approval_messages(tg, 42)))
    first = _approval_messages(tg, 42)[-1]
    second = await _second_request(tg, task_id, run_id, first)
    old, new = (m["reply_markup"]["inline_keyboard"][0][0]["callback_data"] for m in (first, second))
    assert old != new and old.rsplit(":", 1)[0] == new.rsplit(":", 1)[0]  # same run, another request

    tg.ch.dispatch(callback(42, first["_id"], old))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == OLDER_REQUEST
    await tg.app.tasks.drain()
    assert (await tg.app.tasks.get(task_id))["status"] == "waiting_for_user"
    assert not (isolated_home / "files" / "b.txt").exists()
    # a button from before #337 (no request tag) is matched by the message it came on
    tg.ch.dispatch(callback(42, first["_id"], old.rsplit(":", 1)[0]))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == OLDER_REQUEST
    assert (await tg.app.tasks.get(task_id))["status"] == "waiting_for_user"

    tg.ch.dispatch(callback(42, second["_id"], new))
    await tg.ch.wait_idle()
    await tg.app.tasks.drain()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Answered: Allow once"
    assert (await tg.app.tasks.get(task_id))["status"] == "completed"
    assert (isolated_home / "files" / "b.txt").exists()


async def test_a_reply_to_an_older_request_of_the_same_run_is_refused(tg, llm, isolated_home):
    llm.replies[1] = SECOND_WRITE
    await tg.pair(42)
    task_id, run_id = await _waiting_task(tg.app)
    await until(lambda: bool(_approval_messages(tg, 42)))
    first = _approval_messages(tg, 42)[-1]
    second = await _second_request(tg, task_id, run_id, first)

    await tg.say(42, "once", reply_to_message={"message_id": first["_id"]})
    await tg.app.tasks.drain()
    assert OLDER_REQUEST_REPLY in tg.api.screen()[-1]
    assert (await tg.app.tasks.get(task_id))["status"] == "waiting_for_user"
    assert not (isolated_home / "files" / "b.txt").exists()

    await tg.say(42, "once", reply_to_message={"message_id": second["_id"]})
    await tg.app.tasks.drain()
    assert (await tg.app.tasks.get(task_id))["status"] == "completed"
