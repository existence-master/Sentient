from __future__ import annotations

import asyncio

from tests.channels.conftest import callback, until


async def test_task_result_is_delivered(tg):
    await tg.pair(42)
    before = len(tg.api.sent("sendMessage"))
    await tg.app.notify("task", "Task 'Morning digest' has finished with status: completed. Details: https://x.y/z",
                        title="Task completed", payload={"task_id": "t1", "event": "run_completed"})
    await until(lambda: len(tg.api.sent("sendMessage")) > before)
    text = tg.api.sent("sendMessage")[-1]["text"]
    assert text.startswith("<b>Task completed</b>") and "Morning digest" in text and "https://" not in text


async def test_delivery_respects_deliver_flag_and_kinds(tg):
    await tg.pair(42)
    await tg.app.channels.set_deliver("telegram", "42", False)
    before = len(tg.api.sent("sendMessage"))
    await tg.app.notify("task", "done", title="Task completed", payload={"task_id": "t1", "event": "run_completed"})
    await asyncio.sleep(0.3)  # let the delivery listener handle it while delivery is off
    assert len(tg.api.sent("sendMessage")) == before
    await tg.app.channels.set_deliver("telegram", "42", True)
    await tg.app.notify("info", "Just so you know", title="FYI")
    await tg.app.notify("task", "second", title="Task failed", payload={"task_id": "t2", "event": "run_failed"})
    await until(lambda: len(tg.api.sent("sendMessage")) > before)
    texts = [p["text"] for p in tg.api.sent("sendMessage")[before:]]
    assert len(texts) == 1 and "Task failed" in texts[0]


async def test_plan_approval_buttons(tg, monkeypatch):
    await tg.pair(42)
    approved: list[str] = []

    async def approve(task_id: str) -> dict:
        approved.append(task_id)
        return {"task_id": task_id, "status": "processing"}

    monkeypatch.setattr(tg.app.tasks, "approve", approve)
    await tg.app.notify("task", "I've created a new plan for you: 'Weekly report'", title="Plan ready for approval",
                        payload={"task_id": "t9", "event": "approval_needed"})
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    buttons = msg["reply_markup"]["inline_keyboard"][0]
    assert [b["text"] for b in buttons] == ["Approve plan", "Decline"]
    tg.ch.dispatch(callback(42, msg["_id"], buttons[0]["callback_data"]))
    await tg.ch.wait_idle()
    assert approved == ["t9"]
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Plan approved"
    assert "<b>Plan approved</b>" in tg.api.screen()[-1]


async def test_plan_resolved_elsewhere_settles_buttons(tg):
    await tg.pair(42)
    note = await tg.app.notify("task", "New plan", title="Plan ready for approval",
                               payload={"task_id": "t9", "event": "approval_needed"})
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    await tg.app.notifications.update_payload(note["id"], {**note["payload"], "status": "declined"})
    await until(lambda: any(p["message_id"] == msg["_id"] for p in tg.api.sent("editMessageText")))
    assert "<b>Plan declined</b>" in tg.api.screen()[-1]


async def test_suggestion_dismiss_uses_proactivity_service(tg):
    await tg.pair(42)
    suggestion = {"suggestion_type": "draft_email_reply", "description": "Draft a reply to Jane",
                  "action_details": {"action_type": "draft_email"}, "reasoning": "r", "confidence": 0.8,
                  "source_event": {"source": "gmail", "event_type": "new_email", "summary": "Jane: Tuesday?"}}
    note = await tg.app.notify("proactive", "Draft a reply to Jane confirming Tuesday", title="Suggestion from Gmail",
                               payload={"suggestion": suggestion, "status": "pending", "task_id": None})
    await until(lambda: bool(tg.api.button_messages()))
    msg = tg.api.button_messages()[-1]
    buttons = msg["reply_markup"]["inline_keyboard"][0]
    assert [b["text"] for b in buttons] == ["Approve", "Dismiss"]
    assert buttons[1]["callback_data"] == f"sg:d:{note['id']}"
    tg.ch.dispatch(callback(42, msg["_id"], buttons[1]["callback_data"]))
    await tg.ch.wait_idle()
    assert tg.api.sent("answerCallbackQuery")[-1]["text"] == "Dismissed"
    stored = await tg.app.notifications.get(note["id"])
    assert stored["payload"]["status"] == "dismissed" and stored["read"] is True
    # pressing again reports the conflict from the proactivity service
    tg.ch.dispatch(callback(42, msg["_id"], buttons[0]["callback_data"]))
    await tg.ch.wait_idle()
    assert "already dismissed" in tg.api.sent("answerCallbackQuery")[-1]["text"]


async def test_background_subagent_summary_goes_to_its_chat_once(tg):
    chat = await tg.pair(42)
    before = len(tg.api.sent("sendMessage"))
    tg.app.bus.publish("subagent.updated", {
        "subagent_id": "sa1", "session_id": chat["session_id"], "status": "completed", "background": True,
        "goal": "Research flights to Goa", "summary": "Cheapest is IndiGo on Friday.",
    })
    await until(lambda: len(tg.api.sent("sendMessage")) > before)
    assert "Background work finished" in tg.api.screen()[-1] and "IndiGo" in tg.api.screen()[-1]
    await tg.app.notify("info", "Research flights to Goa is done", title="Background work finished",
                        payload={"subagent_id": "sa1", "session_id": chat["session_id"]})
    await tg.app.notify("task", "x", title="Task failed", payload={"task_id": "t", "event": "run_failed"})
    await until(lambda: "Task failed" in tg.api.screen()[-1])
    assert len(tg.api.sent("sendMessage")) == before + 2  # the subagent notification was not repeated
