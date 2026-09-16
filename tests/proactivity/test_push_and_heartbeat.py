"""source.items consumption (feed/webhook), feed-aware polling, duplicate and stale suggestions, heartbeat."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from sentient.proactivity.service import SuggestionError, similar
from tests.proactivity.conftest import ACTIONABLE, MEETING_EMAIL, script_pipeline

HOOK_ITEM = {
    "id": "wh-1", "name": "Deploy alerts", "body": {"status": "failed", "service": "api", "commit": "abc123"},
    "received_at": "2026-09-15T09:00:00+00:00",
}
FAILED_TASK = {"name": "Weekly digest", "status": "error", "description": "Gmail token expired"}
RETRY_JSON = (
    '{"actionable": true, "confidence_score": 0.9, "reasoning": "failed task", '
    '"suggestion_description": "Retry the Weekly digest task", "suggestion_type_description": "retry a failed task", '
    '"suggestion_action_details": {"action_type": "retry_task"}}'
)


def _quiet_now() -> str:
    now = datetime.now(UTC)
    return f"{(now - timedelta(hours=1)).strftime('%H:%M')}-{(now + timedelta(hours=1)).strftime('%H:%M')}"


async def test_feed_items_run_the_pipeline_once(app, fake_tasks):
    pro, llm = app.proactivity, app.fake
    script_pipeline(llm, ACTIONABLE)
    event = {"source": "gmail", "event": "new_email", "origin": "feed", "items": [MEETING_EMAIL]}
    assert await pro.on_source_items(event) == 1
    notes = await app.notifications.list()
    assert len(notes) == 1 and notes[0]["kind"] == "proactive"
    assert notes[0]["title"] == "Gmail: Jane Doe: Meeting about Project Phoenix"
    assert notes[0]["payload"]["suggestion"]["source_event"]["item_id"] == "msg-1"
    assert fake_tasks.events == []  # no more tasks calls from proactivity
    calls = len(llm.calls)
    assert await pro.on_source_items(event) == 0  # already seen
    assert len(llm.calls) == calls


async def test_poll_origin_and_unwatched_sources_are_ignored(app):
    pro, llm = app.proactivity, app.fake
    assert await pro.on_source_items({"source": "gmail", "event": "new_email", "origin": "poll", "items": [MEETING_EMAIL]}) == 0
    assert await pro.on_source_items({"source": "outlook", "event": "new_email", "origin": "feed", "items": [MEETING_EMAIL]}) == 0
    assert await pro.on_source_items({"source": "gmail", "origin": "feed", "items": "not a list"}) == 0
    assert llm.calls == [] and await app.store.fetchall("SELECT * FROM proactive_seen") == []
    # the poll path processes what poll_source returned, exactly once
    script_pipeline(llm, {"actionable": False})
    assert await pro.poll_all() == 1
    assert await pro.on_source_items({"source": "gmail", "event": "new_email", "origin": "feed", "items": [MEETING_EMAIL]}) == 0


async def test_webhook_items_get_a_generic_prompt(app):
    pro, llm = app.proactivity, app.fake
    reasoner = {**ACTIONABLE, "suggestion_description": "Look into the failed api deploy (commit abc123)",
                "suggestion_action_details": {"action_type": "investigate_failure"}}
    script_pipeline(llm, reasoner, type_name="investigate_failed_deploy")
    event = {"source": "webhook", "event": "hook1", "origin": "webhook", "items": [HOOK_ITEM]}
    assert await pro.on_source_items(event) == 1
    queries_msg = next(c for c in llm.calls if c.get("json"))["messages"][1]["content"]
    assert "call to the webhook 'Deploy alerts'" in queries_msg
    reasoner_msg = [c for c in llm.calls if c.get("json")][-1]["messages"][1]["content"]
    assert "webhook named 'Deploy alerts'" in reasoner_msg and "abc123" in reasoner_msg
    note = (await app.notifications.list())[0]
    ev = note["payload"]["suggestion"]["source_event"]
    assert ev["source"] == "webhook" and ev["event_type"] == "hook1" and ev["hook_name"] == "Deploy alerts"
    assert note["title"] == "Webhook: Deploy alerts"


async def test_webhook_handled_by_a_task_or_empty_is_skipped(app, fake_tasks):
    pro, llm = app.proactivity, app.fake
    fake_tasks.tasks = [{"name": "On deploy", "status": "active",
                         "schedule": {"type": "triggered", "source": "webhook", "event": "hook1"}}]
    assert await pro.on_source_items({"source": "webhook", "event": "hook1", "origin": "webhook", "items": [HOOK_ITEM]}) == 1
    assert llm.calls == [] and await app.notifications.list() == []
    empty = {**HOOK_ITEM, "id": "wh-2", "body": ""}
    assert await pro.on_source_items({"source": "webhook", "event": "hook2", "origin": "webhook", "items": [empty]}) == 1
    assert llm.calls == []  # empty body: pre-filtered
    app.config.proactivity.webhook_suggestions = False
    assert await pro.on_source_items({"source": "webhook", "event": "hook2", "origin": "webhook",
                                      "items": [{**HOOK_ITEM, "id": "wh-3"}]}) == 0


async def test_polling_skipped_while_feed_active_but_poll_now_forces(app):
    pro, integ, llm = app.proactivity, app.integrations, app.fake
    integ.feed_active = lambda source: source == "gmail"
    assert await pro.poll_all() == 0
    assert [p[0] for p in integ.polls] == ["gcalendar"]
    status = {s["source"]: s for s in (await pro.status())["sources"]}
    assert status["gmail"]["feed_active"] is True and status["gcalendar"]["feed_active"] is False
    script_pipeline(llm, {"actionable": False})
    assert await pro.poll_now() == {"ok": True, "events": 1}
    assert ("gmail" in [p[0] for p in integ.polls])


async def test_feed_active_errors_fall_back_to_polling(app):
    def boom(source):
        raise RuntimeError("not ready")

    app.integrations.feed_active = boom
    script_pipeline(app.fake, {"actionable": False})
    assert await app.proactivity.poll_all() == 1


async def test_no_duplicate_suggestions_for_the_same_item_or_action(app):
    pro, llm = app.proactivity, app.fake
    script_pipeline(llm, ACTIONABLE)
    assert (await pro.process_event("gmail", "new_email", MEETING_EMAIL))["status"] == "pending"
    calls = len(llm.calls)
    assert await pro.process_event("gmail", "new_email", MEETING_EMAIL) is None
    assert len(llm.calls) == calls  # same item: no model calls at all
    # a second email about the same thing while the first suggestion is still open
    llm.json_replies += [{"q": "Project Phoenix meeting"}, {**ACTIONABLE, "suggestion_description": "Draft a reply to Jane confirming Tuesday 2pm"}]
    text_calls = len(llm.text_calls)
    assert await pro.process_event("gmail", "new_email", {**MEETING_EMAIL, "id": "msg-2"}) is None
    assert len(llm.text_calls) == text_calls  # dropped before type standardization
    assert similar("Draft a reply to Jane confirming Tuesday at 2pm", "Draft a reply to Jane confirming Tuesday 2pm")
    assert not similar("Draft a reply to Jane", "Schedule the design review")


async def test_scratchpad_carries_user_model_and_recent_suggestions(app):
    pro, llm = app.proactivity, app.fake
    app.user_model.context = "Sarthak prefers afternoon meetings."
    script_pipeline(llm, ACTIONABLE)
    await pro.process_event("gmail", "new_email", MEETING_EMAIL)
    script_pipeline(llm, {"actionable": False})
    await pro.process_event("gmail", "new_email", {**MEETING_EMAIL, "id": "msg-9", "subject": "Lunch Friday?"})
    reasoner_msg = [c for c in llm.calls if c.get("json")][-1]["messages"][1]["content"]
    assert "prefers afternoon meetings" in reasoner_msg
    assert "suggestions_already_made" in reasoner_msg and "Draft a reply to Jane" in reasoner_msg
    assert app.user_model.queries and "Project Phoenix" in app.user_model.queries[0]


async def test_stale_suggestions_expire(app):
    pro, llm = app.proactivity, app.fake
    script_pipeline(llm, ACTIONABLE)
    rec = await pro.process_event("gmail", "new_email", MEETING_EMAIL)
    now = datetime.now(UTC)
    assert await pro.expire_stale(now + timedelta(hours=1)) == 0
    assert await pro.expire_stale(now + timedelta(hours=49)) == 1
    note = await app.notifications.get(rec["notification_id"])
    assert note["payload"]["status"] == "expired" and note["read"]
    with pytest.raises(SuggestionError) as exc:
        await pro.act_on_suggestion(rec["notification_id"], "approve")
    assert exc.value.status == 409
    row = await app.store.fetchone("SELECT status FROM proactive_suggestions WHERE id = ?", (rec["id"],))
    assert row["status"] == "expired"


async def test_calendar_suggestion_expires_when_the_event_starts(app):
    pro, llm = app.proactivity, app.fake
    start = datetime.now(UTC) + timedelta(hours=1)
    event = {"id": "ev-7", "summary": "Design review", "status": "confirmed", "organizer_email": "jane@acme.com",
             "start": start.isoformat(), "end": (start + timedelta(hours=1)).isoformat()}
    script_pipeline(llm, {**ACTIONABLE, "suggestion_description": "Prepare notes for the design review"},
                    type_name="prepare_meeting_brief")
    assert await pro.on_source_items({"source": "gcalendar", "event": "new_event", "origin": "feed", "items": [event]}) == 1
    assert await pro.expire_stale(start - timedelta(minutes=5)) == 0
    assert await pro.expire_stale(start + timedelta(minutes=1)) == 1


async def test_heartbeat_skips_quiet_hours_and_respects_daily_cap(app, fake_tasks):
    pro, llm = app.proactivity, app.fake
    fake_tasks.tasks = [FAILED_TASK]
    app.config.proactivity.quiet_hours = _quiet_now()
    assert await pro.heartbeat() is None
    assert llm.text_calls == []  # no model call during quiet hours
    app.config.proactivity.quiet_hours = ""
    app.config.proactivity.heartbeat_daily_cap = 1
    llm.text_replies += [RETRY_JSON, "retry_failed_task"]
    rec = await pro.heartbeat()
    assert rec["status"] == "pending" and await pro.heartbeats_today(datetime.now(UTC)) == 1
    note = await app.notifications.get(rec["notification_id"])
    assert note["title"] == "Suggestion from Check-in"
    fake_tasks.tasks = [FAILED_TASK, {"name": "Invoice run", "status": "approval_pending", "description": "needs ok"}]
    calls = len(llm.text_calls)
    assert await pro.heartbeat() is None
    assert len(llm.text_calls) == calls  # cap reached: no model call


async def test_heartbeat_context_and_no_reply(app, fake_tasks):
    pro, llm = app.proactivity, app.fake
    app.user_model.context = "Sarthak likes a short prep note before design meetings."
    start = datetime.now(UTC) + timedelta(hours=2)
    await pro._mark_seen("gcalendar", {"id": "ev-1", "summary": "Design review", "status": "confirmed",
                                        "start": start.isoformat(), "attendees": [{"email": "jane@acme.com"}]})
    llm.text_replies.append("NO_REPLY")
    assert await pro.heartbeat() is None
    situation = json.loads(llm.text_calls[-1]["messages"][1]["content"])
    assert situation["upcoming_events_next_12h"][0]["summary"] == "Design review"
    assert situation["upcoming_events_next_12h"][0]["attendees"] == ["jane@acme.com"]
    assert situation["user"]["time_of_day"] in {"morning", "afternoon", "evening", "night"}
    assert "short prep note" in situation["about_the_user"]
    # a repeat of a suggestion already made today is dropped
    llm.text_replies += [RETRY_JSON.replace("Retry the Weekly digest task", "Prepare a short note for the Design review"),
                         "prepare_meeting_brief"]
    assert (await pro.heartbeat())["status"] == "pending"
    llm.text_replies.append(RETRY_JSON.replace("Retry the Weekly digest task", "Prepare a short note for Design review"))
    assert await pro.heartbeat() is None
