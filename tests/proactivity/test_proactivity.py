from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from sentient.gateway.app import create_app
from sentient.proactivity.prefilter import event_pre_filter, in_quiet_hours
from sentient.proactivity.service import SuggestionError
from tests.proactivity.conftest import ACTIONABLE, MEETING_EMAIL, script_pipeline


def test_pre_filter():
    assert event_pre_filter(MEETING_EMAIL, "gmail", "sarthak@example.com")
    assert not event_pre_filter({**MEETING_EMAIL, "labels": ["CATEGORY_PROMOTIONS"]}, "gmail")
    assert not event_pre_filter({**MEETING_EMAIL, "subject": "Automatic reply: away"}, "gmail")
    assert not event_pre_filter({**MEETING_EMAIL, "headers": {"List-Unsubscribe": "<x>"}}, "gmail")
    assert not event_pre_filter({**MEETING_EMAIL, "subject": "Invitation: sync @ Tue"}, "gmail")
    assert not event_pre_filter({**MEETING_EMAIL, "body": "Thanks, got it", "snippet": ""}, "gmail")
    assert not event_pre_filter({**MEETING_EMAIL, "sender_email": "no-reply@shop.com"}, "gmail")
    assert not event_pre_filter({**MEETING_EMAIL, "subject": "Big sale this weekend"}, "gmail")
    assert event_pre_filter({**MEETING_EMAIL, "subject": "Salesforce renewal questions"}, "gmail")
    assert not event_pre_filter({**MEETING_EMAIL, "sender_email": "sarthak@example.com"}, "gmail", "sarthak@example.com")
    cal = {"id": "e1", "summary": "Design review", "status": "confirmed", "organizer_email": "jane@acme.com",
           "attendees": [{"email": "sarthak@example.com", "responseStatus": "needsAction"}]}
    assert event_pre_filter(cal, "gcalendar", "sarthak@example.com")
    assert not event_pre_filter({**cal, "status": "cancelled"}, "gcalendar")
    assert not event_pre_filter({**cal, "summary": "Focus time"}, "gcalendar")
    assert not event_pre_filter({**cal, "organizer_email": "sarthak@example.com"}, "gcalendar", "sarthak@example.com")
    assert not event_pre_filter(
        {**cal, "attendees": [{"email": "sarthak@example.com", "responseStatus": "declined"}]}, "gcalendar", "sarthak@example.com"
    )


def test_quiet_hours_parsing():
    at = lambda h, m=0: datetime(2026, 9, 15, h, m, tzinfo=UTC)  # noqa: E731
    assert in_quiet_hours("22:00-07:00", at(23)) and in_quiet_hours("22:00-07:00", at(6, 59))
    assert not in_quiet_hours("22:00-07:00", at(7)) and not in_quiet_hours("22:00-07:00", at(12))
    assert in_quiet_hours("13:00-14:00", at(13, 30)) and not in_quiet_hours("", at(13))
    assert not in_quiet_hours("garbage", at(1))


async def test_full_pipeline_creates_suggestion(app):
    llm = app.fake
    script_pipeline(llm, ACTIONABLE)
    rec = await app.proactivity.process_event("gmail", "new_email", MEETING_EMAIL)
    assert rec["status"] == "pending" and rec["notification_id"]
    note = await app.notifications.get(rec["notification_id"])
    assert note["kind"] == "proactive" and note["message"] == ACTIONABLE["suggestion_description"]
    payload = note["payload"]
    assert payload["status"] == "pending" and payload["task_id"] is None
    s = payload["suggestion"]
    assert s["suggestion_type"] == "draft_meeting_confirmation_email"
    assert s["action_details"]["action_type"] == "draft_email"
    assert s["confidence"] == 0.82 and s["reasoning"]
    assert s["source_event"] == {
        "source": "gmail", "event_type": "new_email", "summary": "Jane Doe: Meeting about Project Phoenix",
        "item_id": "msg-1", "url": MEETING_EMAIL["url"],
    }
    # the reasoner saw the trigger, the queries' context and the preferences
    reasoner_msg = [c for c in llm.calls if c.get("json")][-1]["messages"][1]["content"]
    assert "Project Phoenix" in reasoner_msg and "user_preferences" in reasoner_msg


async def test_not_actionable_and_low_confidence(app):
    llm = app.fake
    llm.json_replies += [{"q": "x"}, {"actionable": False}]
    assert await app.proactivity.process_event("gmail", "new_email", MEETING_EMAIL) is None
    script_pipeline(llm, {**ACTIONABLE, "confidence_score": 0.5})
    assert await app.proactivity.process_event("gmail", "new_email", MEETING_EMAIL) is None
    assert (await app.notifications.list()) == []


async def test_new_type_is_standardized_and_saved(app):
    script_pipeline(app.fake, ACTIONABLE, type_name="The best type is `prepare_meeting_brief`.")
    rec = await app.proactivity.process_event("gmail", "new_email", MEETING_EMAIL)
    assert rec["suggestion"]["suggestion_type"] == "prepare_meeting_brief"
    assert "prepare_meeting_brief" in {t["type_name"] for t in await app.proactivity.known_types()}


async def test_threshold_learning_on_approve_and_dismiss(app, fake_tasks):
    pro, llm = app.proactivity, app.fake
    script_pipeline(llm, ACTIONABLE)
    rec = await pro.process_event("gmail", "new_email", MEETING_EMAIL)
    out = await pro.act_on_suggestion(rec["notification_id"], "approve")
    assert out == {"ok": True, "task_id": "task1"}
    task = fake_tasks.created[0]
    assert task["original_context"]["source"] == "proactive"
    assert "Draft a reply to Jane" in task["prompt"] and "draft_email" in task["prompt"]
    note = await app.notifications.get(rec["notification_id"])
    assert note["payload"]["status"] == "approved" and note["payload"]["task_id"] == "task1" and note["read"]
    prefs = {p["suggestion_type"]: p for p in await pro.preferences()}
    assert prefs["draft_meeting_confirmation_email"]["score"] == 1
    assert prefs["draft_meeting_confirmation_email"]["threshold"] == 0.65

    with pytest.raises(SuggestionError) as conflict:
        await pro.act_on_suggestion(rec["notification_id"], "dismiss")
    assert conflict.value.status == 409

    for i in range(2):  # a new email each time: one item never produces two suggestions
        script_pipeline(llm, ACTIONABLE)
        r = await pro.process_event("gmail", "new_email", {**MEETING_EMAIL, "id": f"msg-d{i}"})
        await pro.act_on_suggestion(r["notification_id"], "dismiss")
    prefs = {p["suggestion_type"]: p for p in await pro.preferences()}
    p = prefs["draft_meeting_confirmation_email"]
    assert (p["score"], p["approvals"], p["dismissals"], p["threshold"]) == (-1, 1, 2, 0.75)
    # 0.82 still clears 0.75; after more dismissals it no longer does
    for _ in range(2):
        await pro.record_feedback("draft_meeting_confirmation_email", False)
    script_pipeline(llm, ACTIONABLE)
    assert await pro.process_event("gmail", "new_email", {**MEETING_EMAIL, "id": "msg-last"}) is None
    assert llm.json_replies == [] and llm.text_replies == []  # the reasoner did run and was suppressed
    assert pro.threshold_for(-20) == 0.95 and pro.threshold_for(20) == 0.40
    assert await pro.reset_preference("draft_meeting_confirmation_email")


async def test_quiet_hours_defer_then_flush(app):
    pro = app.proactivity
    now = datetime.now(UTC)
    app.config.proactivity.quiet_hours = f"{(now - timedelta(hours=1)).strftime('%H:%M')}-{(now + timedelta(hours=1)).strftime('%H:%M')}"
    script_pipeline(app.fake, ACTIONABLE)
    rec = await pro.process_event("gmail", "new_email", MEETING_EMAIL)
    assert rec["status"] == "deferred" and rec["notification_id"] is None
    assert await app.notifications.list() == []
    assert await pro.flush_deferred() == 0
    app.config.proactivity.quiet_hours = ""
    assert await pro.flush_deferred() == 1
    notes = await app.notifications.list()
    assert len(notes) == 1 and notes[0]["payload"]["status"] == "pending"


async def test_poll_dedupes_and_leaves_triggers_to_tasks(app, fake_tasks):
    llm = app.fake
    script_pipeline(llm, {"actionable": False})
    assert await app.proactivity.poll_all() == 1
    assert fake_tasks.events == []  # triggered tasks consume source.items themselves (docs/API.md section 16)
    app.integrations.items["gmail"] = [MEETING_EMAIL]
    assert await app.proactivity.poll_all() == 0  # already seen
    status = await app.proactivity.status()
    assert status["enabled"] and status["last_poll_at"]["gmail"]
    assert {s["source"] for s in status["sources"]} == {"gmail", "gcalendar"}
    since = app.integrations.polls[-2][1]
    assert isinstance(since, str) and since.endswith("+00:00")


async def test_heartbeat_no_reply_and_suggestion(app, fake_tasks):
    pro, llm = app.proactivity, app.fake
    assert await pro.heartbeat() is None  # nothing to look at: no LLM call
    assert llm.text_calls == []
    fake_tasks.tasks = [{"name": "Weekly digest", "status": "error", "description": "Gmail token expired"}]
    llm.text_replies.append("NO_REPLY")
    assert await pro.heartbeat() is None
    assert "Weekly digest" in llm.text_calls[-1]["messages"][1]["content"]
    llm.text_replies += [
        '{"actionable": true, "confidence_score": 0.9, "reasoning": "failed task", '
        '"suggestion_description": "Retry the Weekly digest task", "suggestion_type_description": "retry a failed task", '
        '"suggestion_action_details": {"action_type": "retry_task"}}',
        "retry_failed_task",
    ]
    rec = await pro.heartbeat()
    assert rec["status"] == "pending" and rec["suggestion"]["source_event"]["source"] == "heartbeat"


def test_proactivity_routes(config, isolated_home, monkeypatch, fake_tasks):
    from sentient.app import SentientApp
    from tests.conftest import FakeProvider
    from tests.proactivity.conftest import FakeIntegrations

    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider()
    core = SentientApp(config, llm=llm, db_path=isolated_home / "pr.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        core.tasks = fake_tasks
        core.integrations = FakeIntegrations([MEETING_EMAIL])
        core.config.proactivity.context_agent_rounds = 0
        script_pipeline(llm, ACTIONABLE)
        assert c.post("/api/proactivity/poll-now").json() == {"ok": True, "events": 1}
        notes = c.get("/api/notifications").json()["notifications"]
        nid = next(n["id"] for n in notes if n["kind"] == "proactive")
        assert c.post(f"/api/proactivity/suggestions/{nid}", json={"action": "bogus"}).status_code == 400
        assert c.post("/api/proactivity/suggestions/nope", json={"action": "approve"}).status_code == 404
        assert c.post(f"/api/proactivity/suggestions/{nid}", json={"action": "approve"}).json() == {"ok": True, "task_id": "task1"}
        assert c.post(f"/api/proactivity/suggestions/{nid}", json={"action": "dismiss"}).status_code == 409
        st = c.get("/api/proactivity/status").json()
        assert st["suggestions_today"] == 1 and st["sources"][0]["connected"]
        prefs = c.get("/api/proactivity/preferences").json()
        assert prefs[0]["suggestion_type"] == "draft_meeting_confirmation_email" and prefs[0]["score"] == 1
        assert c.delete("/api/proactivity/preferences/draft_meeting_confirmation_email").json() == {"ok": True}
        assert c.get("/api/proactivity/preferences").json() == []


async def test_integration_error_is_recorded_and_other_sources_continue(app):
    from sentient.integrations.base import IntegrationError

    integ = app.integrations
    integ.items["gcalendar"] = [{"id": "ev-1", "summary": "Design review", "status": "confirmed",
                                 "start": "2026-09-16T10:00:00+00:00", "all_day": False, "meet_link": "https://meet"}]
    recorded = {}

    async def poll_source(source, since=None):
        if source == "gmail":
            recorded["gmail"] = "Gmail token expired"
            raise IntegrationError("gmail: HTTP 401")
        out, integ.items[source] = integ.items.get(source, []), []
        return out

    async def poll_state(source):
        return {"source": source, "last_error": recorded.get(source)}

    integ.poll_source = poll_source
    integ.poll_state = poll_state
    app.config.proactivity.enabled = False  # triggers only: no LLM calls
    assert await app.proactivity.poll_all() == 1  # the calendar item still arrived
    st = await app.proactivity.status()
    by_source = {s["source"]: s for s in st["sources"]}
    assert by_source["gmail"]["last_error"] == "Gmail token expired"
    assert by_source["gcalendar"]["last_error"] is None and st["last_poll_at"]["gcalendar"]
    assert app.tasks.events == []
    seen = await app.store.fetchone("SELECT item_id FROM proactive_seen WHERE source = 'gcalendar'")
    assert seen["item_id"] == "ev-1"


async def test_reasoner_retries_when_model_echoes_the_scratchpad(app):
    llm = app.fake
    llm.json_replies.append({"event_type": "gmail", "event": "new_email"})  # echoed queries -> fallback query
    llm.json_replies.append({"universal_search_results": {}, "trigger_event": {}})  # echoed scratchpad
    llm.json_replies.append(ACTIONABLE)
    llm.text_replies.append("draft_meeting_confirmation_email")
    rec = await app.proactivity.process_event("gmail", "new_email", MEETING_EMAIL)
    assert rec and rec["status"] == "pending"
    reasoner_calls = [c for c in llm.calls if c.get("json")][-2:]
    assert "Decide now" in reasoner_calls[0]["messages"][1]["content"]
    assert reasoner_calls[1]["messages"][-1]["content"].startswith("That was not the decision")


async def test_codes_and_reset_links_never_reach_the_proactive_prompts(app):
    """#128: even an item that skipped the mail plugins is masked before any prompt."""
    llm = app.fake
    script_pipeline(llm, {"actionable": False})
    item = {**MEETING_EMAIL, "id": "msg-otp",
            "snippet": "The login code for the shared account is 482913",
            "body": "Hi Sarthak, the login code for the shared account is 482913. If it fails, reset it at "
                    "https://acme.example/reset?token=abcDEF1234567890xyz and update the launch plan. Jane"}
    await app.proactivity.process_event("gmail", "new_email", item)
    prompts_seen = " ".join(str(m["content"]) for c in llm.calls for m in c["messages"])
    assert "Project Phoenix" in prompts_seen or "launch plan" in prompts_seen
    assert "482913" not in prompts_seen and "abcDEF1234567890xyz" not in prompts_seen
    assert "[one-time code hidden]" in prompts_seen
