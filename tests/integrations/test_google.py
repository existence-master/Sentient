from __future__ import annotations

import base64
import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from datetime import time as dtime
from urllib.parse import parse_qs, urlsplit

import httpx
import respx

from sentient.integrations.plugins.gcalendar import free_slots

GMAIL = "gmail.googleapis.com"
MSG_PATH = "/gmail/v1/users/me/messages"


def b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


async def connect_google(app, keychain, plugin_id: str, *, expires_in: float = 3600) -> None:
    keychain["google_oauth_client"] = json.dumps({"client_id": "cid", "client_secret": "csec"})
    await app.integrations.store_credentials(plugin_id, {"access_token": "at1", "refresh_token": "rt1",
                                                         "expires_at": time.time() + expires_in})
    await app.integrations.mark_connected(plugin_id, "me@example.com")


def gmail_message(mid: str, frm: str, subject: str, *, body: str = "", html: str = "", labels=("INBOX", "UNREAD"),
                  extra_headers: dict | None = None) -> dict:
    headers = [{"name": "From", "value": frm}, {"name": "To", "value": "me@example.com"},
               {"name": "Subject", "value": subject}, *[{"name": k, "value": v} for k, v in (extra_headers or {}).items()]]
    if html:
        payload = {"mimeType": "multipart/alternative", "headers": headers,
                   "parts": [{"mimeType": "text/html", "body": {"data": b64(html)}}]}
    else:
        payload = {"mimeType": "text/plain", "headers": headers, "body": {"data": b64(body)}}
    return {"id": mid, "threadId": f"t-{mid}", "snippet": subject, "labelIds": list(labels),
            "internalDate": str(int(datetime(2026, 9, 15, 9, 0, tzinfo=UTC).timestamp() * 1000)), "payload": payload}


def mock_inbox(router) -> None:
    router.get(host=GMAIL, path=MSG_PATH).mock(return_value=httpx.Response(200, json={"messages": [{"id": "m1"}, {"id": "m2"}]}))
    router.get(host=GMAIL, path=f"{MSG_PATH}/m1").mock(return_value=httpx.Response(200, json=gmail_message(
        "m1", "Boss <boss@x.com>", "Salary review", body="Let's talk about your salary.")))
    router.get(host=GMAIL, path=f"{MSG_PATH}/m2").mock(return_value=httpx.Response(200, json=gmail_message(
        "m2", "Jane Doe <jane@y.com>", "Lunch", html="<p>Lunch at <b>1</b>?</p>",
        extra_headers={"Message-ID": "<abc@y.com>"})))


async def test_loopback_oauth_flow_end_to_end(app, keychain):
    async with app.bus.subscribe() as q:
        started = await app.integrations.connect("gmail", {"client_id": "cid", "client_secret": "csec"})
        assert set(started) == {"auth_url", "state"}
        assert json.loads(keychain["google_oauth_client"]) == {"client_id": "cid", "client_secret": "csec"}
        qs = parse_qs(urlsplit(started["auth_url"]).query)
        assert qs["state"] == [started["state"]] and qs["code_challenge_method"] == ["S256"]
        assert "https://www.googleapis.com/auth/gmail.modify" in qs["scope"][0]
        assert "calendar" not in qs["scope"][0]
        redirect = qs["redirect_uri"][0]
        assert redirect.startswith("http://127.0.0.1:")
        assert (await app.integrations.integration("gmail"))["status"] == "connecting"

        with respx.mock(assert_all_called=True) as router:
            router.route(host="127.0.0.1").pass_through()
            token = router.post("https://oauth2.googleapis.com/token").mock(return_value=httpx.Response(
                200, json={"access_token": "at1", "refresh_token": "rt1", "expires_in": 3599, "scope": "x"}))
            router.get("https://openidconnect.googleapis.com/v1/userinfo").mock(
                return_value=httpx.Response(200, json={"email": "me@example.com"}))
            async with httpx.AsyncClient() as http:
                bad = await http.get(redirect, params={"code": "x", "state": "wrong"})
                ok = await http.get(redirect, params={"code": "authcode", "state": started["state"]})
            form = parse_qs(token.calls.last.request.content.decode())
        events = []
        while not q.empty():
            events.append(q.get_nowait())
    assert bad.status_code == 400 and "expired" in bad.text
    assert ok.status_code == 200 and "You can close this tab" in ok.text
    assert form["grant_type"] == ["authorization_code"] and form["code"] == ["authcode"]
    assert form["redirect_uri"] == [redirect] and form["client_secret"] == ["csec"]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"][0].encode()).digest()).rstrip(b"=").decode()
    assert qs["code_challenge"] == [challenge]
    integ = await app.integrations.integration("gmail")
    assert integ["connected"] and integ["account_label"] == "me@example.com" and integ["status"] == "connected"
    assert json.loads(keychain["integration:gmail"])["refresh_token"] == "rt1"
    assert [e["data"]["status"] for e in events if e["type"] == "integration.updated"][-1] == "connected"
    # second Google integration reuses the stored client: no fields needed
    assert (await app.integrations.integration("gcalendar"))["setup"]["fields"][0]["required"] is False
    assert "auth_url" in await app.integrations.connect("gcalendar", {})


async def test_oauth_user_cancel(app, keychain):
    started = await app.integrations.connect("gdrive", {"client_id": "cid", "client_secret": "csec"})
    redirect = parse_qs(urlsplit(started["auth_url"]).query)["redirect_uri"][0]
    with respx.mock() as router:
        router.route(host="127.0.0.1").pass_through()
        async with httpx.AsyncClient() as http:
            r = await http.get(redirect, params={"error": "access_denied", "state": started["state"]})
    assert r.status_code == 400
    integ = await app.integrations.integration("gdrive")
    assert integ["connected"] is False and integ["error"] == "You cancelled the sign-in."


async def test_token_refresh_when_expired_and_on_401(app, ctx, keychain):
    await connect_google(app, keychain, "gmail", expires_in=-10)
    with respx.mock() as router:
        refresh = router.post("https://oauth2.googleapis.com/token").mock(side_effect=[
            httpx.Response(200, json={"access_token": "at2", "expires_in": 3600}),
            httpx.Response(200, json={"access_token": "at3", "expires_in": 3600}),
        ])
        labels = router.get(host=GMAIL, path="/gmail/v1/users/me/labels").mock(side_effect=[
            httpx.Response(200, json={"labels": []}),
            httpx.Response(401, json={"error": {"message": "expired"}}),
            httpx.Response(200, json={"labels": [{"id": "INBOX", "name": "INBOX", "type": "system"}]}),
        ])
        first = await app.registry.get("gmail_list_labels").call(ctx, {})
        second = await app.registry.get("gmail_list_labels").call(ctx, {})
        assert parse_qs(refresh.calls[0].request.content.decode())["grant_type"] == ["refresh_token"]
        auths = [c.request.headers["authorization"] for c in labels.calls]
    assert first == {"labels": []} and second["labels"][0]["id"] == "INBOX"
    assert auths == ["Bearer at2", "Bearer at2", "Bearer at3"]
    stored = json.loads(keychain["integration:gmail"])
    assert stored["access_token"] == "at3" and stored["refresh_token"] == "rt1"


async def test_revoked_refresh_token_marks_error(app, ctx, keychain):
    await connect_google(app, keychain, "gmail", expires_in=-10)
    with respx.mock() as router:
        router.post("https://oauth2.googleapis.com/token").mock(return_value=httpx.Response(400, json={"error": "invalid_grant"}))
        res = await app.registry.get("gmail_list_labels").call(ctx, {})
    assert "Reconnect" in res["error"]
    assert (await app.integrations.integration("gmail"))["status"] == "error"


async def test_gmail_search_privacy_send_and_reply(app, ctx, keychain):
    await connect_google(app, keychain, "gmail")
    await app.integrations.set_privacy_filters("gmail", {"emails": ["boss@x.com"]})
    with respx.mock() as router:
        mock_inbox(router)
        res = await app.registry.get("gmail_search").call(ctx, {"query": "newer_than:1d"})
        assert router.calls[0].request.url.params["q"] == "newer_than:1d"
        send = router.post(host=GMAIL, path=f"{MSG_PATH}/send").mock(
            return_value=httpx.Response(200, json={"id": "s1", "threadId": "t-m2"}))
        sent = await app.registry.get("gmail_send").call(ctx, {"to": "a@b.com", "subject": "Hi", "body": "Hello there"})
        raw = base64.urlsafe_b64decode(json.loads(send.calls.last.request.content)["raw"]).decode()
        replied = await app.registry.get("gmail_reply").call(ctx, {"message_id": "m2", "body": "Sure!"})
        reply_req = json.loads(send.calls.last.request.content)
        reply_raw = base64.urlsafe_b64decode(reply_req["raw"]).decode()
    assert res["count"] == 1 and res["hidden_by_privacy_filters"] == 1
    msg = res["messages"][0]
    assert msg["sender_email"] == "jane@y.com" and msg["body"] == "Lunch at 1?"
    assert sent == {"sent": True, "id": "s1", "thread_id": "t-m2"}
    assert "To: a@b.com" in raw and "Subject: Hi" in raw and "Hello there" in raw
    assert replied["sent"] is True and reply_req["threadId"] == "t-m2"
    assert "In-Reply-To: <abc@y.com>" in reply_raw and "Subject: Re: Lunch" in reply_raw and "To: Jane Doe <jane@y.com>" in reply_raw


async def test_gmail_poll_source_normalized_filtered_deduped(app, keychain):
    assert await app.integrations.poll_source("gmail", None) == []  # not connected
    await connect_google(app, keychain, "gmail")
    await app.integrations.set_privacy_filters("gmail", {"keywords": ["salary"]})
    with respx.mock() as router:
        mock_inbox(router)
        items = await app.integrations.poll_source("gmail", "2026-09-15T08:00:00+00:00")
        q = router.calls[0].request.url.params["q"]
        again = await app.integrations.poll_source("gmail", None)
    assert q == f"in:inbox after:{int(datetime(2026, 9, 15, 8, tzinfo=UTC).timestamp())}"
    assert len(items) == 1
    item = items[0]
    assert set(item) == {"id", "thread_id", "from", "sender_email", "to", "subject", "snippet", "body", "date", "labels", "url"}
    assert item["id"] == "m2" and item["thread_id"] == "t-m2" and item["subject"] == "Lunch"
    assert item["date"].startswith("2026-09-15T09:00") and item["labels"] == ["INBOX", "UNREAD"]
    assert again == []
    state = await app.integrations.poll_state("gmail")
    assert set(state["seen"]) == {"m1", "m2"} and state["cursor"]


def cal_event(eid: str, summary: str, attendees: list[str], updated: str = "2026-09-15T10:00:00Z") -> dict:
    return {"id": eid, "summary": summary, "description": "", "status": "confirmed", "htmlLink": f"https://cal/{eid}",
            "start": {"dateTime": "2026-09-16T10:00:00+05:30"}, "end": {"dateTime": "2026-09-16T11:00:00+05:30"},
            "attendees": [{"email": a} for a in attendees], "organizer": {"email": "me@example.com"}, "updated": updated}


async def test_calendar_list_privacy_and_poll(app, ctx, keychain):
    app.config.assistant.timezone = "Asia/Kolkata"
    await connect_google(app, keychain, "gcalendar")
    await app.integrations.set_privacy_filters("gcalendar", {"emails": ["doctor@clinic.com"]})
    events = {"items": [cal_event("e1", "Standup", ["team@x.com"]), cal_event("e2", "Checkup", ["doctor@clinic.com"])]}
    with respx.mock() as router:
        route = router.get(host="www.googleapis.com", path="/calendar/v3/calendars/primary/events").mock(
            return_value=httpx.Response(200, json=events))
        res = await app.registry.get("gcal_list_events").call(ctx, {"time_min": "2026-09-16"})
        params = route.calls.last.request.url.params
        polled = await app.integrations.poll_source("gcalendar", None)
        poll_params = route.calls.last.request.url.params
    assert params["timeMin"] == "2026-09-16T00:00:00+05:30" and params["timeMax"].startswith("2026-10-16")
    assert [e["id"] for e in res["events"]] == ["e1"] and res["hidden_by_privacy_filters"] == 1
    assert "updatedMin" in poll_params
    assert [e["id"] for e in polled] == ["e1"]
    assert set(polled[0]) >= {"id", "summary", "description", "start", "end", "location", "attendees",
                              "organizer_email", "url", "status"}
    assert polled[0]["attendees"] == ["team@x.com"]


async def test_calendar_create_event_body(app, ctx, keychain):
    app.config.assistant.timezone = "Asia/Kolkata"
    await connect_google(app, keychain, "gcalendar")
    with respx.mock() as router:
        route = router.post(host="www.googleapis.com", path="/calendar/v3/calendars/primary/events").mock(
            return_value=httpx.Response(200, json=cal_event("new", "Dentist", [])))
        res = await app.registry.get("gcal_create_event").call(ctx, {"summary": "Dentist", "start": "2026-09-17T15:00",
                                                                     "duration_minutes": 45})
        body = json.loads(route.calls.last.request.content)
        assert route.calls.last.request.url.params["sendUpdates"] == "none"
    assert body["start"] == {"dateTime": "2026-09-17T15:00:00+05:30", "timeZone": "Asia/Kolkata"}
    assert body["end"]["dateTime"] == "2026-09-17T15:45:00+05:30"
    assert res["created"] is True


def test_free_slots_within_working_hours():
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Kolkata")
    start = datetime(2026, 9, 16, 8, 0, tzinfo=tz)
    end = datetime(2026, 9, 16, 23, 0, tzinfo=tz)
    busy = [(datetime(2026, 9, 16, 10, 0, tzinfo=tz), datetime(2026, 9, 16, 11, 0, tzinfo=tz)),
            (datetime(2026, 9, 16, 11, 15, tzinfo=tz), datetime(2026, 9, 16, 17, 0, tzinfo=tz))]
    slots = free_slots(busy, start, end, timedelta(minutes=30), dtime(9, 0), dtime(18, 0), tz)
    assert slots == [
        {"start": "2026-09-16T09:00:00+05:30", "end": "2026-09-16T10:00:00+05:30"},
        {"start": "2026-09-16T17:00:00+05:30", "end": "2026-09-16T18:00:00+05:30"},
    ]
