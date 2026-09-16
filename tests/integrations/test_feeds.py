"""Change feeds (Gmail history, Calendar sync tokens), the shared seen record, backoff and feed_active."""

from __future__ import annotations

import base64
import json
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from sentient.integrations import feeds as feeds_mod
from sentient.integrations.base import IntegrationError

GMAIL = "gmail.googleapis.com"
MSG_PATH = "/gmail/v1/users/me/messages"
HISTORY = "/gmail/v1/users/me/history"
PROFILE = "/gmail/v1/users/me/profile"
CAL = "www.googleapis.com"
EVENTS = "/calendar/v3/calendars/primary/events"


def b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


async def connect_google(app, keychain, plugin_id: str) -> None:
    keychain["google_oauth_client"] = json.dumps({"client_id": "cid", "client_secret": "csec"})
    await app.integrations.store_credentials(plugin_id, {"access_token": "at1", "refresh_token": "rt1",
                                                         "expires_at": time.time() + 3600})
    await app.integrations.mark_connected(plugin_id, "me@example.com")


def gmail_message(mid: str, frm: str, subject: str, body: str, labels=("INBOX", "UNREAD")) -> dict:
    headers = [{"name": "From", "value": frm}, {"name": "To", "value": "me@example.com"},
               {"name": "Subject", "value": subject}]
    return {"id": mid, "threadId": f"t-{mid}", "snippet": subject, "labelIds": list(labels),
            "internalDate": str(int(datetime(2026, 9, 15, 9, 0, tzinfo=UTC).timestamp() * 1000)),
            "payload": {"mimeType": "text/plain", "headers": headers, "body": {"data": b64(body)}}}


def drain(q) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def source_items(events: list[dict]) -> list[dict]:
    return [e["data"] for e in events if e["type"] == "source.items"]


def added(*pairs: tuple[str, list[str]]) -> dict:
    return {"messagesAdded": [{"message": {"id": mid, "labelIds": labels}} for mid, labels in pairs]}


# ----------------------------------------------------------------------------- gmail history
async def test_gmail_history_bootstrap_incremental_privacy_and_dedupe(app, keychain):
    mgr = app.integrations
    await connect_google(app, keychain, "gmail")
    await mgr.set_privacy_filters("gmail", {"emails": ["boss@x.com"]})
    with respx.mock() as router:
        profile = router.get(host=GMAIL, path=PROFILE).mock(
            return_value=httpx.Response(200, json={"emailAddress": "me@example.com", "historyId": "100"}))
        history = router.get(host=GMAIL, path=HISTORY).mock(return_value=httpx.Response(200, json={
            "historyId": "105",
            "history": [{"id": "101", **added(("m1", ["INBOX", "UNREAD"]))},
                        {"id": "102", **added(("m2", ["INBOX"]), ("s1", ["SENT"]))}]}))
        router.get(host=GMAIL, path=f"{MSG_PATH}/m1").mock(return_value=httpx.Response(200, json=gmail_message(
            "m1", "Boss <boss@x.com>", "Salary", "Let's talk.")))
        router.get(host=GMAIL, path=f"{MSG_PATH}/m2").mock(return_value=httpx.Response(200, json=gmail_message(
            "m2", "Jane <jane@y.com>", "Lunch", "Lunch at 1?")))
        async with app.bus.subscribe() as q:
            first = await mgr.feeds.sync("gmail")
            assert not history.called and profile.call_count == 1
            second = await mgr.feeds.sync("gmail")
            params = history.calls.last.request.url.params
            third = await mgr.feeds.sync("gmail")  # the same history again: nothing new
            events = drain(q)
    assert first == {"ok": True, "emitted": 0, "rebaselined": True}
    assert params["startHistoryId"] == "100" and params["labelId"] == "INBOX"
    assert params["historyTypes"] == "messageAdded"
    assert second == {"ok": True, "emitted": 1, "rebaselined": False} and third["emitted"] == 0
    batches = source_items(events)
    assert len(batches) == 1
    batch = batches[0]
    assert (batch["source"], batch["event"], batch["origin"]) == ("gmail", "new_email", "feed")
    item = batch["items"][0]
    assert item["id"] == "m2" and item["sender_email"] == "jane@y.com" and "_key" not in item
    assert set(item) == {"id", "thread_id", "from", "sender_email", "to", "subject", "snippet", "body", "date",
                         "labels", "url"}
    st = await mgr.feeds.state("gmail")
    assert st["cursor"] == "105" and st["status"] == "ok" and st["failures"] == 0 and st["emitted"] == 1
    # the privacy-filtered message is recorded as seen so it can never leak later
    keys = {r["item_key"] for r in await app.store.fetchall("SELECT item_key FROM integration_seen WHERE source='gmail'")}
    assert keys == {"m1", "m2"}


async def test_gmail_expired_history_rebaselines_without_flooding(app, keychain):
    mgr = app.integrations
    await connect_google(app, keychain, "gmail")
    await mgr.feeds.save("gmail", cursor="5")
    with respx.mock() as router:  # any message fetch would be an unmocked request and fail the test
        router.get(host=GMAIL, path=HISTORY).mock(
            return_value=httpx.Response(404, json={"error": {"message": "Requested entity was not found."}}))
        router.get(host=GMAIL, path=PROFILE).mock(return_value=httpx.Response(200, json={"historyId": "900"}))
        async with app.bus.subscribe() as q:
            res = await mgr.feeds.sync("gmail")
            events = drain(q)
    assert res == {"ok": True, "emitted": 0, "rebaselined": True}
    assert source_items(events) == []
    st = await mgr.feeds.state("gmail")
    assert st["cursor"] == "900" and st["failures"] == 0 and "expired" in st["note"]


# ----------------------------------------------------------------------------- calendar sync tokens
def cal_event(eid: str, summary: str, *, created: datetime, updated: datetime, start: datetime,
              status: str = "confirmed") -> dict:
    return {"id": eid, "summary": summary, "description": "", "status": status, "htmlLink": f"https://cal/{eid}",
            "created": created.isoformat().replace("+00:00", "Z"), "updated": updated.isoformat().replace("+00:00", "Z"),
            "start": {"dateTime": start.isoformat()}, "end": {"dateTime": (start + timedelta(hours=1)).isoformat()},
            "attendees": [], "organizer": {"email": "me@example.com"}}


async def test_calendar_sync_token_new_updated_privacy_and_410(app, keychain):
    mgr = app.integrations
    await connect_google(app, keychain, "gcalendar")
    await mgr.set_privacy_filters("gcalendar", {"keywords": ["therapy"]})
    now = datetime.now(UTC)
    soon = now + timedelta(days=2)
    fresh = cal_event("e1", "Dinner", created=now, updated=now + timedelta(seconds=5), start=soon)
    moved = cal_event("e0", "Standup", created=now - timedelta(days=30), updated=now, start=soon)
    private = cal_event("e5", "Therapy", created=now, updated=now, start=soon)
    gone = cal_event("e9", "Cancelled", created=now, updated=now, start=soon, status="cancelled")
    past = cal_event("e8", "Yesterday", created=now, updated=now, start=now - timedelta(days=1))
    full_syncs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        if "syncToken" not in p:
            if p.get("pageToken") == "p2":
                full_syncs.append("done")
                return httpx.Response(200, json={"items": [fresh, moved], "nextSyncToken": f"full{len(full_syncs)}"})
            return httpx.Response(200, json={"items": [cal_event("old", "Old", created=now, updated=now, start=soon)],
                                             "nextPageToken": "p2"})
        if p["syncToken"] == "full1":
            return httpx.Response(200, json={"items": [fresh, moved, private, gone, past], "nextSyncToken": "s2"})
        return httpx.Response(410, json={"error": {"message": "Sync token is no longer valid, a full sync is required."}})

    with respx.mock() as router:
        route = router.get(host=CAL, path=EVENTS).mock(side_effect=handler)
        async with app.bus.subscribe() as q:
            baseline = await mgr.feeds.sync("gcalendar")
            baseline_events = drain(q)
            changes = await mgr.feeds.sync("gcalendar")
            change_events = drain(q)
            expired = await mgr.feeds.sync("gcalendar")
            expired_events = drain(q)
        assert all("timeMin" not in c.request.url.params for c in route.calls)
    assert baseline == {"ok": True, "emitted": 0, "rebaselined": True} and source_items(baseline_events) == []
    assert changes == {"ok": True, "emitted": 2, "rebaselined": False}
    by_event = {b["event"]: [i["id"] for i in b["items"]] for b in source_items(change_events)}
    assert by_event == {"new_event": ["e1"], "updated_event": ["e0"]}
    assert all(b["origin"] == "feed" and b["source"] == "gcalendar" for b in source_items(change_events))
    assert expired == {"ok": True, "emitted": 0, "rebaselined": True}
    assert source_items(expired_events) == []  # the full resync re-lists e1/e0 but never re-emits them
    st = await mgr.feeds.state("gcalendar")
    assert st["cursor"] == "full2" and "resync" in st["note"]
    triggers = (await mgr.integration("gcalendar"))["triggers"]
    assert [t["event"] for t in triggers] == ["new_event", "updated_event"]


# ----------------------------------------------------------------------------- poll + feed dedupe
async def test_poll_and_feed_never_emit_the_same_item_twice(app, keychain):
    mgr = app.integrations
    await connect_google(app, keychain, "gmail")
    await mgr.feeds.save("gmail", cursor="100")
    with respx.mock() as router:
        router.get(host=GMAIL, path=MSG_PATH).mock(side_effect=[
            httpx.Response(200, json={"messages": [{"id": "m1"}]}),
            httpx.Response(200, json={"messages": [{"id": "m1"}, {"id": "m2"}]}),
        ])
        router.get(host=GMAIL, path=f"{MSG_PATH}/m1").mock(return_value=httpx.Response(200, json=gmail_message(
            "m1", "Jane <jane@y.com>", "Lunch", "Lunch?")))
        router.get(host=GMAIL, path=f"{MSG_PATH}/m2").mock(return_value=httpx.Response(200, json=gmail_message(
            "m2", "Ravi <ravi@z.com>", "Invoice", "Attached.")))
        router.get(host=GMAIL, path=HISTORY).mock(return_value=httpx.Response(200, json={
            "historyId": "101", "history": [added(("m1", ["INBOX"]), ("m2", ["INBOX"]))]}))
        async with app.bus.subscribe() as q:
            polled = await mgr.poll_source("gmail", "2026-09-15T08:00:00+00:00")
            fed = await mgr.feeds.sync("gmail")
            # forget the legacy per-poll list so only the shared seen record can stop a repeat
            await app.store.execute("DELETE FROM integration_poll_state WHERE source = 'gmail'")
            again = await mgr.poll_source("gmail", "2026-09-15T08:00:00+00:00")
            events = drain(q)
    assert [i["id"] for i in polled] == ["m1"] and "_key" not in polled[0]
    assert fed["emitted"] == 1
    assert again == []
    assert [(b["origin"], [i["id"] for i in b["items"]]) for b in source_items(events)] == [("poll", ["m1"]), ("feed", ["m2"])]
    rows = await app.store.fetchall("SELECT item_key, origin FROM integration_seen WHERE source = 'gmail'")
    assert {r["item_key"]: r["origin"] for r in rows} == {"m1": "poll", "m2": "feed"}


async def test_emit_items_rejects_unknown_origin_and_skips_items_without_id(app):
    mgr = app.integrations
    with pytest.raises(ValueError):
        await mgr.emit_items("webhook", "carrier-pigeon", [{"id": "x"}])
    async with app.bus.subscribe() as q:
        out = await mgr.emit_items("webhook", "webhook", [{"body": "no id"}, {"id": "a"}, {"id": "a"}], event="h1")
        events = drain(q)
    assert out == [{"id": "a"}]
    assert source_items(events) == [{"source": "webhook", "event": "h1", "origin": "webhook", "items": [{"id": "a"}]}]


# ----------------------------------------------------------------------------- backoff, status, feed_active
async def test_feed_failure_backoff_and_readable_error(app, keychain):
    mgr = app.integrations
    app.config.integrations.fast_sync_seconds = 60
    await connect_google(app, keychain, "gmail")
    await mgr.feeds.save("gmail", cursor="100")
    with respx.mock() as router:
        route = router.get(host=GMAIL, path=HISTORY).mock(
            return_value=httpx.Response(500, json={"error": {"message": "Backend Error"}}))
        r1 = await mgr.feeds.sync("gmail")
        r2 = await mgr.feeds.sync("gmail")
        calls = route.call_count
        assert await mgr.sync_feeds_now() == 0  # still waiting out the backoff: no request
        assert route.call_count == calls
    assert r1["ok"] is False and r1["error"] == "Gmail couldn't be checked (HTTP 500: Backend Error)."
    assert (r1["retry_in_s"], r2["retry_in_s"], r2["failures"]) == (60, 120, 2)
    state = await mgr.poll_state("gmail")
    assert state["feed"]["last_error"] == r1["error"] and state["feed"]["failures"] == 2
    assert state["feed"]["next_attempt_at"] > datetime.now(UTC).isoformat()
    st = await mgr.feeds.state("gmail")
    assert st["cursor"] == "100"  # a failure never moves the cursor
    assert feeds_mod.backoff_seconds(20, 60) == feeds_mod.BACKOFF_MAX_S
    with respx.mock() as router:
        router.get(host=GMAIL, path=HISTORY).mock(side_effect=httpx.ConnectError("offline"))
        r3 = await mgr.feeds.sync("gmail")
    assert r3["error"] == "Couldn't reach Gmail (ConnectError). Check the internet connection."


async def test_feed_active_and_status(app, keychain, monkeypatch):
    mgr = app.integrations
    monkeypatch.setattr(feeds_mod, "INITIAL_DELAY_S", 3600)  # the loop exists but never syncs in this test
    assert mgr.feed_active("gmail") is False  # not connected
    await connect_google(app, keychain, "gmail")
    assert mgr.feed_active("gmail") is False  # change feeds not running (background off)
    mgr.feeds.start_loop()
    try:
        assert mgr.feed_active("gmail") is True
        assert mgr.feed_active("gcalendar") is False and mgr.feed_active("weather") is False
        assert mgr.feed_active("email_imap") is False and mgr.feed_active("nope") is False
        app.config.integrations.fast_sync_seconds = 0
        assert mgr.feed_active("gmail") is False
        app.config.integrations.fast_sync_seconds = 60
        for _ in range(feeds_mod.ACTIVE_MAX_FAILURES):
            await mgr.feeds.record_failure("gmail", IntegrationError("Gmail is down."))
        assert mgr.feed_active("gmail") is False  # proactivity falls back to polling
        await mgr.feeds.record_success("gmail", cursor="7")
        assert mgr.feed_active("gmail") is True
        status = {s["source"]: s for s in await mgr.feed_status()}
        assert set(status) == {"gmail", "gcalendar", "email_imap"}
        g = status["gmail"]
        assert (g["status"], g["kind"], g["active"], g["connected"]) == ("ok", "gmail_history", True, True)
        assert status["gcalendar"]["status"] == "disconnected" and status["email_imap"]["kind"] == "imap_idle"
    finally:
        await mgr.feeds.stop()
    assert mgr.feed_active("gmail") is False
    await mgr.disconnect("gmail")
    assert (await mgr.feeds.state("gmail"))["cursor"] is None


def test_feed_routes(config, isolated_home, monkeypatch, keychain):
    from fastapi.testclient import TestClient

    from sentient.app import SentientApp
    from sentient.gateway.app import create_app
    from tests.conftest import FakeProvider

    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "feed-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "feeds.db", enable_background=False)
    with TestClient(create_app(core)) as client:
        client.headers["Authorization"] = "Bearer feed-token"
        rows = client.get("/api/integrations/feeds").json()
        assert {r["source"] for r in rows} == {"gmail", "gcalendar", "email_imap"}
        assert client.post("/api/integrations/feeds/gmail/sync").json() == {"ok": False, "skipped": "not connected",
                                                                             "emitted": 0}
        assert client.post("/api/integrations/feeds/weather/sync").status_code == 404
        assert client.get("/api/integrations/feeds", headers={"Authorization": "Bearer nope"}).status_code == 401
