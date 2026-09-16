"""OAuth cancel and human-readable poll errors (desktop UI findings)."""

import pytest

from sentient.integrations.base import IntegrationError


async def test_cancel_pending_google_sign_in(app):
    mgr = app.integrations
    started = await mgr.start_google_flow("gmail", {"client_id": "cid.apps.googleusercontent.com", "client_secret": "sec"})
    try:
        assert started["auth_url"].startswith("https://accounts.google.com/")
        assert (await mgr.integration("gmail"))["status"] == "connecting"

        events: list[dict] = []
        async with app.bus.subscribe() as q:
            out = await mgr.cancel_connect("gmail")
            while not q.empty():
                events.append(q.get_nowait())

        assert out["status"] == "disconnected" and out["connected"] is False
        assert any(e["type"] == "integration.updated" and e["data"]["id"] == "gmail" for e in events)
        # the abandoned state can no longer complete
        ok, _ = await mgr._oauth_callback({"state": started["state"], "code": "late"})
        assert ok is False
    finally:
        await mgr.listener.stop()


async def test_cancel_unknown_integration_raises(app):
    with pytest.raises(KeyError):
        await app.integrations.cancel_connect("nope")


async def test_poll_error_is_readable(app, monkeypatch):
    mgr = app.integrations
    plugin = mgr.plugin("gcalendar")

    async def broken_poll(manager, since):
        raise KeyError("gcalendar")

    monkeypatch.setattr(plugin, "poll", broken_poll)
    monkeypatch.setattr(mgr, "_connected_sync", lambda pid: True)
    with pytest.raises(IntegrationError) as info:
        await mgr.poll_source("gcalendar", None)

    message = str(info.value)
    assert message != "gcalendar" and message != "'gcalendar'"
    assert "Google Calendar" in message and "KeyError" in message
    assert (await mgr.poll_state("gcalendar"))["last_error"] == message


def test_cancel_route(config, isolated_home, monkeypatch):
    from fastapi.testclient import TestClient

    from sentient.app import SentientApp
    from sentient.gateway.app import create_app
    from tests.conftest import FakeProvider

    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "cancel-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "cancel.db", enable_background=False)
    with TestClient(create_app(core)) as client:
        client.headers["Authorization"] = "Bearer cancel-token"
        assert client.post("/api/integrations/gmail/cancel").json()["status"] == "disconnected"
        assert client.post("/api/integrations/nope/cancel").status_code == 404
