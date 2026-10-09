from __future__ import annotations

import pytest
import respx
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.channels.conftest import TOKEN, FakeTelegram
from tests.conftest import FakeProvider


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    app = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "routes.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def test_requires_token(client):
    assert client.get("/api/channels", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_channel_rest_flow(client, keychain):
    channels = client.get("/api/channels").json()
    assert [c["id"] for c in channels] == ["telegram", "discord", "whatsapp"]
    telegram = channels[0]
    assert telegram["status"] == "disconnected" and telegram["paired"] == []
    assert telegram["setup"]["fields"][0]["secret"] is True and "BotFather" in telegram["setup"]["instructions_md"]
    assert "discord.com/developers/applications" in channels[1]["setup"]["instructions_md"]

    assert client.post("/api/channels/telegram/pairing").status_code == 409
    fake = FakeTelegram()
    with respx.mock(assert_all_called=False) as router:
        fake.mount(router)
        r = client.post("/api/channels/telegram/connect", json={"fields": {"bot_token": "bad"}})
        assert r.status_code == 400 and "doesn't look like" in r.json()["detail"]
        r = client.post("/api/channels/telegram/connect", json={"fields": {"bot_token": "987654321:" + "C" * 35}})
        assert r.status_code == 400 and "didn't accept" in r.json()["detail"]
        r = client.post("/api/channels/telegram/connect", json={"fields": {"bot_token": TOKEN}})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "connected" and r.json()["account_label"] == "@sentient_test_bot"
        assert keychain["channel_telegram_token"] == TOKEN

        pairing = client.post("/api/channels/telegram/pairing").json()
        assert len(pairing["code"]) == 6 and pairing["expires_at"]
        assert "@sentient_test_bot" in pairing["instructions"] and pairing["code"] in pairing["instructions"]

        assert client.post("/api/channels/telegram/test", json={}).json() == {"ok": True}
        assert client.post("/api/channels/telegram/test", json={"chat_id": "5"}).json()["ok"] is False

    assert client.patch("/api/channels/telegram/paired/5", json={"deliver": False}).status_code == 404
    assert client.patch("/api/channels/telegram/paired/5", json={"deliver": "yes"}).status_code == 400
    assert client.delete("/api/channels/telegram/paired/5").status_code == 404
    assert client.post("/api/channels/nope/connect", json={"fields": {}}).status_code == 404

    r = client.post("/api/channels/telegram/disconnect")
    assert r.status_code == 200 and r.json()["status"] == "disconnected"
    assert "channel_telegram_token" not in keychain
    assert client.post("/api/channels/telegram/test").json()["ok"] is False
