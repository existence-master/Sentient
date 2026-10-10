from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    app = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "routes.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def test_requires_token(client):
    assert client.get("/api/integrations", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_list_and_get(client):
    items = client.get("/api/integrations").json()
    ids = [i["id"] for i in items]
    assert ids[:6] == ["internet_search", "web", "weather", "maps", "news", "charts"] and "gmail" in ids
    gmail = client.get("/api/integrations/gmail").json()
    assert gmail["auth_type"] == "oauth" and gmail["setup"]["fields"]
    assert client.get("/api/integrations/nope").status_code == 404


def test_api_key_connect_test_disconnect(client, keychain):
    with respx.mock() as router:
        router.get("https://api.notion.com/v1/users/me").mock(
            return_value=httpx.Response(200, json={"object": "user", "name": "Sentient", "bot": {"workspace_name": "Existence"}}))
        r = client.post("/api/integrations/notion/connect", json={"fields": {"token": "ntn_abc"}})
        assert r.status_code == 200, r.text
        assert r.json()["connected"] is True and r.json()["account_label"] == "Existence"
        t = client.post("/api/integrations/notion/test").json()
    assert t["ok"] is True
    assert "integration:notion" in keychain
    d = client.post("/api/integrations/notion/disconnect").json()
    assert d["connected"] is False and "integration:notion" not in keychain


def test_connect_errors_are_400(client):
    with respx.mock() as router:
        router.get("https://api.github.com/user").mock(return_value=httpx.Response(401))
        r = client.post("/api/integrations/github/connect", json={"fields": {"token": "bad"}})
    assert r.status_code == 400 and "didn't accept" in r.json()["detail"]
    r = client.post("/api/integrations/gmail/connect", json={"fields": {}})
    assert r.status_code == 400 and "Client ID" in r.json()["detail"]
    assert client.post("/api/integrations/nope/connect", json={}).status_code == 404


def test_google_connect_returns_auth_url(client, keychain):
    r = client.post("/api/integrations/gcalendar/connect", json={"fields": {"client_id": "cid", "client_secret": "sec"}})
    assert r.status_code == 200
    body = r.json()
    assert body["auth_url"].startswith("https://accounts.google.com/o/oauth2/v2/auth?") and body["state"]
    assert client.get("/api/integrations/gcalendar").json()["status"] == "connecting"


def test_builtin_test_and_connect_noop(client):
    assert client.post("/api/integrations/weather/test").json()["ok"] is True
    assert client.post("/api/integrations/weather/connect").json()["connected"] is True


def test_privacy_filters_routes(client):
    assert client.get("/api/integrations/gmail/privacy-filters").json() == {"keywords": [], "emails": [], "labels": []}
    body = {"keywords": ["salary"], "emails": ["boss@x.com"], "labels": ["Personal"]}
    assert client.put("/api/integrations/gmail/privacy-filters", json=body).json() == {"ok": True}
    assert client.get("/api/integrations/gmail/privacy-filters").json() == body
    assert client.put("/api/integrations/notion/privacy-filters", json=body).status_code == 400


def test_mcp_routes_validation(client):
    assert client.get("/api/integrations/mcp").json() == []
    assert client.post("/api/integrations/mcp", json={"name": "x", "transport": "stdio"}).status_code == 400
    assert client.post("/api/integrations/mcp", json={"name": "x", "transport": "ftp"}).status_code == 422
    assert client.delete("/api/integrations/mcp/none").status_code == 404
    assert client.post("/api/integrations/mcp/none/test").status_code == 404
    assert client.post("/api/integrations/mcp/none/sign-in").status_code == 404
    assert client.post("/api/integrations/mcp/none/sign-out").status_code == 404
    assert client.post("/api/integrations/mcp", json={"name": "x", "transport": "http", "url": "http://x.test/mcp",
                                                      "auth": "token"}).status_code == 422
    assert client.post("/api/integrations/mcp", json={"name": "x", "transport": "http", "url": "http://x.test/mcp",
                                                      "auth": "headers"}).status_code == 400
    local = client.post("/api/integrations/mcp", json={"name": "local", "command": "x", "enabled": False}).json()
    assert local["auth"] == "none" and local["header_keys"] == [] and local["signed_in"] is False
    assert local["missing_values"] == []
    assert client.post("/api/integrations/mcp/local/sign-in").status_code == 400
    assert client.post("/api/integrations/mcp/none/values", json={"values": {}}).status_code == 404
    assert client.post("/api/integrations/mcp/local/values", json={"values": {"API_KEY": "v"}}).status_code == 400
    assert client.post("/api/integrations/mcp/local/values", json={"enable": "yes"}).status_code == 422
    kept = client.post("/api/integrations/mcp/local/values", json={"values": {}}).json()
    assert kept["name"] == "local" and kept["status"] == "disabled"
