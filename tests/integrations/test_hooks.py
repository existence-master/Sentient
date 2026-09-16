"""Inbound webhooks: management routes, public POST /hooks/{id}, secrets, limits, publishing."""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider


@pytest.fixture
def core(config, isolated_home):
    return SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "hooks.db", enable_background=False)


@pytest.fixture
def published(core, monkeypatch) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    original = core.bus.publish

    def publish(type_, data=None):
        events.append((type_, data))
        original(type_, data)

    monkeypatch.setattr(core.bus, "publish", publish)
    return events


@pytest.fixture
def client(core, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "hook-token")
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer hook-token"})
        yield c


def items_of(published) -> list[dict]:
    return [d for t, d in published if t == "source.items"]


def test_manage_hooks_token_secret_once_and_triggers(client, core, published):
    assert client.get("/api/hooks", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/api/hooks", json={"name": "x"}, headers={"Authorization": ""}).status_code == 401
    r = client.post("/api/hooks", json={"name": "  Build   finished "})
    assert r.status_code == 200, r.text
    hook = r.json()
    assert set(hook) == {"id", "name", "url", "created_at", "last_called_at", "calls", "secret"}
    assert hook["name"] == "Build finished" and hook["calls"] == 0 and hook["last_called_at"] is None
    assert hook["url"] == f"http://testserver/hooks/{hook['id']}" and len(hook["secret"]) >= 32
    listed = client.get("/api/hooks").json()
    assert [h["id"] for h in listed] == [hook["id"]] and "secret" not in listed[0]

    triggers =client.get("/api/integrations/webhook").json()
    assert triggers["connected"] is True and triggers["tools"] == []
    assert triggers["triggers"] == [{"event": hook["id"], "label": 'When "Build finished" is called'}]
    assert any(t == "integration.updated" and d["id"] == "webhook" for t, d in published)

    assert client.post("/api/hooks", json={"name": "   "}).status_code == 400
    assert client.post("/api/hooks", json={"name": "x" * 81}).status_code == 400
    assert client.post("/api/hooks", json={}).status_code == 422
    assert client.delete(f"/api/hooks/{hook['id']}").json() == {"ok": True}
    assert client.delete(f"/api/hooks/{hook['id']}").status_code == 404
    assert client.get("/api/integrations/webhook").json()["triggers"] == []


async def test_secret_is_stored_hashed(app):
    hook = await app.integrations.hooks.create("Door sensor")
    row = await app.store.fetchone("SELECT * FROM hooks WHERE id = ?", (hook["id"],))
    assert row["secret_hash"] == hashlib.sha256(hook["secret"].encode()).hexdigest()
    assert hook["secret"] not in str(dict(row))
    assert hook["url"] == f"/hooks/{hook['id']}"
    stored = await app.integrations.hooks.get(hook["id"])
    assert app.integrations.hooks.secret_ok(stored, hook["secret"]) is True
    assert app.integrations.hooks.secret_ok(stored, hook["secret"] + "x") is False
    assert app.integrations.hooks.secret_ok(stored, None) is False


def test_call_hook_without_bearer_publishes_every_body_kind(client, core, published):
    hook = client.post("/api/hooks", json={"name": "Build finished"}).json()
    secret, url = hook["secret"], f"/hooks/{hook['id']}"
    del client.headers["Authorization"]  # the public endpoint needs only the hook secret

    r = client.post(url, json={"status": "passed"}, headers={"X-Sentient-Secret": secret})
    assert r.status_code == 200 and r.json()["ok"] is True
    batch = items_of(published)[-1]
    assert (batch["source"], batch["event"], batch["origin"]) == ("webhook", hook["id"], "webhook")
    item = batch["items"][0]
    assert item["id"] == r.json()["item_id"] and item["name"] == "Build finished"
    assert item["body"] == {"status": "passed"} and item["content_type"] == "application/json"
    assert item["received_at"] and item["query"] == {}

    client.post(f"{url}?secret={secret}&run=7", data={"a": "1", "b": ["2", "3"]})
    item = items_of(published)[-1]["items"][0]
    assert item["body"] == {"a": "1", "b": ["2", "3"]} and item["query"] == {"run": "7"}

    client.post(url, content=b"door opened", headers={"X-Sentient-Secret": secret, "Content-Type": "text/plain"})
    assert items_of(published)[-1]["items"][0]["body"] == "door opened"

    client.post(url, content=b'{"raw": true}', headers={"X-Sentient-Secret": secret})
    assert items_of(published)[-1]["items"][0]["body"] == {"raw": True}

    client.post(url, data={"k": "v"}, files={"file": ("a.txt", b"abc", "text/plain")},
                headers={"X-Sentient-Secret": secret})
    assert items_of(published)[-1]["items"][0]["body"] == {
        "k": "v", "file": {"filename": "a.txt", "content_type": "text/plain", "size": 3}}

    client.post(url, headers={"X-Sentient-Secret": secret})
    assert items_of(published)[-1]["items"][0]["body"] is None

    client.headers["Authorization"] = "Bearer hook-token"
    listed = client.get("/api/hooks").json()[0]
    assert listed["calls"] == 6 and listed["last_called_at"]
    assert len(items_of(published)) == 6  # every call is its own item, never deduped away


def test_call_hook_errors(client, core, published):
    hook = client.post("/api/hooks", json={"name": "Alarm"}).json()
    url, secret = f"/hooks/{hook['id']}", hook["secret"]
    del client.headers["Authorization"]
    assert client.post("/hooks/unknown", json={}, headers={"X-Sentient-Secret": secret}).status_code == 404
    assert client.post(url, json={}).status_code == 401
    assert client.post(url, json={}, headers={"X-Sentient-Secret": "wrong"}).status_code == 401
    assert client.post(f"{url}?secret=wrong", json={}).status_code == 401
    assert client.post(url, json={}, headers={"Authorization": "Bearer hook-token"}).status_code == 401  # token isn't a secret
    assert client.get(url).status_code in (404, 405)  # only POST calls a hook (the UI mount may answer GET)
    core.config.integrations.webhook_max_body_kb = 1
    big = client.post(url, content=b"x" * 2048, headers={"X-Sentient-Secret": secret, "Content-Type": "text/plain"})
    assert big.status_code == 413 and "1 KB" in big.json()["detail"]
    bad = client.post(url, content=b"{nope", headers={"X-Sentient-Secret": secret, "Content-Type": "application/json"})
    assert bad.status_code == 400
    assert items_of(published) == []
    client.headers["Authorization"] = "Bearer hook-token"
    assert client.get("/api/hooks").json()[0]["calls"] == 0
