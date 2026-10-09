import functools
import json

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider(replies=["hello"])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "r.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        c.llm = llm
        yield c


def test_bootstrap_and_onboarding(client):
    b = client.get("/api/bootstrap").json()
    assert b["assistant"]["onboarding_complete"] is False
    r = client.post(
        "/api/onboarding",
        json={"user_name": "Sarthak", "assistant_name": "Nova", "timezone": "Asia/Kolkata",
              "location": "Pune, India", "professional_context": "Founder of Existence", "persona": "professional"},
    )
    assert r.json() == {"ok": True}
    b = client.get("/api/bootstrap").json()
    assert b["assistant"]["onboarding_complete"] is True and b["assistant"]["name"] == "Nova"
    ws = client.get("/api/memories/workspace").json()
    assert "chief of staff" in ws["soul"] and "Founder of Existence" in ws["user"]


def test_config_patch_validates(client):
    r = client.patch("/api/config", json={"tasks": {"max_concurrent_runs": 4}})
    assert r.json()["config"]["tasks"]["max_concurrent_runs"] == 4
    bad = client.patch("/api/config", json={"tools": {"approvals": {"mode": "sometimes"}}})
    assert bad.status_code == 422


def test_sessions_crud_and_search(client):
    r = client.post("/api/chat", json={"text": "remember the blue notebook"})
    sid = json.loads(r.text.splitlines()[0])["session_id"]
    assert any(s["id"] == sid for s in client.get("/api/sessions").json())
    hits = client.get("/api/sessions/search", params={"q": "blue notebook"}).json()
    assert hits and hits[0]["session_id"] == sid
    assert client.patch(f"/api/sessions/{sid}", json={"title": "Notebook"}).json()["ok"]
    assert client.get("/api/sessions").json()[0]["title"] == "Notebook"
    listed = client.get("/api/sessions").json()[0]
    assert listed["untrusted"] == "" and listed["visited_hosts"] is None  # checked and clean, no web pages yet
    client.delete(f"/api/sessions/{sid}")
    assert client.get("/api/sessions").json() == []


def test_file_upload_and_attachment_reaches_model(client):
    up = client.post("/api/files", files={"file": ("notes.txt", b"the launch code is 42", "text/plain")}).json()
    assert up["name"] == "uploads/notes.txt"
    assert any(f["name"] == "uploads/notes.txt" for f in client.get("/api/files").json())
    client.post("/api/chat", json={"text": "what is the code?", "attachments": [up["name"]]})
    last_user = [m for m in client.llm.calls[-1]["messages"] if m["role"] == "user"][-1]
    assert "the launch code is 42" in last_user["content"]
    assert client.get("/api/files/content/uploads/notes.txt").content == b"the launch code is 42"
    # encoded so the HTTP client does not normalise the traversal away before it reaches the server
    assert client.get("/api/files/content/uploads%2F..%2F..%2Fconfig.yaml").status_code in {400, 404}


def test_notifications_roundtrip(client):
    core = client.app.state.sentient
    client.portal.call(functools.partial(core.notify, "info", "Hello there", title="Hi"))
    data = client.get("/api/notifications").json()
    assert data["unread"] == 1 and data["notifications"][0]["message"] == "Hello there"
    nid = data["notifications"][0]["id"]
    client.post(f"/api/notifications/{nid}/read")
    assert client.get("/api/notifications").json()["unread"] == 0
    client.delete("/api/notifications")
    assert client.get("/api/notifications").json()["notifications"] == []


def test_models_routes(client, monkeypatch):
    provs = client.get("/api/models/providers").json()
    assert any(p["id"] == "anthropic" and p["key_required"] for p in provs)
    r = client.put("/api/models/roles", json={"fast": "ollama_chat/qwen3:4b"}).json()
    assert r["fast"] == "ollama_chat/qwen3:4b"
    assert client.put("/api/models/roles", json={"primary": ""}).status_code == 400
    t = client.post("/api/models/test", json={"model": "fake/any"}).json()
    assert t["ok"] is True
    client.post("/api/chat", json={"text": "hi"})
    usage = client.get("/api/usage").json()
    assert usage["totals"]["prompt_tokens"] >= 1 and usage["by_source"][0]["source"] == "chat"


def test_websocket_forwards_bus_events(client):
    core = client.app.state.sentient
    with client.websocket_connect("/ws?token=test-token") as ws:
        assert ws.receive_json()["type"] == "hello"
        client.portal.call(functools.partial(core.notify, "task", "Task finished", payload={"task_id": "t1"}))
        msg = ws.receive_json()
        assert msg["type"] == "notification.new" and msg["data"]["task_id"] == "t1"


def test_patch_config_null_removes_map_entry_and_structured_errors(client):
    client.patch("/api/config", json={"models": {"temperature": {"primary": 0.3}}})
    cfg = client.patch("/api/config", json={"models": {"temperature": {"primary": None}}}).json()["config"]
    assert "primary" not in cfg["models"]["temperature"]
    # null on a regular nested field still means "set to null" (validated by the schema)
    cfg = client.patch("/api/config", json={"models": {"roles": {"planner": None}}}).json()["config"]
    assert cfg["models"]["roles"]["planner"] is None
    bad = client.patch("/api/config", json={"tools": {"approvals": {"mode": "sometimes"}}})
    assert bad.status_code == 422
    assert bad.json()["detail"][0]["loc"][-1] == "mode"


def test_sessions_list_gives_visited_hosts_as_a_list(client):
    r = client.post("/api/chat", json={"text": "hi"})
    sid = json.loads(r.text.splitlines()[0])["session_id"]
    store = client.app.state.sentient.store
    client.portal.call(store.execute, "UPDATE sessions SET visited_hosts = ? WHERE id = ?", ('["news.example"]', sid))
    listed = next(x for x in client.get("/api/sessions").json() if x["id"] == sid)
    assert listed["visited_hosts"] == ["news.example"]
