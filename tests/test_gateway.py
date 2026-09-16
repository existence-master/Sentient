import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider, tool_call


@pytest.fixture
def client(config, isolated_home):
    llm = FakeProvider(replies=["hello from gateway", [tool_call("current_datetime")], "done"])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "g.db"))
    with TestClient(app) as c:
        c.token = c.app.state.token
        yield c


def test_health_open_but_api_needs_token(client):
    assert client.get("/api/health").json()["ok"] is True
    assert client.get("/api/tools").status_code == 401
    r = client.get("/api/tools", headers={"Authorization": f"Bearer {client.token}"})
    assert r.status_code == 200 and any(p["id"] == "memory" for p in r.json())


def test_chat_ndjson_stream(client):
    r = client.post("/api/chat", json={"text": "hi"}, headers={"Authorization": f"Bearer {client.token}"})
    lines = [json.loads(line) for line in r.text.strip().splitlines()]
    assert lines[0]["type"] == "session"
    assert lines[-1]["type"] == "done" and lines[-1]["content"] == "hello from gateway"


def test_websocket_chat_with_tool(client):
    def turn(ws, text):
        ws.send_json({"type": "chat.send", "text": text})
        seen = []
        while True:
            msg = ws.receive_json()
            seen.append(msg["type"])
            if msg["type"] == "done":
                return seen

    with client.websocket_connect(f"/ws?token={client.token}") as ws:
        assert ws.receive_json()["type"] == "hello"
        first = turn(ws, "hi")           # scripted reply 1: plain text
        assert first[0] == "session" and "text_delta" in first
        second = turn(ws, "time?")       # scripted reply 2: tool call, then reply 3
        assert "tool_call" in second and "tool_result" in second


def test_websocket_rejects_bad_token(client):
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws?token=nope") as ws:
        ws.receive_json()


def test_config_schema_and_update(client):
    h = {"Authorization": f"Bearer {client.token}"}
    schema = client.get("/api/config/schema", headers=h).json()
    assert "AssistantConfig" in schema["$defs"]
    cfg = client.get("/api/config", headers=h).json()
    cfg["assistant"]["name"] = "Jarvis"
    assert client.put("/api/config", json=cfg, headers=h).json()["saved"] is True
    assert client.get("/api/bootstrap", headers=h).json()["assistant"]["name"] == "Jarvis"
