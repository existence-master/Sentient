"""Steering over the WebSocket and the NDJSON fallback (docs/API.md section 10)."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call


@pytest.fixture
def gated(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    started = threading.Event()
    release = threading.Event()

    @tool("slow_lookup", risk=Risk.read)
    async def slow_lookup(ctx: ToolContext) -> dict:
        """Slow lookup."""
        started.set()
        await asyncio.to_thread(release.wait, 10)
        return {"ok": True}

    class Slow(ToolPlugin):
        id = "slow"
        display_name = "Slow"
        tools = [slow_lookup]

    llm = FakeProvider(replies=[[tool_call("slow_lookup")], "final answer", "second reply"])
    s = SentientApp(config, llm=llm, db_path=isolated_home / "steer.db", enable_background=False)
    app = create_app(s)
    with TestClient(app) as c:
        s.registry.register(Slow())
        c.headers.update({"Authorization": "Bearer test-token"})
        c.llm, c.started, c.release = llm, started, release
        try:
            yield c
        finally:
            release.set()


def _until(ws, kind: str, seen: list[dict]) -> dict:
    while True:
        msg = ws.receive_json()
        seen.append(msg)
        if msg["type"] == kind:
            return msg


def test_websocket_send_during_reply_steers_it(gated):
    with gated.websocket_connect("/ws?token=test-token") as ws:
        assert ws.receive_json()["type"] == "hello"
        seen: list[dict] = []
        ws.send_json({"type": "chat.send", "text": "look it up"})
        sid = _until(ws, "session", seen)["session_id"]
        _until(ws, "tool_call", seen)
        assert gated.started.wait(5)
        ws.send_json({"type": "chat.send", "session_id": sid, "text": "and use metric units", "client_id": "k1"})
        ack = _until(ws, "steer_ack", seen)
        assert ack == {"type": "steer_ack", "session_id": sid, "queued": True, "client_id": "k1"}
        gated.release.set()
        inter = _until(ws, "user_interjection", seen)
        assert inter["text"] == "and use metric units"
        done = _until(ws, "done", seen)
        assert done["content"] == "final answer"
        assert {"role": "user", "content": "and use metric units"} in gated.llm.calls[1]["messages"]

        # no reply running: chat.steer behaves like chat.send
        seen = []
        ws.send_json({"type": "chat.steer", "session_id": sid, "text": "hello again"})
        assert _until(ws, "steer_ack", seen)["queued"] is False
        _until(ws, "session", seen)
        assert _until(ws, "done", seen)["content"] == "second reply"


def test_ndjson_chat_steers_running_reply(gated):
    sid = gated.post("/api/sessions").json()["session_id"]
    out: dict = {}

    def first():
        out["lines"] = [json.loads(x) for x in gated.post("/api/chat", json={"text": "look", "session_id": sid}).text.splitlines()]

    t = threading.Thread(target=first)
    t.start()
    try:
        assert gated.started.wait(5)
        r = gated.post("/api/chat", json={"text": "steer me", "session_id": sid})
        assert [json.loads(x) for x in r.text.splitlines()] == [{"type": "steer_ack", "session_id": sid, "queued": True}]
    finally:
        gated.release.set()
        t.join(15)
    types = [x["type"] for x in out["lines"]]
    assert "user_interjection" in types and types[-1] == "done"
