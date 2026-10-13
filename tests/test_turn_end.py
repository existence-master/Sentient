"""Every chat turn ends with exactly one ``done``, also when the model fails (#334, docs/API.md section 1)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.events import Done, Error
from sentient.llm.provider import ProviderError, StreamChunk
from tests.conftest import FakeProvider

FAILURE = "Can't reach the model at localhost:9. Check that it is running, or pick another model."


class FailingProvider(FakeProvider):
    """Fails every chat call, before or after a few words."""

    def __init__(self, *args, partial: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.partial = partial

    async def stream(self, role, messages, tools=None, *, model=None):
        self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
        if self.partial:
            yield StreamChunk(text=self.partial, model="fake")
        raise ProviderError(FAILURE)


@pytest.mark.parametrize("partial", ["", "Let me "])
async def test_a_failed_turn_ends_with_error_then_done(config, isolated_home, partial):
    app = SentientApp(config, llm=FailingProvider(partial=partial), db_path=isolated_home / "t.db",
                      enable_background=False)
    await app.start()
    try:
        sid = await app.store.create_session(channel="desktop")
        events = [e async for e in app.agent.run_turn(sid, "hi")]
        assert [e.type for e in events if isinstance(e, Error | Done)] == ["error", "done"]
        error, done = events[-2], events[-1]
        assert isinstance(error, Error) and not error.recoverable and error.message == FAILURE
        assert isinstance(done, Done) and done.error == FAILURE and done.message_id is None and not done.cancelled
        assert done.turn_id == error.turn_id and done.session_id == sid
        assert not app.agent.is_running(sid)
        rows = await app.store.recent_messages(sid, 10)
        assert [r["role"] for r in rows] == ["user"]  # nothing of the failed reply is saved
    finally:
        await app.stop()


async def test_a_finished_turn_has_no_error_on_its_done(config, isolated_home):
    app = SentientApp(config, llm=FakeProvider(["Hello."]), db_path=isolated_home / "t.db", enable_background=False)
    await app.start()
    try:
        sid = await app.store.create_session(channel="desktop")
        events = [e async for e in app.agent.run_turn(sid, "hi")]
        dones = [e for e in events if isinstance(e, Done)]
        assert len(dones) == 1 and dones[0].error is None and dones[0].message_id
    finally:
        await app.stop()


@pytest.fixture
def make_client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    clients = []

    def make(llm) -> TestClient:
        core = SentientApp(config, llm=llm, db_path=isolated_home / "g.db", enable_background=False)
        c = TestClient(create_app(core))
        c.__enter__()
        c.headers.update({"Authorization": "Bearer test-token"})
        c.core = core
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def ws_turn(client: TestClient, text: str = "hi") -> list[dict]:
    """Send one message over /ws and collect the chat events until its done."""
    out = []
    with client.websocket_connect("/ws?token=test-token") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "chat.send", "text": text, "client_id": "c1"})
        while True:
            msg = ws.receive_json()
            if "." in msg["type"]:  # domain events
                continue
            out.append(msg)
            if msg["type"] == "done":
                return out


def terminal(events: list[dict]) -> list[str]:
    return [e["type"] for e in events if e["type"] in {"error", "done"}]


def test_ws_and_rest_end_a_failed_turn_with_done(make_client):
    client = make_client(FailingProvider())
    events = ws_turn(client)
    assert terminal(events) == ["error", "done"]
    assert events[-2]["recoverable"] is False and events[-1]["error"] == FAILURE
    assert events[-1]["message_id"] is None

    r = client.post("/api/chat", json={"text": "hello again"})
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert terminal(lines) == ["error", "done"] and lines[-1]["error"] == FAILURE


def test_a_turn_that_breaks_still_ends_with_done(make_client, monkeypatch):
    client = make_client(FakeProvider())

    async def broken(*args, **kwargs):
        raise RuntimeError("the store went away")
        yield  # pragma: no cover

    monkeypatch.setattr(client.core.agent, "run_turn", broken)
    events = ws_turn(client)
    assert terminal(events) == ["error", "done"]
    assert events[-2]["recoverable"] is False and events[-1]["error"] == "the store went away"

    r = client.post("/api/chat", json={"text": "hello again"})
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert terminal(lines) == ["error", "done"] and lines[-1]["error"] == "the store went away"


def test_a_break_after_done_sends_no_second_done(make_client, monkeypatch):
    client = make_client(FakeProvider())

    async def late(*args, **kwargs):
        yield Done(content="fine", message_id="m1", session_id=args[0])
        raise RuntimeError("after the reply")

    monkeypatch.setattr(client.core.agent, "run_turn", late)
    r = client.post("/api/chat", json={"text": "hi"})
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert terminal(lines) == ["done"]
    with client.websocket_connect("/ws?token=test-token") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "chat.send", "text": "hi"})
        seen = []
        while not seen or seen[-1] != "done":
            msg = ws.receive_json()
            if "." not in msg["type"]:
                seen.append(msg["type"])
        ws.send_json({"type": "ping"})
        while (msg := ws.receive_json())["type"] != "pong":  # nothing else of that turn came before the pong
            if "." not in msg["type"]:
                seen.append(msg["type"])
    assert terminal([{"type": t} for t in seen]) == ["done"]
