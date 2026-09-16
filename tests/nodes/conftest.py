"""Device (node) test fixtures: a gateway client factory, hello builder and a threaded fake device."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider

TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////"
    "////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)


def hello(**overrides: Any) -> dict:
    msg = {
        "type": "hello",
        "protocol": 1,
        "name": "Test Glasses",
        "kind": "glasses",
        "platform": "pytest",
        "app_version": "0.1",
        "capabilities": ["display.text", "notify.show", "camera.photo", "speak", "button.events", "battery"],
    }
    msg.update(overrides)
    return msg


def wait_for(pred: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


class FakeDevice:
    """Answers ``invoke`` messages on a TestClient websocket from a background thread.

    ``handlers[capability](invoke) -> dict | (dict, bytes) | None``: a dict is merged into the result's
    ``data``; a tuple sends a full result message then a binary frame; None never answers.
    """

    def __init__(self, ws, handlers: dict[str, Callable[[dict], Any]]):
        self.ws = ws
        self.handlers = handlers
        self.invokes: list[dict] = []
        self.other: list[dict] = []
        self.payloads: list[bytes] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            try:
                raw = self.ws.receive()
            except Exception:
                return
            if raw.get("type") == "websocket.close":
                return
            if raw.get("bytes") is not None:
                self.payloads.append(raw["bytes"])  # a binary frame that follows an invoke
                continue
            if raw.get("text") is None:
                continue
            msg = json.loads(raw["text"])
            if msg.get("type") != "invoke":
                self.other.append(msg)
                continue
            self.invokes.append(msg)
            handler = self.handlers.get(msg["capability"])
            if handler is None:
                continue
            reply = handler(msg)
            if reply is None:
                continue
            with contextlib.suppress(Exception):
                if isinstance(reply, tuple):
                    result, payload = reply
                    self.ws.send_json({"type": "result", "id": msg["id"], **result})
                    self.ws.send_bytes(payload)
                elif "ok" in reply:
                    self.ws.send_json({"type": "result", "id": msg["id"], **reply})
                else:
                    self.ws.send_json({"type": "result", "id": msg["id"], "ok": True, "data": reply})


@pytest.fixture
def nodes_client(config, isolated_home):
    stack = contextlib.ExitStack()

    def make(*, llm: Any = None) -> TestClient:
        core = SentientApp(
            config, llm=llm or FakeProvider(), db_path=isolated_home / "nodes.db", enable_background=False
        )
        client = stack.enter_context(TestClient(create_app(core)))
        client.core = core
        client.token = client.app.state.token
        client.api_headers = {"Authorization": f"Bearer {client.token}"}
        return client

    yield make
    stack.close()


def pair_device(client: TestClient, **hello_overrides: Any) -> tuple[str, str]:
    """Pair a device and disconnect. Returns (node_id, token)."""
    code = client.post("/api/nodes/pairing", headers=client.api_headers).json()["code"]
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(pair_code=code, **hello_overrides))
        welcome = ws.receive_json()
    assert welcome["type"] == "welcome", welcome
    node_id = welcome["node_id"]
    wait_for(lambda: node_id not in client.core.nodes._conns)
    return node_id, welcome["token"]
