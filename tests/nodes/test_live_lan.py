"""Real servers: the gateway plus the TLS LAN listener in a thread, and the reference node over wss."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import ssl
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.nodes.reference import ReferenceNode
from tests.conftest import FakeProvider

TOKEN = "live-gateway-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _insecure_ssl() -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


@pytest.fixture
def live(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", TOKEN)
    gw_port, lan_port = _free_port(), _free_port()
    config.nodes.lan_enabled = True
    config.nodes.lan_port = lan_port
    config.nodes.mdns_enabled = False
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "live.db", enable_background=False)
    core.nodes.lan_host = "127.0.0.1"
    server = uvicorn.Server(uvicorn.Config(create_app(core), host="127.0.0.1", port=gw_port, log_level="warning"))
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=lambda: loop.run_until_complete(server.serve()), daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (server.started and core.nodes.lan and core.nodes.lan.running):
        time.sleep(0.05)
    assert server.started and core.nodes.lan and core.nodes.lan.running, core.nodes.lan_status()
    yield SimpleNamespace(
        core=core, loop=loop, gw=f"http://127.0.0.1:{gw_port}", lan=f"127.0.0.1:{lan_port}",
        auth={"Authorization": f"Bearer {TOKEN}"},
    )
    server.should_exit = True
    thread.join(20)


async def test_reference_node_over_lan_tls(live, tmp_path):
    async with httpx.AsyncClient(timeout=20) as http:
        status = (await http.get(f"{live.gw}/api/nodes/lan", headers=live.auth)).json()
        assert status["running"] and len(status["fingerprint"]) == 64
        pairing = (await http.post(f"{live.gw}/api/nodes/pairing", headers=live.auth)).json()
        assert pairing["fingerprint"] == status["fingerprint"]

        lines: list[str] = []
        node = ReferenceNode(
            f"wss://{live.lan}/ws/node", code=pairing["code"], name="Live Glasses", fingerprint=status["fingerprint"],
            speech=False, directory=tmp_path, out=lines.append, interactive=False,
        )
        task = asyncio.create_task(node.run())
        try:
            await asyncio.wait_for(node.welcomed.wait(), 20)
            res = await http.post(
                f"{live.gw}/api/nodes/{node.node_id}/invoke",
                json={"capability": "display.text", "params": {"text": "Hello from the engine"}},
                headers=live.auth,
            )
            assert res.json() == {"ok": True, "data": {"shown": True}}
            assert any("Hello from the engine" in line for line in lines)
            nodes = (await http.get(f"{live.gw}/api/nodes", headers=live.auth)).json()
            assert nodes[0]["connection"] == "lan" and nodes[0]["online"]
        finally:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task

        saved = json.loads(node.file.read_text())
        assert saved["token"] and saved["fingerprint"] == status["fingerprint"]

        # the LAN listener never exposes the main API, but serves the web app
        lan_http = httpx.AsyncClient(verify=False, timeout=10)
        async with lan_http:
            assert (await lan_http.get(f"https://{live.lan}/api/nodes", headers=live.auth)).status_code == 404
            assert (await lan_http.get(f"https://{live.lan}/api/health")).status_code == 404
            web = await lan_http.get(f"https://{live.lan}/node/")
            assert web.status_code == 200 and "Pair" in web.text

        # the gateway token does not authenticate on the LAN
        async with connect(f"wss://{live.lan}/ws/node?token={TOKEN}", ssl=_insecure_ssl()) as ws:
            await ws.send(json.dumps({"type": "hello", "name": "x", "capabilities": []}))
            err = json.loads(await ws.recv())
            assert err["code"] == "pairing_required"

        # voice on the LAN accepts the node token and nothing else
        async with connect(f"wss://{live.lan}/ws/voice?node_token={saved['token']}", ssl=_insecure_ssl()) as ws:
            await ws.send(json.dumps({"type": "start", "sample_rate": 16000}))
            assert json.loads(await ws.recv())["type"] == "ready"
            await ws.send(json.dumps({"type": "stop"}))
        with pytest.raises(InvalidStatus):
            async with connect(f"wss://{live.lan}/ws/voice?token={TOKEN}", ssl=_insecure_ssl()):
                pass

        # a wrong pinned fingerprint stops the node before it sends anything
        bad = ReferenceNode(
            f"wss://{live.lan}/ws/node", name="Live Glasses", fingerprint="00" * 32, speech=False,
            directory=tmp_path, out=lines.append, interactive=False,
        )
        assert await asyncio.wait_for(bad.run(), 20) == 1
        assert any("does not match" in line for line in lines)


def test_lan_listener_follows_config(live):
    live.core.config.nodes.lan_enabled = False
    live.loop.call_soon_threadsafe(live.core.bus.publish, "config.updated", {"sections": ["nodes"]})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and live.core.nodes.lan is not None:
        time.sleep(0.05)
    assert live.core.nodes.lan is None
    assert httpx.get(f"{live.gw}/api/nodes/lan", headers=live.auth).json()["running"] is False
