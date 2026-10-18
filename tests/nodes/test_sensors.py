"""Microphone and camera state from devices, camera-off mode, and the desktop app's live list (#170)."""

import threading

import pytest

from sentient.nodes.service import DeviceError
from tests.nodes.conftest import TINY_JPEG_B64, FakeDevice, hello, pair_device, wait_for


def _connect(client, token, **overrides):
    ctx = client.websocket_connect("/ws/node")
    ws = ctx.__enter__()
    ws.send_json(hello(token=token, **overrides))
    welcome = ws.receive_json()
    assert welcome["type"] == "welcome", welcome
    return ctx, ws


def _active(client) -> list[dict]:
    return client.get("/api/sensors", headers=client.api_headers).json()["active"]


def test_device_reports_mic_and_camera(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    ctx, ws = _connect(client, token)
    try:
        ws.send_json({"type": "state", "mic": True, "mic_for": "  Recording   a voice memo ", "camera": True})
        wait_for(lambda: len(_active(client)) == 2)
        uses = {u["sensor"]: u for u in _active(client)}
        assert uses["mic"]["id"] == f"device:{node_id}:mic" and uses["mic"]["purpose"] == "Recording a voice memo"
        assert uses["camera"]["purpose"] == "Turned on by the device" and uses["camera"]["device"] == "Test Glasses"
        assert uses["camera"]["device_kind"] == "glasses" and uses["camera"]["can_stop"] is True

        # turning one off from Sentient asks the device; it stays shown until the device says it is off
        res = client.post("/api/sensors/stop", json={"id": f"device:{node_id}:mic"}, headers=client.api_headers).json()
        assert res["stopped"] == [] and res["asked"] == [f"device:{node_id}:mic"]
        assert ws.receive_json() == {"type": "stop_sensor", "sensor": "mic"}
        assert len(_active(client)) == 2
        ws.send_json({"type": "state", "mic": False})
        wait_for(lambda: [u["sensor"] for u in _active(client)] == ["camera"])
    finally:
        ctx.__exit__(None, None, None)
    wait_for(lambda: _active(client) == [])  # a device that disconnects has nothing live


def test_camera_off_mode_is_never_overridden(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    ctx, ws = _connect(client, token, state={"battery": 70, "camera_off": True})
    try:
        device = FakeDevice(ws, {"camera.photo": lambda m: {"mime": "image/jpeg", "base64": TINY_JPEG_B64}})
        node = client.get(f"/api/nodes/{node_id}", headers=client.api_headers).json()
        assert node["camera_off"] is True
        state = client.get("/api/sensors", headers=client.api_headers).json()
        assert state["camera_off"] == [{"device_id": node_id, "device": "Test Glasses"}] and state["camera"] is False

        url = f"/api/nodes/{node_id}/invoke"
        res = client.post(url, json={"capability": "camera.photo"}, headers=client.api_headers).json()
        assert res["ok"] is False and res["code"] == "camera_off"
        assert "switched off on the device" in res["error"]
        assert device.invokes == []  # the device was never asked

        # the agent's photo tool gets the same plain refusal
        core = client.core
        with pytest.raises(DeviceError, match="switched off on the device"):
            client.portal.call(core.registry.get("device_take_photo").call, core.agent.tool_context(None, "desktop"), {})
        assert device.invokes == []

        # only the device itself ends camera-off mode
        ws.send_json({"type": "state", "camera_off": False})
        wait_for(lambda: client.get("/api/sensors", headers=client.api_headers).json()["camera_off"] == [])
        ok = client.post(url, json={"capability": "camera.photo"}, headers=client.api_headers).json()
        assert ok["ok"] is True and len(device.invokes) == 1
    finally:
        ctx.__exit__(None, None, None)


def test_a_photo_shows_the_camera_while_it_is_taken(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    ctx, ws = _connect(client, token)
    seen: list[list[dict]] = []

    def photo(_msg):
        seen.append(client.core.sensors.state()["active"])
        return {"mime": "image/jpeg", "base64": TINY_JPEG_B64}

    try:
        FakeDevice(ws, {"camera.photo": photo})
        res = client.post(f"/api/nodes/{node_id}/invoke", json={"capability": "camera.photo"}, headers=client.api_headers)
        assert res.json()["ok"] is True
        [[use]] = seen
        assert use["sensor"] == "camera" and use["purpose"] == "Taking a photo" and use["device_id"] == node_id
        assert use["id"].startswith(f"device:{node_id}:photo:")
        assert _active(client) == []  # off again once the photo arrived
    finally:
        ctx.__exit__(None, None, None)


def test_turning_off_a_camera_mid_photo_drops_the_photo(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    ctx, ws = _connect(client, token)
    try:
        device = FakeDevice(ws, {"camera.photo": lambda m: None})  # never answers
        results: list[dict] = []
        caller = threading.Thread(
            target=lambda: results.append(
                client.post(
                    f"/api/nodes/{node_id}/invoke", json={"capability": "camera.photo", "timeout_ms": 10_000},
                    headers=client.api_headers,
                ).json()
            )
        )  # fmt: skip
        caller.start()
        wait_for(lambda: len(_active(client)) == 1)
        use_id = _active(client)[0]["id"]
        res = client.post("/api/sensors/stop", json={"id": use_id}, headers=client.api_headers).json()
        assert res["stopped"] == [use_id]
        caller.join(5)
        assert results[0]["ok"] is False and results[0]["code"] == "stopped"
        wait_for(lambda: {"type": "stop_sensor", "sensor": "camera"} in device.other)
        assert _active(client) == []
    finally:
        ctx.__exit__(None, None, None)


def test_desktop_app_gets_the_live_list(nodes_client):
    client = nodes_client()
    _, token = pair_device(client)
    with client.websocket_connect(f"/ws/node?token={client.token}") as desktop:
        desktop.send_json(hello(kind="desktop", name="Maya's laptop", capabilities=["notify.show"]))
        assert desktop.receive_json()["type"] == "welcome"
        first = desktop.receive_json()
        assert first == {"type": "sensors", "mic": False, "camera": False, "active": [], "camera_off": []}

        # push to talk on the desktop itself, reported over its own device connection
        desktop.send_json({"type": "state", "mic": True, "mic_for": "Push to talk"})
        mine = desktop.receive_json()
        assert mine["mic"] is True and mine["active"][0]["device_id"] == "desktop"
        assert mine["active"][0]["device"] == "This computer"
        assert mine["active"][0]["purpose"] == "Push to talk"

        ctx, glasses = _connect(client, token)
        try:
            glasses.send_json({"type": "state", "camera_off": True})
            update = desktop.receive_json()
            assert update["type"] == "sensors" and update["camera_off"][0]["device"] == "Test Glasses"
        finally:
            ctx.__exit__(None, None, None)
        assert desktop.receive_json()["camera_off"] == []  # gone with the device

        # the tray's Turn off reaches the desktop app as stop_sensor
        client.post("/api/sensors/stop", json={}, headers=client.api_headers)
        assert desktop.receive_json() == {"type": "stop_sensor", "sensor": "mic"}
        desktop.send_json({"type": "state", "mic": False})
        assert desktop.receive_json()["active"] == []


def test_a_device_that_reports_its_camera_stays_shown_until_it_says_off(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    ctx, ws = _connect(client, token)
    try:
        FakeDevice(ws, {"camera.photo": lambda m: None})  # never answers
        results: list[dict] = []
        caller = threading.Thread(
            target=lambda: results.append(
                client.post(f"/api/nodes/{node_id}/invoke", json={"capability": "camera.photo", "timeout_ms": 10_000},
                            headers=client.api_headers).json()
            )
        )  # fmt: skip
        caller.start()
        wait_for(lambda: len(_active(client)) == 1)
        ws.send_json({"type": "state", "camera": True, "camera_for": "Taking a photo"})
        wait_for(lambda: client.core.nodes._conns[node_id].camera)
        assert len(_active(client)) == 1  # Sentient's own request and the device's report show once
        use_id = _active(client)[0]["id"]
        client.post("/api/sensors/stop", json={"id": use_id}, headers=client.api_headers)
        caller.join(5)
        assert results[0]["code"] == "stopped"
        # the request is over, but the device still says its camera is on: that stays shown
        assert [u["id"] for u in _active(client)] == [f"device:{node_id}:camera"]
        ws.send_json({"type": "state", "camera": False})
        wait_for(lambda: _active(client) == [])
    finally:
        ctx.__exit__(None, None, None)
