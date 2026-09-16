import base64

from tests.nodes.conftest import FakeDevice, hello, pair_device, wait_for


def _connect(client, token, **overrides):
    ws_ctx = client.websocket_connect("/ws/node")
    ws = ws_ctx.__enter__()
    ws.send_json(hello(token=token, **overrides))
    assert ws.receive_json()["type"] == "welcome"
    return ws_ctx, ws


def test_invoke_roundtrip(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    ctx, ws = _connect(client, token)
    try:
        device = FakeDevice(ws, {"display.text": lambda m: {"shown": True, "echo": m["params"]["text"]}})
        res = client.post(
            f"/api/nodes/{node_id}/invoke",
            json={"capability": "display.text", "params": {"text": "Hello"}},
            headers=client.api_headers,
        ).json()
        assert res == {"ok": True, "data": {"shown": True, "echo": "Hello"}}
        inv = device.invokes[0]
        assert inv["type"] == "invoke" and inv["capability"] == "display.text" and inv["timeout_ms"] == 20000
    finally:
        ctx.__exit__(None, None, None)


def test_invoke_timeout_unsupported_offline_and_errors(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    url = f"/api/nodes/{node_id}/invoke"
    assert client.post(url, json={"capability": "display.text"}, headers=client.api_headers).json()["code"] == "offline"
    assert client.post("/api/nodes/nope/invoke", json={"capability": "x"}, headers=client.api_headers).status_code == 404
    ctx, ws = _connect(client, token)
    try:
        FakeDevice(ws, {
            "notify.show": lambda m: {"ok": False, "error": {"code": "permission_denied", "message": "Denied"}},
            "speak": lambda m: {"ok": False, "error": "no speaker"},
        })  # fmt: skip
        slow = client.post(url, json={"capability": "display.text", "timeout_ms": 150}, headers=client.api_headers).json()
        assert slow["ok"] is False and slow["code"] == "timeout"
        unsupported = client.post(url, json={"capability": "location.get"}, headers=client.api_headers).json()
        assert unsupported["code"] == "unsupported"
        denied = client.post(url, json={"capability": "notify.show"}, headers=client.api_headers).json()
        assert denied == {"ok": False, "error": "Denied", "code": "permission_denied"}
        failed = client.post(url, json={"capability": "speak"}, headers=client.api_headers).json()
        assert failed == {"ok": False, "error": "no speaker"}
    finally:
        ctx.__exit__(None, None, None)


def test_binary_follow_frame(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    ctx, ws = _connect(client, token)
    try:
        FakeDevice(ws, {"camera.photo": lambda m: ({"ok": True, "data": {"mime": "image/jpeg", "binary": True}}, b"\xff\xd8jpeg")})
        res = client.post(f"/api/nodes/{node_id}/invoke", json={"capability": "camera.photo"}, headers=client.api_headers).json()
        assert res == {"ok": True, "data": {"mime": "image/jpeg", "base64": base64.b64encode(b"\xff\xd8jpeg").decode()}}
    finally:
        ctx.__exit__(None, None, None)


def test_upload_then_result(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    assert client.post("/api/nodes/upload", content=b"x").status_code == 401
    assert client.post("/api/nodes/upload", content=b"x", headers=client.api_headers).status_code == 401  # gateway token is not a node token
    up = client.post(
        "/api/nodes/upload", content=b"\xff\xd8photo", headers={"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg"}
    ).json()
    assert up["mime"] == "image/jpeg" and up["size"] == 7
    multi = client.post(
        "/api/nodes/upload", files={"file": ("a.png", b"png!", "image/png")}, headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert multi["mime"] == "image/png"
    ctx, ws = _connect(client, token)
    try:
        FakeDevice(ws, {"camera.photo": lambda m: {"mime": "image/jpeg", "upload_id": up["upload_id"]}})
        res = client.post(f"/api/nodes/{node_id}/invoke", json={"capability": "camera.photo"}, headers=client.api_headers).json()
        assert res["ok"] and base64.b64decode(res["data"]["base64"]) == b"\xff\xd8photo"
        again = client.post(f"/api/nodes/{node_id}/invoke", json={"capability": "camera.photo"}, headers=client.api_headers).json()
        assert again["code"] == "bad_upload"  # uploads are single use
    finally:
        ctx.__exit__(None, None, None)


def test_events_state_and_ping(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    with client.websocket_connect(f"/ws?token={client.token}") as main:
        assert main.receive_json()["type"] == "hello"
        with client.websocket_connect("/ws/node") as ws:
            ws.send_json(hello(token=token))
            assert ws.receive_json()["type"] == "welcome"
            ws.send_json({"type": "ping", "ts": 5})
            assert ws.receive_json() == {"type": "pong", "ts": 5}
            ws.send_json({"type": "event", "event": "button", "data": {"action": "press"}})
            ws.send_json({"type": "state", "battery": 42, "charging": True, "worn": True})
            ws.send_json({"type": "event", "event": "nope nope"})
            assert ws.receive_json()["code"] == "protocol"
            ws.send_json({"type": "bogus"})
            assert ws.receive_json()["code"] == "unknown_type"
            seen = []
            while not any(e["type"] == "node.updated" and e["data"]["battery"] == 42 for e in seen):
                seen.append(main.receive_json())
    button = next(e for e in seen if e["type"] == "node.event")
    assert button["data"] == {"node_id": node_id, "event": "button", "data": {"action": "press"}}
    updated = next(e for e in seen if e["type"] == "node.updated" and e["data"]["battery"] == 42)
    assert updated["data"]["charging"] is True and updated["data"]["worn"] is True
    wait_for(lambda: not client.core.nodes._conns)
    assert client.get(f"/api/nodes/{node_id}", headers=client.api_headers).json()["battery"] == 42
