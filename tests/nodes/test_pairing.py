import pytest
from starlette.websockets import WebSocketDisconnect

from tests.nodes.conftest import hello, pair_device, wait_for


def _error(ws) -> dict:
    msg = ws.receive_json()
    assert msg["type"] == "error", msg
    with pytest.raises(WebSocketDisconnect):
        ws.receive_json()
    return msg


def test_pairing_response_shape(nodes_client):
    client = nodes_client()
    res = client.post("/api/nodes/pairing", headers=client.api_headers)
    assert res.status_code == 200
    p = res.json()
    assert len(p["code"]) == 6 and p["code"].isdigit()
    assert p["lan_enabled"] is False and p["fingerprint"] is None
    assert p["urls"] == ["ws://testserver/ws/node"]
    assert p["web_url"] == f"http://testserver/node/#code={p['code']}"
    assert p["qr"].startswith("sentient://pair?url=ws%3A%2F%2Ftestserver%2Fws%2Fnode&code=")
    assert p["qr_svg"].startswith("<svg")
    assert client.post("/api/nodes/pairing").status_code == 401


def test_pair_then_reconnect_with_token(nodes_client):
    client = nodes_client()
    events = []
    code = client.post("/api/nodes/pairing", headers=client.api_headers).json()["code"]
    with client.websocket_connect(f"/ws?token={client.token}") as main:
        assert main.receive_json()["type"] == "hello"
        with client.websocket_connect("/ws/node") as ws:
            ws.send_json(hello(pair_code=code))
            welcome = ws.receive_json()
            assert welcome["type"] == "welcome" and welcome["token"] and welcome["protocol"] == 1
            assert welcome["assistant"] == "Sentient" and welcome["keepalive_s"] == 60
            node_id = welcome["node_id"]
            nodes = client.get("/api/nodes", headers=client.api_headers).json()
            assert [(n["node_id"], n["online"], n["kind"], n["connection"]) for n in nodes] == [
                (node_id, True, "glasses", "local")
            ]
            events.append(main.receive_json())
    assert events[0]["type"] == "node.updated" and events[0]["data"]["name"] == "Test Glasses"
    token = welcome["token"]
    wait_for(lambda: not client.core.nodes._conns)
    assert client.get(f"/api/nodes/{node_id}", headers=client.api_headers).json()["online"] is False

    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token, capabilities=["display.text"]))
        again = ws.receive_json()
        assert again["type"] == "welcome" and again["node_id"] == node_id and "token" not in again
        assert client.get(f"/api/nodes/{node_id}", headers=client.api_headers).json()["capabilities"] == ["display.text"]

    node = client.portal.call(client.core.nodes.verify_token, token)
    assert node["node_id"] == node_id
    assert client.portal.call(client.core.nodes.verify_token, "nope") is None


def test_code_is_single_use(nodes_client):
    client = nodes_client()
    code = client.post("/api/nodes/pairing", headers=client.api_headers).json()["code"]
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(pair_code=code))
        assert ws.receive_json()["type"] == "welcome"
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(pair_code=code))
        assert _error(ws)["code"] == "bad_code"


def test_code_expires(nodes_client):
    client = nodes_client()
    svc = client.core.nodes
    now = [1000.0]
    svc.clock = lambda: now[0]
    code = client.post("/api/nodes/pairing", headers=client.api_headers).json()["code"]
    now[0] += 601
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(pair_code=code))
        assert _error(ws)["code"] == "bad_code"


def test_wrong_codes_are_rate_limited(nodes_client):
    client = nodes_client()
    code = client.post("/api/nodes/pairing", headers=client.api_headers).json()["code"]
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(5):
        with client.websocket_connect("/ws/node") as ws:
            ws.send_json(hello(pair_code=wrong))
            assert _error(ws)["code"] == "bad_code"
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(pair_code=code))  # even the right code waits out the limit
        assert _error(ws)["code"] == "rate_limited"


def test_pairing_required_and_protocol_errors(nodes_client):
    client = nodes_client()
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello())
        err = _error(ws)
        assert err["code"] == "pairing_required" and "code" in err["message"]
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json({"type": "ping"})
        assert _error(ws)["code"] == "protocol"
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token="not-a-token"))
        assert _error(ws)["code"] == "bad_token"


def test_revoke_closes_and_blocks_token(nodes_client):
    client = nodes_client()
    node_id, token = pair_device(client)
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token))
        assert ws.receive_json()["type"] == "welcome"
        assert client.delete(f"/api/nodes/{node_id}", headers=client.api_headers).json() == {"ok": True}
        assert _error(ws)["code"] == "revoked"
    assert client.get("/api/nodes", headers=client.api_headers).json() == []
    assert client.delete(f"/api/nodes/{node_id}", headers=client.api_headers).status_code == 404
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token))
        assert _error(ws)["code"] == "bad_token"


def test_rename(nodes_client):
    client = nodes_client()
    node_id, _ = pair_device(client)
    res = client.patch(f"/api/nodes/{node_id}", json={"name": "Frames"}, headers=client.api_headers)
    assert res.status_code == 200 and res.json()["name"] == "Frames"
    assert client.patch("/api/nodes/missing", json={"name": "x"}, headers=client.api_headers).status_code == 404


def test_desktop_node_uses_gateway_token(nodes_client):
    client = nodes_client()
    with client.websocket_connect(f"/ws/node?token={client.token}") as ws:
        ws.send_json(hello(kind="phone", name="", capabilities=["screen.capture", "notify.show"]))
        welcome = ws.receive_json()
        assert welcome["node_id"] == "desktop" and "token" not in welcome
        node = client.get("/api/nodes/desktop", headers=client.api_headers).json()
        assert node["kind"] == "desktop" and node["name"] == "This computer" and node["online"]
    res = client.delete("/api/nodes/desktop", headers=client.api_headers)
    assert res.status_code == 400


def test_devices_disabled(nodes_client, config):
    client = nodes_client()
    config.nodes.enabled = False
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello())
        assert _error(ws)["code"] == "disabled"


def test_new_connection_replaces_old(nodes_client):
    client = nodes_client()
    _, token = pair_device(client)
    with client.websocket_connect("/ws/node") as first:
        first.send_json(hello(token=token))
        assert first.receive_json()["type"] == "welcome"
        with client.websocket_connect("/ws/node") as second:
            second.send_json(hello(token=token))
            assert second.receive_json()["type"] == "welcome"
            assert _error(first)["code"] == "replaced"
            assert len(client.core.nodes._conns) == 1
