"""Stop everything from a paired device: `stop_all` / `resume` messages and the `stop_state` broadcast."""

from tests.nodes.conftest import hello, pair_device, wait_for


def test_device_can_stop_and_resume(nodes_client):
    client = nodes_client()
    _, token = pair_device(client)
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token))
        welcome = ws.receive_json()
        assert welcome["type"] == "welcome" and welcome["stopped"] is False

        ws.send_json({"type": "stop_all"})
        state = ws.receive_json()
        assert state["type"] == "stop_state" and state["stopped"] is True and state["source"] == "device"
        assert client.core.stopped

        ws.send_json({"type": "stop_all"})  # already stopped: still answered
        assert ws.receive_json()["stopped"] is True

        ws.send_json({"type": "resume"})
        assert ws.receive_json() == {"type": "stop_state", "stopped": False, "stopped_at": None, "source": "device"}

        client.post("/api/stop-all", headers=client.api_headers)  # stopped elsewhere: every device hears it
        assert ws.receive_json()["stopped"] is True
    wait_for(lambda: not client.core.nodes._conns)

    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token))
        assert ws.receive_json()["stopped"] is True  # a device that connects later sees it in welcome
