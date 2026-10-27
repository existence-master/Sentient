import json

import pytest

from sentient.nodes.reference import (
    NodeFatal,
    ReferenceNode,
    normalize_url,
    parse_pair_link,
    state_file,
)
from sentient.nodes.tls import fingerprint_der


def _node(tmp_path, **kw):
    lines: list[str] = []
    node = ReferenceNode(
        kw.pop("url", "ws://127.0.0.1:7777/ws/node"), speech=False, directory=tmp_path, out=lines.append,
        interactive=False, **kw,
    )
    return node, lines


def test_urls_and_links():
    assert normalize_url("192.168.1.20:7778") == "wss://192.168.1.20:7778/ws/node"
    assert normalize_url("https://host:7778/") == "wss://host:7778/ws/node"
    assert normalize_url("ws://127.0.0.1:7777/ws/node") == "ws://127.0.0.1:7777/ws/node"
    url, code, fp = parse_pair_link("sentient://pair?url=wss%3A%2F%2F10.0.0.2%3A7778%2Fws%2Fnode&code=123456&fp=ab12")
    assert (url, code, fp) == ("wss://10.0.0.2:7778/ws/node", "123456", "ab12")
    with pytest.raises(NodeFatal):
        parse_pair_link("sentient://pair?code=1")
    assert state_file("My Glasses!", None).name == "my-glasses.json"


def test_hello_uses_saved_token_and_code(tmp_path):
    node, _ = _node(tmp_path, code="123456", camera=0)
    h = node.hello()
    assert h["pair_code"] == "123456" and "token" not in h and "camera.photo" in h["capabilities"]
    node._save(url=node.url, token="tok")
    node.code = None
    fresh, _ = _node(tmp_path)
    assert fresh.hello()["token"] == "tok" and "camera.photo" not in fresh.hello()["capabilities"]
    other, _ = _node(tmp_path, url="ws://10.0.0.9:7777/ws/node")
    assert "token" not in other.hello()  # a token is only sent to the engine that issued it
    assert json.loads(node.file.read_text())["token"] == "tok"


async def test_handles_invokes(tmp_path):
    node, lines = _node(tmp_path)
    res = await node.handle_invoke({"id": "1", "capability": "display.text", "params": {"text": "Turn left"}})
    assert res == {"type": "result", "id": "1", "ok": True, "data": {"shown": True}}
    assert any("Turn left" in line for line in lines)
    await node.handle_invoke({"id": "2", "capability": "notify.show", "params": {"title": "Mail", "text": "From Ana"}})
    assert "[notification] Mail: From Ana" in lines
    spoken = await node.handle_invoke({"id": "3", "capability": "speak", "params": {"text": "Hi"}})
    assert spoken["data"] == {"spoken": False} and '[says] "Hi"' in lines
    battery = await node.handle_invoke({"id": "4", "capability": "battery"})
    assert battery["data"] == {"battery": 87, "charging": False}
    nope = await node.handle_invoke({"id": "5", "capability": "camera.photo"})
    assert nope["ok"] is False and nope["error"]["code"] == "unsupported"


def test_certificate_pinning(tmp_path):
    der = b"certificate-bytes"
    fp = fingerprint_der(der)
    pinned, _ = _node(tmp_path, url="wss://h:1/ws/node", fingerprint=fp.upper())
    assert pinned.check_pin(der) == fp
    wrong, _ = _node(tmp_path, url="wss://h:1/ws/node", fingerprint="00" * 32)
    with pytest.raises(NodeFatal, match="does not match"):
        wrong.check_pin(der)
    unpinned, _ = _node(tmp_path, url="wss://h:1/ws/node")
    with pytest.raises(NodeFatal, match="--fingerprint"):
        unpinned.check_pin(der)
    insecure, _ = _node(tmp_path, url="wss://h:1/ws/node", insecure=True)
    assert insecure.check_pin(der) == fp
    assert insecure.ssl_context() is not None and _node(tmp_path)[0].ssl_context() is None


def test_auth_errors_are_fatal(tmp_path):
    node, _ = _node(tmp_path)
    node._save(token="old")
    with pytest.raises(NodeFatal, match="--code"):
        node._on_auth_error({"code": "revoked", "message": "removed"})
    assert node.saved["token"] is None
    with pytest.raises(ConnectionError):
        node._on_auth_error({"code": "disabled", "message": "off"})


async def test_camera_off_mode_refuses_on_the_device(tmp_path):
    node, lines = _node(tmp_path, camera_off=True)
    h = node.hello()
    assert "camera.photo" in h["capabilities"] and h["state"]["camera_off"] is True
    res = await node.handle_invoke({"id": "6", "capability": "camera.photo"})
    assert res["ok"] is False and res["error"]["code"] == "camera_off"  # the webcam is never opened
    assert "[camera] refused: camera-off mode is on" in lines
    plain, _ = _node(tmp_path)
    assert "camera_off" not in plain.hello()["state"]  # no camera, nothing to report


async def test_a_photo_turned_off_in_sentient_is_dropped(tmp_path):
    node, lines = _node(tmp_path, camera=0)
    sent: list[dict] = []

    class WS:
        async def send(self, raw):
            sent.append(json.loads(raw))

    def photo(params):  # the webcam is never opened in tests
        node._photo_stops += 1  # stop_sensor camera arrives while the photo is taken
        return {"mime": "image/jpeg", "base64": "abc"}

    node._photo = photo
    await node._answer(WS(), {"type": "invoke", "id": "7", "capability": "camera.photo", "params": {}})
    assert sent[0] == {"type": "state", "camera": True, "camera_for": "Taking a photo"}
    assert sent[1]["ok"] is False and sent[1]["error"]["code"] == "stopped"  # no image sent
    assert sent[2] == {"type": "state", "camera": False}
    assert "[camera] photo dropped: turned off in Sentient" in lines


async def test_a_photo_stopped_before_it_started_is_never_taken(tmp_path):
    node, lines = _node(tmp_path, camera=0)
    sent: list[dict] = []

    class WS:
        async def send(self, raw):
            sent.append(json.loads(raw))

    def photo(params):
        raise AssertionError("the camera must not be used")

    node._photo = photo
    queued_at = node._photo_stops  # the invoke arrived...
    node._photo_stops += 1  # ...then stop_sensor camera, before its task ran
    await node._answer(WS(), {"type": "invoke", "id": "8", "capability": "camera.photo", "params": {}}, queued_at)
    assert sent == [{"type": "result", "id": "8", "ok": False,
                     "error": {"code": "stopped", "message": "The camera was turned off in Sentient."}}]  # fmt: skip
    assert "[camera] photo not taken: turned off in Sentient" in lines
