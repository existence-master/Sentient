import pytest

from sentient import paths
from sentient.nodes.service import DeviceError
from sentient.nodes.tools import device_take_photo
from sentient.tools.base import Risk
from tests.conftest import FakeProvider
from tests.nodes.conftest import TINY_JPEG_B64, FakeDevice, hello, pair_device, wait_for


def _call(client, name, **args):
    core = client.core
    ctx = core.agent.tool_context(None, "desktop")
    return client.portal.call(core.registry.get(name).call, ctx, args)


def test_plugin_hidden_until_a_device_is_online(nodes_client):
    client = nodes_client()
    reg = client.core.registry
    assert reg.plugin("devices") is not None and reg.is_hidden("devices")
    _, token = pair_device(client)
    assert reg.is_hidden("devices")
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token))
        ws.receive_json()
        assert not reg.is_hidden("devices")
        assert "device_take_photo" in {s["function"]["name"] for s in reg.openai_schemas()}
    wait_for(lambda: reg.is_hidden("devices"))


def test_button_only_device_keeps_tools_hidden(nodes_client):
    client = nodes_client()
    _, token = pair_device(client)
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token, capabilities=["button.events", "battery"]))
        ws.receive_json()
        assert client.core.registry.is_hidden("devices")


def test_desktop_screen_capture_visibility(nodes_client, config):
    client = nodes_client()
    reg = client.core.registry
    with client.websocket_connect(f"/ws/node?token={client.token}") as ws:
        ws.send_json(hello(capabilities=["screen.capture", "notify.show"]))
        ws.receive_json()
        assert not reg.is_hidden("devices")
        config.nodes.allow_desktop_screen = False
        client.portal.call(client.core.nodes.apply_config)
        assert reg.is_hidden("devices")
        res = client.post("/api/nodes/desktop/invoke", json={"capability": "screen.capture"}, headers=client.api_headers).json()
        assert res["code"] == "not_allowed"


def test_take_photo_with_vision_description(nodes_client):
    llm = FakeProvider()
    llm.text_replies = ["A red mug on a wooden desk."]
    client = nodes_client(llm=llm)
    _, token = pair_device(client)
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token))
        ws.receive_json()
        device = FakeDevice(ws, {"camera.photo": lambda m: {"mime": "image/jpeg", "base64": TINY_JPEG_B64}})
        out = _call(client, "device_take_photo", question="What is on my desk?")
        plain = _call(client, "device_take_photo", device="glasses")
    assert device.invokes[0]["params"] == {"facing": "back"}
    assert out["description"] == "A red mug on a wooden desk." and out["device"] == "Test Glasses"
    assert out["file"].startswith("outputs/devices/photo-") and out["file"].endswith(".jpg")
    assert (paths.files_dir() / out["file"]).read_bytes()[:2] == b"\xff\xd8"
    call = llm.text_calls[0]
    assert call["role"] == "vision"
    parts = call["messages"][0]["content"]
    assert "What is on my desk?" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert plain["description"] is None and len(llm.text_calls) == 1


class NoVisionProvider(FakeProvider):
    async def complete_text(self, role, messages, *, model=None):
        raise RuntimeError("litellm.BadRequestError: model does not support image input")


def test_photo_with_model_without_vision_is_friendly(nodes_client):
    client = nodes_client(llm=NoVisionProvider())
    _, token = pair_device(client)
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token))
        ws.receive_json()
        FakeDevice(ws, {"camera.photo": lambda m: {"mime": "image/jpeg", "base64": TINY_JPEG_B64}})
        out = _call(client, "device_take_photo", question="what is this")
    assert out["description"] is None and out["file"]
    assert "cannot look at images" in out["note"] and "Settings > Models" in out["note"]


def test_camera_tools_ask_for_approval(config):
    ctx = type("Ctx", (), {"config": config})()
    assert device_take_photo.risk_fn({}, ctx) == Risk.send
    config.nodes.camera_requires_approval = False
    assert device_take_photo.risk_fn({}, ctx) is None


def test_display_notify_speak_and_errors(nodes_client):
    client = nodes_client()
    _, token = pair_device(client, name="Pixel", kind="phone")
    with pytest.raises(DeviceError, match="No device is connected"):
        _call(client, "device_notify", text="hi")

    async def fake_speak(text, voice=None):
        return b"RIFFwav"

    client.core.voice.speak = fake_speak
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token, name="Pixel", kind="phone", capabilities=["display.text", "audio.play", "location.get"]))
        ws.receive_json()
        device = FakeDevice(ws, {
            "display.text": lambda m: {"shown": True},
            "audio.play": lambda m: {"playing": True},
            "location.get": lambda m: {"lat": 18.52, "lon": 73.85, "accuracy_m": 12},
        })  # fmt: skip
        assert _call(client, "device_display", text="Meeting in 5")["device"] == "Pixel"
        assert _call(client, "device_notify", text="Hi", device="phone") == {"ok": True, "device": "Pixel"}
        assert _call(client, "device_speak", text="Hello") == {"ok": True, "device": "Pixel"}
        loc = _call(client, "device_get_location")
        listed = _call(client, "device_list")
        with pytest.raises(DeviceError, match="No connected device matches"):
            _call(client, "device_display", text="x", device="watch")
        with pytest.raises(DeviceError, match="can do that"):
            _call(client, "device_take_photo")
    caps = [(i["capability"], i["params"]) for i in device.invokes]
    assert caps[0] == ("display.text", {"text": "Meeting in 5"})
    assert caps[1] == ("display.text", {"text": "Hi"})
    assert caps[2][0] == "audio.play" and caps[2][1]["mime"] == "audio/wav" and caps[2][1]["text"] == "Hello"
    assert loc["lat"] == 18.52 and "openstreetmap" in loc["map"]
    assert listed["devices"][0]["name"] == "Pixel" and listed["devices"][0]["online"] is True


def test_speak_sends_raw_pcm_to_small_devices(nodes_client):
    """Microcontrollers get PCM16 in a binary frame instead of a large base64 JSON string."""
    client = nodes_client()
    _, token = pair_device(client)
    pcm = b"\x01\x00\x02\x00" * 8
    wav = (
        b"RIFF" + (36 + len(pcm)).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + (24000).to_bytes(4, "little")
        + (48000).to_bytes(4, "little") + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
        + b"data" + len(pcm).to_bytes(4, "little") + pcm
    )

    async def fake_speak(text, voice=None):
        return wav

    client.core.voice.speak = fake_speak
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token, capabilities=["audio.pcm", "button.events"]))
        ws.receive_json()
        device = FakeDevice(ws, {"audio.pcm": lambda m: {"playing": True}})
        assert _call(client, "device_speak", text="Turn left")["ok"] is True
        wait_for(lambda: bool(device.payloads))
    invoke = device.invokes[0]
    assert invoke["capability"] == "audio.pcm"
    assert invoke["params"] == {
        "format": "pcm16", "sample_rate": 24000, "channels": 1, "text": "Turn left",
        "binary": True, "bytes": len(pcm),
    }
    assert device.payloads == [pcm]


def test_wav_without_pcm16_falls_back_to_audio_play(nodes_client):
    client = nodes_client()
    _, token = pair_device(client)

    async def fake_speak(text, voice=None):
        return b"OggS-not-a-wav"

    client.core.voice.speak = fake_speak
    with client.websocket_connect("/ws/node") as ws:
        ws.send_json(hello(token=token, capabilities=["audio.pcm", "audio.play"]))
        ws.receive_json()
        device = FakeDevice(ws, {"audio.play": lambda m: {"playing": True}})
        assert _call(client, "device_speak", text="Hi")["ok"] is True
    assert device.invokes[0]["capability"] == "audio.play" and device.payloads == []
