"""The live microphone and camera registry (#170, docs/API.md section 24)."""

from sentient.events import EventBus
from sentient.sensors import SensorRegistry


async def test_start_update_end_publish_whole_state():
    bus = EventBus()
    reg = SensorRegistry(bus)
    seen: list[dict] = []
    reg.on_change(seen.append)
    async with bus.subscribe() as q:
        reg.start("voice:a", "mic", device_id="desktop", device="This computer", device_kind="desktop", purpose="Voice chat")
        event = q.get_nowait()
        assert event["type"] == "sensors.updated"
        state = event["data"]
        assert state["mic"] is True and state["camera"] is False and state["camera_off"] == []
        [use] = state["active"]
        assert use["id"] == "voice:a" and use["device"] == "This computer" and use["purpose"] == "Voice chat"
        assert use["can_stop"] is False and use["since"]

        reg.start("voice:a", "mic", device_id="desktop", device="This computer", device_kind="desktop", purpose="Voice chat")
        assert q.empty()  # nothing changed, nothing published
        reg.update("voice:a", purpose="Listening for 'Hey Sentient'")
        assert q.get_nowait()["data"]["active"][0]["purpose"] == "Listening for 'Hey Sentient'"
        assert reg.state()["active"][0]["since"] == use["since"]  # still the same use

        reg.end("voice:a")
        assert q.get_nowait()["data"] == {"mic": False, "camera": False, "active": [], "camera_off": []}
        reg.end("voice:a")
        assert q.empty()
    assert len(seen) == 3 and seen[-1]["active"] == []


async def test_end_prefix_and_camera_off():
    reg = SensorRegistry(EventBus())
    for use_id, sensor in (("device:g1:mic", "mic"), ("device:g1:photo:x", "camera"), ("device:g2:mic", "mic")):
        reg.start(use_id, sensor, device_id=use_id.split(":")[1], device="Glasses", device_kind="glasses", purpose="p")
    reg.end_prefix("device:g1:")
    assert [u["id"] for u in reg.state()["active"]] == ["device:g2:mic"]
    assert reg.state()["camera"] is False

    reg.set_camera_off("g1", "Glasses", True)
    assert reg.state()["camera_off"] == [{"device_id": "g1", "device": "Glasses"}]
    reg.set_camera_off("g1", "Glasses", False)
    assert reg.state()["camera_off"] == []


async def test_stop_one_or_all():
    reg = SensorRegistry(EventBus())
    calls: list[str] = []

    async def off_now() -> bool:
        calls.append("now")
        return True

    async def ask_device() -> bool:
        calls.append("asked")
        return False

    reg.start("a", "mic", device_id="d", device="D", device_kind="phone", purpose="p", stop=off_now)
    reg.start("b", "camera", device_id="d", device="D", device_kind="phone", purpose="p", stop=ask_device)
    reg.start("c", "mic", device_id="d", device="D", device_kind="phone", purpose="p")  # nothing can stop it
    assert await reg.stop("missing") == {"stopped": [], "asked": []}
    assert await reg.stop("a") == {"stopped": ["a"], "asked": []}
    assert not reg.is_on("a")
    assert await reg.stop() == {"stopped": [], "asked": ["b"]}
    assert reg.is_on("b") and reg.is_on("c")  # a device that was only asked stays shown until it says it is off
    assert calls == ["now", "asked"]
