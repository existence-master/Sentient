"""The microphone indicator from voice sessions (#170) and where voice goes (status ``privacy``)."""

from sentient.voice import session as voice_session
from tests.nodes.conftest import wait_for
from tests.voice.conftest import SR, FakeWakeDetector, collect_until, frames, recv, silence


def _start(ws, **extra):
    ws.send_json({"type": "start", "sample_rate": SR, **extra})
    ready = recv(ws)
    assert ready["type"] == "ready"
    recv(ws)  # state
    return ready


def _mic(client) -> list[dict]:
    return [u for u in client.core.sensors.state()["active"] if u["sensor"] == "mic"]


def test_audio_turns_the_mic_on_and_end_utterance_off(voice_client):
    client = voice_client()
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        assert _mic(client) == []  # an open session with no audio is not a live microphone
        for f in frames(silence(100)):
            ws.send_bytes(f)
        ws.send_json({"type": "ping"})
        collect_until(ws, lambda m: m["type"] == "pong")
        [use] = _mic(client)
        assert use["device_id"] == "desktop" and use["device"] == "This computer" and use["device_kind"] == "desktop"
        assert use["purpose"] == "Voice chat" and use["can_stop"] is True
        assert client.get("/api/sensors").json()["mic"] is True

        ws.send_json({"type": "end_utterance"})
        ws.send_json({"type": "ping"})
        collect_until(ws, lambda m: m["type"] == "pong")
        assert _mic(client) == []


def test_mic_goes_off_when_audio_stops_and_on_close(voice_client, monkeypatch):
    monkeypatch.setattr(voice_session, "MIC_IDLE_S", 0.15)
    client = voice_client()
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        ws.send_bytes(silence(40))
        wait_for(lambda: len(_mic(client)) == 1)
        wait_for(lambda: _mic(client) == [])  # no frames for MIC_IDLE_S
        ws.send_bytes(silence(40))
        wait_for(lambda: len(_mic(client)) == 1)
    wait_for(lambda: _mic(client) == [])  # socket closed


def test_wake_mode_says_what_the_mic_listens_for(voice_client):
    client = voice_client(wake=lambda: FakeWakeDetector())
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        ws.send_json({"type": "start", "sample_rate": SR, "mode": "wake"})
        collect_until(ws, lambda m: m["type"] == "state" and m["state"] == "standby")
        ws.send_bytes(silence(40))
        wait_for(lambda: len(_mic(client)) == 1)
        assert _mic(client)[0]["purpose"] == "Listening for 'Hey Sentient'"
        ws.send_json({"type": "wake"})
        collect_until(ws, lambda m: m["type"] == "state" and m["state"] == "listening")
        wait_for(lambda: _mic(client)[0]["purpose"] == "Voice chat")


def test_turn_off_from_the_indicator_closes_the_socket(voice_client):
    client = voice_client()
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start(ws)
        ws.send_bytes(silence(40))
        wait_for(lambda: len(_mic(client)) == 1)
        use_id = _mic(client)[0]["id"]
        res = client.post("/api/sensors/stop", json={"id": use_id}).json()
        assert res["stopped"] == [use_id] and res["asked"] == [] and res["state"]["mic"] is False
        msgs = collect_until(ws, lambda m: m["type"] == "error")
        err = msgs[-1]
        assert err["code"] == "mic_stopped" and err["recoverable"] is False
        closed = None
        for _ in range(20):
            msg = ws.receive()
            if msg["type"] == "websocket.close":
                closed = msg
                break
        assert closed == {"type": "websocket.close", "code": 1000, "reason": "mic_stopped"}
    assert _mic(client) == [] and not client.core.voice.sessions
    assert client.post("/api/sensors/stop", json={"id": use_id}).json()["stopped"] == []  # already off: no error


def test_device_session_names_the_device(voice_client, monkeypatch):
    client = voice_client()
    client.headers.pop("Authorization")
    node = {"node_id": "g1", "name": "Maya's glasses", "kind": "glasses", "platform": "esp32-s3"}

    async def verify_token(token: str):
        return dict(node) if token == "good-token" else None

    monkeypatch.setattr(client.core.nodes, "verify_token", verify_token, raising=False)
    with client.websocket_connect("/ws/voice?node_token=good-token") as ws:
        _start(ws)
        ws.send_bytes(silence(40))
        wait_for(lambda: len(_mic(client)) == 1)
        use = _mic(client)[0]
        assert (use["device_id"], use["device"], use["device_kind"]) == ("g1", "Maya's glasses", "glasses")


def test_privacy_follows_the_settings(voice_client):
    client = voice_client()
    cfg = client.core.config
    privacy = client.get("/api/voice/status").json()["privacy"]
    assert privacy["audio_local"] is True and privacy["replies"]["local"] is True
    assert {i["id"] for i in privacy["items"]} == {"dictation", "wake", "voice", "voice_notes"}

    cfg.voice.stt_provider = "openai"
    p = client.core.voice.privacy()
    items = {i["id"]: i for i in p["items"]}
    assert p["audio_local"] is False and items["voice"]["where"] == "Sent to OpenAI to be turned into text."
    assert items["voice_notes"]["local"] is False
    assert items["dictation"]["local"] is True and items["wake"]["local"] is True  # always on this computer

    cfg.models.providers["openai"].api_base = "http://127.0.0.1:8000/v1"  # a speech server on this computer
    assert client.core.voice.privacy()["audio_local"] is True
    cfg.models.providers["openai"].api_base = "https://speech.example.org/v1"
    assert {i["id"]: i for i in client.core.voice.privacy()["items"]}["voice"]["where"] == (
        "Sent to speech.example.org to be turned into text."
    )

    cfg.voice.stt_provider = "deepgram"
    cfg.voice.tts_provider = "elevenlabs"
    p = client.core.voice.privacy()
    assert p["audio_local"] is False and p["replies"] == {
        "local": False, "where": "The text of each reply is sent to ElevenLabs to be read aloud.",
    }  # fmt: skip
    cfg.models.local_only = True  # Local only keeps speech and replies on this computer
    p = client.core.voice.privacy()
    assert p["audio_local"] is True and p["replies"]["local"] is True
