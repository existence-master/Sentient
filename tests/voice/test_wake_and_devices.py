"""transcribe_bytes, device (node_token) sessions, PCM16 output, wake word and talk mode, latency hooks."""

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from sentient.app import SentientApp
from sentient.voice import service as voice_service
from sentient.voice.audio import pcm16_to_wav
from sentient.voice.base import VoiceError
from sentient.voice.text import SentenceSplitter
from sentient.voice.wake import (
    OpenWakeWordDetector,
    WakeHit,
    WhisperWakeDetector,
    earcon_wav,
    match_wake_phrase,
    resolve_openwakeword_model,
)
from tests.conftest import FakeProvider
from tests.voice.conftest import (
    SR,
    FakeSTT,
    FakeTTS,
    FakeWakeDetector,
    SlowProvider,
    collect_until,
    frames,
    recv,
    silence,
    tone,
    types,
)


def _speak(ws, ms: int = 700, tail_ms: int = 1000):
    for f in frames(silence(200) + tone(ms) + silence(tail_ms)):
        ws.send_bytes(f)


def _msgs(items):
    return [m for m in items if isinstance(m, dict)]


def _start_wake(ws, **extra):
    ws.send_json({"type": "start", "sample_rate": SR, "mode": "wake", **extra})
    ready = recv(ws)
    assert ready["type"] == "ready" and ready["mode"] == "wake"
    assert recv(ws) == {"type": "state", "state": "standby", "session_id": ready["session_id"]}
    return ready


# ----------------------------------------------------------------------------- transcribe_bytes
class PathSTT(FakeSTT):
    def __init__(self, error: Exception | None = None):
        super().__init__()
        self.paths: list[Path] = []
        self.error = error

    async def transcribe_file(self, path) -> str:
        p = Path(path)
        self.paths.append(p)
        assert p.is_file()
        if self.error is not None:
            raise self.error
        return await super().transcribe_file(path)


@pytest.fixture
def core(config, isolated_home):
    return SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "v.db", enable_background=False)


async def test_transcribe_bytes_picks_container_and_cleans_up(core):
    stt = PathSTT()
    core.voice.use_providers(stt=stt, tts=FakeTTS())
    cases = [
        (b"OggS\x00\x02" + b"\0" * 64, "voice-note.opus", ".ogg"),
        (b"\x1aE\xdf\xa3" + b"\0" * 64, "", ".webm"),
        (b"\0\0\0\x20ftypM4A " + b"\0" * 64, "clip", ".m4a"),
        (b"ID3\x04" + b"\0" * 64, "audio", ".mp3"),
        (pcm16_to_wav(tone(100), SR), "rec.WAV", ".wav"),
    ]
    for data, name, _suffix in cases:
        assert await core.voice.transcribe_bytes(data, name) == "dictated text"
    assert [p.suffix for p in stt.paths] == [c[2] for c in cases]
    assert not any(p.exists() for p in stt.paths)  # temp files removed


async def test_transcribe_bytes_friendly_errors(core, monkeypatch):
    core.voice.use_providers(stt=PathSTT(), tts=FakeTTS())
    with pytest.raises(VoiceError, match="empty"):
        await core.voice.transcribe_bytes(b"", "a.ogg")
    with pytest.raises(VoiceError, match="unsupported audio type"):
        await core.voice.transcribe_bytes(b"hello world, not audio", "notes.txt")
    monkeypatch.setattr(voice_service, "MAX_AUDIO_BYTES", 10)
    with pytest.raises(VoiceError, match="too large"):
        await core.voice.transcribe_bytes(b"OggS" + b"\0" * 20, "a.ogg")
    monkeypatch.undo()

    av_error = type("InvalidDataError", (Exception,), {"__module__": "av.error"})
    stt = PathSTT(error=av_error("Invalid data found when processing input"))
    core.voice.use_providers(stt=stt, tts=FakeTTS())
    with pytest.raises(VoiceError, match="could not decode the audio file"):
        await core.voice.transcribe_bytes(b"OggS" + b"\0" * 20, "a.ogg")
    core.voice.use_providers(stt=PathSTT(error=RuntimeError("boom")), tts=FakeTTS())
    with pytest.raises(VoiceError, match="transcription failed: boom"):
        await core.voice.transcribe_bytes(b"OggS" + b"\0" * 20, "a.ogg")
    assert not any(p.exists() for p in stt.paths)


# ----------------------------------------------------------------------------- devices
NODE = {"node_id": "node-1", "name": "Frames", "kind": "glasses", "platform": "esp32"}


def _node_client(voice_client, monkeypatch, **kwargs):
    client = voice_client(**kwargs)
    client.headers.pop("Authorization")

    async def verify_token(token: str):
        return dict(NODE) if token == "good-token" else None

    monkeypatch.setattr(client.core.nodes, "verify_token", verify_token, raising=False)
    return client


async def _session(client, sid):
    return await client.core.store.get_session(sid)


def test_node_token_session_uses_device_channel_and_voice_role(voice_client, monkeypatch):
    client = _node_client(voice_client, monkeypatch, replies=["Hi from the glasses.", "Hi from the phone."])
    with client.websocket_connect("/ws/voice?node_token=good-token") as ws:
        ws.send_json({"type": "start", "sample_rate": SR})
        ready = recv(ws)
        assert ready["channel"] == "glasses" and ready["node_id"] == "node-1" and ready["audio_format"] == "wav"
        assert recv(ws)["state"] == "listening"
        ws.send_json({"type": "text", "text": "hello"})
        collect_until(ws, lambda m: m["type"] == "audio_end")
        ws.send_json({"type": "start", "sample_rate": SR, "channel": "phone"})
        phone = collect_until(ws, lambda m: m["type"] == "ready")[-1]
        assert phone["channel"] == "phone" and phone["session_id"] != ready["session_id"]
        ws.send_json({"type": "text", "text": "and you"})
        collect_until(ws, lambda m: m["type"] == "audio_end")
    assert asyncio.run(_session(client, ready["session_id"]))["channel"] == "glasses"
    assert asyncio.run(_session(client, phone["session_id"]))["channel"] == "phone"
    roles = [c["role"] for c in client.core.llm.calls]
    assert roles.count("voice") >= 2 and "primary" not in roles


def test_node_token_rejected(voice_client, monkeypatch):
    client = _node_client(voice_client, monkeypatch)
    for url in ("/ws/voice?node_token=wrong", "/ws/voice?node_token=", "/ws/voice?token=nope", "/ws/voice"):
        with pytest.raises(WebSocketDisconnect), client.websocket_connect(url) as ws:
            ws.receive_json()
    # nodes service without token verification: device tokens are refused
    monkeypatch.setattr(client.core.nodes, "verify_token", None, raising=False)
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/voice?node_token=good-token") as ws:
        ws.receive_json()


async def test_authorize_uses_node_validated_by_lan_listener():
    from types import SimpleNamespace

    from sentient.voice.socket import authorize_voice_socket

    ws = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(token="lan-secret")),
        query_params={"token": "lan-secret"},
        headers={},
        scope={"state": {"node": dict(NODE)}},
    )
    core = SimpleNamespace(nodes=None)  # no second token lookup
    assert await authorize_voice_socket(ws, core) == (True, NODE)
    ws.scope = {}
    assert await authorize_voice_socket(ws, core) == (True, None)
    ws.query_params = {"token": "wrong"}
    assert await authorize_voice_socket(ws, core) == (False, None)


def test_pcm16_output_framing(voice_client):
    client = voice_client(replies=["Hello there. Bye."])
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        ws.send_json({"type": "start", "sample_rate": SR, "audio_format": "pcm16",
                      "output_sample_rate": 8000, "max_frame_bytes": 51})  # fmt: skip
        ready = recv(ws)
        assert ready["audio_format"] == "pcm16" and ready["output_sample_rate"] == 8000
        recv(ws)
        ws.send_json({"type": "text", "text": "hi"})
        items = collect_until(ws, lambda m: m["type"] == "audio_end")
    headers = [(i, m) for i, m in enumerate(items) if isinstance(m, dict) and m["type"] == "audio"]
    assert [h["text"] for _, h in headers] == ["Hello there.", "Bye."]
    for i, h in headers:
        # FakeTTS: 320 samples at 16 kHz -> 160 samples at 8 kHz -> 320 bytes, frames of 50 bytes
        assert h["format"] == "pcm16" and h["sample_rate"] == 8000 and h["bytes"] == 320 and h["frames"] == 7
        body = items[i + 1 : i + 1 + h["frames"]]
        assert all(isinstance(b, bytes) for b in body)
        assert [len(b) for b in body] == [50] * 6 + [20]
        assert isinstance(items[i + 1 + h["frames"]], dict)  # nothing extra before the next message
        assert not body[0].startswith(b"RIFF")
        assert set(np.frombuffer(b"".join(body), dtype="<i2").tolist()) == {1}


# ----------------------------------------------------------------------------- wake mode
def test_wake_mode_standby_wake_reply_follow_up_standby(voice_client):
    detector = FakeWakeDetector(hits=[None, WakeHit("hey sentient")])
    client = voice_client(
        stt_texts=["what time is it"], replies=["It is noon."], wake=lambda: detector,
        voice={"follow_up_seconds": 0.4},
    )  # fmt: skip
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        ready = _start_wake(ws)
        assert ready["wake"] == {"engine": "fake_wake", "phrase": "hey sentient", "ready": True}

        _speak(ws, ms=500, tail_ms=900)  # somebody talking, no wake word
        _speak(ws, ms=500, tail_ms=900)  # "hey sentient"
        woke = collect_until(ws, lambda m: m["type"] == "state")
        msgs = _msgs(woke)
        assert msgs[0]["type"] == "wake" and msgs[0]["phrase"] == "hey sentient" and msgs[0]["source"] == "voice"
        earcon = next(m for m in msgs if m["type"] == "audio")
        assert earcon["earcon"] is True and earcon["sentence_index"] == -1
        assert woke[woke.index(earcon) + 1][:4] == b"RIFF"
        assert msgs[-1]["state"] == "listening"
        assert client.stt.calls == [] and len(detector.segments) == 2

        _speak(ws)  # the request: no wake word needed now
        turn = collect_until(ws, lambda m: m["type"] == "audio_end")
        assert [m["text"] for m in _msgs(turn) if m["type"] == "transcript"] == ["what time is it"]
        t0 = time.monotonic()
        after = collect_until(ws, lambda m: m["type"] == "state" and m["state"] == "standby")
        assert time.monotonic() - t0 >= 0.3
        assert [m["state"] for m in _msgs(after) if m["type"] == "state"] == ["listening", "standby"]
        assert len(detector.segments) == 2 and len(client.stt.calls) == 1 and detector.resets == 1

        _speak(ws, ms=500, tail_ms=900)  # standby again: not transcribed as a request
        ws.send_json({"type": "ping"})
        assert types(collect_until(ws, lambda m: m["type"] == "pong")) == ["pong"]
    assert len(client.stt.calls) == 1


def test_client_wake_message_then_follow_up_timeout(voice_client):
    client = voice_client(wake=lambda: FakeWakeDetector(), voice={"follow_up_seconds": 0.2, "wake_earcon": False})
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start_wake(ws)
        ws.send_json({"type": "wake"})
        items = collect_until(ws, lambda m: m["type"] == "state" and m["state"] == "standby")
    assert types(items) == ["wake", "state", "state"]
    assert items[0]["source"] == "client"
    assert [m["state"] for m in items[1:]] == ["listening", "standby"]


def test_one_breath_wake_and_command_with_whisper_engine(voice_client):
    wake_stt = FakeSTT(["hey sentence what's the weather"])
    client = voice_client(
        stt_texts=["Hey Sentient, what's the weather?"], replies=["Sunny."],
        wake=lambda: WhisperWakeDetector(wake_stt, "hey sentient"), voice={"wake_earcon": False},
    )  # fmt: skip
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        ready = _start_wake(ws)
        assert ready["wake"] == {"engine": "whisper", "phrase": "hey sentient", "ready": True, "model": "fake"}
        _speak(ws, ms=1200, tail_ms=900)
        items = collect_until(ws, lambda m: m["type"] == "audio_end")
    msgs = _msgs(items)
    assert msgs[0]["type"] == "wake"
    assert [m["state"] for m in msgs if m["type"] == "state"][:3] == ["listening", "transcribing", "thinking"]
    assert [m["text"] for m in msgs if m["type"] == "transcript"] == ["what's the weather?"]
    assert len(wake_stt.calls) == 1 and len(client.stt.calls) == 1
    assert wake_stt.calls[0][0] == client.stt.calls[0][0]  # short segment: the detector saw all of it
    assert client.tts.texts == ["Sunny."]


def test_long_breath_is_retranscribed_in_full(voice_client):
    wake_stt = FakeSTT(["hey sentient set a"])  # the detector only hears the first 0.5 s
    client = voice_client(
        stt_texts=["Hey Sentient, set a timer for five minutes."], replies=["Done."],
        wake=lambda: WhisperWakeDetector(wake_stt, "hey sentient", window_s=0.5), voice={"wake_earcon": False},
    )  # fmt: skip
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start_wake(ws)
        _speak(ws, ms=1500, tail_ms=900)
        items = collect_until(ws, lambda m: m["type"] == "audio_end")
    assert [m["text"] for m in _msgs(items) if m["type"] == "transcript"] == ["set a timer for five minutes."]
    assert wake_stt.calls[0][0] == SR  # 0.5 s of PCM16
    assert client.stt.calls[0][0] > 2 * SR


def test_wake_phrase_alone_waits_for_the_request(voice_client):
    wake_stt = FakeSTT(["Hi, Sentient."])
    client = voice_client(
        stt_texts=["play some jazz"], replies=["Playing."],
        wake=lambda: WhisperWakeDetector(wake_stt, "hey sentient"), voice={"wake_earcon": False},
    )  # fmt: skip
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start_wake(ws)
        _speak(ws, ms=500, tail_ms=900)
        woke = collect_until(ws, lambda m: m["type"] == "state")
        assert types(woke) == ["wake", "state"] and woke[-1]["state"] == "listening"
        assert client.stt.calls == []
        _speak(ws)
        items = collect_until(ws, lambda m: m["type"] == "audio_end")
    assert [m["text"] for m in _msgs(items) if m["type"] == "transcript"] == ["play some jazz"]


def test_streaming_wake_keeps_rest_of_breath(voice_client):
    detector = FakeWakeDetector(streaming=True, trigger_after=int(SR * 0.3) * 2)
    client = voice_client(stt_texts=["turn on the lights"], wake=lambda: detector, voice={"wake_earcon": False})
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        _start_wake(ws)
        for f in frames(silence(200) + tone(1000) + silence(1000)):
            ws.send_bytes(f)
        items = collect_until(ws, lambda m: m["type"] == "audio_end")
    msgs = _msgs(items)
    assert msgs[0]["type"] == "wake" and msgs[0]["score"] == 0.9
    (n_bytes, _rate), = client.stt.calls
    # the phrase (first ~100 ms of the tone plus pre-roll) was dropped; the rest of the breath was kept
    assert 800 <= n_bytes / 32 <= 1250


def test_wake_engine_unavailable_reports_error(voice_client):
    client = voice_client(voice={"wake_engine": "openwakeword", "wake_word": "hey sentient"})
    with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
        ready = _start_wake(ws)
        assert ready["wake"]["ready"] is False and "no pretrained model" in ready["wake"]["error"]
        err = recv(ws)
        assert err["type"] == "error" and err["recoverable"] is True and "wake word unavailable" in err["message"]
        ws.send_json({"type": "wake"})  # a button still works
        assert recv(ws)["type"] == "wake"


def test_status_wake_field_loads_nothing(voice_client, isolated_home):
    client = voice_client()
    assert client.get("/api/voice/status").json()["wake"] == {
        "engine": "whisper", "phrase": "hey sentient", "ready": False, "model": "base",
    }  # fmt: skip
    client.core.config.voice.wake_engine = "openwakeword"
    client.core.config.voice.wake_word = "Hey Jarvis"
    assert client.get("/api/voice/status").json()["wake"] == {
        "engine": "openwakeword", "phrase": "hey jarvis", "ready": False, "model": "hey_jarvis",
    }  # fmt: skip
    client.core.config.voice.wake_word = "hey sentient"
    wake = client.get("/api/voice/status").json()["wake"]
    assert wake["ready"] is False and "no pretrained model" in wake["error"]
    assert not (isolated_home / "models" / "openwakeword").exists()
    client.core.voice.use_wake_detector(lambda: FakeWakeDetector())
    assert client.get("/api/voice/status").json()["wake"] == {"engine": "fake_wake", "phrase": "hey sentient", "ready": True}


# ----------------------------------------------------------------------------- matching and helpers
@pytest.mark.parametrize(
    ("text", "remainder"),
    [
        ("hey sentient what's the weather", "what's the weather"),
        ("Hey, Sentence! What's up?", "What's up?"),
        ("hi sentient", ""),
        ("Hey sentience.", ""),
        ("High Sentient, lights on.", "lights on."),
        ("hey sent ient play music", "play music"),
        ("Heysentient, lights on", "lights on"),
        ("okay so hey sentient remind me", "remind me"),
        ("Hey-Sentient: stop", "stop"),
    ],
)
def test_wake_phrase_fuzzy_matches(text, remainder):
    m = match_wake_phrase(text, "hey sentient")
    assert m is not None and m.remainder == remainder


@pytest.mark.parametrize(
    "text", ["this sentence is wrong", "hey patient", "hey there", "", "sentient beings are rare", "hey send it"]
)
def test_wake_phrase_rejects(text):
    assert match_wake_phrase(text, "hey sentient") is None


def test_wake_sensitivity_and_other_phrases():
    assert match_wake_phrase("hey senti", "hey sentient", sensitivity=1.0) is not None
    assert match_wake_phrase("hey senti", "hey sentient", sensitivity=0.0) is None
    assert match_wake_phrase("ok jarvis turn it on", "jarvis").remainder == "turn it on"
    assert match_wake_phrase("Okay computer, stop.", "ok computer").remainder == "stop."


def test_openwakeword_model_resolution(tmp_path):
    assert resolve_openwakeword_model("", "Hey Jarvis") == ("pretrained", "hey_jarvis")
    assert resolve_openwakeword_model("alexa", "hey sentient") == ("pretrained", "alexa")
    custom = tmp_path / "hey_sentient.onnx"
    custom.write_bytes(b"\0" * 2048)
    assert resolve_openwakeword_model(str(custom), "hey sentient") == ("path", str(custom))
    with pytest.raises(VoiceError, match="not found"):
        resolve_openwakeword_model(str(tmp_path / "missing.onnx"), "")
    with pytest.raises(VoiceError, match="no pretrained model"):
        resolve_openwakeword_model("", "hey sentient")
    det = OpenWakeWordDetector("hey jarvis", 0.7)
    assert det.streaming and det.phrase == "hey jarvis" and abs(det.threshold - 0.3) < 1e-9 and not det.ready


def test_earcon_is_short_wav():
    wav = earcon_wav()
    assert wav[:4] == b"RIFF" and 3000 < len(wav) < 20000


def test_first_clause_split_only_for_the_first_chunk():
    s = SentenceSplitter(first_clause_chars=20)
    text = "The weather in Pune today is cloudy, with a high of 29 degrees. Later, it may rain, so take an umbrella."
    out = []
    for i in range(0, len(text), 4):
        out += s.feed(text[i : i + 4])
    out += s.flush()
    assert out == ["The weather in Pune today is cloudy", "with a high of 29 degrees.",
                   "Later, it may rain, so take an umbrella."]  # fmt: skip
    short = SentenceSplitter(first_clause_chars=20)
    assert [*short.feed("Sure, here it is. "), *short.flush()] == ["Sure, here it is."]
    plain = SentenceSplitter()
    assert [*plain.feed(text), *plain.flush()][0] == "The weather in Pune today is cloudy, with a high of 29 degrees."


def test_first_clause_starts_speech_earlier_and_metrics_break_down(voice_client, config):
    reply = "The short answer to your question is simple, because the sky scatters blue light far more than red light."
    results = {}
    for chars in (40, 0):
        config.voice.tts_first_clause_chars = chars
        client = voice_client(llm=SlowProvider(replies=[reply], chunk_delay=0.02), tts_delay=0.05)
        with client.websocket_connect(f"/ws/voice?token={client.token}") as ws:
            ws.send_json({"type": "start", "sample_rate": SR})
            recv(ws)
            recv(ws)
            ws.send_json({"type": "text", "text": "why is the sky blue"})
            end = collect_until(ws, lambda m: m["type"] == "audio_end")[-1]
        results[chars] = (end["metrics"], list(client.tts.texts))
    clause, sentence = results[40], results[0]
    assert clause[1][0] == "The short answer to your question is simple"  # trailing comma is not spoken
    assert sentence[1] == [reply]
    for metrics, _ in (clause, sentence):
        assert {"first_token_ms", "first_sentence_ms", "first_tts_ms", "first_audio_ms", "tts_ms", "total_ms"} <= set(metrics)
        assert metrics["first_audio_ms"] >= metrics["first_sentence_ms"] + metrics["first_tts_ms"] - 5
    # speaking the first clause starts earlier; the margin stays small because shared CI runners are slow
    assert clause[0]["first_sentence_ms"] + 40 < sentence[0]["first_sentence_ms"]
    assert clause[0]["first_audio_ms"] + 40 < sentence[0]["first_audio_ms"]
