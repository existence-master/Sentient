import json
import os
import sys
import types as pytypes

import httpx
import pytest
import respx

from sentient import secrets
from sentient.voice.audio import pcm16_to_wav, wav_to_pcm16
from sentient.voice.base import VoiceError
from sentient.voice.stt import (
    DeepgramSTT,
    ElevenLabsSTT,
    FasterWhisperSTT,
    OpenAISTT,
    build_stt,
    stt_language,
)
from sentient.voice.tts import (
    KOKORO_MODELS,
    KOKORO_RELEASE,
    KOKORO_VOICES_FILE,
    ElevenLabsTTS,
    KokoroTTS,
    OpenAITTS,
    SystemTTS,
    build_tts,
)
from tests.voice.conftest import tone


@pytest.fixture
def fake_keys(monkeypatch):
    monkeypatch.setattr(secrets, "get_secret", lambda name, env=None: f"key-{name}")


@pytest.fixture
def no_keys(monkeypatch):
    monkeypatch.setattr(secrets, "get_secret", lambda name, env=None: None)


# ----------------------------------------------------------------------------- REST
def test_rest_requires_token(voice_client):
    client = voice_client()
    client.headers.pop("Authorization")
    assert client.get("/api/voice/status").status_code == 401
    assert client.post("/api/voice/speak", json={"text": "hi"}).status_code == 401


def test_status(voice_client):
    client = voice_client()
    body = client.get("/api/voice/status").json()
    assert body["stt"]["provider"] == "fake_stt" and body["stt"]["ready"] is True
    assert body["tts"]["provider"] == "fake_tts"
    assert body["tts"]["voices"] == [{"id": "v1", "name": "Voice One", "language": "en-US"}]


def test_status_with_real_default_providers_loads_nothing(config, isolated_home):
    from fastapi.testclient import TestClient

    from sentient.app import SentientApp
    from sentient.gateway.app import create_app
    from tests.conftest import FakeProvider

    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "s.db", enable_background=False)
    with TestClient(create_app(core)) as client:
        body = client.get("/api/voice/status", headers={"Authorization": f"Bearer {client.app.state.token}"}).json()
    assert body["stt"] == {"provider": "faster_whisper", "model": "base", "ready": False, "device": None}
    assert body["tts"]["provider"] == "system"


def test_speak_returns_wav_and_strips_markdown(voice_client):
    client = voice_client()
    r = client.post("/api/voice/speak", json={"text": "**Hello** [world](https://x.io)!", "voice": "v1"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert client.tts.texts == ["Hello world!"] and client.tts.voices_used == ["v1"]
    assert client.post("/api/voice/speak", json={"text": "```\ncode\n```"}).status_code == 400


def test_transcribe_upload(voice_client):
    client = voice_client()
    wav = pcm16_to_wav(tone(300), 16000)
    r = client.post("/api/voice/transcribe", files={"file": ("clip.wav", wav, "audio/wav")})
    assert r.status_code == 200 and r.json() == {"text": "dictated text"}
    r = client.post("/api/voice/transcribe", files={"file": ("blob", b"\x1aE\xdf\xa3", "audio/webm;codecs=opus")})
    assert r.status_code == 200
    assert client.stt.files == [".wav", ".webm"]
    assert client.post("/api/voice/transcribe", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415


def test_prepare_streams_ndjson(voice_client):
    client = voice_client()
    r = client.post("/api/voice/prepare", json={"target": "all"})
    lines = [json.loads(line) for line in r.text.strip().splitlines()]
    assert [ln["component"] for ln in lines[:-1]] == ["stt", "tts"]
    assert lines[-1] == {"stage": "done", "progress": 1.0, "ok": True}


# ----------------------------------------------------------------------------- provider selection
def test_build_providers_from_config(config):
    assert isinstance(build_stt(config), FasterWhisperSTT)
    assert isinstance(build_tts(config), SystemTTS)
    config.voice.stt_provider, config.voice.tts_provider = "openai", "kokoro"
    stt = build_stt(config)
    assert isinstance(stt, OpenAISTT) and stt.model == "whisper-1"  # 'base' is a local size
    assert isinstance(build_tts(config), KokoroTTS)
    config.voice.stt_provider, config.voice.tts_provider = "deepgram", "elevenlabs"
    assert isinstance(build_stt(config), DeepgramSTT) and isinstance(build_tts(config), ElevenLabsTTS)
    config.voice.stt_provider, config.voice.tts_provider = "elevenlabs", "openai"
    assert isinstance(build_stt(config), ElevenLabsSTT) and isinstance(build_tts(config), OpenAITTS)


def test_language_resolution(config):
    assert stt_language("en-US") == "en" and stt_language("auto") is None and stt_language("") is None
    config.assistant.language = "hi-IN"
    assert build_stt(config).language == "hi"
    config.voice.stt_language = "auto"
    assert build_stt(config).language is None


async def test_service_rebuilds_on_config_change(config, isolated_home):
    from sentient.app import SentientApp
    from tests.conftest import FakeProvider

    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False)
    svc = core.voice
    stt1, tts1 = svc.stt, svc.tts
    assert svc.stt is stt1 and svc.tts is tts1
    config.voice.tts_voice, config.voice.tts_speed = "Zira", 1.3
    assert svc.tts is tts1 and tts1.voice == "Zira" and tts1.speed == 1.3  # in place
    config.voice.stt_model = "tiny"
    svc.refresh()
    assert svc._stt is None
    assert svc.stt is not stt1 and svc.stt.model == "tiny"
    config.voice.tts_provider = "openai"
    assert isinstance(svc.tts, OpenAITTS)
    await svc.stop()


# ----------------------------------------------------------------------------- faster-whisper device fallback
def test_faster_whisper_falls_back_to_cpu(monkeypatch, tmp_path):
    created = []

    class FakeWhisperModel:
        def __init__(self, name, device, compute_type, download_root, **kw):
            created.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("CUDA failed")

    monkeypatch.setitem(sys.modules, "faster_whisper", pytypes.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr("sentient.voice.stt.cuda_unavailable_reason", lambda: None)
    stt = FasterWhisperSTT("large-v3", "auto", "en", download_root=tmp_path)
    stt._load()
    assert created == [("cuda", "float16"), ("cpu", "int8")]
    assert stt.device == "cpu" and "CUDA failed" in stt.fallback_reason
    assert stt.status()["note"].startswith("CPU fallback")

    created.clear()
    monkeypatch.setattr("sentient.voice.stt.cuda_unavailable_reason", lambda: "missing cublas64_12.dll")
    stt2 = FasterWhisperSTT("medium", "auto", None, download_root=tmp_path)
    stt2._load()
    assert created == [("cpu", "int8")] and stt2.fallback_reason == "missing cublas64_12.dll"

    # auto keeps small models on the CPU without probing the GPU at all
    created.clear()
    monkeypatch.setattr("sentient.voice.stt.cuda_unavailable_reason", lambda: pytest.fail("probed CUDA"))
    stt3 = FasterWhisperSTT("base", "auto", None, download_root=tmp_path)
    stt3._load()
    assert created == [("cpu", "int8")] and stt3.fallback_reason is None
    for s in (stt, stt2, stt3):
        s.close()


# ----------------------------------------------------------------------------- cloud request shapes
@respx.mock
async def test_openai_stt_request(fake_keys):
    route = respx.post("https://api.openai.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": " hello world "})
    )
    stt = OpenAISTT("gpt-4o-transcribe", "en")
    assert await stt.transcribe(tone(500), 16000) == "hello world"
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer key-openai"
    body = req.content
    assert b'name="model"\r\n\r\ngpt-4o-transcribe' in body
    assert b'name="language"\r\n\r\nen' in body
    assert b'filename="audio.wav"' in body and b"RIFF" in body


@respx.mock
async def test_deepgram_stt_request(fake_keys):
    route = respx.post("https://api.deepgram.com/v1/listen").mock(
        return_value=httpx.Response(200, json={"results": {"channels": [{"alternatives": [{"transcript": "hi there"}]}]}})
    )
    assert await DeepgramSTT("", "en").transcribe(tone(500), 16000) == "hi there"
    req = route.calls.last.request
    assert req.headers["authorization"] == "Token key-deepgram"
    assert req.url.params["model"] == "nova-3" and req.url.params["language"] == "en"
    assert req.headers["content-type"] == "audio/wav"


@respx.mock
async def test_elevenlabs_stt_request(fake_keys):
    route = respx.post("https://api.elevenlabs.io/v1/speech-to-text").mock(
        return_value=httpx.Response(200, json={"text": "scribed"})
    )
    assert await ElevenLabsSTT("base").transcribe(tone(500), 16000) == "scribed"
    req = route.calls.last.request
    assert req.headers["xi-api-key"] == "key-elevenlabs"
    assert b'name="model_id"\r\n\r\nscribe_v1' in req.content


@respx.mock
async def test_openai_tts_request(fake_keys):
    pcm = tone(200, sr=24000)
    route = respx.post("https://api.openai.com/v1/audio/speech").mock(return_value=httpx.Response(200, content=pcm))
    wav = await OpenAITTS("nova", 1.25).synthesize("Hello there.")
    body = json.loads(route.calls.last.request.content)
    assert body == {"model": "gpt-4o-mini-tts", "input": "Hello there.", "voice": "nova", "response_format": "pcm", "speed": 1.25}
    assert route.calls.last.request.headers["authorization"] == "Bearer key-openai"
    assert wav_to_pcm16(wav) == (pcm, 24000)


@respx.mock
async def test_elevenlabs_tts_request(fake_keys):
    pcm = tone(200, sr=22050)
    route = respx.post("https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM").mock(
        return_value=httpx.Response(200, content=pcm)
    )
    wav = await ElevenLabsTTS("", 1.5).synthesize("Hi.")
    req = route.calls.last.request
    assert req.url.params["output_format"] == "pcm_22050"
    assert req.headers["xi-api-key"] == "key-elevenlabs"
    body = json.loads(req.content)
    assert body["model_id"] == "eleven_flash_v2_5" and body["text"] == "Hi."
    assert body["voice_settings"]["speed"] == 1.2  # clamped to the API range
    assert wav_to_pcm16(wav) == (pcm, 22050)


@respx.mock
async def test_cloud_http_error_is_voice_error(fake_keys):
    respx.post("https://api.openai.com/v1/audio/speech").mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(VoiceError, match="HTTP 401"):
        await OpenAITTS().synthesize("hi")


async def test_missing_key_is_voice_error(no_keys):
    with pytest.raises(VoiceError, match="No OpenAI API key"):
        await OpenAITTS().synthesize("hi")
    stt = OpenAISTT()
    assert stt.ready is False and "error" in stt.status()


# ----------------------------------------------------------------------------- kokoro download
@respx.mock
async def test_kokoro_prepare_downloads_with_progress(tmp_path, monkeypatch):
    model_bytes, voices_bytes = b"m" * 5000, b"v" * 3000
    respx.get(KOKORO_RELEASE + KOKORO_MODELS["int8"]).mock(return_value=httpx.Response(200, content=model_bytes))
    respx.get(KOKORO_RELEASE + KOKORO_VOICES_FILE).mock(return_value=httpx.Response(200, content=voices_bytes))
    tts = KokoroTTS("af_heart", 1.0, "int8", root=tmp_path)
    monkeypatch.setattr(tts, "_load", lambda: setattr(tts, "_kokoro", object()))
    assert tts.downloaded is False and tts.ready is False
    steps = [s async for s in tts.prepare()]
    assert steps[-1]["stage"] == "ready"
    files = [s["file"] for s in steps if s["stage"] == "download" and s["progress"] == 1.0]
    assert files == [KOKORO_MODELS["int8"], KOKORO_VOICES_FILE]
    assert tts.model_path.read_bytes() == model_bytes and tts.voices_path.read_bytes() == voices_bytes
    assert not list(tmp_path.glob("*.part"))
    tts.close()


# ----------------------------------------------------------------------------- real OS voice (offline)
@pytest.mark.skipif(sys.platform != "win32", reason="SAPI voices are always present on Windows")
async def test_system_tts_real_windows_voice():
    tts = SystemTTS("", 1.0)
    try:
        wav = await tts.synthesize("Testing one two.")
        again = await tts.synthesize("Second call works too.")  # no run-loop reentrancy errors
    finally:
        tts.close()
    pcm, sr = wav_to_pcm16(wav)
    if not pcm and os.environ.get("CI"):
        pytest.skip("this machine has no speech voices installed (headless CI runner)")
    assert sr > 8000 and len(pcm) > sr * 2 * 0.3
    assert again[:4] == b"RIFF" and tts.backend in {"pyttsx3", "powershell"}
