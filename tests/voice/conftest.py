"""Voice test fixtures: fake STT/TTS, synthetic PCM, a gateway client factory."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.provider import StreamChunk
from sentient.voice.audio import pcm16_to_wav
from sentient.voice.base import STTProvider, TTSProvider
from sentient.voice.wake import WakeDetector, WakeHit
from tests.conftest import FakeProvider

SR = 16000


def tone(ms: int, sr: int = SR, freq: float = 220.0, amp: float = 0.3) -> bytes:
    n = int(sr * ms / 1000)
    t = np.arange(n, dtype=np.float32) / sr
    wave = amp * np.sin(2 * math.pi * freq * t)
    return (wave * 32767).astype("<i2").tobytes()


def silence(ms: int, sr: int = SR) -> bytes:
    return b"\x00\x00" * int(sr * ms / 1000)


def noise(ms: int, sr: int = SR, amp: float = 0.003, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    n = int(sr * ms / 1000)
    return (rng.uniform(-amp, amp, n) * 32767).astype("<i2").tobytes()


def frames(pcm: bytes, sr: int = SR, ms: int = 20) -> list[bytes]:
    step = int(sr * ms / 1000) * 2
    return [pcm[i : i + step] for i in range(0, len(pcm), step)]


class FakeSTT(STTProvider):
    name = "fake_stt"
    model = "fake"

    def __init__(self, texts: list[str] | None = None, delay: float = 0.0):
        self.texts = list(texts or [])
        self.delay = delay
        self.calls: list[tuple[int, int]] = []
        self.files: list[str] = []
        self.device = "cpu"

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        self.calls.append((len(pcm16), sample_rate))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.texts.pop(0) if self.texts else "hello"

    async def transcribe_file(self, path) -> str:
        self.files.append(Path(path).suffix)
        return "dictated text"


class FakeTTS(TTSProvider):
    name = "fake_tts"

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.texts: list[str] = []
        self.voices_used: list[str | None] = []

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        self.texts.append(text)
        self.voices_used.append(voice)
        if self.delay:
            await asyncio.sleep(self.delay)
        return pcm16_to_wav(b"\x01\x00" * 320, 16000)

    async def voices(self) -> list[dict[str, Any]]:
        return [{"id": "v1", "name": "Voice One", "language": "en-US"}]


class FakeWakeDetector(WakeDetector):
    """Scripted wake detector. Segment mode pops ``hits`` (WakeHit or None) per VAD segment;
    streaming mode fires once after ``trigger_after`` bytes of audio."""

    engine = "fake_wake"

    def __init__(
        self,
        hits: list[WakeHit | None] | None = None,
        *,
        streaming: bool = False,
        trigger_after: int = 0,
        phrase: str = "hey sentient",
    ):
        self.hits = list(hits or [])
        self.streaming = streaming
        self.trigger_after = trigger_after
        self.phrase = phrase
        self.segments: list[int] = []
        self.fed = 0
        self.resets = 0

    async def check_segment(self, pcm16: bytes, sample_rate: int) -> WakeHit | None:
        self.segments.append(len(pcm16))
        return self.hits.pop(0) if self.hits else None

    async def feed(self, pcm16: bytes, sample_rate: int) -> WakeHit | None:
        self.fed += len(pcm16)
        if self.streaming and self.trigger_after and self.fed >= self.trigger_after:
            self.trigger_after = 0
            return WakeHit(self.phrase, score=0.9)
        return None

    def reset(self) -> None:
        self.resets += 1


class SlowProvider(FakeProvider):
    """FakeProvider that trickles text so a turn can be interrupted mid-generation."""

    def __init__(self, *args, chunk_delay: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_delay = chunk_delay

    async def stream(self, role, messages, tools=None, *, model=None):
        self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
        reply = self.replies.pop(0) if self.replies else "ok"
        for i in range(0, len(reply), 5):
            await asyncio.sleep(self.chunk_delay)
            yield StreamChunk(text=reply[i : i + 5], model="fake")
        yield StreamChunk(done=True, usage={"prompt_tokens": 1, "completion_tokens": 1}, model="fake")


def recv(ws) -> dict | bytes:
    msg = ws.receive()
    if msg.get("text") is not None:
        return json.loads(msg["text"])
    if msg.get("bytes") is not None:
        return msg["bytes"]
    raise AssertionError(f"socket closed or unexpected message: {msg}")


def collect_until(ws, pred, limit: int = 2000) -> list[dict | bytes]:
    items: list[dict | bytes] = []
    for _ in range(limit):
        item = recv(ws)
        items.append(item)
        if isinstance(item, dict) and pred(item):
            return items
    raise AssertionError(f"condition not met; got {[i.get('type') if isinstance(i, dict) else 'bytes' for i in items]}")


def types(items: list[dict | bytes]) -> list[str]:
    return [i["type"] if isinstance(i, dict) else "<bytes>" for i in items]


@pytest.fixture
def voice_client(config, isolated_home):
    stack = contextlib.ExitStack()

    def make(
        *,
        replies: list[Any] | None = None,
        llm: Any = None,
        stt_texts: list[str] | None = None,
        tts_delay: float = 0.0,
        approvals: str = "off",
        wake: Any = None,
        voice: dict[str, Any] | None = None,
    ) -> TestClient:
        config.tools.approvals.mode = approvals
        for key, value in (voice or {}).items():
            setattr(config.voice, key, value)
        core = SentientApp(
            config,
            llm=llm or FakeProvider(replies=replies or ["Hello there. How can I help?"]),
            db_path=isolated_home / "voice.db",
            enable_background=False,
        )
        stt, tts = FakeSTT(stt_texts), FakeTTS(delay=tts_delay)
        core.voice.use_providers(stt=stt, tts=tts)
        if wake is not None:
            core.voice.use_wake_detector(wake)
        client = stack.enter_context(TestClient(create_app(core)))
        client.token = client.app.state.token
        client.core, client.stt, client.tts = core, stt, tts
        client.headers.update({"Authorization": f"Bearer {client.token}"})
        return client

    yield make
    stack.close()
