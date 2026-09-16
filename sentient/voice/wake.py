"""Wake word detection for hands-free ("wake") voice sessions (docs/API.md section 16).

Two engines, selected by ``voice.wake_engine``:

- ``whisper``: the session's energy VAD cuts speech into segments; the first
  ``WAKE_WINDOW_S`` seconds of each segment are transcribed with a tiny/base
  faster-whisper model on the CPU and fuzzy-matched against ``voice.wake_word``
  (tolerating common mishearings such as "hey sentence" or "hi sentient"). Works
  for any phrase with no training; costs one tiny transcription per segment.
- ``openwakeword``: a streaming neural detector fed every audio frame (80 ms
  windows). Pretrained phrases (hey_jarvis, alexa, hey_mycroft, hey_rhasspy) or a
  custom ``.onnx``/``.tflite`` model path in ``voice.wake_model``. Model files are
  downloaded once from the openWakeWord GitHub release on first use.

A detector never loads anything in its constructor or ``status``; ``prepare``
(or the first detection) loads it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sentient.voice.audio import float_to_wav, pcm16_to_float, resample
from sentient.voice.base import STTProvider, VoiceError, models_dir

log = logging.getLogger(__name__)

WAKE_WINDOW_S = 3.0  # only the start of a segment is transcribed to look for the phrase

# ----------------------------------------------------------------------------- fuzzy phrase matching
_GREETING_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"hey", "hay", "hei", "hej", "heh", "he", "hi", "hy", "high", "eh", "ey", "ay", "yo", "hiya"}),
    frozenset({"ok", "okay", "okey", "okie", "k"}),
    frozenset({"hello", "hallo", "hullo", "helo"}),
)
# what speech recognisers commonly write instead of a wake name (compared without spaces)
MISHEARINGS: dict[str, frozenset[str]] = {
    "sentient": frozenset({
        "sentence", "sentences", "sentience", "sentients", "sentiant", "sentiens", "sentien", "centient",
        "sensient", "sendient", "sentinent", "sentin", "sentiment", "sentiments", "sentia", "sencient",
    }),
}  # fmt: skip


def _norm_token(piece: str) -> str:
    return re.sub(r"[^a-z0-9]", "", piece.lower())


def _pieces(text: str) -> list[str]:
    return re.sub(r"[-_/]", " ", text or "").split()


@dataclass
class WakeMatch:
    start: int  # index of the first word of the phrase (whitespace-split words of the text)
    end: int  # index just past the phrase
    score: float
    remainder: str  # what was said after the phrase, original casing and punctuation


def _greeting_group(word: str) -> frozenset[str] | None:
    for group in _GREETING_GROUPS:
        if word in group:
            return group
    return None


def _name_score(candidate: str, name: str) -> float:
    if not candidate:
        return 0.0
    if candidate == name or candidate in MISHEARINGS.get(name, ()):
        return 1.0
    if len(candidate) < 0.6 * len(name):
        return 0.0
    return SequenceMatcher(None, candidate, name).ratio()


def match_threshold(sensitivity: float) -> float:
    """Similarity needed for the name part: 0.9 at sensitivity 0, 0.75 at 0.5, 0.6 at 1."""
    return 0.9 - 0.3 * max(0.0, min(1.0, float(sensitivity)))


def match_wake_phrase(text: str, phrase: str, sensitivity: float = 0.5) -> WakeMatch | None:
    """Find ``phrase`` in a transcript, tolerating greeting swaps, split words and mishearings.

    A leading greeting in the phrase ("hey", "ok", "hello") must be heard as some
    greeting, so "this sentence is wrong" never wakes "hey sentient" while
    "hi sentence, what's up" does.
    """
    words = _pieces(phrase)
    p_tokens = [t for t in (_norm_token(w) for w in words) if t]
    if not p_tokens:
        return None
    pieces = _pieces(text)
    norm = [_norm_token(p) for p in pieces]
    idx = [i for i, t in enumerate(norm) if t]  # positions of real words
    tokens = [norm[i] for i in idx]
    greeting = p_tokens[0] if len(p_tokens) > 1 else None
    group = _greeting_group(greeting) if greeting else None
    if greeting and group is None and len(p_tokens) == 1:
        greeting = None
    name_tokens = p_tokens[1:] if greeting else p_tokens
    name = "".join(name_tokens)
    full = "".join(p_tokens)
    threshold = match_threshold(sensitivity)

    def greeting_ok(tok: str) -> bool:
        if group is not None:
            return tok in group
        return SequenceMatcher(None, tok, greeting).ratio() >= 0.8 if greeting else True

    for k in range(len(tokens)):
        best: tuple[float, int] | None = None
        if greeting:
            # merged into one word: "heysentient"
            s = SequenceMatcher(None, tokens[k], full).ratio()
            if s >= threshold and len(tokens[k]) >= len(full) - 1:  # the bare name ("sentient beings") is not enough
                best = (s, k + 1)
            if greeting_ok(tokens[k]):
                for n in range(1, len(name_tokens) + 2):
                    if k + 1 + n > len(tokens):
                        break
                    s = _name_score("".join(tokens[k + 1 : k + 1 + n]), name)
                    if s >= threshold and (best is None or s > best[0]):
                        best = (s, k + 1 + n)
        else:
            for n in range(1, len(name_tokens) + 2):
                if k + n > len(tokens):
                    break
                s = _name_score("".join(tokens[k : k + n]), name)
                if s >= threshold and (best is None or s > best[0]):
                    best = (s, k + n)
        if best is not None:
            score, end_tok = best
            start_piece = idx[k]
            end_piece = idx[end_tok - 1] + 1
            remainder = " ".join(pieces[end_piece:]).strip().lstrip(",.!?;:").strip()
            return WakeMatch(start=start_piece, end=end_piece, score=round(score, 3), remainder=remainder)
    return None


def has_words(text: str) -> bool:
    return any(ch.isalnum() for ch in text or "")


# ----------------------------------------------------------------------------- detectors
@dataclass
class WakeHit:
    phrase: str
    remainder: str = ""  # speech heard after the phrase in the same segment (segment engines)
    heard: str = ""  # raw detector transcript (segment engines)
    truncated: bool = False  # the segment was longer than what the detector looked at
    score: float | None = None

    @property
    def has_command(self) -> bool:
        return self.truncated or has_words(self.remainder)


class WakeDetector:
    """One per voice session. ``streaming`` engines get every frame through ``feed``;
    the others get finished VAD segments through ``check_segment``."""

    engine: str = "wake"
    streaming: bool = False
    phrase: str = ""
    error: str | None = None

    @property
    def ready(self) -> bool:
        return True

    async def prepare(self) -> None:
        """Load (and on first use download) whatever the engine needs. Raises VoiceError."""

    async def prepare_steps(self) -> AsyncIterator[dict[str, Any]]:
        """Progress dicts for ``POST /api/voice/prepare`` (component ``wake``)."""
        if self.ready:
            yield {"stage": "ready", "component": "wake", "progress": 1.0}
            return
        yield {"stage": "loading", "component": "wake", "progress": None}
        try:
            await self.prepare()
        except Exception as exc:
            yield {"stage": "error", "component": "wake", "message": str(exc)}
            return
        yield {"stage": "ready", "component": "wake", "progress": 1.0}

    async def feed(self, pcm16: bytes, sample_rate: int) -> WakeHit | None:
        return None

    async def check_segment(self, pcm16: bytes, sample_rate: int) -> WakeHit | None:
        return None

    def reset(self) -> None:
        """Forget streaming state (called when a session returns to standby)."""

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"engine": self.engine, "phrase": self.phrase, "ready": self.ready}
        if self.error:
            out["error"] = self.error
        return out

    def close(self) -> None:
        """Release per-session resources."""


class WhisperWakeDetector(WakeDetector):
    engine = "whisper"
    streaming = False

    def __init__(self, stt: STTProvider, phrase: str, sensitivity: float = 0.5, window_s: float = WAKE_WINDOW_S):
        self.stt = stt
        self.phrase = (phrase or "").strip()
        self.sensitivity = sensitivity
        self.window_s = window_s
        self.error = None if has_words(self.phrase) else "set voice.wake_word to use wake mode"

    @property
    def ready(self) -> bool:
        return self.error is None and self.stt.ready

    async def prepare(self) -> None:
        if self.error and not has_words(self.phrase):
            raise VoiceError(self.error)
        async for step in self.stt.prepare():
            if step.get("stage") == "error":
                self.error = str(step.get("message") or "wake model failed to load")
                raise VoiceError(self.error)

    async def prepare_steps(self) -> AsyncIterator[dict[str, Any]]:
        if not has_words(self.phrase):
            yield {"stage": "error", "component": "wake", "message": self.error}
            return
        async for step in self.stt.prepare():
            yield {**step, "component": "wake"}

    async def check_segment(self, pcm16: bytes, sample_rate: int) -> WakeHit | None:
        if not has_words(self.phrase):
            return None
        limit = int(self.window_s * sample_rate) * 2
        head = pcm16[:limit]
        text = await self.stt.transcribe(head, sample_rate)
        if not text.strip():
            return None
        m = match_wake_phrase(text, self.phrase, self.sensitivity)
        if m is None:
            log.debug("wake: heard %r (no match for %r)", text, self.phrase)
            return None
        return WakeHit(self.phrase, remainder=m.remainder, heard=text, truncated=len(pcm16) > limit, score=m.score)

    def status(self) -> dict[str, Any]:
        out = super().status()
        out["model"] = self.stt.model
        return out


OPENWAKEWORD_PRETRAINED = ("alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy", "timer", "weather")
OPENWAKEWORD_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
OPENWAKEWORD_FEATURES = ("melspectrogram.onnx", "embedding_model.onnx")
_loaded_openwakeword: set[str] = set()  # models loaded at least once in this process


def resolve_openwakeword_model(model: str, phrase: str) -> tuple[str, str]:
    """-> ("pretrained", name) or ("path", file). Raises VoiceError with a friendly message."""
    spec = (model or "").strip()
    if spec and (spec.lower().endswith((".onnx", ".tflite")) or "/" in spec or "\\" in spec):
        p = Path(spec).expanduser()
        if p.suffix.lower() == ".tflite" and p.with_suffix(".onnx").is_file():
            p = p.with_suffix(".onnx")  # tflite-runtime is not available on Windows; use the onnx twin
        if not p.is_file():
            raise VoiceError(f"wake model file not found: {spec}")
        if p.suffix.lower() != ".onnx":
            raise VoiceError("custom openWakeWord models must be .onnx (or a .tflite with an .onnx next to it)")
        return "path", str(p)
    name = "_".join(t for t in (_norm_token(w) for w in _pieces(spec or phrase)) if t)
    for known in OPENWAKEWORD_PRETRAINED:
        if name == known or name == known.replace("_", ""):
            return "pretrained", known
    raise VoiceError(
        f"openWakeWord has no pretrained model for '{spec or phrase}'. Use one of "
        + ", ".join(n.replace("_", " ") for n in OPENWAKEWORD_PRETRAINED[:4])
        + ", set voice.wake_model to a custom .onnx model, or use the whisper wake engine."
    )


class OpenWakeWordDetector(WakeDetector):
    engine = "openwakeword"
    streaming = True
    FRAME = 1280  # 80 ms at 16 kHz

    def __init__(self, phrase: str, sensitivity: float = 0.5, model: str = "", root: Path | None = None):
        self.sensitivity = sensitivity
        self.root = root
        self.threshold = min(0.95, max(0.05, 1.0 - float(sensitivity)))
        self.kind: str | None = None
        self.target: str | None = None
        self.error = None
        try:
            self.kind, self.target = resolve_openwakeword_model(model, phrase)
        except VoiceError as exc:
            self.error = str(exc)
        if self.kind == "pretrained" and self.target:
            self.phrase = self.target.replace("_", " ")
        elif self.target:
            self.phrase = (phrase or Path(self.target).stem).strip()
        else:
            self.phrase = (phrase or "").strip()
        self._model: Any = None
        self._buf = np.zeros(0, dtype=np.int16)
        self._cooldown_until = 0.0
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.target}"

    @property
    def ready(self) -> bool:
        return self._model is not None or (self.error is None and self.key in _loaded_openwakeword)

    def _dir(self) -> Path:
        return self.root or models_dir("openwakeword")

    def _model_file(self) -> Path:
        assert self.target is not None
        if self.kind == "path":
            return Path(self.target)
        return self._dir() / f"{self.target}_v0.1.onnx"

    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if self.error:
                raise VoiceError(self.error)
            root = self._dir()
            needed = [root / f for f in OPENWAKEWORD_FEATURES] + [self._model_file()]
            missing = [p for p in needed if not p.is_file() or p.stat().st_size < 1024]
            if missing:
                try:
                    from openwakeword.utils import download_models

                    log.info("openWakeWord: downloading %s", ", ".join(p.name for p in missing))
                    for p in missing:  # the helper writes error pages as files; clear them first
                        with contextlib.suppress(OSError):
                            p.unlink()
                    download_models([self.target] if self.kind == "pretrained" else ["__features_only__"], str(root))
                except Exception as exc:
                    self.error = f"could not download openWakeWord models: {exc}"
                    raise VoiceError(self.error) from exc
                still = [p.name for p in needed if not p.is_file() or p.stat().st_size < 1024]
                if still:
                    self.error = f"openWakeWord model files missing after download: {', '.join(still)}"
                    raise VoiceError(self.error)
            try:
                from openwakeword.model import Model

                self._model = Model(
                    wakeword_models=[str(self._model_file())],
                    inference_framework="onnx",
                    melspec_model_path=str(root / OPENWAKEWORD_FEATURES[0]),
                    embedding_model_path=str(root / OPENWAKEWORD_FEATURES[1]),
                )
            except Exception as exc:
                self.error = f"could not load openWakeWord model: {exc}"
                raise VoiceError(self.error) from exc
            _loaded_openwakeword.add(self.key)
            self.error = None
            return self._model

    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentient-wake")
        return self._executor

    async def prepare(self) -> None:
        if self._model is not None:
            return
        await asyncio.get_running_loop().run_in_executor(self._pool(), self._load)

    def _predict(self, chunks: list[np.ndarray]) -> float:
        best = 0.0
        for chunk in chunks:
            scores = self._model.predict(chunk)
            if scores:
                best = max(best, float(max(scores.values())))
        return best

    async def feed(self, pcm16: bytes, sample_rate: int) -> WakeHit | None:
        if self._model is None:
            return None
        audio = np.frombuffer(pcm16[: len(pcm16) - len(pcm16) % 2], dtype="<i2")
        if sample_rate != 16000:
            audio = np.clip(resample(pcm16_to_float(audio.tobytes()), sample_rate, 16000) * 32768, -32768, 32767)
            audio = audio.astype(np.int16)
        self._buf = np.concatenate([self._buf, audio.astype(np.int16)])
        n = len(self._buf) // self.FRAME
        if n == 0:
            return None
        chunks = [self._buf[i * self.FRAME : (i + 1) * self.FRAME] for i in range(n)]
        self._buf = self._buf[n * self.FRAME :]
        score = await asyncio.get_running_loop().run_in_executor(self._pool(), self._predict, chunks)
        now = time.monotonic()
        if score >= self.threshold and now >= self._cooldown_until:
            self._cooldown_until = now + 1.5
            self.reset()
            return WakeHit(self.phrase, score=round(score, 3))
        return None

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)
        if self._model is not None:
            with contextlib.suppress(Exception):
                self._model.reset()

    def status(self) -> dict[str, Any]:
        out = super().status()
        if self.target:
            out["model"] = self.target if self.kind == "pretrained" else Path(self.target).name
        return out

    def close(self) -> None:
        self._model = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


# ----------------------------------------------------------------------------- earcon
EARCON_RATE = 24000


@lru_cache(maxsize=1)
def earcon_wav() -> bytes:
    """A short rising two-note chime (~180 ms) played when the wake word is heard."""
    parts = []
    for freq, ms in ((880.0, 70), (1318.5, 110)):
        n = int(EARCON_RATE * ms / 1000)
        t = np.arange(n, dtype=np.float32) / EARCON_RATE
        env = np.minimum(1.0, np.minimum(t / 0.008, (t[-1] - t + 1e-4) / 0.03)).astype(np.float32)
        parts.append(0.22 * env * np.sin(2 * math.pi * freq * t))
    return float_to_wav(np.concatenate(parts), EARCON_RATE)
