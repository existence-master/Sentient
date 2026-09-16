"""Speech-to-text providers: local faster-whisper plus OpenAI, Deepgram and ElevenLabs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import mimetypes
import os
import sys
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from sentient.voice.audio import pcm16_to_float, pcm16_to_wav, resample
from sentient.voice.base import (
    STTProvider,
    VoiceError,
    has_key,
    models_dir,
    raise_for_status,
    require_key,
)

log = logging.getLogger(__name__)

WHISPER_SIZES_MB = {
    "tiny": 75, "tiny.en": 75, "base": 145, "base.en": 145, "small": 484, "small.en": 484,
    "distil-small.en": 336, "medium": 1530, "medium.en": 1530, "distil-medium.en": 789,
    "large-v1": 3090, "large-v2": 3090, "large-v3": 3090, "large": 3090,
    "distil-large-v2": 1510, "distil-large-v3": 1510, "large-v3-turbo": 1620, "turbo": 1620,
}  # fmt: skip
CPU_FRIENDLY_MODELS = {"tiny", "tiny.en", "base", "base.en", "small", "small.en", "distil-small.en"}
MIN_AUDIO_S = 0.25
_AUDIO_MIME = {
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
}


def mime_for(path: Path) -> str:
    return (
        _AUDIO_MIME.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def _dir_size(root: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(dirpath, f))
    return total


# ----------------------------------------------------------------------------- CUDA probing
_dll_dirs_registered = False


def _register_nvidia_wheel_dlls() -> None:
    """Make cuBLAS/cuDNN from ``nvidia-*-cu12`` pip wheels visible to CTranslate2 on Windows."""
    global _dll_dirs_registered
    if _dll_dirs_registered or sys.platform != "win32":
        return
    _dll_dirs_registered = True
    for base in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            with contextlib.suppress(OSError):
                os.add_dll_directory(str(bin_dir))  # type: ignore[attr-defined]
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def cuda_unavailable_reason() -> str | None:
    """Why local CUDA inference cannot work, or None.

    Probing the libraries up front matters: CTranslate2 can abort the whole
    process when cuDNN is missing at inference time instead of raising.
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() < 1:
            return "no CUDA device visible to CTranslate2"
    except Exception as exc:
        return f"ctranslate2 unavailable: {exc}"
    if sys.platform == "darwin":
        return "CUDA is not supported on macOS"
    _register_nvidia_wheel_dlls()
    import ctypes

    if sys.platform == "win32":
        libs = ["cublas64_12.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"]
        loader: Any = ctypes.WinDLL  # type: ignore[attr-defined]
    else:
        libs = ["libcublas.so.12", "libcudnn_ops.so.9"]
        loader = ctypes.CDLL
    missing = []
    for lib in libs:
        try:
            loader(lib)
        except OSError:
            missing.append(lib)
    if missing:
        return (
            "missing " + ", ".join(missing)
            + " (install CUDA 12 cuBLAS and cuDNN 9, e.g. pip install nvidia-cublas-cu12 nvidia-cudnn-cu12)"
        )
    return None


# ----------------------------------------------------------------------------- faster-whisper
class FasterWhisperSTT(STTProvider):
    name = "faster_whisper"

    def __init__(
        self,
        model: str = "base",
        device: str = "auto",
        language: str | None = None,
        download_root: Path | None = None,
    ):
        self.model = model or "base"
        self.requested_device = device
        self.language = language
        self.download_root = download_root
        self.device = None
        self.compute_type: str | None = None
        self.fallback_reason: str | None = None
        self.error = None
        self.load_seconds: float | None = None
        self._model: Any = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentient-stt")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _root(self) -> Path:
        return self.download_root or models_dir("whisper")

    # ---------------------------------------------------------------- loading (worker thread)
    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            from faster_whisper import WhisperModel

            started = time.perf_counter()
            root = str(self._root())
            model = None
            # auto: small models run faster on the CPU than on a GPU that is also serving the
            # local LLM (measured on an RTX 4060 laptop), so only medium/large try CUDA first
            use_gpu = self.requested_device == "cuda" or (
                self.requested_device == "auto" and self.model not in CPU_FRIENDLY_MODELS
            )
            if use_gpu:
                reason = cuda_unavailable_reason()
                if reason is None:
                    try:
                        model = WhisperModel(
                            self.model, device="cuda", compute_type="float16", download_root=root
                        )
                        self._warmup(model)
                        self.device, self.compute_type = "cuda", "float16"
                    except Exception as exc:
                        model = None
                        reason = f"{type(exc).__name__}: {exc}"
                if model is None:
                    self.fallback_reason = reason
                    log.warning("faster-whisper: CUDA not used (%s); falling back to CPU int8", reason)
            if model is None:
                try:
                    model = WhisperModel(
                        self.model,
                        device="cpu",
                        compute_type="int8",
                        download_root=root,
                        cpu_threads=max(1, min(8, (os.cpu_count() or 4) // 2)),
                    )
                except Exception as exc:
                    self.error = f"could not load faster-whisper '{self.model}': {exc}"
                    raise VoiceError(self.error) from exc
                self.device, self.compute_type = "cpu", "int8"
                try:  # the first real transcription otherwise pays CTranslate2's one-time setup
                    self._warmup(model)
                except Exception as exc:
                    log.debug("faster-whisper CPU warm-up failed: %s", exc)
            self._model = model
            self.error = None
            self.load_seconds = round(time.perf_counter() - started, 2)
            log.info(
                "faster-whisper %s loaded on %s/%s in %.1fs",
                self.model, self.device, self.compute_type, self.load_seconds,
            )
            return model

    @staticmethod
    def _warmup(model: Any) -> None:
        import numpy as np

        segments, _info = model.transcribe(
            np.zeros(16000, dtype=np.float32), beam_size=1, vad_filter=False, language="en"
        )
        list(segments)

    def _run(self, audio: Any, beam_size: int) -> str:
        model = self._load()
        segments, _info = model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        parts = []
        for seg in segments:
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                continue  # classic whisper hallucination on noise ("Thank you.")
            parts.append(seg.text.strip())
        return " ".join(p for p in parts if p).strip()

    async def _submit(self, fn, *args) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    # ---------------------------------------------------------------- public
    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        audio = resample(pcm16_to_float(pcm16), sample_rate, 16000)
        if audio.size < MIN_AUDIO_S * 16000:
            return ""
        return await self._submit(self._run, audio, 1)

    async def transcribe_file(self, path: str | Path) -> str:
        return await self._submit(self._run, str(path), 5)

    async def prepare(self) -> AsyncIterator[dict[str, Any]]:
        if self.ready:
            yield {"stage": "ready", "component": "stt", "progress": 1.0, "device": self.device}
            return
        root = self._root()
        expected = WHISPER_SIZES_MB.get(self.model, 0) * 1024 * 1024
        fut = asyncio.ensure_future(self._submit(self._load))
        yield {"stage": "download", "component": "stt", "model": self.model, "progress": 0.0}
        while not fut.done():
            await asyncio.wait({fut}, timeout=0.5)
            if fut.done():
                break
            size = await asyncio.to_thread(_dir_size, root)
            if expected:
                progress: float | None = min(0.99, size / expected)
                stage = "loading" if size >= expected * 0.97 else "download"
            else:
                progress, stage = None, "download"
            yield {"stage": stage, "component": "stt", "model": self.model, "progress": progress, "bytes": size}
        try:
            fut.result()
        except Exception as exc:
            yield {"stage": "error", "component": "stt", "message": str(exc)}
            return
        out: dict[str, Any] = {"stage": "ready", "component": "stt", "progress": 1.0, "device": self.device}
        if self.fallback_reason:
            out["note"] = f"CPU fallback: {self.fallback_reason}"
        yield out

    def status(self) -> dict[str, Any]:
        out = super().status()
        if self.compute_type:
            out["compute_type"] = self.compute_type
        if self.fallback_reason:
            out["note"] = f"CPU fallback: {self.fallback_reason}"
        return out

    def close(self) -> None:
        self._model = None
        self._executor.shutdown(wait=False, cancel_futures=True)


# ----------------------------------------------------------------------------- cloud
class _CloudSTT(STTProvider):
    label = ""
    secret = ""
    env_var = ""
    default_model = ""
    default_base = ""

    def __init__(self, model: str = "", language: str | None = None, api_base: str | None = None):
        self.model = model if model and model not in WHISPER_SIZES_MB else self.default_model
        self.language = language
        self.api_base = (api_base or self.default_base).rstrip("/")
        self.device = "cloud"

    @property
    def ready(self) -> bool:
        return has_key(self.secret, self.env_var)

    def status(self) -> dict[str, Any]:
        out = super().status()
        if not self.ready:
            out["error"] = f"No {self.label} API key"
        return out

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        if len(pcm16) < MIN_AUDIO_S * sample_rate * 2:
            return ""
        return await self._send(pcm16_to_wav(pcm16, sample_rate), "audio.wav", "audio/wav")

    async def transcribe_file(self, path: str | Path) -> str:
        p = Path(path)
        data = await asyncio.to_thread(p.read_bytes)
        return await self._send(data, p.name, mime_for(p))

    async def _send(self, data: bytes, filename: str, mime: str) -> str:
        raise NotImplementedError


class OpenAISTT(_CloudSTT):
    name = "openai"
    label = "OpenAI"
    secret = "openai"
    env_var = "OPENAI_API_KEY"
    default_model = "whisper-1"
    default_base = "https://api.openai.com/v1"

    async def _send(self, data: bytes, filename: str, mime: str) -> str:
        key = require_key(self.secret, self.env_var, self.label)
        form = {"model": self.model, "response_format": "json"}
        if self.language:
            form["language"] = self.language
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.api_base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                data=form,
                files={"file": (filename, data, mime)},
            )
        raise_for_status(r, self.label)
        return str(r.json().get("text") or "").strip()


class DeepgramSTT(_CloudSTT):
    name = "deepgram"
    label = "Deepgram"
    secret = "deepgram"
    env_var = "DEEPGRAM_API_KEY"
    default_model = "nova-3"
    default_base = "https://api.deepgram.com/v1"

    async def _send(self, data: bytes, filename: str, mime: str) -> str:
        key = require_key(self.secret, self.env_var, self.label)
        params = {"model": self.model, "smart_format": "true", "punctuate": "true"}
        if self.language:
            params["language"] = self.language
        else:
            params["detect_language"] = "true"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.api_base}/listen",
                params=params,
                headers={"Authorization": f"Token {key}", "Content-Type": mime},
                content=data,
            )
        raise_for_status(r, self.label)
        try:
            return str(r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise VoiceError(f"Deepgram returned an unexpected response: {r.text[:200]}") from exc


class ElevenLabsSTT(_CloudSTT):
    name = "elevenlabs"
    label = "ElevenLabs"
    secret = "elevenlabs"
    env_var = "ELEVENLABS_API_KEY"
    default_model = "scribe_v1"
    default_base = "https://api.elevenlabs.io/v1"

    async def _send(self, data: bytes, filename: str, mime: str) -> str:
        key = require_key(self.secret, self.env_var, self.label)
        form = {"model_id": self.model}
        if self.language:
            form["language_code"] = self.language
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.api_base}/speech-to-text",
                headers={"xi-api-key": key},
                data=form,
                files={"file": (filename, data, mime)},
            )
        raise_for_status(r, self.label)
        return str(r.json().get("text") or "").strip()


def stt_language(language: str) -> str | None:
    """BCP-47 ('en-US') -> ISO-639-1 ('en'); empty or 'auto' -> None (detect)."""
    lang = (language or "").strip()
    if not lang or lang.lower() == "auto":
        return None
    return lang.replace("_", "-").split("-")[0].lower()


def build_stt(config: Any) -> STTProvider:
    """``config`` is the root SentientConfig."""
    v = config.voice
    language = stt_language(v.stt_language or config.assistant.language)
    if v.stt_provider == "faster_whisper":
        return FasterWhisperSTT(v.stt_model, v.stt_device, language)
    if v.stt_provider == "openai":
        pc = config.models.providers.get("openai")
        return OpenAISTT(v.stt_model, language, pc.api_base if pc is not None else None)
    if v.stt_provider == "deepgram":
        return DeepgramSTT(v.stt_model, language)
    if v.stt_provider == "elevenlabs":
        return ElevenLabsSTT(v.stt_model, language)
    raise VoiceError(f"unknown stt provider {v.stt_provider}")
