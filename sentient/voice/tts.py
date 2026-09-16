"""Text-to-speech providers: OS voices, Kokoro (local neural), OpenAI and ElevenLabs.

Every provider returns one complete PCM16 mono WAV per call.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from sentient.voice.audio import (
    float_to_wav,
    pcm16_to_float,
    pcm16_to_wav,
    trim_silence,
    wav_to_pcm16,
)
from sentient.voice.base import (
    TTSProvider,
    VoiceError,
    download_file,
    has_key,
    models_dir,
    raise_for_status,
    require_key,
)

log = logging.getLogger(__name__)


def _tidy_wav(data: bytes) -> bytes:
    """Normalize to PCM16 mono and trim the OS engines' leading/trailing silence."""
    pcm, sr = wav_to_pcm16(data)
    audio = trim_silence(pcm16_to_float(pcm), sr)
    if audio.size == 0:
        return pcm16_to_wav(pcm, sr)
    return float_to_wav(audio, sr)


# ----------------------------------------------------------------------------- system (OS voices)
class _SpeechThread:
    """One long-lived thread that owns the OS speech engine.

    pyttsx3 is not reentrant (``runAndWait`` from two threads, or while a loop is
    already running, raises "run loop already started") and SAPI5 needs COM
    initialised on the thread that uses it, so every job goes through this
    single thread's queue.
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="sentient-tts-system", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if sys.platform == "win32":
            try:
                import pythoncom  # type: ignore[import-not-found]

                pythoncom.CoInitialize()
            except Exception:
                with contextlib.suppress(Exception):
                    import ctypes

                    ctypes.windll.ole32.CoInitialize(None)  # type: ignore[attr-defined]
        while True:
            item = self._q.get()
            if item is None:
                return
            fn, args, fut = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn(*args))
            except BaseException as exc:
                fut.set_exception(exc)

    async def submit(self, fn: Callable, *args: Any) -> Any:
        fut: Future = Future()
        self._q.put((fn, args, fut))
        return await asyncio.wrap_future(fut)

    def close(self) -> None:
        self._q.put(None)


def _match_voice(voices: list[Any], wanted: str) -> Any | None:
    w = wanted.strip().lower()
    if not w:
        return None
    for v in voices:
        if str(v.id).lower() == w or str(v.name).lower() == w:
            return v
    for v in voices:
        if w in str(v.name).lower() or w in str(v.id).lower():
            return v
    return None


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class SystemTTS(TTSProvider):
    """Zero-download OS voices.

    pyttsx3 (SAPI5 / NSSpeechSynthesizer / eSpeak) first; if it fails once, the
    provider switches to PowerShell ``System.Speech`` on Windows, ``say`` on macOS
    or espeak-ng on Linux for the rest of its life.
    """

    name = "system"

    def __init__(self, voice: str = "", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self.error = None
        self.backend: str | None = None
        self._thread: _SpeechThread | None = None
        self._base_rate: int | None = None
        self._default_voice_id: str | None = None
        self._pyttsx3_failed = False

    def _worker(self) -> _SpeechThread:
        if self._thread is None:
            self._thread = _SpeechThread()
        return self._thread

    @staticmethod
    def _tmp_wav() -> str:
        fd, path = tempfile.mkstemp(prefix="sentient-tts-", suffix=".wav")
        os.close(fd)
        return path

    # ---------------------------------------------------------------- pyttsx3 (speech thread only)
    def _engine(self) -> Any:
        import pyttsx3

        engine = pyttsx3.init()  # cached per driver; properties persist, so always set them
        if self._base_rate is None:
            self._base_rate = int(engine.getProperty("rate") or 200)
            self._default_voice_id = engine.getProperty("voice")
        return engine

    def _pyttsx3_save(self, text: str, voice: str, path: str) -> None:
        engine = self._engine()
        try:
            match = _match_voice(engine.getProperty("voices") or [], voice) if voice else None
            voice_id = match.id if match is not None else self._default_voice_id
            if voice_id:
                engine.setProperty("voice", voice_id)
            engine.setProperty("rate", int((self._base_rate or 200) * self.speed))
            engine.save_to_file(text, path)
            engine.runAndWait()
        finally:
            with contextlib.suppress(Exception):
                engine.stop()

    def _fallback_save(self, text: str, voice: str, path: str) -> str:
        if sys.platform == "win32":
            rate = max(-10, min(10, round((self.speed - 1.0) * 10)))
            select = f"try{{$s.SelectVoice({_ps_quote(voice)})}}catch{{}};" if voice else ""
            script = (
                "[Console]::InputEncoding=[Text.Encoding]::UTF8;"
                "Add-Type -AssemblyName System.Speech;"
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                f"{select}$s.Rate={rate};"
                f"$s.SetOutputToWaveFile({_ps_quote(path)});"
                "$s.Speak([Console]::In.ReadToEnd());$s.Dispose()"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=120,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return "powershell"
        if sys.platform == "darwin" and shutil.which("say"):
            cmd = ["say", "-o", path, "--file-format=WAVE", "--data-format=LEI16@22050"]
            cmd += ["-r", str(int(180 * self.speed))]
            if voice:
                cmd += ["-v", voice]
            subprocess.run([*cmd, text], capture_output=True, timeout=120, check=True)
            return "say"
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if exe:
            cmd = [exe, "-w", path, "-s", str(int(175 * self.speed))]
            if voice:
                cmd += ["-v", voice]
            subprocess.run([*cmd, text], capture_output=True, timeout=120, check=True)
            return "espeak"
        raise VoiceError("no system speech engine available")

    def _synth_job(self, text: str, voice: str) -> bytes:
        path = self._tmp_wav()
        try:
            if not self._pyttsx3_failed:
                try:
                    self._pyttsx3_save(text, voice, path)
                except Exception as exc:
                    log.warning("pyttsx3 failed (%s); using the fallback system speech backend", exc)
                    self._pyttsx3_failed = True
                else:
                    if os.path.getsize(path) <= 44:
                        log.warning("pyttsx3 produced an empty file; using the fallback system speech backend")
                        self._pyttsx3_failed = True
                    else:
                        self.backend = "pyttsx3"
            if self._pyttsx3_failed:
                self.backend = self._fallback_save(text, voice, path)
            with open(path, "rb") as fh:
                data = fh.read()
            if len(data) <= 44:
                raise VoiceError("system speech engine produced no audio")
            return _tidy_wav(data)
        finally:
            with contextlib.suppress(OSError):
                os.remove(path)

    def _voices_job(self) -> list[dict[str, Any]]:
        try:
            engine = self._engine()
            out = []
            for v in engine.getProperty("voices") or []:
                langs = [
                    x.decode(errors="ignore") if isinstance(x, bytes) else str(x)
                    for x in (getattr(v, "languages", None) or [])
                ]
                out.append({"id": str(v.id), "name": str(v.name), "language": langs[0] if langs else ""})
            return out
        except Exception as exc:
            log.debug("pyttsx3 voice listing failed: %s", exc)
            return []

    # ---------------------------------------------------------------- public
    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        if not text.strip():
            raise VoiceError("nothing to say")
        chosen = voice if voice is not None else self.voice
        try:
            data = await self._worker().submit(self._synth_job, text, chosen)
        except VoiceError as exc:
            self.error = str(exc)
            raise
        except Exception as exc:
            self.error = f"system speech failed: {exc}"
            raise VoiceError(self.error) from exc
        self.error = None
        return data

    def _warm_job(self) -> None:
        if not self._pyttsx3_failed:
            self._engine()  # SAPI5/COM start-up takes ~0.5-1 s the first time

    async def prepare(self) -> AsyncIterator[dict[str, Any]]:
        try:
            await self._worker().submit(self._warm_job)
        except Exception as exc:  # synthesis falls back on its own; warming is best effort
            log.debug("system speech warm-up failed: %s", exc)
        yield {"stage": "ready", "component": "tts", "progress": 1.0}

    async def voices(self) -> list[dict[str, Any]]:
        return await self._worker().submit(self._voices_job)

    async def status(self) -> dict[str, Any]:
        out = await super().status()
        out["backend"] = self.backend
        return out

    def close(self) -> None:
        if self._thread is not None:
            self._thread.close()
            self._thread = None


# ----------------------------------------------------------------------------- kokoro (local neural)
KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_MODELS = {
    "fp32": "kokoro-v1.0.onnx",
    "fp16": "kokoro-v1.0.fp16.onnx",
    "int8": "kokoro-v1.0.int8.onnx",
}
KOKORO_VOICES_FILE = "voices-v1.0.bin"
KOKORO_DEFAULT_VOICE = "af_heart"
# first letter of a voice id is its language
KOKORO_LANG = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br", "j": "ja", "z": "cmn"}
KOKORO_LANG_NAME = {"a": "en-US", "b": "en-GB", "e": "es", "f": "fr-FR", "h": "hi", "i": "it", "p": "pt-BR", "j": "ja", "z": "zh"}
KOKORO_KNOWN_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova", "af_river", "af_alloy",
    "af_aoede", "af_jessica", "af_kore", "am_adam", "am_michael", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_onyx", "am_puck", "bf_emma", "bf_isabella", "bf_alice", "bf_lily", "bm_george",
    "bm_lewis", "bm_daniel", "bm_fable", "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
]  # fmt: skip


def _kokoro_voice_entry(vid: str) -> dict[str, str]:
    gender = {"f": "female", "m": "male"}.get(vid[1:2], "")
    pretty = vid.split("_", 1)[-1].capitalize()
    return {
        "id": vid,
        "name": f"{pretty} ({gender})" if gender else pretty,
        "language": KOKORO_LANG_NAME.get(vid[:1], ""),
    }


class KokoroTTS(TTSProvider):
    name = "kokoro"

    def __init__(self, voice: str = "", speed: float = 1.0, variant: str = "fp32", root: Path | None = None):
        self.voice = voice
        self.speed = speed
        # fp32 synthesizes ~4x faster than int8 on CPU onnxruntime (RTF 1.0 vs 3.9 on an i7 laptop)
        self.variant = variant if variant in KOKORO_MODELS else "fp32"
        self.root = root
        self.error = None
        self._kokoro: Any = None
        self._lock = threading.Lock()
        self._prepare_lock: asyncio.Lock | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentient-tts-kokoro")

    def _dir(self) -> Path:
        return self.root or models_dir("kokoro")

    @property
    def model_path(self) -> Path:
        return self._dir() / KOKORO_MODELS[self.variant]

    @property
    def voices_path(self) -> Path:
        return self._dir() / KOKORO_VOICES_FILE

    @property
    def downloaded(self) -> bool:
        return self.model_path.is_file() and self.voices_path.is_file()

    @property
    def ready(self) -> bool:
        return self._kokoro is not None

    def _load(self) -> Any:
        with self._lock:
            if self._kokoro is None:
                from kokoro_onnx import Kokoro

                started = time.perf_counter()
                kokoro = Kokoro(str(self.model_path), str(self.voices_path))
                try:  # onnxruntime's first run is slow; pay it while loading, not on the first reply
                    kokoro.create("Hi.", voice=self._resolve_voice(kokoro, self.voice), speed=1.0, lang="en-us")
                except Exception as exc:
                    log.debug("kokoro warm-up failed: %s", exc)
                self._kokoro = kokoro
                log.info("kokoro %s loaded in %.1fs", self.variant, time.perf_counter() - started)
            return self._kokoro

    @staticmethod
    def _resolve_voice(k: Any, voice: str) -> str:
        wanted = (voice or "").strip()
        available = set(k.get_voices())
        if wanted in available:
            return wanted
        return KOKORO_DEFAULT_VOICE if KOKORO_DEFAULT_VOICE in available else sorted(available)[0]

    def _synth(self, text: str, voice: str) -> bytes:
        k = self._load()
        vid = self._resolve_voice(k, voice)
        lang = KOKORO_LANG.get(vid[:1], "en-us")
        audio, sr = k.create(text, voice=vid, speed=float(self.speed), lang=lang)
        return float_to_wav(audio, sr)

    async def prepare(self) -> AsyncIterator[dict[str, Any]]:
        if self._prepare_lock is None:
            self._prepare_lock = asyncio.Lock()
        async with self._prepare_lock:
            if self.ready:
                yield {"stage": "ready", "component": "tts", "progress": 1.0}
                return
            try:
                files = ((KOKORO_MODELS[self.variant], self.model_path), (KOKORO_VOICES_FILE, self.voices_path))
                for fname, dest in files:
                    if dest.is_file():
                        continue
                    last = 0.0
                    async for done, total in download_file(KOKORO_RELEASE + fname, dest):
                        now = time.monotonic()
                        if now - last >= 0.25 and not (total and done >= total):
                            last = now
                            yield {
                                "stage": "download", "component": "tts", "file": fname,
                                "progress": round(done / total, 4) if total else None,
                                "bytes": done, "total": total,
                            }
                    yield {"stage": "download", "component": "tts", "file": fname, "progress": 1.0}
                yield {"stage": "loading", "component": "tts", "progress": None}
                await asyncio.get_running_loop().run_in_executor(self._executor, self._load)
            except Exception as exc:
                self.error = f"Kokoro setup failed: {exc}"
                yield {"stage": "error", "component": "tts", "message": self.error}
                return
            self.error = None
            yield {"stage": "ready", "component": "tts", "progress": 1.0}

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        if not text.strip():
            raise VoiceError("nothing to say")
        if not self.ready:
            async for step in self.prepare():  # first use downloads and loads on demand
                if step["stage"] == "error":
                    raise VoiceError(step["message"])
        chosen = voice if voice is not None else self.voice
        try:
            return await asyncio.get_running_loop().run_in_executor(self._executor, self._synth, text, chosen)
        except Exception as exc:
            self.error = f"Kokoro synthesis failed: {exc}"
            raise VoiceError(self.error) from exc

    async def voices(self) -> list[dict[str, Any]]:
        ids = list(self._kokoro.get_voices()) if self._kokoro is not None else KOKORO_KNOWN_VOICES
        return [_kokoro_voice_entry(v) for v in ids]

    async def status(self) -> dict[str, Any]:
        out = await super().status()
        out["downloaded"] = self.downloaded
        out["variant"] = self.variant
        return out

    def close(self) -> None:
        self._kokoro = None
        self._executor.shutdown(wait=False, cancel_futures=True)


# ----------------------------------------------------------------------------- cloud
OPENAI_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]
ELEVENLABS_DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel
ELEVENLABS_PREMADE = {
    "21m00Tcm4TlvDq8ikWAM": "Rachel",
    "EXAVITQu4vr4xnSDxMaL": "Sarah",
    "ErXwobaYiN019PkySvjV": "Antoni",
    "TxGEqnHWrfWFTfGW9XjX": "Josh",
    "pNInz6obpgDQGcFmaJgB": "Adam",
    "XB0fDUnXU5powFXDhCwa": "Charlotte",
    "onwK4e9ZLuTAKqWW03F9": "Daniel",
    "pFZP5JQG7iQjIQuC4Bku": "Lily",
}


class OpenAITTS(TTSProvider):
    name = "openai"
    SAMPLE_RATE = 24000  # response_format=pcm is raw 24 kHz 16-bit mono

    def __init__(self, voice: str = "", speed: float = 1.0, model: str = "", api_base: str | None = None):
        self.voice = voice
        self.speed = speed
        self.model = model or "gpt-4o-mini-tts"
        self.api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self.error = None

    @property
    def ready(self) -> bool:
        return has_key("openai", "OPENAI_API_KEY")

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        key = require_key("openai", "OPENAI_API_KEY", "OpenAI")
        v = (voice if voice is not None else self.voice) or "alloy"
        if v not in OPENAI_VOICES:
            v = "alloy"
        body = {
            "model": self.model,
            "input": text,
            "voice": v,
            "response_format": "pcm",
            "speed": max(0.25, min(4.0, float(self.speed))),
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.api_base}/audio/speech", headers={"Authorization": f"Bearer {key}"}, json=body
            )
        raise_for_status(r, "OpenAI")
        return pcm16_to_wav(r.content, self.SAMPLE_RATE)

    async def voices(self) -> list[dict[str, Any]]:
        return [{"id": v, "name": v.capitalize(), "language": ""} for v in OPENAI_VOICES]


class ElevenLabsTTS(TTSProvider):
    name = "elevenlabs"
    SAMPLE_RATE = 22050

    def __init__(self, voice: str = "", speed: float = 1.0, model: str = "", api_base: str | None = None):
        self.voice = voice
        self.speed = speed
        self.model = model or "eleven_flash_v2_5"
        self.api_base = (api_base or "https://api.elevenlabs.io/v1").rstrip("/")
        self.error = None
        self._voices_cache: list[dict[str, Any]] | None = None

    @property
    def ready(self) -> bool:
        return has_key("elevenlabs", "ELEVENLABS_API_KEY")

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        key = require_key("elevenlabs", "ELEVENLABS_API_KEY", "ElevenLabs")
        vid = (voice if voice is not None else self.voice) or ELEVENLABS_DEFAULT_VOICE
        body = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": max(0.7, min(1.2, float(self.speed))),
            },
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.api_base}/text-to-speech/{vid}",
                params={"output_format": f"pcm_{self.SAMPLE_RATE}"},
                headers={"xi-api-key": key},
                json=body,
            )
        raise_for_status(r, "ElevenLabs")
        return pcm16_to_wav(r.content, self.SAMPLE_RATE)

    async def voices(self) -> list[dict[str, Any]]:
        if self._voices_cache is not None:
            return self._voices_cache
        premade = [{"id": k, "name": v, "language": ""} for k, v in ELEVENLABS_PREMADE.items()]
        if not self.ready:
            return premade
        try:
            key = require_key("elevenlabs", "ELEVENLABS_API_KEY", "ElevenLabs")
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.api_base}/voices", headers={"xi-api-key": key})
            raise_for_status(r, "ElevenLabs")
            self._voices_cache = [
                {
                    "id": v["voice_id"],
                    "name": v.get("name", v["voice_id"]),
                    "language": (v.get("labels") or {}).get("language", ""),
                }
                for v in r.json().get("voices", [])
            ]
        except Exception as exc:
            log.debug("elevenlabs voice listing failed: %s", exc)
            return premade
        return self._voices_cache


def build_tts(config: Any) -> TTSProvider:
    """``config`` is the root SentientConfig."""
    v = config.voice
    if v.tts_provider == "system":
        return SystemTTS(v.tts_voice, v.tts_speed)
    if v.tts_provider == "kokoro":
        return KokoroTTS(v.tts_voice, v.tts_speed, v.kokoro_variant)
    if v.tts_provider == "openai":
        pc = config.models.providers.get("openai")
        return OpenAITTS(v.tts_voice, v.tts_speed, v.tts_model, pc.api_base if pc is not None else None)
    if v.tts_provider == "elevenlabs":
        return ElevenLabsTTS(v.tts_voice, v.tts_speed, v.tts_model)
    raise VoiceError(f"unknown tts provider {v.tts_provider}")
