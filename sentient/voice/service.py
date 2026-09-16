"""Voice service: STT/TTS providers plus the live voice conversations.

Providers are built lazily from ``config.voice`` on first use and rebuilt when
the relevant settings change (``config.updated`` on the bus, or detected on the
next access). Nothing heavy loads at app start unless ``voice.preload_on_start``
is on: faster-whisper and Kokoro load on the first transcription / synthesis,
when a voice session opens (background warm-up), or when the UI calls ``prepare``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sentient.services import Service
from sentient.voice.audio import MAX_AUDIO_BYTES, audio_suffix
from sentient.voice.base import STTProvider, TTSProvider, VoiceError
from sentient.voice.session import SendBytes, SendJSON, VoiceSession
from sentient.voice.stt import FasterWhisperSTT, build_stt, stt_language
from sentient.voice.text import clean_for_speech, is_speakable
from sentient.voice.tts import build_tts
from sentient.voice.vad import EnergyVAD
from sentient.voice.wake import OpenWakeWordDetector, WakeDetector, WhisperWakeDetector

if TYPE_CHECKING:  # pragma: no cover
    from sentient.config.schema import VoiceConfig

log = logging.getLogger(__name__)


class VoiceService(Service):
    name = "voice"

    def __init__(self, app):
        super().__init__(app)
        self._stt: STTProvider | None = None
        self._tts: TTSProvider | None = None
        self._stt_sig: tuple | None = None
        self._tts_sig: tuple | None = None
        self._stt_override: STTProvider | None = None
        self._tts_override: TTSProvider | None = None
        self._wake_stt: STTProvider | None = None
        self._wake_stt_sig: tuple | None = None
        self._wake_factory: Callable[[], WakeDetector] | None = None
        self._warm_task: asyncio.Task | None = None
        self._warmed: tuple[int, int] | None = None
        self.sessions: set[VoiceSession] = set()
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ config / providers
    @property
    def config(self) -> VoiceConfig:
        return self.app.config.voice

    def use_providers(self, *, stt: STTProvider | None = None, tts: TTSProvider | None = None) -> None:
        """Inject providers (tests, embedders). They win over config until cleared with None."""
        self._stt_override = stt
        self._tts_override = tts

    def use_wake_detector(self, factory: Callable[[], WakeDetector] | None) -> None:
        """Inject a wake detector factory (one detector per session). None restores the configured engine."""
        self._wake_factory = factory

    def _stt_signature(self) -> tuple:
        cfg = self.app.config
        v = cfg.voice
        openai = cfg.models.providers.get("openai")
        return (v.stt_provider, v.stt_model, v.stt_device, v.stt_language or cfg.assistant.language,
                openai.api_base if openai else None)

    def _tts_signature(self) -> tuple:
        cfg = self.app.config
        v = cfg.voice
        openai = cfg.models.providers.get("openai")
        return (v.tts_provider, v.tts_model, v.kokoro_variant, openai.api_base if openai else None)

    def _wake_stt_signature(self) -> tuple:
        cfg = self.app.config
        return (cfg.voice.wake_whisper_model, cfg.voice.stt_language or cfg.assistant.language)

    @property
    def stt(self) -> STTProvider:
        if self._stt_override is not None:
            return self._stt_override
        sig = self._stt_signature()
        if self._stt is None or sig != self._stt_sig:
            if self._stt is not None:
                self._stt.close()
            self._stt, self._stt_sig = build_stt(self.app.config), sig
        return self._stt

    @property
    def tts(self) -> TTSProvider:
        if self._tts_override is not None:
            return self._tts_override
        sig = self._tts_signature()
        if self._tts is None or sig != self._tts_sig:
            if self._tts is not None:
                self._tts.close()
            self._tts, self._tts_sig = build_tts(self.app.config), sig
        # voice and speed are cheap to change in place (no model reload)
        self._tts.voice = self.config.tts_voice
        self._tts.speed = self.config.tts_speed
        return self._tts

    @property
    def wake_stt(self) -> STTProvider:
        """The small local whisper model the ``whisper`` wake engine listens with (always on the CPU,
        never a cloud STT: ambient speech does not leave the machine)."""
        v = self.config
        if (
            self._stt_override is None
            and v.stt_provider == "faster_whisper"
            and v.stt_model == v.wake_whisper_model
            and v.stt_device in ("auto", "cpu")
        ):
            return self.stt  # same model already used for utterances: load it once
        sig = self._wake_stt_signature()
        if self._wake_stt is None or sig != self._wake_stt_sig:
            if self._wake_stt is not None:
                self._wake_stt.close()
            language = stt_language(v.stt_language or self.app.config.assistant.language)
            self._wake_stt, self._wake_stt_sig = FasterWhisperSTT(v.wake_whisper_model, "cpu", language), sig
        return self._wake_stt

    def make_wake_detector(self) -> WakeDetector:
        if self._wake_factory is not None:
            return self._wake_factory()
        v = self.config
        if v.wake_engine == "openwakeword":
            return OpenWakeWordDetector(v.wake_word, v.wake_sensitivity, v.wake_model)
        return WhisperWakeDetector(self.wake_stt, v.wake_word, v.wake_sensitivity)

    def refresh(self) -> None:
        """Drop providers whose settings changed so their models free memory now."""
        if self._stt is not None and self._stt_signature() != self._stt_sig:
            self._stt.close()
            self._stt = self._stt_sig = None
        if self._tts is not None and self._tts_signature() != self._tts_sig:
            self._tts.close()
            self._tts = self._tts_sig = None
        if self._wake_stt is not None and self._wake_stt_signature() != self._wake_stt_sig:
            self._wake_stt.close()
            self._wake_stt = self._wake_stt_sig = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self._loops.append(asyncio.create_task(self._watch_config(), name="voice:config"))
        if self.config.preload_on_start and self.app.enable_background:
            self.warm(force=True)

    async def _watch_config(self) -> None:
        async with self.app.bus.subscribe() as q:
            while True:
                event = await q.get()
                if event.get("type") == "config.updated":
                    try:
                        self.refresh()
                    except Exception:
                        log.exception("voice: provider refresh failed")

    async def stop(self) -> None:
        for session in list(self.sessions):
            with contextlib.suppress(Exception):
                await session.close()
        self.sessions.clear()
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        await super().stop()
        for provider in (self._stt, self._tts, self._wake_stt):
            if provider is not None:
                with contextlib.suppress(Exception):
                    provider.close()
        self._stt = self._tts = self._wake_stt = None
        self._stt_sig = self._tts_sig = self._wake_stt_sig = None

    def track(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def warm(self, *, force: bool = False) -> None:
        """Load STT and TTS in the background so the first utterance does not pay for it.

        Called when a voice session opens (and at startup with ``preload_on_start``). Skipped for
        apps without background work (tests) unless forced; idempotent per provider pair.
        """
        if not (force or self.app.enable_background):
            return
        if self._warm_task is not None and not self._warm_task.done():
            return
        try:
            key = (id(self.stt), id(self.tts))
        except Exception as exc:
            log.debug("voice warm-up skipped: %s", exc)
            return
        if key == self._warmed:
            return
        self._warmed = key
        self._warm_task = asyncio.get_running_loop().create_task(self._warm(), name="voice:warm")
        self.track(self._warm_task)

    async def _warm(self) -> None:
        async for step in self.prepare("all"):
            if step.get("stage") == "error":
                log.warning("voice warm-up: %s %s", step.get("component"), step.get("message"))

    # ------------------------------------------------------------------ public API
    def wake_status(self) -> dict[str, Any]:
        v = self.config
        try:
            detector = self.make_wake_detector()
            try:
                return detector.status()
            finally:
                detector.close()
        except Exception as exc:
            return {"engine": v.wake_engine, "phrase": v.wake_word, "ready": False, "error": str(exc)}

    async def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sessions": len(self.sessions)}
        try:
            out["stt"] = self.stt.status()
        except Exception as exc:
            out["stt"] = {"provider": self.config.stt_provider, "model": self.config.stt_model,
                          "ready": False, "device": None, "error": str(exc)}
        try:
            out["tts"] = await self.tts.status()
        except Exception as exc:
            out["tts"] = {"provider": self.config.tts_provider, "voice": self.config.tts_voice,
                          "ready": False, "voices": [], "error": str(exc)}
        out["wake"] = self.wake_status()
        return out

    async def prepare(self, target: str = "all") -> AsyncIterator[dict[str, Any]]:
        """Download and load local models, yielding progress dicts ``{stage, component, progress, ...}``.

        ``all`` covers STT and TTS; the wake engine is only prepared when asked for (``wake``).
        """
        ok = True
        names = {"all": ["stt", "tts"], "stt": ["stt"], "tts": ["tts"], "wake": ["wake"]}.get(target, [])
        for name in names:
            detector: WakeDetector | None = None
            try:
                if name == "wake":
                    detector = self.make_wake_detector()
                    steps = detector.prepare_steps()
                else:
                    steps = (self.stt if name == "stt" else self.tts).prepare()
                async for step in steps:
                    ok = ok and step.get("stage") != "error"
                    yield step
            except Exception as exc:
                ok = False
                yield {"stage": "error", "component": name, "message": str(exc)}
            finally:
                if detector is not None:
                    detector.close()
        yield {"stage": "done", "progress": 1.0, "ok": ok}

    async def transcribe_pcm(self, pcm16: bytes, sample_rate: int) -> str:
        try:
            return await self.stt.transcribe(pcm16, sample_rate)
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError(f"transcription failed: {exc}") from exc

    async def transcribe_file(self, path: str | Path) -> str:
        try:
            return await self.stt.transcribe_file(path)
        except VoiceError:
            raise
        except Exception as exc:
            if type(exc).__module__.split(".")[0] == "av":  # PyAV could not read the container
                raise VoiceError(f"could not decode the audio file: {exc}") from exc
            raise VoiceError(f"transcription failed: {exc}") from exc

    async def transcribe_bytes(self, data: bytes, filename: str = "") -> str:
        """Transcribe a complete audio file held in memory (ogg/opus voice notes, webm, m4a, mp3, wav, flac).

        The container is taken from ``filename``'s extension, or sniffed from the bytes. Raises
        ``VoiceError`` with a message that can be shown to the user.
        """
        if not data:
            raise VoiceError("the audio file is empty")
        if len(data) > MAX_AUDIO_BYTES:
            raise VoiceError("the audio file is too large (max 50 MB)")
        suffix = audio_suffix(filename, data)
        if not suffix:
            raise VoiceError("unsupported audio type; send ogg/opus, webm, m4a, mp3, wav or flac")
        fd, tmp = tempfile.mkstemp(prefix="sentient-audio-", suffix=suffix)
        os.close(fd)
        try:
            await asyncio.to_thread(Path(tmp).write_bytes, data)
            return await self.transcribe_file(tmp)
        finally:
            with contextlib.suppress(OSError):
                os.remove(tmp)

    async def synthesize(self, sentence: str, voice: str | None = None) -> bytes:
        """Synthesize text that is already speech-clean (one sentence from the splitter)."""
        try:
            return await self.tts.synthesize(sentence, voice)
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError(f"speech synthesis failed: {exc}") from exc

    async def speak(self, text: str, voice: str | None = None) -> bytes:
        """Markdown in, WAV out (the REST ``/api/voice/speak`` path)."""
        spoken = clean_for_speech(text)
        if not is_speakable(spoken):
            raise VoiceError("nothing to say")
        return await self.synthesize(spoken, voice or None)

    def make_vad(self, sample_rate: int) -> EnergyVAD:
        v = self.config
        return EnergyVAD(
            sample_rate,
            silence_ms=v.vad_silence_ms,
            min_speech_ms=v.vad_min_speech_ms,
            max_utterance_ms=int(v.vad_max_utterance_s * 1000),
        )

    def open_session(
        self, send_json: SendJSON, send_bytes: SendBytes, *, node: dict[str, Any] | None = None
    ) -> VoiceSession:
        session = VoiceSession(self, send_json, send_bytes, node=node)
        self.sessions.add(session)
        return session

    async def close_session(self, session: VoiceSession, *, cancel_turn: bool = True) -> None:
        try:
            await session.close(cancel_turn=cancel_turn)
        finally:
            self.sessions.discard(session)

    def publish_state(self, session_id: str | None, state: str) -> None:
        self.app.bus.publish("voice.state", {"state": state, "session_id": session_id})
