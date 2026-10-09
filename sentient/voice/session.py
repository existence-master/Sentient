"""One live voice conversation (the logic behind ``WS /ws/voice``).

Transport-agnostic: the gateway hands in two coroutines, ``send_json`` and
``send_bytes``, and feeds client messages to ``handle_json`` / ``handle_audio``.

Pipeline per utterance::

    PCM frames -> EnergyVAD -> utterance -> STT (thread) -> transcript
      -> Agent.run_turn(channel="voice"|"glasses") -> every chat event forwarded
      -> text_delta -> SentenceSplitter (first clause early) -> TTS per chunk (thread)
      -> audio header + WAV frame (or raw PCM16 frames for devices)
      -> audio_end -> listening

Wake mode (``start.mode: "wake"``)::

    standby --wake word--> wake (+ earcon) -> listening -> ... reply ...
      -> listening for voice.follow_up_seconds (after estimated playback) -> standby

Interruptions:
- ``interrupt`` (or server-detected barge-in while speaking) stops synthesis and
  sending at once; the model may keep writing text, which is still forwarded
  and persisted.
- A new utterance while a turn is still running cancels that turn (``done`` with
  ``cancelled: true``) and starts a new one. An utterance that arrives while the
  previous one is still being transcribed is merged with it.
- While an approval is pending, a spoken "yes" / "no" answers it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from sentient.voice.audio import pcm16_resample, wav_duration_s, wav_to_pcm16
from sentient.voice.base import VoiceError
from sentient.voice.text import SentenceSplitter
from sentient.voice.wake import WakeDetector, WakeHit, earcon_wav, has_words, match_wake_phrase

if TYPE_CHECKING:  # pragma: no cover
    from sentient.voice.service import VoiceService
    from sentient.voice.vad import EnergyVAD

log = logging.getLogger(__name__)

SendJSON = Callable[[dict[str, Any]], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]

BARGE_IN_MS = 300  # sustained voiced audio needed to cut the assistant off
STATES = ("idle", "standby", "listening", "transcribing", "thinking", "speaking")
SPOKEN_CHANNELS = ("voice", "glasses", "phone")
# Channel name handed to Agent.run_turn: it picks the voice role for "voice" and "glasses".
# TODO(core): treat "phone" as a spoken channel in Agent.run_turn / prompt guidance; until then
# phone sessions are stored with channel "phone" but run the agent as "voice".
AGENT_CHANNEL = {"voice": "voice", "glasses": "glasses", "phone": "voice"}
AUDIO_FORMATS = ("wav", "pcm16")
MODES = ("conversation", "wake")

_YES = re.compile(
    r"^(yes|yeah|yep|yup|sure|ok|okay|go ahead|do it|allow( it)?|approve[d]?|confirm(ed)?|please do|affirmative)\b"
)
_NO = re.compile(r"^(no|nope|nah|don'?t|do not|stop|cancel|deny|denied|never ?mind|negative)\b")


def parse_decision(text: str) -> str | None:
    t = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if _NO.match(t):
        return "deny"
    if _YES.match(t):
        return "allow"
    return None


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class VoiceSession:
    def __init__(
        self,
        service: VoiceService,
        send_json: SendJSON,
        send_bytes: SendBytes,
        *,
        node: dict[str, Any] | None = None,
    ):
        self.service = service
        self.app = service.app
        self._send_json = send_json
        self._send_bytes = send_bytes
        self._send_lock = asyncio.Lock()
        self.node = node  # set when a device authenticated with node_token
        self.session_id: str | None = None
        self.sample_rate = 16000
        self.channel = "voice"
        self.mode = "conversation"
        self.audio_format = "wav"
        self.output_sample_rate = 16000
        self.max_frame_bytes = 0
        self.state = "idle"
        self.vad: EnergyVAD | None = None
        self.wake: WakeDetector | None = None
        self.closed = False
        self.metrics: dict[str, Any] = {}
        self._turn: asyncio.Task | None = None
        self._speaker: asyncio.Task | None = None
        self._generating = False
        self._muted = False
        self._audio_started = False
        self._audio_end_sent = False
        self._sentence_index = 0
        self._transcribing_pcm: bytes | None = None
        self._approvals: dict[str, str] = {}  # call_id -> approval_id
        self._side: set[asyncio.Task] = set()
        self._warned_not_started = False
        self._wake_lock = asyncio.Lock()
        self._follow_up: asyncio.Task | None = None
        self._playback_until = 0.0  # monotonic time the client should finish playing what was sent

    # ------------------------------------------------------------------ sending
    async def send(self, obj: dict[str, Any]) -> None:
        if self.closed:
            return
        async with self._send_lock:
            try:
                await self._send_json(obj)
            except Exception as exc:  # socket went away
                log.debug("voice send failed: %s", exc)
                self.closed = True

    async def _send_frames(self, header: dict[str, Any], frames: list[bytes]) -> None:
        if self.closed:
            return
        async with self._send_lock:  # the header and its binary frames must stay adjacent
            try:
                await self._send_json(header)
                for frame in frames:
                    await self._send_bytes(frame)
            except Exception as exc:
                log.debug("voice audio send failed: %s", exc)
                self.closed = True

    def _wav_to_output(self, wav: bytes) -> bytes:
        pcm, sr = wav_to_pcm16(wav)
        return pcm16_resample(pcm, sr, self.output_sample_rate)

    async def _deliver_audio(self, header: dict[str, Any], wav: bytes) -> None:
        """Send one synthesized chunk as a WAV frame, or as raw PCM16 frames for devices."""
        if self.audio_format == "pcm16":
            pcm = await asyncio.to_thread(self._wav_to_output, wav)
            step = self.max_frame_bytes or max(2, len(pcm))
            frames = [pcm[i : i + step] for i in range(0, len(pcm), step)] or [b""]
            header = {**header, "format": "pcm16", "sample_rate": self.output_sample_rate,
                      "bytes": len(pcm), "frames": len(frames)}  # fmt: skip
            duration = len(pcm) / 2 / self.output_sample_rate
        else:
            frames, duration = [wav], wav_duration_s(wav)
        self._playback_until = max(self._playback_until, time.monotonic()) + duration
        await self._send_frames(header, frames)

    async def set_state(self, state: str) -> None:
        if state == self.state or self.closed:  # a closed session's leftover turn must not flip idle back
            return
        self.state = state
        await self.send({"type": "state", "state": state, "session_id": self.session_id})
        self.service.publish_state(self.session_id, state)
        if self.mode == "wake":
            if state == "listening":
                self._arm_follow_up()
            else:
                self._cancel_follow_up()

    # ------------------------------------------------------------------ client messages
    async def handle_json(self, msg: dict[str, Any]) -> str | None:
        kind = msg.get("type")
        if kind == "start":
            await self.start(msg)
        elif kind == "end_utterance":
            await self.end_utterance()
        elif kind == "interrupt":
            await self.interrupt("client")
        elif kind == "approval.respond":
            await self._resolve_approval(str(msg.get("approval_id")), str(msg.get("decision", "deny")))
        elif kind == "text":
            if not await self._require_started():
                return None
            text = str(msg.get("text") or "").strip()
            if text:
                await self._begin(None, text=text)
        elif kind == "wake":
            if not await self._require_started():
                return None
            if self.state == "standby":  # a device button or UI tap instead of the phrase
                await self._on_wake(WakeHit(""), source="client")
        elif kind == "ping":
            await self.send({"type": "pong"})
        elif kind == "stop":
            await self.close()
            return "stop"
        else:
            await self.send({"type": "error", "message": f"unknown message type {kind}", "recoverable": True})
        return None

    async def _error(self, message: str, *, recoverable: bool = True) -> None:
        await self.send({"type": "error", "message": message, "recoverable": recoverable, "session_id": self.session_id})

    async def start(self, msg: dict[str, Any]) -> None:
        if self._turn is not None:
            await self._cancel_turn()
        self._cancel_follow_up()
        self._close_wake()
        try:
            rate = int(msg.get("sample_rate") or 16000)
        except (TypeError, ValueError):
            rate = 16000
        if not 8000 <= rate <= 96000:
            await self.send({"type": "error", "message": f"unsupported sample_rate {rate}", "recoverable": False})
            return
        self.sample_rate = rate
        cfg = self.service.config

        channel = str(msg.get("channel") or "").strip().lower()
        if channel not in SPOKEN_CHANNELS:
            kind = str((self.node or {}).get("kind") or "")
            channel = kind if kind in ("glasses", "phone") else "voice"
        fmt = str(msg.get("audio_format") or "wav").strip().lower()
        if fmt not in AUDIO_FORMATS:
            await self.send({"type": "error", "message": f"unsupported audio_format {fmt}; using wav", "recoverable": True})
            fmt = "wav"
        try:
            out_rate = int(msg.get("output_sample_rate") or rate)
        except (TypeError, ValueError):
            out_rate = rate
        try:
            max_frame = max(0, int(msg.get("max_frame_bytes") or 0))
        except (TypeError, ValueError):
            max_frame = 0
        mode = str(msg.get("mode") or "conversation").strip().lower()
        if mode not in MODES:
            await self.send({"type": "error", "message": f"unknown mode {mode}; using conversation", "recoverable": True})
            mode = "conversation"
        self.channel, self.audio_format, self.mode = channel, fmt, mode
        self.output_sample_rate = min(48000, max(8000, out_rate))
        self.max_frame_bytes = max_frame - max_frame % 2

        store = self.app.store
        sid = msg.get("session_id") or None
        if sid and await store.get_session(str(sid)) is None:
            await self.send({"type": "error", "message": f"unknown session {sid}; starting a new one", "recoverable": True})
            sid = None
        self.session_id = str(sid) if sid else await store.create_session(channel=self.channel)
        self.vad = self.service.make_vad(rate)
        ready: dict[str, Any] = {
            "type": "ready",
            "session_id": self.session_id,
            "stt": cfg.stt_provider,
            "tts": cfg.tts_provider,
            "sample_rate": rate,
            "channel": self.channel,
            "mode": self.mode,
            "audio_format": self.audio_format,
        }
        if self.audio_format == "pcm16":
            ready["output_sample_rate"] = self.output_sample_rate
        if self.node is not None:
            ready["node_id"] = self.node.get("node_id")
        wake_problem: str | None = None
        if self.mode == "wake":
            try:
                self.wake = self.service.make_wake_detector()
                ready["wake"] = self.wake.status()
                wake_problem = self.wake.error
            except Exception as exc:
                wake_problem = str(exc)
                ready["wake"] = {"engine": cfg.wake_engine, "phrase": cfg.wake_word, "ready": False, "error": wake_problem}
        await self.send(ready)
        self.service.warm()
        if self.mode != "wake":
            await self.set_state("listening")
            return
        await self.set_state("standby")
        if wake_problem:
            await self._error(f"wake word unavailable: {wake_problem}")
        elif self.wake is not None and not self.wake.ready:
            self._spawn(self._prepare_wake(self.wake))

    async def _prepare_wake(self, detector: WakeDetector) -> None:
        try:
            await detector.prepare()
        except Exception as exc:
            log.warning("wake detector failed to load: %s", exc)
            await self._error(f"wake word unavailable: {exc}")

    async def _require_started(self) -> bool:
        if self.vad is not None and self.session_id is not None:
            return True
        if not self._warned_not_started:
            self._warned_not_started = True
            await self.send({"type": "error", "message": "send {\"type\": \"start\"} first", "recoverable": True})
        return False

    async def handle_audio(self, pcm16: bytes) -> None:
        if not await self._require_started():
            return
        assert self.vad is not None
        events = self.vad.feed(pcm16)
        if self.state == "standby":
            await self._standby_audio(pcm16, events)
            return
        if (
            self.state == "speaking"
            and not self._muted
            and self.service.config.barge_in
            and self.vad.in_speech
            and self.vad.voiced_ms >= BARGE_IN_MS
        ):
            await self.interrupt("barge_in")
        for ev in events:
            if ev.kind == "utterance":
                await self._on_utterance(ev.pcm)

    async def end_utterance(self) -> None:
        """Push-to-talk release. Always processes what was said, even in standby."""
        if not await self._require_started():
            return
        assert self.vad is not None
        ev = self.vad.flush()
        if ev is None:
            if self.state != "standby" and (self._turn is None or self._turn.done()):
                await self.set_state("listening")
            return
        await self._on_utterance(ev.pcm)

    # ------------------------------------------------------------------ wake word
    async def _standby_audio(self, pcm16: bytes, events: list) -> None:
        detector = self.wake
        if detector is None or detector.error:
            return
        if detector.streaming:
            try:
                hit = await detector.feed(pcm16, self.sample_rate)
            except Exception as exc:
                log.warning("wake detector failed: %s", exc)
                return
            if hit is not None and self.state == "standby" and not self.closed:
                assert self.vad is not None
                self.vad.restart_segment()  # the rest of this breath becomes the utterance
                await self._on_wake(hit)
            return
        for ev in events:
            if ev.kind == "utterance":
                self._spawn(self._check_wake_segment(ev.pcm))

    async def _check_wake_segment(self, pcm: bytes) -> None:
        async with self._wake_lock:  # one detector transcription at a time, in order
            if self.closed or self.wake is None:
                return
            if self.state != "standby":
                # woken while this segment waited (e.g. "hey sentient" ... pause ... "what's the time")
                await self._on_utterance(pcm)
                return
            try:
                hit = await self.wake.check_segment(pcm, self.sample_rate)
            except VoiceError as exc:
                await self._error(f"wake word check failed: {exc}")
                return
            except Exception:
                log.exception("wake word check failed")
                return
            if hit is None or self.state != "standby" or self.closed:
                return
            await self._on_wake(hit)
            if hit.has_command:  # "hey sentient what's the weather" in one breath
                await self._begin(pcm, wake_hit=hit)

    async def _on_wake(self, hit: WakeHit, *, source: str = "voice") -> None:
        msg: dict[str, Any] = {"type": "wake", "phrase": hit.phrase, "source": source, "session_id": self.session_id}
        if hit.score is not None:
            msg["score"] = hit.score
        await self.send(msg)
        if self.service.config.wake_earcon:
            header = {"type": "audio", "format": "wav", "sentence_index": -1, "text": "", "earcon": True,
                      "session_id": self.session_id}  # fmt: skip
            try:
                await self._deliver_audio(header, earcon_wav())
            except Exception as exc:
                log.debug("earcon failed: %s", exc)
        await self.set_state("listening")

    def _arm_follow_up(self) -> None:
        self._cancel_follow_up()
        if self.closed:
            return
        self._follow_up = asyncio.get_running_loop().create_task(
            self._follow_up_timer(), name=f"voice-follow-up:{self.session_id}"
        )
        self.service.track(self._follow_up)

    def _cancel_follow_up(self) -> None:
        task, self._follow_up = self._follow_up, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    def _follow_up_delay(self) -> float:
        window = float(self.service.config.follow_up_seconds)
        return window + max(0.0, self._playback_until - time.monotonic())

    async def _follow_up_timer(self) -> None:
        """Back to standby once nothing happened for follow_up_seconds after playback ended."""
        delay = self._follow_up_delay()
        was_busy = False
        while True:
            await asyncio.sleep(delay)
            if self.closed or self.mode != "wake" or self.state != "listening":
                return
            busy = (
                (self._turn is not None and not self._turn.done())
                or (self.vad is not None and self.vad.in_speech)
                or self._wake_lock.locked()
            )
            if busy:
                was_busy, delay = True, 0.25
                continue
            if was_busy:  # a full window again after whatever kept us busy
                was_busy, delay = False, self._follow_up_delay()
                continue
            break
        self._follow_up = None
        if self.vad is not None:
            self.vad.reset()
        if self.wake is not None:
            self.wake.reset()
        await self.set_state("standby")

    def _close_wake(self) -> None:
        if self.wake is not None:
            with contextlib.suppress(Exception):
                self.wake.close()
            self.wake = None

    # ------------------------------------------------------------------ turn control
    async def _on_utterance(self, pcm: bytes) -> None:
        if self._turn is not None and not self._turn.done():
            if self._approvals:
                self._spawn(self._answer_approval_by_voice(pcm))
                return
            if self._transcribing_pcm is not None:
                pcm = self._transcribing_pcm + pcm  # still transcribing: treat as one utterance
        await self._begin(pcm)

    async def _begin(self, pcm: bytes | None, *, text: str | None = None, wake_hit: WakeHit | None = None) -> None:
        await self._cancel_turn()
        self._turn = asyncio.create_task(self._pipeline(pcm, text, wake_hit), name=f"voice-turn:{self.session_id}")
        self.service.track(self._turn)

    async def _cancel_turn(self) -> None:
        turn = self._turn
        if turn is not None and not turn.done():
            turn.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await turn
        self._turn = None

    async def stop_turn(self) -> bool:
        """Stop everything: cancel the reply being heard or spoken and go back to listening."""
        active = (self._turn is not None and not self._turn.done()) or self.state in {"transcribing", "thinking", "speaking"}
        await self._cancel_turn()
        if self._speaker is not None and not self._speaker.done():
            self._speaker.cancel()
        if active:
            await self.set_state("listening")
        return active

    async def interrupt(self, reason: str = "client") -> None:
        """Stop speaking now. Text generation (if any) continues silently."""
        active = self._turn is not None and not self._turn.done()
        if not active:
            if self.state != "standby":
                await self.set_state("listening")
            return
        if self._muted:
            return
        self._muted = True
        if self._speaker is not None and not self._speaker.done():
            self._speaker.cancel()
        if not self._audio_end_sent:
            self._audio_end_sent = True
            await self.send({"type": "audio_end", "session_id": self.session_id, "interrupted": True, "reason": reason})
        await self.set_state("listening")

    # ------------------------------------------------------------------ pipeline
    async def _pipeline(self, pcm: bytes | None, text: str | None, wake_hit: WakeHit | None = None) -> None:
        started = time.perf_counter()
        self.metrics = {}
        try:
            if text is None:
                assert pcm is not None
                self._transcribing_pcm = pcm
                await self.set_state("transcribing")
                self.metrics["audio_ms"] = int(len(pcm) / 2 / self.sample_rate * 1000)
                try:
                    text = await self.service.transcribe_pcm(pcm, self.sample_rate)
                except VoiceError as exc:
                    await self._error(str(exc))
                    await self.set_state("listening")
                    return
                finally:
                    self._transcribing_pcm = None
                self.metrics["stt_ms"] = _ms(started)
                if wake_hit is not None:  # drop the wake phrase heard at the start of the same breath
                    m = match_wake_phrase(text, wake_hit.phrase, self.service.config.wake_sensitivity)
                    if m is not None:
                        text = m.remainder
                if not has_words(text):
                    await self.set_state("listening")
                    return
                await self.send(
                    {"type": "transcript", "text": text, "final": True, "session_id": self.session_id,
                     "stt_ms": self.metrics["stt_ms"]}
                )
            else:
                await self.send({"type": "transcript", "text": text, "final": True, "session_id": self.session_id})
            await self._run_turn(text, started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("voice turn failed")
            await self._error(str(exc), recoverable=False)
            await self.set_state("listening")

    async def _run_turn(self, text: str, started: float) -> None:
        agent = self.app.agent
        if agent is None:
            raise VoiceError("the assistant is not running")
        assert self.session_id is not None
        self._generating = True
        self._muted = False
        self._audio_started = False
        self._audio_end_sent = False
        self._sentence_index = 0
        self._approvals = {}
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._speaker = asyncio.create_task(self._speak_loop(queue, started))
        splitter = SentenceSplitter(first_clause_chars=self.service.config.tts_first_clause_chars)
        await self.set_state("thinking")

        def enqueue(sentences: list[str]) -> None:
            if not self._muted:
                for s in sentences:
                    self.metrics.setdefault("first_sentence_ms", _ms(started))
                    queue.put_nowait(s)

        gen = agent.run_turn(self.session_id, text, channel=AGENT_CHANNEL.get(self.channel, "voice"))
        try:
            async for event in gen:
                etype = event.type
                if etype == "text_delta":
                    self.metrics.setdefault("first_token_ms", _ms(started))
                    enqueue(splitter.feed(event.text))
                elif etype == "tool_call":
                    enqueue(splitter.flush())
                elif etype == "approval_request":
                    enqueue(splitter.flush())
                    self._approvals[event.call_id] = event.approval_id
                    enqueue([f"I need your approval to use {event.name.replace('_', ' ')}. Say yes or no."])
                elif etype == "tool_result":
                    self._approvals.pop(event.call_id, None)
                elif etype in {"done", "error"}:
                    enqueue(splitter.flush())
                await self.send(event.model_dump())
        except asyncio.CancelledError:
            self._generating = False
            self._approvals = {}
            if self._speaker is not None and not self._speaker.done():
                self._speaker.cancel()
            await self.send({"type": "done", "content": "", "session_id": self.session_id, "cancelled": True})
            if self._audio_started and not self._audio_end_sent:
                self._audio_end_sent = True
                await self.send({"type": "audio_end", "session_id": self.session_id, "interrupted": True, "reason": "cancelled"})
            raise
        finally:
            self._generating = False
            with contextlib.suppress(BaseException):
                await gen.aclose()

        self._approvals = {}
        queue.put_nowait(None)
        try:
            await self._speaker
        except asyncio.CancelledError:
            me = asyncio.current_task()
            if me is not None and me.cancelling():  # the turn itself was cancelled, not just the speaker
                if self._audio_started and not self._audio_end_sent:
                    self._audio_end_sent = True
                    await self.send({"type": "audio_end", "session_id": self.session_id, "interrupted": True, "reason": "cancelled"})
                raise
        self.metrics["total_ms"] = _ms(started)
        if not self._audio_end_sent:
            self._audio_end_sent = True
            await self.send(
                {"type": "audio_end", "session_id": self.session_id, "sentences": self._sentence_index,
                 "metrics": dict(self.metrics)}
            )
        await self.set_state("listening")

    async def _speak_loop(self, queue: asyncio.Queue[str | None], started: float) -> None:
        while True:
            sentence = await queue.get()
            if sentence is None:
                return
            if self._muted:
                continue
            t0 = time.perf_counter()
            try:
                wav = await self.service.synthesize(sentence)
            except VoiceError as exc:
                await self._error(str(exc))
                continue
            self.metrics["tts_ms"] = self.metrics.get("tts_ms", 0) + _ms(t0)
            if self._muted:
                continue
            if not self._audio_started:
                self._audio_started = True
                self.metrics["first_tts_ms"] = _ms(t0)
                self.metrics["first_audio_ms"] = _ms(started)
            await self.set_state("speaking")
            header = {"type": "audio", "format": "wav", "sentence_index": self._sentence_index,
                      "text": sentence, "session_id": self.session_id}
            self._sentence_index += 1
            try:
                await self._deliver_audio(header, wav)
            except Exception as exc:  # undecodable provider output
                log.warning("voice audio delivery failed: %s", exc)
                await self._error(f"could not send audio: {exc}")
            if queue.empty() and self._approvals:
                await self.set_state("listening")  # waiting for the user's answer

    # ------------------------------------------------------------------ approvals
    async def _resolve_approval(self, approval_id: str, decision: str) -> None:
        ok = self.app.approvals.resolve(approval_id, decision)
        for call_id, aid in list(self._approvals.items()):
            if aid == approval_id:
                self._approvals.pop(call_id, None)
        await self.send({"type": "approval.ack", "approval_id": approval_id, "resolved": ok})
        if ok and self._generating:
            await self.set_state("thinking")

    async def _answer_approval_by_voice(self, pcm: bytes) -> None:
        await self.set_state("transcribing")
        try:
            text = await self.service.transcribe_pcm(pcm, self.sample_rate)
        except VoiceError as exc:
            await self._error(str(exc))
            await self.set_state("listening")
            return
        if not text.strip():
            await self.set_state("listening")
            return
        await self.send({"type": "transcript", "text": text, "final": True, "session_id": self.session_id})
        decision = parse_decision(text)
        pending = list(self._approvals.values())
        if decision is not None and pending:
            for approval_id in pending:
                await self._resolve_approval(approval_id, decision)
            return
        await self._begin(None, text=text)  # not an answer: treat it as a new request

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._side.add(task)
        task.add_done_callback(self._side.discard)
        self.service.track(task)

    # ------------------------------------------------------------------ shutdown
    async def close(self, *, cancel_turn: bool = True) -> None:
        if self.closed:
            return
        if cancel_turn:
            await self._cancel_turn()
        else:
            # socket dropped: stop speaking but let the reply finish so it is persisted
            self._muted = True
            if self._speaker is not None and not self._speaker.done():
                self._speaker.cancel()
        self._cancel_follow_up()
        for t in list(self._side):
            t.cancel()
        await self.set_state("idle")
        self._close_wake()
        self.closed = True
