"""Energy (RMS) voice activity detection with an adaptive noise floor.

Deliberately simple and dependency-free: it runs inline on the event loop for
every 20 ms frame and costs microseconds. faster-whisper's own Silero VAD
filter then removes any non-speech that slipped through before transcription.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from sentient.voice.audio import pcm16_to_float, rms


@dataclass
class VADEvent:
    kind: Literal["speech_start", "utterance"]
    pcm: bytes = b""
    duration_ms: int = 0
    reason: str = ""  # utterance: silence | max_length | manual


class EnergyVAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        *,
        silence_ms: int = 700,
        min_speech_ms: int = 250,
        max_utterance_ms: int = 30_000,
        frame_ms: int = 20,
        preroll_ms: int = 300,
        onset_ms: int = 60,
        start_ratio: float = 3.0,
        stop_ratio: float = 2.0,
        min_start_rms: float = 0.012,
        min_stop_rms: float = 0.006,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_ms = frame_ms
        self.frame_bytes = max(2, int(self.sample_rate * frame_ms / 1000) * 2)
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_ms = max_utterance_ms
        self.onset_ms = onset_ms
        self.start_ratio = start_ratio
        self.stop_ratio = stop_ratio
        self.min_start_rms = min_start_rms
        self.min_stop_rms = min_stop_rms
        self.noise_floor: float | None = None
        self.level = 0.0
        self._preroll: deque[bytes] = deque(maxlen=max(1, preroll_ms // frame_ms))
        self._since_reset: deque[bytes] = deque(maxlen=max(1, max_utterance_ms // frame_ms))
        self._voiced_since_reset = 0
        self._pending = b""
        self.in_speech = False
        self.voiced_ms = 0
        self._speech: list[bytes] = []
        self._silence_run = 0
        self._voiced_run = 0

    # ------------------------------------------------------------------ state
    def _reset_speech(self) -> None:
        self.in_speech = False
        self.voiced_ms = 0
        self._speech = []
        self._silence_run = 0
        self._voiced_run = 0

    def reset(self) -> None:
        """Forget any partial utterance. The learned noise floor is kept."""
        self._reset_speech()
        self._preroll.clear()
        self._since_reset.clear()
        self._voiced_since_reset = 0
        self._pending = b""

    def restart_segment(self) -> None:
        """Drop the audio collected so far but keep following the current breath.

        Used right after a streaming wake word detection: the phrase is thrown
        away and whatever the user keeps saying becomes the utterance.
        """
        self._speech = []
        self.voiced_ms = 0
        self._silence_run = 0
        self._preroll.clear()
        self._since_reset.clear()
        self._voiced_since_reset = 0

    @property
    def speech_ms(self) -> int:
        return len(self._speech) * self.frame_ms

    # ------------------------------------------------------------------ processing
    def feed(self, pcm16: bytes) -> list[VADEvent]:
        data = self._pending + pcm16
        n = len(data) - len(data) % self.frame_bytes
        self._pending = data[n:]
        events: list[VADEvent] = []
        for i in range(0, n, self.frame_bytes):
            ev = self._frame(data[i : i + self.frame_bytes])
            if ev is not None:
                events.append(ev)
        return events

    def _frame(self, frame: bytes) -> VADEvent | None:
        energy = rms(pcm16_to_float(frame))
        self.level = energy
        if self.noise_floor is None:
            self.noise_floor = min(max(energy, 1e-4), self.min_start_rms / self.start_ratio)
        floor = self.noise_floor
        start_th = max(self.min_start_rms, floor * self.start_ratio)
        stop_th = max(self.min_stop_rms, floor * self.stop_ratio)
        self._since_reset.append(frame)
        if energy >= stop_th:
            self._voiced_since_reset += 1

        if not self.in_speech:
            self._preroll.append(frame)
            if energy >= start_th:
                self._voiced_run += 1
            else:
                self._voiced_run = 0
                # track the room: fall fast, rise slowly
                rate = 0.2 if energy < floor else 0.02
                self.noise_floor = max(1e-4, floor + rate * (energy - floor))
            if self._voiced_run * self.frame_ms >= self.onset_ms:
                self.in_speech = True
                self._speech = list(self._preroll)
                self.voiced_ms = self._voiced_run * self.frame_ms
                self._silence_run = 0
                return VADEvent("speech_start")
            return None

        self._speech.append(frame)
        if energy >= stop_th:
            self.voiced_ms += self.frame_ms
            self._silence_run = 0
        else:
            self._silence_run += self.frame_ms
        if self._silence_run >= self.silence_ms:
            return self._finish("silence")
        if self.speech_ms >= self.max_utterance_ms:
            return self._finish("max_length")
        return None

    def _finish(self, reason: str) -> VADEvent | None:
        frames = self._speech
        if reason == "silence":
            drop = max(0, (self._silence_run - 200) // self.frame_ms)  # keep ~200 ms of tail
            if drop:
                frames = frames[: len(frames) - drop]
        voiced = self.voiced_ms
        self._reset_speech()
        self._preroll.clear()
        self._since_reset.clear()
        self._voiced_since_reset = 0
        if voiced < self.min_speech_ms or not frames:
            return None
        return VADEvent("utterance", pcm=b"".join(frames), duration_ms=len(frames) * self.frame_ms, reason=reason)

    def flush(self) -> VADEvent | None:
        """Push-to-talk release: return what was said even if silence never arrived."""
        if self.in_speech:
            if self._pending:
                self._speech.append(self._pending)
                self._pending = b""
            voiced = self.voiced_ms
            frames = self._speech
            self.reset()
            if voiced < min(self.min_speech_ms, 120) or not frames:
                return None
            pcm = b"".join(frames)
            return VADEvent("utterance", pcm=pcm, duration_ms=len(pcm) * 500 // self.sample_rate, reason="manual")
        voiced_ms = self._voiced_since_reset * self.frame_ms
        frames = list(self._since_reset)
        self.reset()
        if voiced_ms < 120 or not frames:
            return None
        pcm = b"".join(frames)
        return VADEvent("utterance", pcm=pcm, duration_ms=len(pcm) * 500 // self.sample_rate, reason="manual")
