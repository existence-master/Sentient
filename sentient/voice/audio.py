"""Small, dependency-light audio helpers shared by STT, TTS and the voice socket.

All live audio inside the voice package is PCM16 little-endian mono. WAV is the
wire format for synthesized speech because every browser can decode it with
``AudioContext.decodeAudioData`` without extra codecs.
"""

from __future__ import annotations

import io
import wave

import numpy as np


def pcm16_to_float(pcm16: bytes) -> np.ndarray:
    if len(pcm16) % 2:
        pcm16 = pcm16[:-1]
    return np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(np.round(audio * 32768.0), -32768, 32767).astype("<i2").tobytes()


def pcm16_to_wav(pcm16: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16)
    return buf.getvalue()


def float_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    return pcm16_to_wav(float_to_pcm16(audio), sample_rate)


def wav_to_pcm16(data: bytes) -> tuple[bytes, int]:
    """Decode a WAV (any PCM width, any channel count) to mono PCM16 + sample rate."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return _decode_with_soundfile(data)
    if width == 2 and ch == 1:
        return raw[: len(raw) - len(raw) % 2], sr  # already the target format: no lossy round trip
    if width == 2:
        arr = np.frombuffer(raw[: len(raw) - len(raw) % 2], dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        arr = np.frombuffer(raw[: len(raw) - len(raw) % 4], dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return _decode_with_soundfile(data)
    if ch > 1:
        arr = arr[: len(arr) - len(arr) % ch].reshape(-1, ch).mean(axis=1)
    return float_to_pcm16(arr), sr


def _decode_with_soundfile(data: bytes) -> tuple[bytes, int]:
    """Float WAV, WAVE_FORMAT_EXTENSIBLE, 24-bit and other formats the wave module rejects."""
    try:
        import soundfile as sf

        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"could not decode audio: {exc}") from exc
    return float_to_pcm16(arr.mean(axis=1)), int(sr)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float audio. Integer down-ratios use box-filter decimation
    (48k/32k -> 16k); everything else low-passes lightly and interpolates."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    if src_rate > dst_rate and src_rate % dst_rate == 0:
        k = src_rate // dst_rate
        n = audio.size - audio.size % k
        return audio[:n].reshape(-1, k).mean(axis=1)
    if src_rate > dst_rate:
        k = max(1, round(src_rate / dst_rate))
        if k > 1:
            audio = np.convolve(audio, np.ones(k, dtype=np.float32) / k, mode="same")
    n_out = round(audio.size * dst_rate / src_rate)
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def pcm16_resample(pcm16: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm16[: len(pcm16) - len(pcm16) % 2]
    return float_to_pcm16(resample(pcm16_to_float(pcm16), src_rate, dst_rate))


MAX_AUDIO_BYTES = 50 * 1024 * 1024
AUDIO_SUFFIXES = {".wav", ".webm", ".ogg", ".oga", ".opus", ".m4a", ".mp4", ".aac", ".mp3", ".mpeg", ".mpga", ".flac"}
_SUFFIX_ALIAS = {".oga": ".ogg", ".opus": ".ogg", ".mpeg": ".mp3", ".mpga": ".mp3"}
CONTENT_TYPE_SUFFIX = {
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/vnd.wave": ".wav",
    "audio/webm": ".webm", "video/webm": ".webm", "audio/ogg": ".ogg", "audio/opus": ".ogg",
    "application/ogg": ".ogg", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/m4a": ".m4a",
    "audio/aac": ".aac", "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/flac": ".flac", "audio/x-flac": ".flac",
}  # fmt: skip


def audio_suffix(filename: str = "", data: bytes = b"", content_type: str = "") -> str:
    """Pick the file suffix used to decode an audio file: its name, then its content type, then its bytes."""
    name = (filename or "").strip()
    suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if suffix in AUDIO_SUFFIXES:
        return _SUFFIX_ALIAS.get(suffix, suffix)
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype in CONTENT_TYPE_SUFFIX:
        return CONTENT_TYPE_SUFFIX[ctype]
    return sniff_audio_suffix(data) if data else ""


def sniff_audio_suffix(data: bytes) -> str:
    """Container type from magic bytes: '.wav', '.ogg', '.webm', '.m4a', '.mp3', '.flac' or ''."""
    head = data[:16]
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if head[:4] == b"OggS":
        return ".ogg"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    if head[4:8] == b"ftyp":
        return ".m4a"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return ".mp3"
    return ""


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))


def trim_silence(audio: np.ndarray, sample_rate: int, threshold: float = 0.005, keep_ms: int = 40) -> np.ndarray:
    """Drop leading and trailing near-silence so the first word plays sooner."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio[:0]
    pad = int(sample_rate * keep_ms / 1000)
    return audio[max(0, loud[0] - pad) : min(audio.size, loud[-1] + pad)]


def wav_duration_s(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return 0.0
