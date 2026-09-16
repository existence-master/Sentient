"""Provider interfaces for speech-to-text and text-to-speech.

Every provider is cheap to construct: nothing heavy (model weights, network
clients, OS speech engines) is touched until the first ``transcribe`` /
``synthesize`` / ``prepare`` call. Blocking work runs in worker threads so the
gateway event loop never stalls.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from sentient import paths, secrets

log = logging.getLogger(__name__)


class VoiceError(RuntimeError):
    """A provider could not do its job (missing key, model failed to load, HTTP error)."""


def models_dir(*parts: str) -> Path:
    """Local voice model weights live under ``$SENTIENT_HOME/models``."""
    p = paths.home().joinpath("models", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_key(name: str, env_var: str, label: str) -> str:
    key = secrets.get_secret(name, env_var)
    if not key:
        raise VoiceError(
            f"No {label} API key. Add it in Settings > Models > API keys (stored as '{name}')."
        )
    return key


def has_key(name: str, env_var: str) -> bool:
    return bool(secrets.get_secret(name, env_var))


def raise_for_status(r: httpx.Response, label: str) -> None:
    if r.status_code >= 400:
        try:
            detail = r.text[:300]
        except Exception:
            detail = r.reason_phrase
        raise VoiceError(f"{label} returned HTTP {r.status_code}: {detail}")


async def download_file(
    url: str, dest: Path, *, chunk: int = 1 << 20
) -> AsyncIterator[tuple[int, int | None]]:
    """Stream ``url`` to ``dest`` atomically, yielding ``(bytes_done, total)``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    timeout = httpx.Timeout(60.0, read=120.0)
    async with (
        httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client,
        client.stream("GET", url) as r,
    ):
        if r.status_code >= 400:
            raise VoiceError(f"download of {url} failed with HTTP {r.status_code}")
        total = int(r.headers.get("content-length") or 0) or None
        done = 0
        fh = await asyncio.to_thread(part.open, "wb")
        try:
            async for block in r.aiter_bytes(chunk):
                await asyncio.to_thread(fh.write, block)
                done += len(block)
                yield done, total
        finally:
            await asyncio.to_thread(fh.close)
    part.replace(dest)


class STTProvider:
    name: str = "stt"
    model: str = ""
    device: str | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return True

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        raise NotImplementedError

    async def transcribe_file(self, path: str | Path) -> str:
        raise NotImplementedError

    async def prepare(self) -> AsyncIterator[dict[str, Any]]:
        yield {"stage": "ready", "component": "stt", "progress": 1.0}

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "provider": self.name,
            "model": self.model,
            "ready": self.ready,
            "device": self.device,
        }
        if self.error:
            out["error"] = self.error
        return out

    def close(self) -> None:
        """Release models / threads. Called when config swaps the provider out."""


class TTSProvider:
    name: str = "tts"
    voice: str = ""
    speed: float = 1.0
    error: str | None = None

    @property
    def ready(self) -> bool:
        return True

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Return a complete WAV file (PCM16 mono)."""
        raise NotImplementedError

    async def voices(self) -> list[dict[str, Any]]:
        return []

    async def prepare(self) -> AsyncIterator[dict[str, Any]]:
        yield {"stage": "ready", "component": "tts", "progress": 1.0}

    async def status(self) -> dict[str, Any]:
        try:
            voices = await self.voices()
        except Exception as exc:  # voice listing must never break status
            log.debug("voice listing failed: %s", exc)
            voices = []
        out: dict[str, Any] = {
            "provider": self.name,
            "voice": self.voice,
            "ready": self.ready,
            "voices": voices,
        }
        if self.error:
            out["error"] = self.error
        return out

    def close(self) -> None:
        """Release models / threads."""
