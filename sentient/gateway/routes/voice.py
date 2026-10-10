"""Voice routes. OWNER: VOICE AGENT. Contract: docs/API.md sections 9 and 16.

REST under ``/api/voice/*`` (token via the shared AUTH dependency) and the live
``WS /ws/voice`` socket. The socket transport lives in ``sentient.voice.socket`` so
the nodes LAN listener can mount the same endpoint; it authenticates with the
gateway token or a device ``node_token``.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from sentient.gateway.deps import AUTH, get_core
from sentient.voice.audio import MAX_AUDIO_BYTES, audio_suffix
from sentient.voice.base import VoiceError
from sentient.voice.service import DictationStopped
from sentient.voice.socket import voice_socket_endpoint

log = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])


@router.get("/api/voice/status", dependencies=AUTH)
async def voice_status(request: Request):
    return await get_core(request).voice.status()


@router.post("/api/voice/transcribe", dependencies=AUTH)
async def voice_transcribe(request: Request, file: UploadFile = File(...)):
    suffix = audio_suffix(file.filename or "", content_type=file.content_type or "")
    if not suffix:
        raise HTTPException(415, "unsupported audio type; send wav, webm, ogg, m4a, mp3 or flac")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio file")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio file too large (max 50 MB)")
    try:
        text = await get_core(request).voice.transcribe_bytes(data, f"upload{suffix}")
    except VoiceError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"text": text}


@router.post("/api/voice/dictate", dependencies=AUTH)
async def voice_dictate(
    request: Request,
    file: UploadFile = File(...),
    cleanup: Literal["raw", "tidy", "polish"] | None = Form(None),
):
    """Push to talk and dictation (#169): local speech recognition, then ``cleanup`` (default voice.dictation's)."""
    suffix = audio_suffix(file.filename or "", content_type=file.content_type or "")
    if not suffix:
        raise HTTPException(415, "unsupported audio type; send wav, webm, ogg, m4a, mp3 or flac")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio file")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio file too large (max 50 MB)")
    try:
        return await get_core(request).voice.dictate(data, f"dictation{suffix}", cleanup=cleanup)
    except DictationStopped as exc:
        raise HTTPException(409, str(exc)) from exc
    except VoiceError as exc:
        raise HTTPException(503, str(exc)) from exc


class SpeakBody(BaseModel):
    text: str
    voice: str | None = None


@router.post("/api/voice/speak", dependencies=AUTH)
async def voice_speak(request: Request, body: SpeakBody):
    if not body.text.strip():
        raise HTTPException(400, "text is empty")
    try:
        wav = await get_core(request).voice.speak(body.text, body.voice)
    except VoiceError as exc:
        status = 400 if str(exc) == "nothing to say" else 503
        raise HTTPException(status, str(exc)) from exc
    return Response(content=wav, media_type="audio/wav")


class PrepareBody(BaseModel):
    target: Literal["all", "stt", "tts", "wake"] = "all"


@router.post("/api/voice/prepare", dependencies=AUTH)
async def voice_prepare(request: Request, body: PrepareBody | None = None):
    svc = get_core(request).voice
    target = body.target if body is not None else "all"

    async def gen():
        async for step in svc.prepare(target):
            yield json.dumps(step, default=str) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@router.websocket("/ws/voice")
async def voice_socket(ws: WebSocket):
    await voice_socket_endpoint(ws)
