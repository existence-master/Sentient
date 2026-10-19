"""OpenAI-compatible chat API (#211): other apps on this computer talk to Sentient, with its memory and tools.

``POST /v1/chat/completions`` (streaming and not) and ``GET /v1/models`` answer only while ``openai_api.enabled`` is
on, and only with Sentient's own API key (``POST /api/openai-api/key``), never the window's token. Each ``user`` of
the request gets its own named chat, and only the last user message is sent: Sentient keeps the chat's history itself.
"""

from __future__ import annotations

import json
import logging
import secrets as pysecrets
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from sentient import secrets
from sentient.gateway.deps import AUTH, get_core
from sentient.llm.events import Done, TextDelta, Usage
from sentient.store.db import new_id

log = logging.getLogger(__name__)

router = APIRouter(tags=["openai-api"])

KEY_NAME = "openai_api_key"  # keychain entry
SESSIONS_META = "openai_api.sessions"  # {name: session_id}, the named chats
CHANNEL = "api"
MODEL = "sentient"
DEFAULT_NAME = "API"


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionBody(BaseModel):
    model: str = MODEL
    messages: list[ChatMessage]
    stream: bool = False
    user: str | None = None  # which named chat; other OpenAI fields (temperature...) are ignored


def _error(status: int, message: str, code: str) -> JSONResponse:
    """OpenAI's error shape, so clients show the message. OpenAI's SDKs retry a 5xx twice by default, which would run
    the turn (and its tools) again: ``x-should-retry: false`` tells them not to."""
    kind = "server_error" if status >= 500 else "invalid_request_error"
    return JSONResponse(
        {"error": {"message": message, "type": kind, "param": None, "code": code}},
        status_code=status, headers={"x-should-retry": "false"},
    )


def _refusal(request: Request) -> JSONResponse | None:
    if not get_core(request).config.openai_api.enabled:
        return _error(404, "Sentient's OpenAI-compatible API is off.", "api_off")
    expected = secrets.get_secret(KEY_NAME)
    header = request.headers.get("authorization", "")
    given = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not expected or not given or not pysecrets.compare_digest(given, expected):
        return _error(401, "Invalid or missing API key.", "invalid_api_key")
    return None


def _text(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, list):  # [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
        return "\n".join(str(p.get("text") or "") for p in content if p.get("type") == "text")
    return content or ""


async def _session(core: Any, name: str) -> str:
    """The named chat, made on first use and again after it was deleted."""
    # ponytail: two first requests for one new name at once can make two chats; a lock if clients ever do that
    known = json.loads(await core.store.get_meta(SESSIONS_META) or "{}")
    sid = known.get(name)
    if sid and await core.store.get_session(sid) is not None:
        return sid
    sid = await core.store.create_session(channel=CHANNEL, title=name)
    known[name] = sid
    await core.store.set_meta(SESSIONS_META, json.dumps(known))
    return sid


@router.post("/api/openai-api/key", dependencies=AUTH)
async def new_key(request: Request):
    """Make a new API key, replacing the old one, and return it. Only this answer ever shows it."""
    key = "sk-sentient-" + pysecrets.token_urlsafe(32)
    if not secrets.set_secret(KEY_NAME, key):
        raise HTTPException(503, "Sentient couldn't save the key in this computer's keychain.")
    return {"key": key, "base_url": str(request.base_url).rstrip("/") + "/v1"}


@router.get("/v1/models")
async def models(request: Request):
    if (refused := _refusal(request)) is not None:
        return refused
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "sentient"}]}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionBody):
    if (refused := _refusal(request)) is not None:
        return refused
    text = next((_text(m.content) for m in reversed(body.messages) if m.role == "user"), "").strip()
    if not text:
        return _error(400, "Send at least one user message with text.", "no_user_message")
    core = get_core(request)
    session_id = await _session(core, (body.user or "").strip() or DEFAULT_NAME)
    cid, created = f"chatcmpl-{new_id()}", int(time.time())

    def chunk(delta: dict, finish: str | None = None) -> str:
        data = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(data)}\n\n"

    if body.stream:

        async def gen():
            yield chunk({"role": "assistant"})
            try:
                async for event in core.agent.run_turn(session_id, text, channel=CHANNEL):
                    if isinstance(event, TextDelta) and event.text:
                        yield chunk({"content": event.text})
                    elif isinstance(event, Done) and event.error:
                        yield f"data: {json.dumps({'error': {'message': event.error, 'type': 'server_error'}})}\n\n"
            except Exception as exc:
                log.exception("API chat turn failed")
                yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error'}})}\n\n"
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    parts: list[str] = []
    done: Done | None = None
    prompt_tokens = completion_tokens = 0
    try:
        async for event in core.agent.run_turn(session_id, text, channel=CHANNEL):
            if isinstance(event, TextDelta):
                parts.append(event.text)
            elif isinstance(event, Usage):
                prompt_tokens += event.prompt_tokens
                completion_tokens += event.completion_tokens
            elif isinstance(event, Done):
                done = event
    except Exception as exc:
        log.exception("API chat turn failed")
        return _error(500, str(exc), "turn_failed")
    if done is not None and done.error:
        return _error(500, done.error, "turn_failed")
    content = (done.content if done is not None else "") or "".join(parts)
    return {
        "id": cid, "object": "chat.completion", "created": created, "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }
