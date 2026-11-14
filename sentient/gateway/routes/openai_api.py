"""OpenAI-compatible chat API (#211): other apps on this computer talk to Sentient, with its memory and tools.

``POST /v1/chat/completions`` (streaming and not) and ``GET /v1/models`` answer only while ``openai_api.enabled`` is
on, and only with Sentient's own API key (``POST /api/openai-api/key``), never the window's token. Each ``user`` of
the request gets its own named chat, and only the last user message is sent: Sentient keeps the chat's history itself.

An app is never a way around approvals: a call that needs the user's OK posts a ``kind: "approval"`` notification
(the window's Allow and Deny card, and the paired chats when the user is away from the app, like a task's approval),
and the request waits for the answer. A call the user declines, or nobody answers in time, ends the reply with a line
saying it was not done. Each turn runs in its own task, so Stop everything ends it with "Stopped." and a client that
hangs up doesn't cut it short.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import secrets as pysecrets
import time
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from sentient import secrets
from sentient.gateway.deps import AUTH, get_core
from sentient.llm.events import ApprovalRequest, Done, TextDelta, ToolResultEvent, Usage
from sentient.store.db import new_id
from sentient.tools.rules import tool_title

log = logging.getLogger(__name__)

router = APIRouter(tags=["openai-api"])

KEY_NAME = "openai_api_key"  # keychain entry
SESSIONS_META = "openai_api.sessions"  # {name: session_id}, the named chats
CHANNEL = "api"
MODEL = "sentient"
DEFAULT_NAME = "API"
STOPPED = "Stopped."
NOT_DONE = "Not done: {what} needed your OK in Sentient and didn't get it."
TURN_FAILED = "Sentient couldn't finish that reply. The details are in Sentient's log."


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


def _loopback(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    return parsed.is_loopback


def _refusal(request: Request) -> JSONResponse | None:
    if not get_core(request).config.openai_api.enabled:
        return _error(404, "Sentient's OpenAI-compatible API is off.", "api_off")
    # Check the transport's local address as well as the peer: proxy headers can rewrite the peer, but not the
    # socket's local address. Host/forwarded headers must never make a network-facing connection count as local.
    server = request.scope.get("server")
    if not server or not request.client or not _loopback(server[0]) or not _loopback(request.client.host):
        return _error(403, "Sentient's OpenAI-compatible API only accepts connections on this computer.", "local_only")
    expected = secrets.get_secret(KEY_NAME)
    header = request.headers.get("authorization", "")
    given = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not expected or not given or not pysecrets.compare_digest(given.encode(), expected.encode()):
        return _error(401, "Invalid or missing API key.", "invalid_api_key")
    return None


class _PublicAPIRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def guarded(request: Request) -> Response:
            # FastAPI parses JSON before resolving dependencies. Refuse here before parsing; the endpoints
            # recheck afterwards in case the setting or key changed while the body was arriving.
            if (refused := _refusal(request)) is not None:
                return refused
            try:
                return await handler(request)
            except RequestValidationError:
                return _error(400, "Invalid request body. Send a valid JSON chat completion request.", "invalid_request")
            except StarletteHTTPException as exc:
                if exc.status_code != 400:
                    raise
                # Invalid text encoding is a body-parsing HTTP 400, not a RequestValidationError.
                return _error(400, "Invalid request body. Send a valid JSON chat completion request.", "invalid_request")

        return guarded


public_router = APIRouter(
    route_class=_PublicAPIRoute,
    responses={"4XX": {"description": "OpenAI-shaped request, locality, or authentication error."}},
)


def _text(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, list):  # [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
        return "\n".join(str(p.get("text") or "") for p in content if p.get("type") == "text")
    return content or ""


async def _session(core: Any, name: str) -> str:
    """The named chat, made on first use and again after it was deleted."""
    # All names share one metadata value, so different names must serialize their updates too.
    lock = getattr(core.store, "_openai_api_sessions_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        core.store._openai_api_sessions_lock = lock
    async with lock:
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


@public_router.get("/v1/models")
async def models(request: Request):
    if (refused := _refusal(request)) is not None:
        return refused
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "sentient"}]}


class _Approvals:
    """The approvals one API turn asks for, to tell the client which calls were not done. The approval itself, its
    ``kind: "approval"`` notification and that note's ``status`` (``allow``, ``deny``, ``expired``) come from the
    agent's one path for turns another app steers, the same one MCP turns use (#443), so every place that shows it
    settles it."""

    def __init__(self, core: Any, session_id: str) -> None:
        self.core, self.session_id = core, session_id
        self.open: dict[str, str] = {}  # call_id -> what it was, while it waits
        self.not_done: list[str] = []
        self.told = 0  # how many of not_done the client has been given

    def what(self, name: str) -> str:
        tool = self.core.registry.get(name)
        return f'"{tool_title(tool) if tool is not None else name}"'

    async def see(self, event: Any) -> None:
        if isinstance(event, ApprovalRequest):
            self.open[event.call_id] = self.what(event.name)
        elif isinstance(event, ToolResultEvent) and event.call_id in self.open:
            what = self.open.pop(event.call_id)
            if isinstance(event.result, dict) and event.result.get("declined"):
                self.not_done.append(what)

    async def close(self) -> None:
        """Stopped or broken while still waiting: the agent settled its notes and forgot the requests."""
        self.open.clear()

    def note(self) -> str:
        """The "Not done" lines not given to the client yet. They go before the model's words, so a client that reads
        only the first line isn't told a declined call ran when a small model says it did (#479 review)."""
        fresh, self.told = self.not_done[self.told:], len(self.not_done)
        return " ".join(NOT_DONE.format(what=w) for w in fresh)


async def _run_turn(core: Any, session_id: str, text: str, out: asyncio.Queue, approvals: _Approvals) -> None:
    """One API turn, in its own task; _start's completion callback finishes its event queue."""
    try:
        async with contextlib.aclosing(core.agent.run_turn(session_id, text, channel=CHANNEL)) as turn:
            async for event in turn:
                await approvals.see(event)
                out.put_nowait(event)
    except Exception as exc:
        log.exception("API chat turn failed")
        out.put_nowait(exc)
    finally:
        await approvals.close()


def _start(core: Any, session_id: str, text: str) -> tuple[asyncio.Queue, _Approvals] | None:
    """Reserve one API turn per chat without queuing another request behind it."""
    turns = getattr(core, "_openai_api_turns", None)
    if turns is None:
        turns = core._openai_api_turns = {}
    active = turns.get(session_id)
    if (active is not None and not active.done()) or core.agent.is_busy(session_id):
        return None
    out: asyncio.Queue = asyncio.Queue()
    approvals = _Approvals(core, session_id)
    # No await between the busy check and reservation, including before Agent.run_turn starts tracking it.
    task = core.agent.create_turn_task(_run_turn(core, session_id, text, out, approvals))
    turns[session_id] = task  # also keeps detached turns alive after their HTTP client disconnects

    def finished(done: asyncio.Task) -> None:
        if turns.get(session_id) is done:
            turns.pop(session_id)
        # A task cancelled before its first step never enters _run_turn's finally block.
        if done.cancelled():
            out.put_nowait(asyncio.CancelledError())
        elif (error := done.exception()) is not None:
            out.put_nowait(error)
        out.put_nowait(None)

    task.add_done_callback(finished)
    return out, approvals


@public_router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionBody):
    if (refused := _refusal(request)) is not None:
        return refused
    text = next((_text(m.content) for m in reversed(body.messages) if m.role == "user"), "").strip()
    if not text:
        return _error(400, "Send at least one user message with text.", "no_user_message")
    core = get_core(request)
    generation = core.stop_generation
    session_id = await _session(core, (body.user or "").strip() or DEFAULT_NAME)
    if core.stop_generation != generation:
        return _error(409, STOPPED, "stopped")
    started = _start(core, session_id, text)
    if started is None:
        return _error(409, "This chat is already answering. Wait for it to finish, then try again.", "chat_busy")
    cid, created = f"chatcmpl-{new_id()}", int(time.time())
    out, approvals = started

    def chunk(delta: dict, finish: str | None = None) -> str:
        data = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(data)}\n\n"

    def failure(message: str) -> str:
        return f"data: {json.dumps({'error': {'message': message, 'type': 'server_error'}})}\n\n"

    if body.stream:

        async def gen():
            yield chunk({"role": "assistant"})
            while (item := await out.get()) is not None:
                if note := approvals.note():  # a declined call: said at once, before the model writes about it
                    yield chunk({"content": note + "\n\n"})
                if isinstance(item, TextDelta) and item.text:
                    yield chunk({"content": item.text})
                elif isinstance(item, Done) and item.error:
                    yield failure(item.error)
                elif isinstance(item, asyncio.CancelledError):
                    yield failure(STOPPED)
                elif isinstance(item, Exception):  # logged in _run_turn; its text can hold internal details
                    yield failure(TURN_FAILED)
            if note := approvals.note():
                yield chunk({"content": "\n\n" + note})
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    parts: list[str] = []
    done: Done | None = None
    failed: str | None = None
    prompt_tokens = completion_tokens = 0
    while (item := await out.get()) is not None:
        if isinstance(item, TextDelta):
            parts.append(item.text)
        elif isinstance(item, Usage):
            prompt_tokens += item.prompt_tokens
            completion_tokens += item.completion_tokens
        elif isinstance(item, Done):
            done = item
        elif isinstance(item, asyncio.CancelledError):
            return _error(409, STOPPED, "stopped")
        elif isinstance(item, Exception):  # logged in _run_turn; its text can hold internal details
            failed = TURN_FAILED
    if failed or (done is not None and done.error):
        return _error(500, failed or done.error, "turn_failed")
    note = approvals.note()
    content = (note + "\n\n" if note else "") + ((done.content if done is not None else "") or "".join(parts))
    return {
        "id": cid, "object": "chat.completion", "created": created, "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


router.include_router(public_router)
