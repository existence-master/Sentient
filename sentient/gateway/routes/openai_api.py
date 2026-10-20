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
_turns: set[asyncio.Task] = set()  # turns still running after their client hung up


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


class _Approvals:
    """The approvals one API turn asks for, as ``kind: "approval"`` notifications whose ``status`` follows the answer
    (the window's card words: ``allow``, ``deny``, ``expired``), so every place that shows one settles it."""

    def __init__(self, core: Any, session_id: str) -> None:
        self.core, self.session_id = core, session_id
        self.open: dict[str, tuple[str, str, float]] = {}  # call_id -> (notification id, what, asked at)
        self.not_done: list[str] = []

    def what(self, name: str) -> str:
        tool = self.core.registry.get(name)
        return f'"{tool_title(tool) if tool is not None else name}"'

    async def see(self, event: Any) -> None:
        if isinstance(event, ApprovalRequest):
            what = self.what(event.name)
            note = await self.core.notify(
                "approval", f"Another app asked Sentient to use {what}. Allow it?", title="Approval needed",
                payload={**event.model_dump(mode="json"), "session_id": self.session_id, "source": CHANNEL},
            )
            self.open[event.call_id] = (str(note["id"]), what, time.monotonic())
        elif isinstance(event, ToolResultEvent) and event.call_id in self.open:
            nid, what, asked = self.open.pop(event.call_id)
            if isinstance(event.result, dict) and event.result.get("declined"):
                self.not_done.append(what)
                # ponytail: unanswered means it waited the whole timeout; the broker doesn't say which it was
                waited = time.monotonic() - asked >= self.core.approvals.timeout_s - 1
                await self.settle(nid, "expired" if waited else "deny")
            else:
                await self.settle(nid, "allow")

    async def settle(self, nid: str, status: str) -> None:
        try:
            note = await self.core.notifications.get(nid)
            if note is not None and not (note.get("payload") or {}).get("status"):
                await self.core.notifications.update_payload(nid, {**(note.get("payload") or {}), "status": status})
        except Exception:
            log.exception("could not settle approval notification %s", nid)

    async def close(self) -> None:
        """Stopped or broken while still waiting: no longer answerable anywhere."""
        for nid, _, _ in list(self.open.values()):
            await self.settle(nid, "expired")
        self.open.clear()

    def note(self) -> str:
        return "\n\n" + " ".join(NOT_DONE.format(what=w) for w in self.not_done) if self.not_done else ""


async def _run_turn(core: Any, session_id: str, text: str, out: asyncio.Queue, approvals: _Approvals) -> None:
    """One API turn, in its own task (Stop everything cancels it). Its events go to ``out``, then None."""
    try:
        async for event in core.agent.run_turn(session_id, text, channel=CHANNEL):
            await approvals.see(event)
            out.put_nowait(event)
    except asyncio.CancelledError:
        out.put_nowait(asyncio.CancelledError())
        raise
    except Exception as exc:
        log.exception("API chat turn failed")
        out.put_nowait(exc)
    finally:
        await approvals.close()
        out.put_nowait(None)


def _start(core: Any, session_id: str, text: str) -> tuple[asyncio.Queue, _Approvals]:
    out: asyncio.Queue = asyncio.Queue()
    approvals = _Approvals(core, session_id)
    task = asyncio.create_task(_run_turn(core, session_id, text, out, approvals))
    _turns.add(task)
    task.add_done_callback(_turns.discard)
    return out, approvals


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
    out, approvals = _start(core, session_id, text)

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
                if isinstance(item, TextDelta) and item.text:
                    yield chunk({"content": item.text})
                elif isinstance(item, Done) and item.error:
                    yield failure(item.error)
                elif isinstance(item, asyncio.CancelledError):
                    yield failure(STOPPED)
                elif isinstance(item, Exception):
                    yield failure(str(item))
            if note := approvals.note():
                yield chunk({"content": note})
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
        elif isinstance(item, Exception):
            failed = str(item)
    if failed or (done is not None and done.error):
        return _error(500, failed or done.error, "turn_failed")
    content = ((done.content if done is not None else "") or "".join(parts)) + approvals.note()
    return {
        "id": cid, "object": "chat.completion", "created": created, "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }
