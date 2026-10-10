"""Typed events streamed from the agent loop to every client.

These replace the legacy approach of diffing a serialized transcript and
asking the UI to regex ``<think>``/``<tool_code>`` tags out of a text stream.
Every client (web UI, CLI, glasses node, a Telegram adapter) consumes the
same event stream; the gateway serializes them as JSON with a ``type`` field.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class _Event(BaseModel):
    session_id: str | None = None
    turn_id: str | None = None


class TextDelta(_Event):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(_Event):
    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


class ToolCallEvent(_Event):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolProgress(_Event):
    """Output of a tool that is still running (code stdout, browser frames, subagent steps)."""

    type: Literal["tool_progress"] = "tool_progress"
    call_id: str
    name: str
    kind: Literal["stdout", "stderr", "status", "frame", "subagent"] = "status"
    text: str | None = None
    image: str | None = None
    data: Any = None


class ToolResultEvent(_Event):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    result: Any = None
    is_error: bool = False
    duration_ms: int | None = None


class ApprovalRequest(_Event):
    type: Literal["approval_request"] = "approval_request"
    approval_id: str
    call_id: str
    name: str
    arguments: dict[str, Any]
    risk: str
    reason: str = ""
    risk_label: str | None = None  # "Purchase", "Sends", "Runs code"... (Tool.describe_fn or the effective risk)
    target: str | None = None      # short human label of what is acted on, e.g. "Place order"
    # why this asks although rules or modes would let it run: the chat read outside content (ADR 0018)
    untrusted: str | None = None


class UserInterjection(_Event):
    """A steer message the model received in the middle of a running reply."""

    type: Literal["user_interjection"] = "user_interjection"
    text: str


class Usage(_Event):
    type: Literal["usage"] = "usage"
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # context meter (#131), set when the model's context length is known: tokens in its context after this call,
    # what it reads at once, the share in percent, and a plain warning from 85%
    context_used: int | None = None
    context_length: int | None = None
    context_percent: int | None = None
    context_warning: str | None = None


class Error(_Event):
    type: Literal["error"] = "error"
    message: str
    recoverable: bool = True


class Done(_Event):
    type: Literal["done"] = "done"
    content: str = ""
    message_id: str | None = None
    cancelled: bool = False
    # memories this reply had in mind (sentient.memory.sources): what was in its prompt or returned by memory tools
    memory_sources: list[dict[str, Any]] = Field(default_factory=list)


AgentEvent = (
    TextDelta
    | ThinkingDelta
    | ToolCallEvent
    | ToolProgress
    | ToolResultEvent
    | ApprovalRequest
    | UserInterjection
    | Usage
    | Error
    | Done
)

PROGRESS_KINDS = ("stdout", "stderr", "status", "frame", "subagent")


def tool_progress_event(call_id: str, name: str, payload: dict, **ev: Any) -> ToolProgress:
    """Build a ToolProgress from a free-form ``ctx.progress`` payload. Unknown keys go into ``data``."""
    payload = dict(payload or {})
    kind = payload.pop("kind", "status")
    if kind not in PROGRESS_KINDS:
        kind = "status"
    text = payload.pop("text", None)
    image = payload.pop("image", None)
    data = payload.pop("data", None)
    if payload:  # extra keys are kept rather than dropped
        data = {**(data if isinstance(data, dict) else ({"value": data} if data is not None else {})), **payload}
    return ToolProgress(
        call_id=call_id,
        name=name,
        kind=kind,
        text=None if text is None else str(text),
        image=None if image is None else str(image),
        data=data,
        **ev,
    )
