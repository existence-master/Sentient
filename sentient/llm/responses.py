"""A small streaming client for OpenAI's Responses API, used for ChatGPT plan models (issue #205).

LiteLLM can bridge chat calls to the Responses API, but plan usage needs every request streamed (including the
text and JSON jobs that don't stream), ``store: false``, no system items, no sampling or length settings, and tool
call ids taken from ``call_id``. This module does exactly that and nothing more: chat messages in, simple events out.
Rules: https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

USAGE_LIMIT = ("Usage limit reached. You've used what your ChatGPT plan allows Sentient for now. See or change the "
               "limit under Manage usage: https://chatgpt.com/settings/usage")
NOT_ELIGIBLE = "This ChatGPT account can't share its plan with other apps. Plan usage needs ChatGPT Plus or Pro."
SIGNED_OUT = "ChatGPT signed you out. Sign in with ChatGPT again in Settings > Models."


class ResponsesError(RuntimeError):
    """The Responses API refused or broke off a request. The message is a plain sentence."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------- request
def _text_parts(content: Any, role: str) -> list[dict]:
    kind = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": kind, "text": content}] if content else []
    parts: list[dict] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and part.get("text"):
            parts.append({"type": kind, "text": part["text"]})
        elif part.get("type") == "image_url" and role != "assistant":
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if url:
                parts.append({"type": "input_image", "image_url": url})
    return parts


def to_input(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Chat Completions messages as Responses ``instructions`` and ``input`` items.

    Plan usage refuses ``system`` items: the leading system prompts become ``instructions`` and a later system note
    becomes a ``developer`` message in its place.
    """
    instructions: list[str] = []
    items: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            text = content if isinstance(content, str) else " ".join(
                p.get("text", "") for p in content or [] if isinstance(p, dict))
            if text and not items:
                instructions.append(text)
            elif text:
                items.append({"type": "message", "role": "developer", "content": [{"type": "input_text", "text": text}]})
        elif role == "tool":
            out = content if isinstance(content, str) else json.dumps(content)
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id") or "", "output": out})
        elif role in {"user", "assistant"}:
            parts = _text_parts(content, role)
            if parts:
                items.append({"type": "message", "role": role, "content": parts})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                items.append({"type": "function_call", "call_id": tc.get("id") or "", "name": fn.get("name") or "",
                              "arguments": args if isinstance(args, str) else json.dumps(args or {})})
    return ("\n\n".join(instructions) or None), items


def to_tools(tools: list[dict] | None) -> list[dict]:
    """Chat Completions function tools as Responses function tools (flat, not nested under ``function``)."""
    out = []
    for t in tools or []:
        fn = t.get("function") if t.get("type") == "function" else None
        if not fn:
            continue
        tool = {"type": "function", "name": fn["name"],
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}}}
        if fn.get("description"):
            tool["description"] = fn["description"]
        out.append(tool)
    return out


def request_body(model: str, messages: list[dict], tools: list[dict] | None = None, *,
                 reasoning_effort: str | None = None) -> dict:
    """The body plan usage accepts: streamed, never stored, no temperature, token limit or previous response."""
    instructions, items = to_input(messages)
    body: dict[str, Any] = {"model": model, "input": items, "stream": True, "store": False}
    if instructions:
        body["instructions"] = instructions
    sent_tools = to_tools(tools)
    if sent_tools:
        body["tools"] = sent_tools
        body["tool_choice"] = "auto"
    if reasoning_effort and reasoning_effort != "none":
        body["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
    return body


# ---------------------------------------------------------------------------- errors
def problem_text(status: int, code: str, detail: str) -> str:
    """A plain sentence for a refused request (https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery)."""
    if code == "subscription_sharing_usage_limit_exceeded" or status == 429:
        return USAGE_LIMIT
    if code in {"subscription_sharing_user_not_eligible", "chatpass_v2_scope_not_authorized"}:
        return NOT_ELIGIBLE
    if code in {"subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable"} or status == 503:
        return "ChatGPT plan usage is unavailable right now. Try again in a little while."
    if status == 401 or code == "subscription_sharing_invalid_user":
        return SIGNED_OUT
    if code == "subscription_sharing_unsupported_capability":
        return f"ChatGPT plans don't support part of this request ({detail or 'an unsupported feature'})."
    if status:
        return f"ChatGPT answered with an error ({status}): {detail or code}"
    return f"ChatGPT stopped the reply: {detail or code or 'unknown error'}"


def _error_fields(err: Any) -> tuple[str, str]:
    if isinstance(err, dict):
        return str(err.get("code") or err.get("type") or ""), str(err.get("message") or "")[:200]
    return "", str(err or "")[:200]


def _problem(status: int, raw: bytes) -> str:
    try:
        data = json.loads(raw)
    except ValueError:
        return problem_text(status, "", raw[:200].decode("utf-8", "replace"))
    code, detail = _error_fields(data.get("error") if isinstance(data, dict) else data)
    return problem_text(status, code, detail)


# ---------------------------------------------------------------------------- streaming
async def _events(response: httpx.Response) -> AsyncIterator[dict]:
    """Server-sent events as dicts. Only ``data:`` lines matter; each carries its own ``type``."""
    data: list[str] = []

    def parse(raw: str) -> dict | None:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return event if isinstance(event, dict) else None

    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line.strip() and data:
            raw, data = "\n".join(data), []
            if raw == "[DONE]":
                return
            event = parse(raw)
            if event is not None:
                yield event
    if data:
        event = parse("\n".join(data))
        if event is not None:
            yield event


async def open_stream(url: str, headers: dict[str, str], body: dict, *, timeout: float) -> AsyncIterator[dict]:
    """Send one streamed request. A refused request raises here, before any event; then iterate the result for

    ``{"text": ...}``, ``{"thinking": ...}`` and finally
    ``{"done": True, "tool_calls": [{"id", "name", "arguments"}], "usage": {...}, "model": ...}``.
    """
    http = httpx.AsyncClient(timeout=timeout)
    try:
        r = await http.send(http.build_request("POST", url, headers=headers, json=body), stream=True)
    except httpx.HTTPError as exc:
        await http.aclose()
        raise ResponsesError("Couldn't reach ChatGPT. Check your internet connection.") from exc
    if r.status_code >= 400:
        raw = await r.aread()
        await r.aclose()
        await http.aclose()
        raise ResponsesError(_problem(r.status_code, raw), r.status_code)
    return _read(http, r)


async def _read(http: httpx.AsyncClient, r: httpx.Response) -> AsyncIterator[dict]:
    calls: dict[str, dict] = {}  # output item id -> {"id": call_id, "name", "arguments"}

    def call_for(key: str) -> dict:
        if key not in calls:
            calls[key] = {"id": "", "name": "", "arguments": ""}
        return calls[key]

    try:
        async for ev in _events(r):
            kind = ev.get("type") or ""
            if kind == "response.output_text.delta":
                if ev.get("delta"):
                    yield {"text": ev["delta"]}
            elif kind in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
                if ev.get("delta"):
                    yield {"thinking": ev["delta"]}
            elif kind in {"response.output_item.added", "response.output_item.done"}:
                item = ev.get("item") or {}
                if item.get("type") != "function_call":
                    continue
                call = call_for(item.get("id") or item.get("call_id") or str(ev.get("output_index")))
                call["id"] = item.get("call_id") or call["id"]
                call["name"] = item.get("name") or call["name"]
                if item.get("arguments"):
                    call["arguments"] = item["arguments"]
            elif kind == "response.function_call_arguments.delta":
                call_for(ev.get("item_id") or str(ev.get("output_index")))["arguments"] += ev.get("delta") or ""
            elif kind == "response.function_call_arguments.done":
                if ev.get("arguments") is not None:
                    call_for(ev.get("item_id") or str(ev.get("output_index")))["arguments"] = ev["arguments"]
            elif kind in {"response.completed", "response.incomplete"}:
                resp = ev.get("response") or {}
                usage = resp.get("usage") or {}
                yield {"done": True, "tool_calls": [c for c in calls.values() if c["name"]],
                       "model": resp.get("model") or "",
                       "usage": {"prompt_tokens": int(usage.get("input_tokens") or 0),
                                 "completion_tokens": int(usage.get("output_tokens") or 0)}}
                return
            elif kind in {"response.failed", "error"}:
                code, detail = _error_fields((ev.get("response") or {}).get("error") or ev.get("error") or ev)
                raise ResponsesError(problem_text(0, code, detail))
        raise ResponsesError("ChatGPT ended the reply early. Please try again.")
    finally:
        await r.aclose()
        await http.aclose()
