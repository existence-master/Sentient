"""Provider-agnostic LLM access.

One adapter (LiteLLM SDK) reaches Ollama, OpenAI, Anthropic, Gemini, OpenRouter,
Groq, local OpenAI-compatible servers, and so on, with native tool calling.

The rest of Sentient never mentions a provider: it asks for a *role*
(``primary``, ``fast``, ``embedding``...) and the provider resolves the role
to a model string plus a fallback chain from config.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from sentient import secrets
from sentient.config.schema import SentientConfig

log = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass
class StreamChunk:
    """What the provider yields while streaming."""

    text: str = ""
    thinking: str = ""
    done: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    cost: float | None = None  # US dollars for this call when the provider knows the model's price


class LLMProvider(Protocol):
    """``role`` picks config (reasoning effort, fallbacks); ``model`` overrides the role's model."""

    async def stream(
        self, role: str, messages: list[dict], tools: list[dict] | None = None, *, model: str | None = None
    ) -> AsyncIterator[StreamChunk]: ...

    async def complete_json(self, role: str, messages: list[dict], *, model: str | None = None) -> Any: ...

    async def complete_text(self, role: str, messages: list[dict], *, model: str | None = None) -> str: ...

    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...

    def model_for(self, role: str) -> str: ...


def _provider_prefix(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else ""


def _response_cost(litellm: Any, response: Any, model: str) -> float | None:
    """Price of one completion from LiteLLM's bundled price list; None when the model's price is unknown."""
    if response is None or not getattr(response, "usage", None):
        return None
    try:
        cost = litellm.completion_cost(completion_response=response, model=model)
    except Exception:  # unknown or local model: no price
        return None
    return float(cost) if cost else None


CACHE_PREFIXES = {"anthropic"}


def apply_prompt_cache(model: str, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[dict], list[dict] | None]:
    """Anthropic prompt caching: mark the system prompt and the tool list with ``cache_control``.

    Returns new lists (the caller's are not mutated). Other providers get their inputs back unchanged.
    """
    if _provider_prefix(model) not in CACHE_PREFIXES:
        return messages, tools
    marker = {"type": "ephemeral"}
    out: list[dict] = []
    marked = False
    for m in messages:
        if not marked and m.get("role") == "system":
            content = m.get("content")
            if isinstance(content, str) and content:
                m = {**m, "content": [{"type": "text", "text": content, "cache_control": marker}]}
                marked = True
            elif isinstance(content, list) and content and isinstance(content[-1], dict):
                blocks = [*content[:-1], {**content[-1], "cache_control": marker}]
                m = {**m, "content": blocks}
                marked = True
        out.append(m)
    new_tools = tools
    if tools:
        new_tools = [*tools[:-1], {**tools[-1], "cache_control": marker}]
    return out, new_tools


class LiteLLMProvider:
    def __init__(self, config: SentientConfig):
        self.config = config

    # ------------------------------------------------------------------ resolution
    def model_for(self, role: str) -> str:
        roles = self.config.models.roles
        model = getattr(roles, role, None)
        if not model:
            model = roles.primary
        return model

    def _chain(self, role: str, model: str | None = None) -> list[str]:
        if model:  # an explicit pick is strict: no silent fallback to something else
            return [model]
        primary = self.model_for(role)
        fallbacks = self.config.models.fallbacks.get(role) or (
            self.config.models.fallbacks.get("primary", []) if role in {"planner", "executor", "vision", "voice"} else []
        )
        return [primary, *[m for m in fallbacks if m != primary]]

    def _kwargs_for(self, model: str, role: str | None = None) -> dict:
        prefix = _provider_prefix(model)
        pc = self.config.models.providers.get(prefix)
        kwargs: dict[str, Any] = {"timeout": self.config.models.request_timeout_s}
        effort = self.config.models.reasoning.get(role or "")
        temp = self.config.models.temperature.get(role or "")
        if temp is not None:
            kwargs["temperature"] = temp
        local = prefix in {"ollama", "ollama_chat"}
        if effort and (effort != "none" or local):
            # Ollama maps this to its `think` flag (none -> off). Cloud models already answer without
            # extended thinking by default, and some reject "none", so it is only sent to local models.
            kwargs["reasoning_effort"] = effort
        if pc:
            if pc.api_base:
                kwargs["api_base"] = pc.api_base
            key = secrets.get_secret(prefix, pc.api_key_env)
            if key:
                kwargs["api_key"] = key
        return kwargs

    # ------------------------------------------------------------------ streaming chat
    async def stream(
        self, role: str, messages: list[dict], tools: list[dict] | None = None, *, model: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        import litellm

        litellm.drop_params = True
        litellm.suppress_debug_info = True
        last_error: Exception | None = None
        override = model
        for model in self._chain(role, override):
            try:
                kwargs = self._kwargs_for(model, role)
                sent_messages, sent_tools = apply_prompt_cache(model, messages, tools)
                if sent_tools:
                    kwargs["tools"] = sent_tools
                    kwargs["tool_choice"] = "auto"
                response = await litellm.acompletion(
                    model=model, messages=sent_messages, stream=True, **kwargs
                )
                chunks: list[Any] = []
                in_think = False
                emitted = False
                async for chunk in response:
                    chunks.append(chunk)
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        emitted = True
                        yield StreamChunk(thinking=reasoning, model=model)
                    text = delta.content or ""
                    if not text:
                        continue
                    # Models like qwen3 emit <think>...</think> inline; route it to the thinking stream.
                    pieces, in_think = split_think(text, in_think)
                    emitted = True
                    for piece, is_think in pieces:
                        if is_think:
                            yield StreamChunk(thinking=piece, model=model)
                        else:
                            yield StreamChunk(text=piece, model=model)
                full = litellm.stream_chunk_builder(chunks, messages=messages)
                tool_calls: list[ToolCall] = []
                msg = full.choices[0].message if full and full.choices else None
                for tc in (getattr(msg, "tool_calls", None) or []):
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        try:
                            args = parse_json_loose(tc.function.arguments or "")
                        except ValueError:
                            args = {"_raw": tc.function.arguments}
                    if not isinstance(args, dict):
                        args = {"_raw": tc.function.arguments}
                    tool_calls.append(ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:12]}", name=tc.function.name, arguments=args))
                usage = {}
                if full is not None and getattr(full, "usage", None):
                    usage = {
                        "prompt_tokens": full.usage.prompt_tokens or 0,
                        "completion_tokens": full.usage.completion_tokens or 0,
                    }
                yield StreamChunk(
                    done=True, tool_calls=tool_calls, usage=usage, model=model, cost=_response_cost(litellm, full, model)
                )
                return
            except Exception as exc:
                last_error = exc
                log.warning("model %s failed for role %s: %s", model, role, exc)
                if "emitted" in locals() and emitted:
                    # part of a reply already reached the user; switching models would duplicate it
                    raise ProviderError(f"{model} stopped mid-reply: {exc}") from exc
                continue
        raise ProviderError(f"All models failed for role '{role}': {last_error}")

    # ------------------------------------------------------------------ non-streaming JSON
    async def complete_text(self, role: str, messages: list[dict], *, model: str | None = None) -> str:
        import re

        import litellm

        litellm.drop_params = True
        last_error: Exception | None = None
        override = model
        for model in self._chain(role, override):
            try:
                kwargs = self._kwargs_for(model, role)
                sent, _ = apply_prompt_cache(model, messages)
                resp = await litellm.acompletion(model=model, messages=sent, **kwargs)
                text = resp.choices[0].message.content or ""
                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            except Exception as exc:
                last_error = exc
                log.warning("model %s failed for role %s: %s", model, role, exc)
        raise ProviderError(f"All models failed for role '{role}': {last_error}")

    async def complete_json(self, role: str, messages: list[dict], *, model: str | None = None) -> Any:
        import litellm

        litellm.drop_params = True
        last_error: Exception | None = None
        override = model
        for model in self._chain(role, override):
            try:
                kwargs = self._kwargs_for(model, role)
                # Ollama's JSON mode corrupts qwen3 output ({"{"name": ...); ask for plain text and parse loosely
                if _provider_prefix(model) not in {"ollama", "ollama_chat"}:
                    kwargs["response_format"] = {"type": "json_object"}
                sent, _ = apply_prompt_cache(model, messages)
                resp = await litellm.acompletion(model=model, messages=sent, **kwargs)
                text = resp.choices[0].message.content or ""
                return parse_json_loose(text)
            except Exception as exc:
                last_error = exc
                log.warning("model %s failed for role %s: %s", model, role, exc)
        raise ProviderError(f"All models failed for role '{role}': {last_error}")

    # ------------------------------------------------------------------ embeddings
    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        import litellm

        model = model or self.model_for("embedding")
        kwargs = self._kwargs_for(model)
        kwargs.pop("timeout", None)
        resp = await litellm.aembedding(model=model, input=texts, **kwargs)
        return [d["embedding"] for d in resp.data]


# ---------------------------------------------------------------------- helpers
def split_think(text: str, in_think: bool) -> tuple[list[tuple[str, bool]], bool]:
    """Split a text delta on <think> / </think> boundaries.

    Returns ``([(piece, is_thinking), ...], state_after)``. Pieces are never empty.
    Tags split across two deltas are not handled; in practice Ollama emits them whole.
    """
    out: list[tuple[str, bool]] = []
    pos = 0
    state = in_think
    while pos < len(text):
        tag = "</think>" if state else "<think>"
        idx = text.find(tag, pos)
        if idx == -1:
            out.append((text[pos:], state))
            break
        if idx > pos:
            out.append((text[pos:idx], state))
        state = not state
        pos = idx + len(tag)
    return out, state


def _json_candidates(text: str) -> list[Any]:
    """Every complete JSON object/array embedded in ``text`` (outermost first)."""
    decoder = json.JSONDecoder()
    found: list[Any] = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                obj, end = decoder.raw_decode(text, i)
                found.append(obj)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return found


def parse_json_loose(text: str, expect_keys: tuple[str, ...] | list[str] | None = None) -> Any:
    """Parse JSON from a model reply, tolerating think tags, code fences, prose around the JSON,
    and corrupted wrappers like ``{"{"name": ...}`` that Ollama JSON mode produces with qwen3.

    With several candidates, prefers an object containing ``expect_keys``, then the largest object,
    then the largest array. Falls back to the optional ``json-repair`` package.
    """
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    candidates = _json_candidates(cleaned)
    if candidates:
        def rank(obj: Any) -> tuple[int, int, int]:
            keys_hit = sum(1 for k in (expect_keys or ()) if isinstance(obj, dict) and k in obj)
            return (keys_hit, 1 if isinstance(obj, dict) else 0, len(json.dumps(obj, default=str)))

        return max(candidates, key=rank)
    try:
        import json_repair  # optional

        repaired = json_repair.loads(cleaned)
        if repaired not in ("", None, [], {}):
            return repaired
    except Exception:
        pass
    raise ValueError(f"Model did not return JSON: {text[:200]!r}")
