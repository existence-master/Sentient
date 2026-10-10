"""Context meter (#131): how full the model's context is after each model call, and a plain warning near the limit.

``used`` is the prompt (the larger of what the provider reported and a local count, because Ollama reports only the
part of a prompt it had not cached) plus the reply. ``length`` is what the model reads at once for that role:
Ollama's ``num_ctx`` or a cloud model's input window (``LLMProvider.context_window``). Nothing is ever blocked.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

WARN_PERCENT = 85
LOCAL_PREFIXES = {"ollama", "ollama_chat"}  # where a longer context length in Settings helps


def count_prompt_tokens(messages: list[dict], tools: list[dict] | None = None) -> int:
    """About how many tokens ``messages`` and ``tools`` take, with LiteLLM's bundled tokenizer (offline). 0 on failure."""
    try:
        import litellm

        # no model name: LiteLLM's default tokenizer, never one it would download
        return int(litellm.token_counter(model="", messages=messages, tools=tools, use_default_image_token_count=True))
    except Exception as exc:
        log.debug("token count failed: %s", exc)
        return 0


def warning(percent: int, model: str, source: str) -> str | None:
    if percent < WARN_PERCENT:
        return None
    name = model.split("/", 1)[-1] or model
    local = model.split("/", 1)[0] in LOCAL_PREFIXES
    if source == "task":
        text = (f"This task's work is getting long for {name} ({percent}% of what it reads at once). "
                "Earlier steps may be left out.")
        return text + (" A longer context length in Settings > Models helps." if local else "")
    text = (f"This chat is getting long for {name} ({percent}% of what it reads at once). "
            "Older messages may be left out")
    return text + (": start a new chat, or set a longer context length in Settings > Models." if local
                   else ", so a new chat works best.")


def meter(used: int, length: int, model: str, source: str) -> dict[str, Any]:
    """The ``usage`` event's context fields."""
    percent = round(100 * used / length) if length > 0 else 0
    return {"context_used": used, "context_length": length, "context_percent": percent,
            "context_warning": warning(percent, model, source)}


async def measure(
    llm: Any, role: str, model: str, messages: list[dict], tools: list[dict] | None, usage: dict, source: str
) -> dict[str, Any]:
    """The ``usage`` event's context fields for one model call, or {} when the model's context length is unknown."""
    window = getattr(llm, "context_window", None)
    if not callable(window):
        return {}
    try:
        length = await window(role, model)
    except Exception as exc:
        log.debug("context length for %s unknown: %s", model, exc)
        return {}
    if not length:
        return {}
    counted = await asyncio.to_thread(count_prompt_tokens, messages, tools)  # large chats take a moment
    prompt = max(int(usage.get("prompt_tokens") or 0), counted)
    return meter(prompt + int(usage.get("completion_tokens") or 0), int(length), model, source)
