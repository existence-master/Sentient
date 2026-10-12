"""Can a model answer right now? A quick check that never calls the model (#312).

A chat can keep its own model (``sessions.model``). Before each turn that uses it, ``unavailable`` says in one plain
sentence why it can't answer (Claude Code turned off or missing, an Ollama model that isn't downloaded, a provider key
or sign-in that isn't set), so the turn can use the default model instead and say so. Anything this can't tell
without calling the model (Claude Code not signed in, a provider that is down) shows up when the call fails, and the
agent loop falls back then.
"""

from __future__ import annotations

import re
import time

import httpx

from sentient.config.schema import SentientConfig
from sentient.llm import chatgpt, claude_code
from sentient.llm.checkup import LABELS, LOCAL, OLLAMA
from sentient.llm.presets import _installed, _ollama_base, key_set
from sentient.llm.provider import CLAUDE_CODE, provider_config

TAGS_TTL_S = 15.0  # how long the list of downloaded Ollama models is reused between turns
TAGS_TIMEOUT_S = 2.0
_tags: dict[str, tuple[float, list[str] | None]] = {}
MODEL_ID = re.compile(r"^[A-Za-z0-9_.\-]+/\S{1,200}$")
MODEL_ID_HINT = "A model looks like provider/model, for example ollama_chat/qwen3:8b or claude-code/sonnet."


def model_id(value: str | None) -> str | None:
    """A chat's model as given, or None for "follow the default". Raises ``ValueError`` (with a plain hint) when it
    doesn't look like ``provider/model``."""
    value = (value or "").strip()
    if not value:
        return None
    if not MODEL_ID.match(value):
        raise ValueError(MODEL_ID_HINT)
    return value


def label(model: str | None) -> str:
    """A short name people recognise: ``qwen3:8b``, ``Claude Sonnet``, ``gpt-5-mini``."""
    if not model:
        return "the default model"
    prefix, _, name = model.partition("/")
    if not name:
        return model
    if prefix == CLAUDE_CODE and name in {"sonnet", "opus", "haiku", "fable"}:
        return f"Claude {name.capitalize()}"
    return name


async def _installed_models(base: str) -> list[str] | None:
    """Ollama's downloaded models, or None when Ollama can't be reached (then nothing is known)."""
    hit = _tags.get(base)
    if hit is not None and time.monotonic() - hit[0] < TAGS_TTL_S:
        return hit[1]
    try:
        async with httpx.AsyncClient(timeout=TAGS_TIMEOUT_S) as client:
            r = await client.get(f"{base}/api/tags")
            r.raise_for_status()
            names: list[str] | None = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        names = None
    _tags[base] = (time.monotonic(), names)
    return names


def forget_cache() -> None:
    _tags.clear()


async def unavailable(config: SentientConfig, model: str) -> str | None:
    """Why ``model`` can't answer a chat right now, as a plain sentence, or None when it looks usable."""
    prefix, _, name = model.partition("/")
    if not name:
        return None
    if prefix == CLAUDE_CODE:
        return claude_code.refusal(config, "primary")
    if prefix in OLLAMA:
        installed = await _installed_models(_ollama_base(config, prefix))
        if installed is not None and not _installed(name, installed):
            return f"{name} isn't downloaded on this computer. Download it in Settings > Models, or pick another model."
        return None
    if prefix in LOCAL:
        return None
    if prefix == chatgpt.PREFIX:
        return None if chatgpt.signed_in() else "You're not signed in with ChatGPT. Sign in under Settings > Models."
    pc = provider_config(config, prefix)
    if pc and pc.api_key_env and not key_set(config, prefix):
        provider = LABELS.get(prefix, prefix)
        return f"No {provider} key is set. Add it in Settings > Models."
    return None
