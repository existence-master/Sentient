"""Where a model runs, and how it is paid for (#148).

``runs_here`` decides, without any network call, whether a request to a model stays on this computer. It is strict
on purpose, because Local only (``models.local_only``) must never let a request out:

- only a local runtime counts (Ollama, LM Studio, llamafile, vLLM). An OpenAI-compatible address is treated as the
  cloud even on this computer, because it may be a proxy that forwards to one;
- its address must be this computer (``localhost``, ``127.x.x.x``, ``::1``), read from config, else from the
  environment variable LiteLLM itself would read, else LiteLLM's own default. A name that would need a lookup
  counts as the cloud;
- an Ollama cloud model (``gpt-oss:120b-cloud``) is the cloud: Ollama sends it to ollama.com.

``billing`` sorts models for the cost meter: ``local`` (free), ``plan`` (Claude Code and ChatGPT plans, paid by the
month, no price per call) and ``paid`` (an API key, priced per token from LiteLLM's bundled price list).
"""

from __future__ import annotations

import contextlib
import contextvars
import ipaddress
import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

# local runtimes and the address LiteLLM uses when none is set (None: the call can't be made without one)
LOCAL_RUNTIMES: dict[str, str | None] = {
    "ollama": "http://localhost:11434",
    "ollama_chat": "http://localhost:11434",
    "lm_studio": None,
    "llamafile": "http://127.0.0.1:8080/v1",
    "hosted_vllm": None,
    "vllm": "in-process",  # LiteLLM runs the vLLM library inside this process
}
# the environment variables LiteLLM reads for the address when none is passed
ENV_BASES = {
    "ollama": "OLLAMA_API_BASE",
    "ollama_chat": "OLLAMA_API_BASE",
    "lm_studio": "LM_STUDIO_API_BASE",
    "llamafile": "LLAMAFILE_API_BASE",
    "hosted_vllm": "HOSTED_VLLM_API_BASE",
}
PLAN_PREFIXES = {"claude-code", "chatgpt"}
LOCAL_SOURCE_DEFAULT = "background"

# what the model calls inside are for (chat, task...), for the usage table; set by the agent loop
_source: contextvars.ContextVar[str | None] = contextvars.ContextVar("sentient_usage_source", default=None)


@contextlib.contextmanager
def usage_source(source: str | None) -> Iterator[None]:
    """Model calls inside are counted under ``source`` (``chat``, ``task``...) in the usage table."""
    token = _source.set(source)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):  # reset from another context: an async generator moved tasks
            _source.reset(token)


def current_source() -> str:
    return _source.get() or LOCAL_SOURCE_DEFAULT


def _prefix(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else ""


def _loopback(base: str) -> bool:
    if base == "in-process":
        return True
    try:
        host = (urlsplit(base if "//" in base else f"http://{base}").hostname or "").strip().lower()
    except ValueError:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # any other name could point anywhere


def ollama_cloud(model: str) -> bool:
    """An Ollama cloud model (``gpt-oss:120b-cloud``, ``glm-4.6:cloud``): Ollama forwards it to ollama.com."""
    tag = model.split("/", 1)[-1].lower()
    return tag.endswith(":cloud") or tag.endswith("-cloud")


def runs_here(config: Any, model: str) -> bool:
    """True only when a request to ``model`` provably stays on this computer (see the module notes)."""
    prefix = _prefix(model or "")
    if prefix not in LOCAL_RUNTIMES:
        return False
    if prefix in {"ollama", "ollama_chat"} and ollama_cloud(model):
        return False
    pc = config.models.providers.get(prefix)
    env = ENV_BASES.get(prefix)
    base = (pc.api_base if pc is not None else None) or (os.environ.get(env) if env else None) or LOCAL_RUNTIMES[prefix]
    return bool(base) and _loopback(str(base))


def billing(model: str) -> str:
    """``local``, ``plan`` or ``paid``, from the model's name alone (the usage table keeps no address)."""
    prefix = _prefix(model or "")
    if prefix in PLAN_PREFIXES:
        return "plan"
    if prefix in LOCAL_RUNTIMES and not ollama_cloud(model):
        return "local"
    return "paid"


def short_name(model: str) -> str:
    """A name people recognise: ``gpt-4o``, ``qwen3:8b``, ``Claude Sonnet``."""
    prefix, _, name = (model or "").partition("/")
    if prefix == "claude-code" and name in {"sonnet", "opus", "haiku", "fable"}:
        return f"Claude {name.capitalize()}"
    return name or model


def local_only_message(model: str) -> str:
    """The plain refusal when Local only stops a request to ``model``."""
    return (f"Local only is on, so nothing was sent to {short_name(model)}, which runs outside this computer. "
            "Pick a local model, or switch to May use cloud in the model menu.")
