"""Model configuration and secrets. OWNER: core. Contract: docs/API.md section 3.

Everything a non-technical user needs to switch models: which providers exist,
whether a key is set, which local models are installed, a one-click test, role
assignment, fallback chains, pulling Ollama models, and keychain secrets.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sentient import secrets
from sentient.gateway.deps import AUTH, get_core
from sentient.llm import claude_code, connect, presets
from sentient.llm.presets import PresetError
from sentient.llm.provider import provider_config

router = APIRouter(prefix="/api", tags=["models"], dependencies=AUTH)

PROVIDERS: list[dict[str, Any]] = [
    {"id": "ollama_chat", "label": "Ollama (local)", "kind": "local", "key_required": False,
     "docs_url": "https://ollama.com/download", "suggested": ["ollama_chat/qwen3:8b", "ollama_chat/qwen3:14b", "ollama_chat/llama3.1:8b", "ollama_chat/gpt-oss:20b"]},
    {"id": "lm_studio", "label": "LM Studio (local)", "kind": "local", "key_required": False,
     "docs_url": "https://lmstudio.ai/", "suggested": []},
    {"id": "anthropic", "label": "Anthropic", "kind": "cloud", "key_required": True,
     "docs_url": "https://console.anthropic.com/settings/keys", "suggested": ["anthropic/claude-sonnet-5-5", "anthropic/claude-opus-5-5", "anthropic/claude-haiku-5-5"]},
    {"id": "openai", "label": "OpenAI", "kind": "cloud", "key_required": True,
     "docs_url": "https://platform.openai.com/api-keys", "suggested": ["openai/gpt-5", "openai/gpt-5-mini", "openai/text-embedding-3-small"]},
    {"id": "gemini", "label": "Google Gemini", "kind": "cloud", "key_required": True,
     "docs_url": "https://aistudio.google.com/apikey", "suggested": ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash", "gemini/gemini-embedding-001"]},
    {"id": "openrouter", "label": "OpenRouter", "kind": "cloud", "key_required": True,
     "docs_url": "https://openrouter.ai/keys", "suggested": ["openrouter/anthropic/claude-sonnet-5.5", "openrouter/meta-llama/llama-4-maverick"]},
    {"id": "groq", "label": "Groq", "kind": "cloud", "key_required": True,
     "docs_url": "https://console.groq.com/keys", "suggested": ["groq/llama-3.3-70b-versatile"]},
    {"id": "mistral", "label": "Mistral", "kind": "cloud", "key_required": True,
     "docs_url": "https://console.mistral.ai/api-keys", "suggested": ["mistral/mistral-large-latest"]},
    {"id": "deepseek", "label": "DeepSeek", "kind": "cloud", "key_required": True,
     "docs_url": "https://platform.deepseek.com/api_keys", "suggested": ["deepseek/deepseek-chat"]},
    {"id": "xai", "label": "xAI", "kind": "cloud", "key_required": True,
     "docs_url": "https://console.x.ai/", "suggested": ["xai/grok-4"]},
    {"id": "nous", "label": "Nous Portal", "kind": "cloud", "key_required": True,
     "docs_url": "https://portal.nousresearch.com/", "suggested": []},
]


def _provider_key_status(s, pid: str) -> tuple[bool, str | None]:
    pc = provider_config(s.config, pid)
    env = pc.api_key_env if pc else None
    import os

    kr_val = secrets.get_secret(pid, None)
    if kr_val:
        return True, "keychain"
    if env and os.environ.get(env):
        return True, "env"
    return False, None


@router.get("/models/providers")
async def providers(request: Request):
    s = get_core(request)
    out = []
    for p in PROVIDERS:
        key_set, _ = _provider_key_status(s, p["id"])
        pc = s.config.models.providers.get(p["id"])
        out.append({**p, "key_set": key_set if p["key_required"] else True, "api_base": pc.api_base if pc else None})
    return out


def _ollama_base(s) -> str:
    pc = s.config.models.providers.get("ollama_chat") or s.config.models.providers.get("ollama")
    return (pc.api_base if pc and pc.api_base else "http://localhost:11434").rstrip("/")


@router.get("/models/local")
async def local_models(request: Request):
    s = get_core(request)
    result: dict[str, Any] = {"ollama": {"reachable": False, "models": []}, "lm_studio": {"reachable": False, "models": []}}
    async with httpx.AsyncClient(timeout=3) as client:
        try:
            r = await client.get(f"{_ollama_base(s)}/api/tags")
            r.raise_for_status()
            models = []
            for m in r.json().get("models", []):
                details = m.get("details") or {}
                family = (details.get("family") or "").lower()
                name = m.get("name", "")
                models.append(
                    {
                        "name": name,
                        "size": m.get("size", 0),
                        "family": family,
                        "parameter_size": details.get("parameter_size"),
                        "is_embedding": "embed" in name or "bert" in family or "minilm" in name,
                    }
                )
            # capabilities from /api/show (completion, tools, thinking, vision, embedding). A hint only:
            # some old model downloads report "tools" but still fail tool calls; POST /models/test is authoritative.
            import asyncio

            async def _caps(name: str) -> list[str]:
                try:
                    shown = await client.post(f"{_ollama_base(s)}/api/show", json={"model": name})
                    shown.raise_for_status()
                    return list(shown.json().get("capabilities") or [])
                except Exception:
                    return []

            caps = await asyncio.gather(*[_caps(m["name"]) for m in models])
            for m, c in zip(models, caps, strict=False):
                m["capabilities"] = c
                if c:
                    m["is_embedding"] = "embedding" in c and "completion" not in c
            result["ollama"] = {"reachable": True, "models": models}
        except Exception:
            pass
        lm = s.config.models.providers.get("lm_studio")
        if lm and lm.api_base:
            try:
                r = await client.get(f"{lm.api_base.rstrip('/')}/models")
                r.raise_for_status()
                result["lm_studio"] = {"reachable": True, "models": [{"name": m.get("id")} for m in r.json().get("data", [])]}
            except Exception:
                pass
    return result


class TestBody(BaseModel):
    model: str
    role: str | None = None


@router.post("/models/test")
async def test_model(request: Request, body: TestBody):
    s = get_core(request)
    started = time.perf_counter()
    text = ""
    probe_tool = {
        "type": "function",
        "function": {
            "name": "report_ready",
            "description": "Call this to report that you are ready.",
            "parameters": {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]},
        },
    }
    try:
        tool_called = False
        with claude_code.attended():  # the user pressed Test: the only dry run Claude Code gets (ADR 0022)
            async for chunk in s.llm.stream(
                body.role or "fast",
                [{"role": "user", "content": "Call the report_ready tool with status 'ready'."}],
                [probe_tool],
                model=body.model,
            ):
                text += chunk.text
                if chunk.done and chunk.tool_calls:
                    tool_called = True
        return {
            "ok": True,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "reply": text.strip()[:200] or ("(called report_ready)" if tool_called else ""),
            "supports_tools": tool_called,
        }
    except Exception as exc:
        return {"ok": False, "latency_ms": int((time.perf_counter() - started) * 1000), "error": str(exc)[:500]}


@router.get("/models/claude-code")
async def claude_code_status(request: Request):
    """Claude through the user's own Claude Code (experimental, ADR 0022): turned on, installed, version. Runs
    ``claude --version`` only while it is turned on; never a model call, never reads Claude's login."""
    from sentient.config.schema import CLAUDE_CODE_MODELS

    return {**await claude_code.status(get_core(request).config), "models": list(CLAUDE_CODE_MODELS)}


class EmbedTestBody(BaseModel):
    model: str


@router.post("/models/test-embedding")
async def test_embedding(request: Request, body: EmbedTestBody):
    s = get_core(request)
    try:
        [vec] = await s.llm.embed(["hello world"], model=body.model)
        return {"ok": True, "dim": len(vec)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:500]}


ROLE_KEYS = {"primary", "fast", "planner", "executor", "embedding", "vision", "voice"}


class CheckupBody(BaseModel):
    roles: dict[str, str | None] | None = None


@router.post("/models/checkup")
async def model_checkup(request: Request, body: CheckupBody | None = None):
    """Check every role's model (or only ``roles``) and stream NDJSON progress. Never changes config."""
    from sentient.llm.checkup import run_checkup

    s = get_core(request)
    roles = body.roles if body else None
    if roles and set(roles) - ROLE_KEYS:
        raise HTTPException(400, f"unknown role {sorted(set(roles) - ROLE_KEYS)[0]}")

    async def gen():
        hardware = await s.hardware.get()
        async for event in run_checkup(s.config, s.llm, roles, hardware=hardware):
            yield json.dumps(event) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@router.put("/models/roles")
async def set_roles(request: Request, body: dict):
    s = get_core(request)
    cfg = s.config.model_copy(deep=True)
    roles = cfg.models.roles.model_dump()
    for k, v in body.items():
        if k not in ROLE_KEYS:
            raise HTTPException(400, f"unknown role {k}")
        if k in {"primary", "fast", "embedding"} and not v:
            raise HTTPException(400, f"role {k} cannot be empty")
        roles[k] = v or None
    cfg.models.roles = type(cfg.models.roles).model_validate(roles)
    s.save_config(cfg)
    return cfg.models.roles.model_dump()


@router.put("/models/fallbacks")
async def set_fallbacks(request: Request, body: dict[str, list[str]]):
    s = get_core(request)
    cfg = s.config.model_copy(deep=True)
    for role, chain in body.items():
        if role not in ROLE_KEYS:
            raise HTTPException(400, f"unknown role {role}")
        cfg.models.fallbacks[role] = [m for m in chain if m]
    s.save_config(cfg)
    return {"ok": True, "fallbacks": cfg.models.fallbacks}


# ----------------------------------------------------------------------------- presets (#212)
class PresetBody(BaseModel):
    name: str
    overwrite: bool = False


class RenameBody(BaseModel):
    name: str


def _preset_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except PresetError as exc:
        raise HTTPException(exc.status, exc.message) from None


async def _preset_await(coro):
    try:
        return await coro
    except PresetError as exc:
        raise HTTPException(exc.status, exc.message) from None


@router.get("/models/presets")
async def list_presets(request: Request):
    return await presets.listing(get_core(request))


@router.post("/models/presets")
async def save_preset(request: Request, body: PresetBody):
    """Save the current models as a preset of your own."""
    return _preset_call(presets.save_current, get_core(request), body.name, overwrite=body.overwrite)


@router.post("/models/presets/undo")
async def undo_preset(request: Request):
    return await _preset_await(presets.undo(get_core(request)))


@router.post("/models/presets/{name}/apply")
async def apply_preset(request: Request, name: str):
    return await _preset_await(presets.apply(get_core(request), name))


@router.patch("/models/presets/{name}")
async def rename_preset(request: Request, name: str, body: RenameBody):
    return _preset_call(presets.rename, get_core(request), name, body.name)


@router.delete("/models/presets/{name}")
async def delete_preset(request: Request, name: str):
    _preset_call(presets.delete, get_core(request), name)
    return {"ok": True}


class PullBody(BaseModel):
    name: str


@router.post("/models/ollama/pull")
async def pull_ollama(request: Request, body: PullBody):
    s = get_core(request)
    base = _ollama_base(s)

    async def gen():
        try:
            async with httpx.AsyncClient(timeout=None) as client, client.stream(
                "POST", f"{base}/api/pull", json={"model": body.name, "stream": True}
            ) as r:
                async for line in r.aiter_lines():
                    if line.strip():
                        yield line + "\n"
        except Exception as exc:
            yield json.dumps({"status": "error", "error": str(exc)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ----------------------------------------------------------------------------- connecting plans
@router.post("/models/connect/openrouter")
async def connect_openrouter(request: Request):
    """Start OpenRouter's browser sign-in. The window opens ``auth_url``; the key lands in the keychain."""
    return await get_core(request).connections.start_openrouter()


@router.get("/models/connect/openrouter/{state}")
async def connect_openrouter_status(request: Request, state: str):
    status = get_core(request).connections.flow_status(state)
    if status is None:
        raise HTTPException(404, "unknown sign-in")
    return status


@router.post("/models/connect/{provider}/check")
async def check_provider_key(request: Request, provider: str):
    if provider not in connect.CHECKABLE:
        raise HTTPException(404, f"no key check for {provider}")
    return await connect.check_key(get_core(request).config, provider)


@router.get("/models/catalog/{provider}")
async def provider_catalog(request: Request, provider: str):
    if provider not in connect.CATALOGS:
        raise HTTPException(404, f"no model list for {provider}")
    try:
        return await get_core(request).connections.catalog(provider)
    except connect.ConnectError as exc:
        raise HTTPException(502, str(exc)) from exc


# ----------------------------------------------------------------------------- secrets
@router.get("/secrets")
async def list_secrets(request: Request):
    s = get_core(request)
    out = []
    for p in PROVIDERS:
        if not p["key_required"]:
            continue
        key_set, source = _provider_key_status(s, p["id"])
        out.append({"name": p["id"], "set": key_set, "source": source, "kind": "provider"})
    names = getattr(s.integrations, "secret_names", None)
    if callable(names):
        for name in names():
            present = secrets.get_secret(name, None) is not None
            out.append({"name": name, "set": present, "source": "keychain" if present else None, "kind": "integration"})
    return out


class SecretBody(BaseModel):
    value: str


@router.put("/secrets/{name}")
async def put_secret(request: Request, name: str, body: SecretBody):
    if not body.value.strip():
        raise HTTPException(400, "empty value")
    if not secrets.set_secret(name, body.value.strip()):
        raise HTTPException(500, "the OS keychain is unavailable")
    get_core(request).connections.forget(name)  # a new key can mean a different account's models
    get_core(request).bus.publish("config.updated", {"sections": ["secrets"]})
    return {"ok": True}


@router.delete("/secrets/{name}")
async def delete_secret(request: Request, name: str):
    secrets.delete_secret(name)
    get_core(request).connections.forget(name)
    get_core(request).bus.publish("config.updated", {"sections": ["secrets"]})
    return {"ok": True}
