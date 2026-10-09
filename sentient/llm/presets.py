"""Model presets: switch every model role between local and cloud in one step (#212).

The built-in presets ("Local only", "Cloud", "Mixed") are generated from ``config/schema.py`` and the cloud keys that
are set; the user's own presets live in ``models.presets``. Applying a preset is one config save. The setup it
replaced is kept (store meta) so the switch can be undone, and anything still missing (an Ollama model that isn't
downloaded, a provider key that isn't set) comes back with a plain fix and, where possible, a one-click action.

Shared by ``/api/models/presets`` and the ``/model`` command in messaging channels.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from sentient import secrets
from sentient.config.schema import (
    CLOUD_PRESET,
    LOCAL_PRESET,
    MIXED_PRESET,
    MODEL_ROLE_NAMES,
    PRESET_CLOUD_MODELS,
    ModelPreset,
    ModelRoles,
    SentientConfig,
)
from sentient.llm.checkup import LABELS, LOCAL, OLLAMA
from sentient.llm.provider import provider_config

UNDO_META_KEY = "models.preset_undo"
BUILTIN = (LOCAL_PRESET, CLOUD_PRESET, MIXED_PRESET)
SETUP_FIELDS = ("fallbacks", "reasoning", "context_length", "context_length_per_role")
MAX_NAME = 40
_BAD_NAME = re.compile(r"[/\\\x00-\x1f]")


class PresetError(Exception):
    """A plain-language problem; ``status`` is the HTTP status for routes."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def _prefix(model: str | None) -> str:
    return model.split("/", 1)[0] if model and "/" in model else ""


def key_set(config: SentientConfig, provider: str) -> bool:
    pc = provider_config(config, provider)
    return bool(secrets.get_secret(provider, pc.api_key_env if pc else None))


def cloud_provider(config: SentientConfig) -> str | None:
    """The first of Anthropic, OpenAI and OpenRouter with a key set."""
    return next((pid for pid in PRESET_CLOUD_MODELS if key_set(config, pid)), None)


def _builtin(config: SentientConfig) -> list[dict[str, Any]]:
    defaults = ModelRoles()
    current_embedding = config.models.roles.embedding
    # keep a local embedding model the user picked: changing it re-indexes memory
    embedding = current_embedding if _prefix(current_embedding) in LOCAL else defaults.embedding
    optional = {"planner": None, "executor": None, "vision": None}
    local = {"primary": defaults.primary, "fast": defaults.fast, **optional, "voice": None, "embedding": embedding}
    pid = cloud_provider(config)
    cloud = PRESET_CLOUD_MODELS[pid or next(iter(PRESET_CLOUD_MODELS))]
    label = LABELS.get(pid or "", "")
    reason = None if pid else "Needs a key for Anthropic, OpenAI or OpenRouter."
    return [
        {"name": LOCAL_PRESET, "builtin": True, "available": True, "reason": None, "provider": None,
         "description": "Everything runs on this computer with Ollama. Private, and works offline.",
         "roles": local, "fallbacks": {}},
        {"name": CLOUD_PRESET, "builtin": True, "available": pid is not None, "reason": reason, "provider": pid,
         # embedding is left as it is: a cloud embedding model would re-index all of memory
         "description": f"{label or 'A cloud provider'} for every job. Needs the internet.",
         "roles": {"primary": cloud["main"], "fast": cloud["fast"], **optional, "voice": cloud["fast"]},
         "fallbacks": {}},
        {"name": MIXED_PRESET, "builtin": True, "available": pid is not None, "reason": reason, "provider": pid,
         "description": f"{label or 'The cloud'} for chat and planning, this computer for background jobs and memory.",
         "roles": {"primary": cloud["main"], "fast": defaults.fast, **optional, "voice": cloud["fast"],
                   "embedding": embedding},
         "fallbacks": {}},
    ]


def _custom(name: str, preset: ModelPreset) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "builtin": False, "available": True, "reason": None, "provider": None,
                           "description": None, "roles": dict(preset.roles)}
    for f in SETUP_FIELDS:
        value = getattr(preset, f)
        if value is not None:
            out[f] = value
    return out


def presets(config: SentientConfig) -> list[dict[str, Any]]:
    """Built-in presets first, then the user's own in saved order. A saved preset can't shadow a built-in."""
    builtin_names = {n.lower() for n in BUILTIN}
    custom = [_custom(n, p) for n, p in config.models.presets.items() if n.lower() not in builtin_names]
    return _builtin(config) + custom


def find(config: SentientConfig, name: str) -> dict[str, Any]:
    wanted = name.strip().lower()
    for p in presets(config):
        if p["name"].lower() == wanted:
            return p
    raise PresetError(f"There's no model setup called {name.strip()!r}.", 404)


def matches(config: SentientConfig, preset: dict[str, Any]) -> bool:
    """True when every role the preset sets has that model now."""
    roles = config.models.roles.model_dump()
    return all((roles.get(r) or None) == (m or None) for r, m in preset["roles"].items())


def _snapshot(config: SentientConfig) -> dict[str, Any]:
    m = config.models
    return {"roles": m.roles.model_dump(), "fallbacks": dict(m.fallbacks), "reasoning": dict(m.reasoning),
            "context_length": m.context_length, "context_length_per_role": dict(m.context_length_per_role)}


def _changes(before: SentientConfig, after: SentientConfig) -> list[dict[str, Any]]:
    a, b = before.models.roles.model_dump(), after.models.roles.model_dump()
    return [{"role": r, "from": a.get(r), "to": b.get(r)} for r in MODEL_ROLE_NAMES if a.get(r) != b.get(r)]


async def listing(app: Any) -> dict[str, Any]:
    """``GET /api/models/presets``: every preset, which one is active, and whether a switch can be undone."""
    config = app.config
    items = presets(config)
    active = config.models.active_preset
    current = next((p for p in items if active and p["name"].lower() == active.lower()), None)
    for p in items:
        p["active"] = p is current
    undo = await _undo_entry(app)
    return {
        "active": current["name"] if current else None,
        "modified": bool(current and not matches(config, current)),
        "can_undo": undo is not None,
        "undo_preset": undo.get("preset") if undo else None,
        "presets": items,
    }


# ----------------------------------------------------------------------------- applying and undo
async def apply(app: Any, name: str) -> dict[str, Any]:
    """Switch every role (and the preset's other settings) at once. Returns what changed and what is missing."""
    before = app.config
    preset = find(before, name)
    if not preset["available"]:
        raise PresetError(preset["reason"] or f"{preset['name']} can't be used yet.", 409)
    cfg = before.model_copy(deep=True)
    roles = cfg.models.roles.model_dump()
    roles.update({r: m or None for r, m in preset["roles"].items()})
    cfg.models.roles = ModelRoles.model_validate(roles)
    for f in SETUP_FIELDS:
        if f in preset:
            value = preset[f]
            setattr(cfg.models, f, dict(value) if isinstance(value, dict) else value)
    cfg.models.active_preset = preset["name"]
    if _snapshot(cfg) == _snapshot(before) and cfg.models.active_preset == before.models.active_preset:
        # already in use, unchanged: nothing to save, and the undo step still points at the setup before it
        return {"preset": preset["name"], "changed": [], "missing": await missing(cfg),
                "can_undo": await _undo_entry(app) is not None}
    previous = await app.store.get_meta(UNDO_META_KEY)
    await app.store.set_meta(UNDO_META_KEY, json.dumps({"preset": before.models.active_preset,
                                                        "setup": _snapshot(before)}))
    try:
        app.save_config(cfg)
    except Exception:
        await app.store.set_meta(UNDO_META_KEY, previous or "null")
        raise
    return {"preset": preset["name"], "changed": _changes(before, cfg), "missing": await missing(cfg),
            "can_undo": True}


async def _undo_entry(app: Any) -> dict[str, Any] | None:
    try:
        data = json.loads(await app.store.get_meta(UNDO_META_KEY) or "null")
    except ValueError:
        return None
    return data if isinstance(data, dict) and isinstance(data.get("setup"), dict) else None


async def undo(app: Any) -> dict[str, Any]:
    """Put back the setup the last switch replaced. One step: undoing twice is refused."""
    entry = await _undo_entry(app)
    if entry is None:
        raise PresetError("There's no model switch to undo.", 409)
    before = app.config
    cfg = before.model_copy(deep=True)
    setup = entry["setup"]
    data = cfg.models.model_dump()
    data.update({k: v for k, v in setup.items() if k in {"roles", *SETUP_FIELDS}})
    data["active_preset"] = entry.get("preset")
    cfg.models = type(cfg.models).model_validate(data)
    app.save_config(cfg)
    await app.store.set_meta(UNDO_META_KEY, "null")
    return {"preset": cfg.models.active_preset, "changed": _changes(before, cfg), "missing": await missing(cfg),
            "can_undo": False}


# ----------------------------------------------------------------------------- saving your own
def clean_name(name: str) -> str:
    name = " ".join(str(name or "").split())
    if not name:
        raise PresetError("Give the setup a name.")
    if len(name) > MAX_NAME:
        raise PresetError(f"Keep the name under {MAX_NAME} characters.")
    if _BAD_NAME.search(name):
        raise PresetError("A name can't contain slashes.")
    if name.lower() in {n.lower() for n in BUILTIN}:
        raise PresetError(f"{name} is a built-in setup. Pick another name.")
    return name


def _custom_key(config: SentientConfig, name: str) -> str | None:
    return next((n for n in config.models.presets if n.lower() == name.strip().lower()), None)


def save_current(app: Any, name: str, *, overwrite: bool = False) -> dict[str, Any]:
    """Save the current models as a preset (and mark it active)."""
    name = clean_name(name)
    cfg = app.config.model_copy(deep=True)
    existing = _custom_key(cfg, name)
    if existing and not overwrite:
        raise PresetError(f"You already have a setup called {existing}.", 409)
    if existing:
        del cfg.models.presets[existing]
    snap = _snapshot(cfg)
    cfg.models.presets[name] = ModelPreset(roles=snap.pop("roles"), **snap)
    cfg.models.active_preset = name
    app.save_config(cfg)
    return _custom(name, cfg.models.presets[name])


def rename(app: Any, old: str, new: str) -> dict[str, Any]:
    cfg = app.config.model_copy(deep=True)
    if old.strip().lower() in {n.lower() for n in BUILTIN}:
        raise PresetError("Built-in setups can't be renamed.")
    key = _custom_key(cfg, old)
    if key is None:
        raise PresetError(f"There's no model setup called {old.strip()!r}.", 404)
    new = clean_name(new)
    clash = _custom_key(cfg, new)
    if clash and clash != key:
        raise PresetError(f"You already have a setup called {clash}.", 409)
    cfg.models.presets = {(new if n == key else n): p for n, p in cfg.models.presets.items()}
    if (cfg.models.active_preset or "").lower() == key.lower():
        cfg.models.active_preset = new
    app.save_config(cfg)
    return _custom(new, cfg.models.presets[new])


def delete(app: Any, name: str) -> None:
    cfg = app.config.model_copy(deep=True)
    if name.strip().lower() in {n.lower() for n in BUILTIN}:
        raise PresetError("Built-in setups can't be deleted.")
    key = _custom_key(cfg, name)
    if key is None:
        raise PresetError(f"There's no model setup called {name.strip()!r}.", 404)
    del cfg.models.presets[key]
    if (cfg.models.active_preset or "").lower() == key.lower():
        cfg.models.active_preset = None
    app.save_config(cfg)


# ----------------------------------------------------------------------------- what is missing
def _ollama_base(config: SentientConfig, prefix: str) -> str:
    providers = config.models.providers
    pc = providers.get(prefix) or providers.get("ollama_chat") or providers.get("ollama")
    return (pc.api_base if pc and pc.api_base else "http://localhost:11434").rstrip("/")


def _installed(name: str, installed: list[str]) -> bool:
    return name in installed or (":" not in name and f"{name}:latest" in installed)


async def missing(config: SentientConfig) -> list[dict[str, Any]]:
    """Models the roles use that can't work yet, with a plain fix and an optional one-click ``action``:
    ``{kind: "pull_model", name, label}`` or ``{kind: "add_key", provider, label}``."""
    by_model: dict[str, list[str]] = {}
    for role, model in config.models.roles.model_dump().items():
        if model:
            by_model.setdefault(model, []).append(role)
    out: list[dict[str, Any]] = []
    tags: dict[str, list[str] | None] = {}
    down: dict[str, dict[str, Any]] = {}
    keyless: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=3) as client:
        for model, roles in by_model.items():
            prefix, name = _prefix(model), model.split("/", 1)[-1]
            if prefix in OLLAMA:
                base = _ollama_base(config, prefix)
                if base not in tags:
                    try:
                        r = await client.get(f"{base}/api/tags")
                        r.raise_for_status()
                        tags[base] = [m.get("name", "") for m in r.json().get("models", [])]
                    except Exception:
                        tags[base] = None
                if tags[base] is None:
                    item = down.get(base)
                    if item is None:
                        item = down[base] = {
                            "kind": "start_ollama", "roles": [], "model": None,
                            "detail": "Ollama isn't running, so the local models can't be checked or used yet.",
                            "fix": "Start the Ollama app, or install it from ollama.com.", "action": None}
                        out.append(item)
                    item["roles"] += roles
                elif not _installed(name, tags[base] or []):
                    out.append({"kind": "pull_model", "roles": roles, "model": model,
                                "detail": f"{name} isn't downloaded yet.",
                                "fix": f"Download {name}. It can take a few minutes.",
                                "action": {"kind": "pull_model", "name": name, "label": f"Download {name}"}})
                continue
            pc = provider_config(config, prefix)
            if prefix in LOCAL or not (pc and pc.api_key_env) or key_set(config, prefix):
                continue
            label = LABELS.get(prefix, prefix)
            item = keyless.get(prefix)
            if item is None:
                item = keyless[prefix] = {
                    "kind": "add_key", "roles": [], "model": model, "provider": prefix,
                    "detail": f"No {label} key is set.", "fix": f"Add your {label} key.",
                    "action": {"kind": "add_key", "provider": prefix, "label": f"Add {label} key"}}
                out.append(item)
            item["roles"] += roles
    return out
