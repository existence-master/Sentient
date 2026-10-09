"""Model check-up: does each role's model actually work on this computer?

Shared by ``POST /api/models/checkup`` and ``sentient doctor --models``. Per role it checks that the model is
reachable, sends one short reply, a scripted tool call (and a second step for roles that run long tasks), a JSON
reply for roles that need one, how the model handles thinking, the context length in use and, for Ollama, how
much of the model sits on the graphics card. Every problem comes with a plain-language fix and, where one is
obvious, a one-click ``action``. It is informational only: it never changes configuration.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from sentient import secrets
from sentient.config.schema import ModelRoles, SentientConfig
from sentient.llm.provider import ToolCall

ROLES = ("primary", "fast", "planner", "executor", "vision", "voice", "embedding")
TOOL_ROLES = {"primary", "fast", "executor", "vision", "voice"}  # roles that run the agent loop with tools
CHAIN_ROLES = {"primary", "executor"}  # roles that run multi-step work
JSON_ROLES = {"fast", "planner"}  # roles whose callers parse JSON replies
OLLAMA = {"ollama", "ollama_chat"}
LOCAL = OLLAMA | {"lm_studio"}
SUGGESTED_LOCAL = ModelRoles.model_fields["primary"].default.split("/", 1)[1]  # the default local model
STEP_TIMEOUT_S = 60.0
MIN_CONTEXT = 8192
SMALL_QWEN = re.compile(r"^qwen3:(0\.6|1\.7|4)b")
LABELS = {
    "ollama": "Ollama", "ollama_chat": "Ollama", "lm_studio": "LM Studio", "anthropic": "Anthropic",
    "openai": "OpenAI", "gemini": "Google Gemini", "openrouter": "OpenRouter", "groq": "Groq",
    "mistral": "Mistral", "deepseek": "DeepSeek", "xai": "xAI",
}
ORDER = {"fail": 3, "warn": 2, "pass": 1, "skip": 0}

FIND_CITY = {
    "type": "function",
    "function": {
        "name": "find_city",
        "description": "Look up a city by name and return its city_id.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
}
GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city_id returned by find_city.",
        "parameters": {"type": "object", "properties": {"city_id": {"type": "string"}}, "required": ["city_id"]},
    },
}
CITY_ID = "c-42"
TOOL_PROMPT = (
    {"role": "user",
     "content": "What is the weather in Paris right now? Use the tools: find the city first, then get its weather."},
)


def _check(cid: str, label: str, status: str, detail: str, fix: str | None = None, action: dict | None = None) -> dict:
    out: dict[str, Any] = {"id": cid, "label": label, "status": status, "detail": detail}
    if fix:
        out["fix"] = fix
    if action:
        out["action"] = action
    return out


def _worst(checks: list[dict]) -> str:
    return max((c["status"] for c in checks), key=ORDER.__getitem__, default="skip")


def _short(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text if len(text) <= 240 else text[:237] + "..."


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


class _Ollama:
    """Ollama's native API, cached for one check-up run (the same model is often used by several roles)."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self._up: dict[str, bool] = {}
        self._show: dict[str, dict | None] = {}
        self._tags: dict[str, list[str]] = {}

    async def up(self, base: str) -> bool:
        if base not in self._up:
            try:
                r = await self.client.get(f"{base}/api/version")
                self._up[base] = r.status_code == 200
            except Exception:
                self._up[base] = False
        return self._up[base]

    async def show(self, base: str, name: str) -> dict | None:
        """``/api/show`` for an installed model, None when it is not downloaded."""
        key = f"{base}|{name}"
        if key not in self._show:
            try:
                r = await self.client.post(f"{base}/api/show", json={"model": name})
                self._show[key] = r.json() if r.status_code == 200 else None
            except Exception:
                self._show[key] = None
        return self._show[key]

    async def installed(self, base: str) -> list[str]:
        if base not in self._tags:
            try:
                r = await self.client.get(f"{base}/api/tags")
                self._tags[base] = [m.get("name", "") for m in r.json().get("models", [])]
            except Exception:
                self._tags[base] = []
        return self._tags[base]

    async def loaded(self, base: str, name: str) -> dict | None:
        """This model's entry in ``/api/ps`` (what is loaded right now), or None."""
        try:
            r = await self.client.get(f"{base}/api/ps")
            models = r.json().get("models", [])
        except Exception:
            return None
        wanted = {name, f"{name}:latest"} if ":" not in name else {name}
        return next((m for m in models if (m.get("name") or m.get("model")) in wanted), None)


class _RoleCheck:
    def __init__(self, run: CheckupRun, role: str, model: str):
        self.run = run
        self.role = role
        self.model = model
        self.prefix = model.split("/", 1)[0] if "/" in model else ""
        self.name = model.split("/", 1)[1] if "/" in model else model
        self.label = LABELS.get(self.prefix, self.prefix or "the provider")
        self.ollama = self.prefix in OLLAMA
        self.checks: list[dict] = []
        self.thinking_seen = False
        self.show: dict | None = None
        self.first_call: ToolCall | None = None

    # ------------------------------------------------------------------ helpers
    def add(self, *args, **kwargs) -> dict:
        c = _check(*args, **kwargs)
        self.checks.append(c)
        return c

    async def _collect(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[str, list]:
        async def go() -> tuple[str, list]:
            text, calls = "", []
            async for chunk in self.run.llm.stream(self.role, messages, tools, model=self.model):
                text += chunk.text
                if chunk.thinking:
                    self.thinking_seen = True
                if chunk.done:
                    calls = list(chunk.tool_calls)
            return text, calls

        return await asyncio.wait_for(go(), self.run.timeout_s)

    def _base(self) -> str:
        return self.run.ollama_base(self.prefix)

    def _failure_fix(self, exc: BaseException) -> str:
        if isinstance(exc, TimeoutError):
            return (f"No reply within {int(self.run.timeout_s)} seconds. The model may be too big for this "
                    "computer: try a smaller model, or a shorter context length.")
        msg = str(exc).lower()
        if any(k in msg for k in ("401", "403", "api key", "api_key", "authentication", "unauthorized")):
            return f"{self.label} turned the key down. Check your {self.label} key under Providers."
        if self.ollama:
            return "Make sure Ollama is running and this model is downloaded, then try again."
        if self.prefix == "lm_studio":
            return "Make sure LM Studio's server is running with this model loaded."
        return f"Check the model name and your {self.label} key under Providers."

    async def _better_model(self) -> tuple[str, dict | None]:
        """Fix text and, for Ollama, a one-click switch (or download) to a model that can call tools."""
        if self.prefix not in LOCAL:
            return "Pick a model from this provider that supports tool calls.", None
        if self.name == SUGGESTED_LOCAL:
            return "Try a larger local model or a cloud model for tools and tasks.", None
        if SMALL_QWEN.match(self.name):
            text = f"{self.name} can't call tools reliably; try {SUGGESTED_LOCAL}."
        else:
            text = f"Pick a model that handles tools well, like {SUGGESTED_LOCAL}."
        if not self.ollama:
            return text, None
        installed = await self.run.ollama.installed(self._base())
        if SUGGESTED_LOCAL in installed:
            return text, {"kind": "use_model", "role": self.role, "model": f"ollama_chat/{SUGGESTED_LOCAL}",
                          "label": f"Use {SUGGESTED_LOCAL}"}
        return text, {"kind": "pull_model", "name": SUGGESTED_LOCAL, "label": f"Download {SUGGESTED_LOCAL}"}

    def _status(self, cid: str) -> str | None:
        return next((c["status"] for c in self.checks if c["id"] == cid), None)

    def _settings_key(self) -> tuple:
        """Everything a model test depends on. Roles with the same key get the same answers, so they share results."""
        models = self.run.config.models
        pc = models.providers.get(self.prefix)
        base = self._base() if self.ollama else (pc.api_base if pc else None)
        ctx = (models.context_length_per_role.get(self.role) or models.context_length) if self.ollama else None
        return (self.model, base, models.reasoning.get(self.role), ctx, models.temperature.get(self.role))

    async def _shared(self, group: str, check) -> None:
        """Run a model test once per model and settings; another role with the same settings reuses the result."""
        entry = self.run.shared.setdefault(self._settings_key(), {})
        if group in entry:
            source, checks, thinking = entry[group]
            self.checks += [self._reused(c, source) for c in checks]
            self.thinking_seen |= thinking
            return
        start, thought = len(self.checks), self.thinking_seen
        self.thinking_seen = False
        await check()
        entry[group] = (self.role, [dict(c) for c in self.checks[start:]], self.thinking_seen)
        self.thinking_seen |= thought

    def _reused(self, check: dict, source: str) -> dict:
        out = {**check, "detail": f"Same as {source}. {check['detail']}"}
        action = check.get("action")
        if action and action.get("role") and action["kind"] != "set_context_length":
            out["action"] = {**action, "role": self.role}  # the fix applies to this role
        return out

    # ------------------------------------------------------------------ checks
    async def connection(self) -> bool:
        """Can the model be reached at all? False stops the remaining checks."""
        if self.ollama:
            base = self._base()
            if not await self.run.ollama.up(base):
                self.add("connection", "Connection", "fail", f"Ollama isn't answering at {base}.",
                         "Start the Ollama app (or install it from ollama.com), then run the check again.")
                return False
            self.show = await self.run.ollama.show(base, self.name)
            if self.show is None:
                self.add("connection", "Connection", "fail", f"{self.name} isn't downloaded in Ollama.",
                         f"Download {self.name}, or pick a model you already have.",
                         {"kind": "pull_model", "name": self.name, "label": f"Download {self.name}"})
                return False
            self.add("connection", "Connection", "pass", f"Ollama is running and {self.name} is downloaded.")
            return True
        pc = self.run.config.models.providers.get(self.prefix)
        if self.prefix not in LOCAL and pc and pc.api_key_env:
            if not secrets.get_secret(self.prefix, pc.api_key_env):
                self.add("connection", "Connection", "fail", f"No {self.label} key is set.",
                         f"Add your {self.label} key under Providers.")
                return False
            self.add("connection", "Connection", "pass", f"{self.label} key is set.")
        return True

    async def reply(self) -> None:
        self.run.step(self.role, "Asking for a short reply")
        started = time.perf_counter()
        try:
            text, _ = await self._collect([{"role": "user", "content": "Reply with the single word: ready"}])
        except Exception as exc:
            self.add("reply", "Reply", "fail", _short(exc) if not isinstance(exc, TimeoutError) else "No reply.",
                     self._failure_fix(exc))
            return
        ms = int((time.perf_counter() - started) * 1000)
        if not text.strip():
            self.add("reply", "Reply", "warn", f"Answered in {ms / 1000:.1f} s but the reply was empty.",
                     "The model may be spending its whole reply thinking. Try turning Reasoning down for this role.")
        else:
            self.add("reply", "Reply", "pass", f"Answered in {ms / 1000:.1f} s.")

    async def tools(self) -> None:
        self.run.step(self.role, "Trying a tool call")
        try:
            _, calls = await self._collect(list(TOOL_PROMPT), [FIND_CITY, GET_WEATHER])
        except Exception as exc:
            self.add("tools", "Tool call", "fail", _short(exc) if not isinstance(exc, TimeoutError) else "No reply.",
                     self._failure_fix(exc))
            return
        first = calls[0] if calls else None
        if first is None or first.name != "find_city" or "paris" not in str(first.arguments.get("name", "")).lower():
            fix, action = await self._better_model()
            detail = ("It answered in text instead of calling the test tool." if first is None
                      else f"It called {first.name} with the wrong details.")
            self.add("tools", "Tool call", "fail", detail + " Tools and tasks won't work well.", fix, action)
            return
        self.first_call = first
        self.add("tools", "Tool call", "pass", "Called the test tool correctly.")

    async def chain(self) -> None:
        """Second tool step: feed the first call's result back and expect a call that uses it."""
        self.run.step(self.role, "Trying a second tool step")
        # the role whose first step passed may be another role with the same settings: then use a canonical call
        first = self.first_call or ToolCall(id="call_find_city", name="find_city", arguments={"name": "Paris"})
        messages = [
            *TOOL_PROMPT,
            {"role": "assistant", "content": "", "tool_calls": [first.to_openai()]},
            {"role": "tool", "tool_call_id": first.id, "content": json.dumps({"city_id": CITY_ID, "name": "Paris"})},
        ]
        try:
            _, calls = await self._collect(messages, [FIND_CITY, GET_WEATHER])
        except Exception as exc:
            self.add("chain", "Two tool steps", "warn", _short(exc), self._failure_fix(exc))
            return
        if any(c.name == "get_weather" and str(c.arguments.get("city_id", "")) == CITY_ID for c in calls):
            self.add("chain", "Two tool steps", "pass", "Used the first tool's answer in a second tool call.")
            return
        fix, action = await self._better_model()
        self.add("chain", "Two tool steps", "warn",
                 "It made one tool call but lost track on the second step, so longer tasks may stall.", fix, action)

    async def json_reply(self) -> None:
        self.run.step(self.role, "Asking for a JSON reply")
        prompt = 'Reply with only this JSON and nothing else: {"ok": true, "word": "ready"}'

        async def go() -> Any:
            return await self.run.llm.complete_json(self.role, [{"role": "user", "content": prompt}], model=self.model)

        try:
            data = await asyncio.wait_for(go(), self.run.timeout_s)
        except Exception as exc:
            fix = self._failure_fix(exc)
            if not isinstance(exc, TimeoutError) and "json" in str(exc).lower():
                fix = f"Memory and background jobs need clean JSON. Try {SUGGESTED_LOCAL} or a cloud model."
            self.add("json", "JSON reply", "fail", _short(exc) if not isinstance(exc, TimeoutError) else "No reply.", fix)
            return
        if isinstance(data, dict) and "ok" in data:
            self.add("json", "JSON reply", "pass", "Returned clean JSON.")
        else:
            self.add("json", "JSON reply", "warn", "The reply wasn't the JSON it was asked for.",
                     f"Memory and background jobs may miss things. Try {SUGGESTED_LOCAL} or a cloud model.")

    def thinking(self) -> None:
        caps = (self.show or {}).get("capabilities") or []
        if not ("thinking" in caps or self.name.startswith("qwen3")):
            return
        effort = self.run.config.models.reasoning.get(self.role)
        if effort == "none" and self.thinking_seen:
            self.add("thinking", "Thinking", "warn", "It kept thinking even though reasoning is off for this role.",
                     "Replies will be slower than they should be. Update Ollama to the latest version.")
        elif not effort and self.thinking_seen and self.role != "primary":
            self.add("thinking", "Thinking", "warn",
                     "Reasoning isn't set for this role, so the model thinks before every reply.",
                     "Turn Reasoning off for this role to make it faster.",
                     {"kind": "set_reasoning", "role": self.role, "value": "none", "label": "Turn reasoning off"})
        elif effort and effort != "none":
            self.add("thinking", "Thinking", "pass", f"Thinking is on ({effort}), as set.")
        else:
            self.add("thinking", "Thinking", "pass", "Thinking is off, as set." if effort else "Model default.")

    def context(self) -> int | None:
        models = self.run.config.models
        per_role = models.context_length_per_role.get(self.role)
        configured = per_role or models.context_length
        info = (self.show or {}).get("model_info") or {}
        limit = next((v for k, v in info.items() if k.endswith(".context_length") and isinstance(v, int) and v > 0), None)
        in_use = min(configured, limit) if limit else configured
        detail = f"Reads {in_use:,} tokens at a time" + (f" (this model can do up to {limit:,})." if limit else ".")
        if in_use < MIN_CONTEXT and self.role != "embedding" and (limit is None or limit >= MIN_CONTEXT):
            self.add("context", "Context length", "warn", detail,
                     "Long tasks, big inboxes and many tools may get cut off. 8,192 tokens is a good minimum.",
                     {"kind": "set_context_length", "value": MIN_CONTEXT, "role": self.role if per_role else None,
                      "label": "Use 8,192 tokens"})
        else:
            self.add("context", "Context length", "pass", detail)
        return in_use

    async def gpu(self, in_use: int | None) -> None:
        entry = await self.run.ollama.loaded(self._base(), self.name)
        size, vram = int((entry or {}).get("size") or 0), int((entry or {}).get("size_vram") or 0)
        if not entry or size <= 0:
            self.add("gpu", "Graphics card", "skip", "The model isn't loaded right now, so this couldn't be checked.")
            return
        if vram >= size * 0.99:
            self.add("gpu", "Graphics card", "pass", f"Runs fully on the graphics card ({_gb(size)}).")
            return
        action = None
        if in_use and in_use > MIN_CONTEXT:
            per_role = self.role in self.run.config.models.context_length_per_role
            action = {"kind": "set_context_length", "value": MIN_CONTEXT, "role": self.role if per_role else None,
                      "label": "Use 8,192 tokens"}
        if vram <= 0:
            detail = f"Ollama is running all of this model ({_gb(size)}) on the processor."
        else:
            detail = (f"Ollama is running {round(100 * (size - vram) / size)}% of this model on the processor "
                      f"({_gb(vram)} of {_gb(size)} fits on the graphics card).")
        self.add("gpu", "Graphics card", "warn", detail,
                 "It will be slow. Pick a smaller model or a shorter context length.", action)

    async def embedding(self) -> None:
        self.run.step(self.role, "Trying an embedding")

        async def go() -> list[list[float]]:
            return await self.run.llm.embed(["hello world"], model=self.model)

        try:
            [vec] = await asyncio.wait_for(go(), self.run.timeout_s)
        except Exception as exc:
            self.add("embedding", "Embedding", "fail", _short(exc) if not isinstance(exc, TimeoutError) else "No reply.",
                     self._failure_fix(exc))
            return
        self.add("embedding", "Embedding", "pass", f"Works ({len(vec)} dimensions).")

    # ------------------------------------------------------------------ run
    async def result(self) -> dict:
        try:
            await self._run_checks()
        except Exception as exc:  # a check-up never crashes: report it as one more failed row
            self.add("error", "Check-up", "fail", _short(exc), "Run the check again. If it keeps failing, check the logs.")
        return {"role": self.role, "model": self.model, "provider": self.prefix, "local": self.prefix in LOCAL,
                "inherits": None, "status": _worst(self.checks), "checks": self.checks}

    async def _run_checks(self) -> None:
        if await self.connection():
            if self.role == "embedding":
                await self.embedding()
            else:
                await self._shared("reply", self.reply)
                ok = self._status("reply") != "fail"
                if ok and self.role in TOOL_ROLES:
                    await self._shared("tools", self.tools)
                    if self.role in CHAIN_ROLES and self._status("tools") == "pass":
                        await self._shared("chain", self.chain)
                if ok and self.role in JSON_ROLES:
                    await self._shared("json", self.json_reply)
                if self.ollama:
                    if ok:
                        self.thinking()
                    in_use = self.context()
                    await self.gpu(in_use)


class CheckupRun:
    def __init__(self, config: SentientConfig, llm: Any, client: httpx.AsyncClient, timeout_s: float):
        self.config = config
        self.llm = llm
        self.ollama = _Ollama(client)
        self.timeout_s = timeout_s
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self.shared: dict[tuple, dict[str, tuple[str, list[dict], bool]]] = {}  # settings key -> group -> result

    def ollama_base(self, prefix: str) -> str:
        providers = self.config.models.providers
        pc = providers.get(prefix) or providers.get("ollama_chat") or providers.get("ollama")
        return (pc.api_base if pc and pc.api_base else "http://localhost:11434").rstrip("/")

    def step(self, role: str, label: str) -> None:
        self.events.put_nowait({"type": "step", "role": role, "label": label})


def plan(config: SentientConfig, roles: dict[str, str | None] | None = None) -> list[tuple[str, str | None]]:
    """``[(role, model)]`` to check, in display order. ``model`` is None for an optional role that uses the main model.

    ``roles`` checks only these roles with these models (onboarding checks its picks before they are saved); an
    empty mapping checks nothing. ``None`` checks every role in the saved config.
    """
    configured = config.models.roles
    if roles is not None:
        return [(r, roles[r] or None) for r in ROLES if r in roles]
    return [(r, getattr(configured, r, None) or None) for r in ROLES]


async def run_checkup(
    config: SentientConfig, llm: Any, roles: dict[str, str | None] | None = None, *, timeout_s: float = STEP_TIMEOUT_S
) -> AsyncIterator[dict]:
    """Stream check-up events: ``start``, then ``step`` and ``role`` per role, then ``done``. Roles run one at a
    time so local models are not loaded side by side."""
    todo = plan(config, roles)
    yield {"type": "start", "roles": [{"role": r, "model": m} for r, m in todo]}
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=5) as client:
        run = CheckupRun(config, llm, client, timeout_s)
        for role, model in todo:
            if not model:
                result = {"role": role, "model": None, "provider": None, "local": None, "inherits": "primary",
                          "status": "skip", "checks": []}
            else:
                task = asyncio.ensure_future(_RoleCheck(run, role, model).result())
                try:
                    while not task.done():
                        getter = asyncio.ensure_future(run.events.get())
                        await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
                        if getter.done():
                            yield getter.result()
                        else:
                            getter.cancel()
                    while not run.events.empty():
                        yield run.events.get_nowait()
                finally:
                    task.cancel()  # the window went away mid-check: stop calling the model
                result = task.result()
            results.append(result)
            yield {"type": "role", **result}
    yield {"type": "done", "status": _worst([{"status": r["status"]} for r in results]), "roles": results}


async def checkup(config: SentientConfig, llm: Any, roles: dict[str, str | None] | None = None, **kwargs) -> dict:
    """The final ``done`` event of :func:`run_checkup`."""
    final: dict = {}
    async for event in run_checkup(config, llm, roles, **kwargs):
        if event["type"] == "done":
            final = event
    return final
