"""Pick which tools to offer the model on a turn.

Small local models degrade when handed dozens of tool schemas: every schema
costs prompt-processing time on a laptop GPU, and choice quality drops. v2
solved this with an extra LLM call per turn (the "Stage 1" router). v3 does it
without an LLM call:

- a few core tools are always offered (clock, memory recall/remember, skills,
  task delegation);
- every other visible plugin is scored against the user's message by embedding
  similarity to its description and selection hint, plus a keyword boost;
- plugins used recently in the same conversation stay available;
- a few capabilities that embeddings match poorly (browser, code, devices,
  subagents, messaging channels) get a boost when trigger phrases appear;
- the best plugins are added until the tool budget is reached.

Cloud models get a large budget (effectively all tools); local models a small
one. ``chat.tool_selection = "all"`` turns selection off.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

log = logging.getLogger(__name__)

ALWAYS_TOOLS = (
    "current_datetime",
    "memory_recall",
    "memory_remember",
    "skill_view",
    "create_task_from_prompt",
)
LOCAL_PREFIXES = ("ollama", "ollama_chat", "lm_studio", "llamafile", "vllm", "hosted_vllm")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "my", "me", "i", "you",
    "it", "this", "that", "what", "whats", "how", "can", "please", "with", "at", "be", "do", "does",
    "there", "now", "right", "your", "from", "about", "tell", "get", "show", "some", "any",
}


# Trigger phrases for capabilities small embedding models match poorly. Keys are tool-name prefixes, so
# the boost applies to whichever plugin registers those tools.
TOOL_TRIGGERS: dict[str, tuple[str, ...]] = {
    "browser_": (
        "browser", "website", "web site", "webpage", "web page", "site", "log in", "login", "sign in", "signed in",
        "click", "fill in", "fill out", "form", "checkout", "portal", "url", "http", "https", "www", "book a",
        "add to cart", "online",
    ),
    "execute_code": (
        "code", "script", "python", "calculate", "calculation", "compute", "csv", "excel", "spreadsheet", "json",
        "parse", "convert", "plot", "chart", "statistics", "average", "median", "percentage", "regex", "analyze",
        "analyse", "dataset", "run a program",
    ),
    "device_": (
        "phone", "glasses", "camera", "photo", "picture", "looking at", "what do you see", "can you see", "screen",
        "screenshot", "location", "where am i", "device", "devices", "nearby", "speak on", "show on",
    ),
    "delegate_": (
        "in parallel", "parallel", "background", "delegate", "subagent", "subagents", "sub-agent", "research",
        "deep dive", "investigate", "compare", "each of", "meanwhile", "while i", "several",
    ),
    "channel_": ("telegram", "discord", "message me", "text me"),
}
TRIGGER_BOOST = 0.45


def _compile_triggers() -> dict[str, re.Pattern]:
    out = {}
    for prefix, phrases in TOOL_TRIGGERS.items():
        alts = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
        out[prefix] = re.compile(rf"(?<![a-z0-9])(?:{alts})(?![a-z0-9])")
    return out


_TRIGGERS = _compile_triggers()


def triggered_prefixes(text: str) -> set[str]:
    low = text.lower()
    return {prefix for prefix, rx in _TRIGGERS.items() if rx.search(low)}


def is_local_model(model: str) -> bool:
    return model.split("/", 1)[0] in LOCAL_PREFIXES


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class ToolSelector:
    """``owner`` is the Agent: its ``registry``, ``llm`` and ``config`` are read on every call,
    so configuration changes apply immediately."""

    def __init__(self, owner: Any):
        self.owner = owner
        self._plugin_vecs: dict[str, list[float]] = {}
        self._vec_key: tuple = ()

    def budget(self, model: str) -> int:
        c = self.owner.config.chat
        return c.max_tools_local if is_local_model(model) else c.max_tools_cloud

    @staticmethod
    def _plugin_text(p: dict) -> str:
        tools = ", ".join(t["name"].replace("_", " ") for t in p["tools"])
        return f"{p['display_name']}. {p['description']} Use for: {p['selection_hint']}. Tools: {tools}"

    async def _ensure_vectors(self, plugins: list[dict]) -> bool:
        llm = self.owner.llm
        key = (*sorted(p["id"] for p in plugins), llm.model_for("embedding"))
        if key == self._vec_key and self._plugin_vecs:
            return True
        try:
            vecs = await llm.embed([self._plugin_text(p) for p in plugins])
        except Exception as exc:
            log.debug("tool selection without embeddings: %s", exc)
            return False
        self._plugin_vecs = {p["id"]: v for p, v in zip(plugins, vecs, strict=False)}
        self._vec_key = key
        return True

    async def select(self, text: str, *, model: str, recent_plugins: set[str] | None = None) -> list[str] | None:
        """Tool names to offer, or None to offer every visible tool."""
        registry = self.owner.registry
        if self.owner.config.chat.tool_selection == "all":
            return None
        visible = registry.tools()
        budget = self.budget(model)
        if len(visible) <= budget:
            return None

        visible_names = {t.name for t in visible}
        chosen: list[str] = [n for n in ALWAYS_TOOLS if n in visible_names]
        plugins = [p for p in registry.catalog() if p["tools"]]
        words = _words(text)
        triggers = triggered_prefixes(text) if text.strip() else set()
        qvec: list[float] | None = None
        if text.strip() and await self._ensure_vectors(plugins):
            try:
                [qvec] = await self.owner.llm.embed([text[:2000]])
            except Exception:
                qvec = None

        scores: dict[str, float] = {}
        for p in plugins:
            score = _cos(qvec, self._plugin_vecs[p["id"]]) if qvec is not None and p["id"] in self._plugin_vecs else 0.0
            vocab = _words(
                f"{p['id']} {p['display_name']} {p['selection_hint']} " + " ".join(t["name"] for t in p["tools"])
            )
            score += min(len(words & vocab), 3) * 0.15
            if recent_plugins and p["id"] in recent_plugins:
                score += 0.5
            if triggers and any(t["name"].startswith(prefix) for prefix in triggers for t in p["tools"]):
                score += TRIGGER_BOOST
            scores[p["id"]] = score

        ranked = sorted(plugins, key=lambda p: scores[p["id"]], reverse=True)
        top = scores[ranked[0]["id"]] if ranked else 0.0
        # only plugins that are plausibly relevant: close to the best match, or used recently
        floor = max(0.15, top * 0.6)
        for p in ranked:
            relevant = scores[p["id"]] >= floor or bool(recent_plugins and p["id"] in recent_plugins)
            if not relevant:
                continue
            names = [t["name"] for t in p["tools"] if t["name"] not in chosen]
            if not names:
                continue
            room = budget - len(chosen)
            if room <= 0:
                break
            if len(names) > room:
                if room >= 2 and p is ranked[0]:
                    chosen.extend(names[:room])  # the best match may be offered partially
                continue
            chosen.extend(names)
        return chosen
