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
  subagents, messaging channels, the terminal) get a boost when trigger phrases appear;
- a plugin the user names ("using Composio", "in Notion") comes first;
- the best plugins are added until the tool budget is reached. A plugin with
  more tools than the room left (a large MCP server) offers its best tools,
  scored one by one from their names and descriptions, keeping a server's
  "search/list" and "execute/run" tools together, and leaves room for the
  smaller relevant plugins after it.

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
    "channel_": ("telegram", "discord", "whatsapp", "message me", "text me"),
    "terminal_": (
        "terminal", "command", "commands", "command line", "shell", "powershell", "bash", "git", "npm", "pip",
        "compile", "run the tests", "my repo", "repository",
    ),
}
TRIGGER_BOOST = 0.45
# A plugin with more tools than the room left keeps at least this many slots when it is offered partially.
MIN_PARTIAL = 4
# Words that mark a server's discovery tools and its action tools (search-then-execute MCP servers).
DISCOVER_WORDS = frozenset({"search", "list", "schema", "find", "discover"})
ACT_WORDS = frozenset({"execute", "run", "call", "invoke"})


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


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in _words(text)}


def _name_pattern(name: str) -> re.Pattern | None:
    parts = _WORD.findall(name.lower())
    if not parts or len("".join(parts)) < 3:
        return None
    return re.compile(r"(?<![a-z0-9])" + r"[\s_-]*".join(re.escape(x) for x in parts) + r"(?![a-z0-9])")


def named_plugins(text: str, plugins: list[dict]) -> set[str]:
    """Ids of the apps and MCP servers the message names ("using Composio", "in Notion").
    Built-in core plugins (memory, files...) are matched by keywords instead: their names are everyday words."""
    low = text.lower()
    out = set()
    for p in plugins:
        if p.get("category", "core") == "core":
            continue
        candidates = {p["display_name"], p["id"].removeprefix("mcp_")}
        if any((rx := _name_pattern(c)) is not None and rx.search(low) for c in candidates):
            out.add(p["id"])
    return out


def _tool_scores(plugin: dict, words: set[str]) -> list[tuple[float, float, str]]:
    """(keyword score, order key, kind) per tool. The keyword score counts the message's words in the tool's
    name and the start of its description; the order key adds a tie-break that puts entry points first."""
    own = _stems(f"{plugin['id']} {plugin['display_name']}")
    out = []
    for t in plugin["tools"]:
        name = t["name"].removeprefix(plugin["id"] + "_")
        name_words = _stems(name.replace("_", " ")) - own
        desc_words = _stems((t.get("description") or "")[:300]) - own - name_words
        score = 0.3 * len(words & name_words) + 0.1 * min(len(words & desc_words), 3)
        kind = "discover" if name_words & DISCOVER_WORDS else "act" if name_words & ACT_WORDS else ""
        out.append((score, score + (0.05 if kind else 0.0), kind))
    return out


def _best_tools(tools: list[dict], scored: list[tuple[float, float, str]], k: int) -> list[str]:
    """The ``k`` best tools in score order. A server's discovery tool (search/list/schema) is useless without its
    action tool (execute/run/call) and vice versa, so when one kind is picked the best of the other comes too."""
    order = sorted(range(len(tools)), key=lambda i: (-scored[i][1], i))
    picked = order[:k]
    if k >= 2:
        for need, have in (("act", "discover"), ("discover", "act")):
            kinds = {scored[i][2] for i in picked}
            if have in kinds and need not in kinds:
                extra = next((i for i in order if scored[i][2] == need), None)
                if extra is None:
                    continue
                if len(picked) >= k:
                    # drop the weakest pick that is not the last of its kind
                    for j in reversed(range(len(picked))):
                        kind = scored[picked[j]][2]
                        if not kind or sum(scored[i][2] == kind for i in picked) > 1:
                            picked.pop(j)
                            break
                    else:
                        continue
                picked.append(extra)
    picked.sort(key=lambda i: (-scored[i][1], i))
    return [tools[i]["name"] for i in picked]


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

        named = named_plugins(text, plugins) if text.strip() else set()
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

        top = max(scores.values(), default=0.0)
        # only plugins that are plausibly relevant: named, close to the best match, or used recently
        floor = max(0.15, top * 0.6)
        relevant = [
            p for p in sorted(plugins, key=lambda p: (p["id"] not in named, -scores[p["id"]]))
            if p["id"] in named or scores[p["id"]] >= floor or bool(recent_plugins and p["id"] in recent_plugins)
        ]
        stems = {_stem(w) for w in words}
        for i, p in enumerate(relevant):
            room = budget - len(chosen)
            if room <= 0:
                break
            tools = [t for t in p["tools"] if t["name"] not in chosen]
            if not tools:
                continue
            if len(tools) <= room:
                chosen.extend(t["name"] for t in tools)
                continue
            # too big for the room left: offer its best tools when it was named, is the best match, or some
            # of its tools match the message by name or description
            scored = _tool_scores({**p, "tools": tools}, stems)
            if room < 2 or not (p["id"] in named or i == 0 or any(s[0] > 0 for s in scored)):
                continue
            keep = min(room, MIN_PARTIAL)
            reserve = 0
            for q in relevant[i + 1:]:
                n = sum(1 for t in q["tools"] if t["name"] not in chosen)
                if n and reserve + n <= room - keep:
                    reserve += n
            chosen.extend(_best_tools(tools, scored, room - reserve))
        return chosen
