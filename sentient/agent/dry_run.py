"""Try runs of tasks: "Try it first" (#142, ADR 0026).

A try run is a task run in which every call that would send, post, buy, delete, change or run something is noted
instead of done. Look-ups (effective risk ``read``) still run for real, so the run sees real data and the user sees
exactly what a real run would do with it. The agent loop decides this in code, from the call's effective risk, after
Never rules and Read only connections and before anything else (``Agent._plan_call``): nothing a model says turns a
noted call into a real one. The only exception is the run's own question to the user (``ask_user``), which still
asks, because the answer shapes the rest of the run.

``DryRun`` lives in ``ToolContext.extra[KEY]`` for the run. ``record`` keeps one entry per noted call, with a preview
built in code (``sentient.agent.preview``), and hands the model a result that says it was only noted. ``report``
writes the run's summary from those entries, in code, so a model can't claim something was sent.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from sentient.agent import preview
from sentient.tools.base import Risk, Tool
from sentient.tools.rules import tool_title

KEY = "dry_run"
REAL_TOOLS = frozenset({"ask_user"})  # the run's own question to the user still asks
MAX_ACTIONS = 100
MAX_ARGUMENTS_CHARS = 2000
NOTED = (
    "TRY RUN: this was not really done, only noted so the user can see what a real run would do. Treat it as done "
    "and carry on with the next step. Don't try it again."
)

OPENING = (
    "Try run started. Anything that would send, post, buy, delete, change or run something is only noted, not done. "
    "Look-ups run for real."
)

_VERBS = {
    "email": "Send an email", "message": "Send a message", "post": "Post", "delete": "Delete for good",
    "trash": "Move to the trash", "purchase": "Buy", "run": "Run",
}


def simulates(tool: Tool, risk: Risk) -> bool:
    """True when a try run notes this call instead of making it: anything above a look-up, except ``ask_user``."""
    return Risk(risk) > Risk.read and tool.name not in REAL_TOOLS


def _kind(name: str, arguments: dict, risk: Risk, purchase: bool) -> str:
    if purchase:
        return "purchase"
    kind = preview.action_kind(name, arguments)
    if kind != "action":
        return kind
    return "run" if Risk(risk) >= Risk.exec else "change"


_NAME_KEYS = ("name", "filename", "path", "file", "command", "title", "purpose", "query")


def _first_text(arguments: dict) -> str:
    """A short text argument ("notes.txt", "git status") for a call whose preview names nothing: a name, path or
    command first, else the first short one."""
    def short(value: Any) -> str:
        ok = isinstance(value, str) and value.strip() and len(value) <= 120 and "\n" not in value
        return value.strip() if ok else ""

    for key in _NAME_KEYS:
        if text := short(arguments.get(key)):
            return text
    return next((text for value in arguments.values() if (text := short(value))), "")


def summary_line(entry: dict) -> str:
    """One plain line for a noted call: 'Send an email in Gmail: To sam@example.com, Subject Lunch'."""
    title = str(entry.get("title") or entry.get("tool") or "Something")
    kind = str(entry.get("kind") or "change")
    app = str(entry.get("app") or "")
    head = _VERBS.get(kind, title)
    if kind == "run" and title:
        head = title
    head += f" in {app}" if app and app.lower() not in head.lower() else ""
    prev = entry.get("preview") if isinstance(entry.get("preview"), dict) else {}
    fields = [f"{f.get('label')} {f.get('value')}" for f in prev.get("fields") or [] if f.get("value")]
    detail = ", ".join(fields) or _first_text(entry.get("arguments") or {})
    return f"{head}: {detail}" if detail else head


def _arguments(arguments: dict) -> dict:
    """The call's arguments as kept on the run; a very long one is cut so one run can't grow without bound."""
    try:
        text = json.dumps(arguments, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return {}
    if len(text) <= MAX_ARGUMENTS_CHARS:
        return dict(arguments)
    return {k: (v[:400] + "..." if isinstance(v, str) and len(v) > 400 else v) for k, v in arguments.items()}


class DryRun:
    """What a try run noted so far. ``on_change(actions)`` saves the list on the run after each new entry."""

    def __init__(
        self, actions: list[dict] | None = None, on_change: Callable[[list[dict]], Awaitable[Any]] | None = None
    ):
        self.actions: list[dict] = [a for a in (actions or []) if isinstance(a, dict)]
        self.on_change = on_change

    async def record(
        self, tool: Tool, name: str, arguments: dict | None, risk: Risk, *, purchase: bool = False,
        asks: bool = False, app: str | None = None, at: str | None = None,
    ) -> dict:
        """Note one call a real run would make and return what the model gets instead of its result."""
        args = dict(arguments or {})
        entry = {
            "tool": name,
            "plugin": tool.plugin,
            "app": app or None,
            "title": tool_title(tool),
            "kind": _kind(name, args, risk, purchase),
            "risk": Risk(risk).name,
            "preview": preview.build(name, args, purchase=purchase, app=app),
            "arguments": _arguments(args),
            "asks": bool(asks),  # a real run would stop and ask the user first (an Ask rule, outside content)
            "at": at,
        }
        entry["summary"] = summary_line(entry)
        if len(self.actions) < MAX_ACTIONS:
            self.actions.append(entry)
            if self.on_change is not None:
                await self.on_change(self.actions)
        return {"ok": True, "try_run": True, "note": NOTED}


def report(actions: list[dict] | None) -> str:
    """The summary of a finished try run, written in code from what it noted."""
    lines = [str(a.get("summary") or "") for a in actions or [] if isinstance(a, dict)]
    lines = [line for line in lines if line]
    if not lines:
        return ("Try run finished. A real run would only look things up: it wouldn't send, change, buy, delete or "
                "run anything.")
    count = f"{len(lines)} thing{'s' if len(lines) != 1 else ''}"
    listed = "\n".join(f"- {line}" for line in lines)
    return f"Try run finished. Nothing was sent, changed or run. A real run would do {count}:\n{listed}"
