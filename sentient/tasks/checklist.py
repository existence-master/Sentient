"""The setup checklist of a routine (#161): a recurring or triggered task confirms, in plain words, what it reads,
what it gives the user, what it may do without asking and what happens when what it reads isn't there.

Stored as ``tasks.setup`` (docs/API.md section 4, "Setup checklist")::

    {"source": {"kind": "page|app|folder|none", "target": "...", "label": "..."},
     "output": "What the user gets", "approval": "plan|ask", "acts": ["Gmail"],
     "if_missing": "skip|notify|retry", "confirmed_at": "...|null"}

The schedule and its time zone are the task's own ``schedule``; the checklist shows them and edits them there.

``prefill`` fills it in code from the request, the plan and the schedule (no model), so the user mostly confirms it.
``check_source`` looks, in code and before a run starts, whether the source is there: the page answers, the app is
connected, the folder exists. When it isn't, the service follows ``if_missing``: the run is ``skipped`` quietly,
skipped with a notification, or skipped and tried again later (``tasks.source_retry_minutes``, at most
``tasks.source_retries`` times, then the user is told).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from sentient.tools.base import Risk

SOURCE_KINDS = ("page", "app", "folder", "none")
APPROVALS = ("plan", "ask")
IF_MISSING = ("skip", "notify", "retry")
DEFAULT_IF_MISSING = "notify"
MAX_OUTPUT = 300
MAX_TARGET = 500
PROBE_TIMEOUT_S = 15.0
# plugins a run always has: never "the source"
_HELPERS = frozenset({"memory", "time", "files", "skills", "task_questions", "task_notepad", "tasks"})
# built-in look-ups with nothing to connect: a step using one isn't a source the checklist can check
_LOOKUPS = frozenset({"web", "internet_search", "search", "news", "weather", "maps", "charts"})
_URL = re.compile(r"https?://[^\s<>\"'`)\]]+", re.IGNORECASE)
_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/]|~[\\/])[^\s\"'<>|*?]+")


def applies(task: dict) -> bool:
    """True for a routine the checklist covers: a recurring or triggered task run by its plan or a check script, not a
    swarm or an already approved one-call task (the Daily Brief)."""
    kind = (task.get("schedule") or {}).get("type")
    fixed = (task.get("original_context") or {}).get("fixed_call")
    return kind in {"recurring", "triggered"} and task.get("task_type") in {None, "single", "script"} and not fixed


def _page_label(url: str) -> str:
    parts = urlsplit(url)
    label = (parts.netloc + parts.path).rstrip("/") or url
    return label if len(label) <= 80 else label[:77] + "..."


def _first_url(*texts: Any) -> str | None:
    for text in texts:
        for match in _URL.finditer(str(text or "")):
            url = match.group(0).rstrip(".,;:!?")
            if urlsplit(url).hostname:
                return url
    return None


def _first_path(*texts: Any) -> str | None:
    for text in texts:
        if (match := _PATH.search(str(text or ""))) is not None:
            return match.group(0).rstrip(".,;:!?")
    return None


def _display(registry: Any, plugin_id: str) -> str:
    plugin = registry.plugin(plugin_id) if plugin_id else None
    return str(getattr(plugin, "display_name", None) or plugin_id)


def _app_source(registry: Any, plugin_id: str) -> dict:
    return {"kind": "app", "target": plugin_id, "label": _display(registry, plugin_id)}


def _is_app(registry: Any, plugin_id: str) -> bool:
    """An app the user connects (an integration or an MCP server), whose connection can be checked."""
    if not plugin_id or plugin_id in _HELPERS or plugin_id in _LOOKUPS:
        return False
    plugin = registry.plugin(plugin_id)
    if plugin is None:
        return False
    return plugin_id.startswith("mcp_") or str(getattr(plugin, "auth_type", "builtin")) != "builtin"


def _reads(registry: Any, plugin_id: str) -> bool:
    """The app has look-ups (so a step using it can be where the task reads from)."""
    plugin = registry.plugin(plugin_id)
    return any(Risk(t.risk) == Risk.read for t in getattr(plugin, "tools", []) or [])


def guess_source(task: dict, registry: Any) -> dict:
    """Where the task reads from, guessed in code: a web address in the request, plan or check script; the app whose
    event starts a triggered task; a folder named in the request; else the first connected app the plan reads."""
    plan = [s for s in task.get("plan") or [] if isinstance(s, dict)]
    script = task.get("script") if isinstance(task.get("script"), dict) else {}
    texts = [task.get("original_prompt"), task.get("description"), *(s.get("description") for s in plan),
             script.get("code")]
    if (url := _first_url(*texts)) is not None:
        return {"kind": "page", "target": url, "label": _page_label(url)}
    schedule = task.get("schedule") or {}
    if schedule.get("type") == "triggered":
        source = str(schedule.get("source") or "")
        if _is_app(registry, source):
            return _app_source(registry, source)
    if (path := _first_path(task.get("original_prompt"), task.get("description"))) is not None:
        return {"kind": "folder", "target": path, "label": path}
    for step in plan:
        pid = str(step.get("tool") or "")
        if _is_app(registry, pid) and _reads(registry, pid):
            return _app_source(registry, pid)
    return {"kind": "none", "target": "", "label": ""}


def guess_output(task: dict) -> str:
    """What the user gets: the plan's last step, else the task's description."""
    plan = [s for s in task.get("plan") or [] if isinstance(s, dict) and str(s.get("description") or "").strip()]
    text = str(plan[-1]["description"] if plan else task.get("description") or task.get("name") or "").strip()
    text = " ".join(text.split())
    return text if len(text) <= MAX_OUTPUT else text[: MAX_OUTPUT - 3] + "..."


def acting_apps(task: dict, registry: Any) -> list[str]:
    """The apps in the plan that can send, post, buy, delete or run something: what "go ahead" lets it do."""
    out: list[str] = []
    for step in task.get("plan") or []:
        pid = str((step or {}).get("tool") or "") if isinstance(step, dict) else ""
        plugin = registry.plugin(pid) if pid and pid not in _HELPERS else None
        if plugin is None:
            continue
        if any(Risk(t.risk) >= Risk.send and not getattr(t, "internal", False) for t in plugin.tools or []):
            name = _display(registry, pid)
            if name not in out:
                out.append(name)
    return out


def prefill(task: dict, registry: Any, previous: dict | None = None) -> dict:
    """The checklist filled in from the request, plan and schedule. Choices the user already made (what it may do,
    what happens when the source is missing) are kept; a changed plan has to be confirmed again."""
    previous = previous if isinstance(previous, dict) else {}
    return {
        "source": guess_source(task, registry),
        "output": guess_output(task),
        "approval": previous.get("approval") if previous.get("approval") in APPROVALS else "plan",
        "acts": acting_apps(task, registry),
        "if_missing": previous.get("if_missing") if previous.get("if_missing") in IF_MISSING else DEFAULT_IF_MISSING,
        "confirmed_at": None,
    }


def _clean_source(value: Any, registry: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("The source must be an object with a kind and a target.")  # noqa: TRY004 (a 400)
    kind = str(value.get("kind") or "").strip()
    target = str(value.get("target") or "").strip()[:MAX_TARGET]
    if kind not in SOURCE_KINDS:
        raise ValueError("The source kind must be page, app, folder or none.")
    if kind == "none":
        return {"kind": "none", "target": "", "label": ""}
    if not target:
        raise ValueError("Say which page, app or folder the task reads from.")
    if kind == "page":
        url = target if "://" in target else f"https://{target}"
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("The page must be a web address starting with http:// or https://.")
        return {"kind": "page", "target": url, "label": _page_label(url)}
    if kind == "app":
        if registry.plugin(target) is None:
            raise ValueError(f"There is no app called '{target}'.")
        return _app_source(registry, target)
    return {"kind": "folder", "target": target, "label": target}


def merge(current: dict | None, answers: Any, registry: Any, now: str) -> dict:
    """The checklist after the user's answers (``PATCH {setup}``): any of ``source``, ``output``, ``approval``,
    ``if_missing``, and ``confirm: true`` to mark it confirmed. Raises ValueError for a value it can't take."""
    if not isinstance(answers, dict):
        raise ValueError("The setup checklist must be an object.")  # noqa: TRY004 (a 400)
    out = dict(current or {})
    if "source" in answers:
        out["source"] = _clean_source(answers["source"], registry)
    if "output" in answers:
        text = " ".join(str(answers["output"] or "").split())
        out["output"] = text[:MAX_OUTPUT]
    if "approval" in answers:
        if answers["approval"] not in APPROVALS:
            raise ValueError("Choose whether it goes ahead with the plan (plan) or asks you first (ask).")
        out["approval"] = answers["approval"]
    if "if_missing" in answers:
        if answers["if_missing"] not in IF_MISSING:
            raise ValueError("Choose skip, notify or retry for when the source is missing.")
        out["if_missing"] = answers["if_missing"]
    out.setdefault("source", {"kind": "none", "target": "", "label": ""})
    out.setdefault("output", "")
    out.setdefault("approval", "plan")
    out.setdefault("acts", [])
    out.setdefault("if_missing", DEFAULT_IF_MISSING)
    out.setdefault("confirmed_at", None)
    if answers.get("confirm"):
        out["confirmed_at"] = now
    return out


def asks_first(task: dict) -> bool:
    """The user set this routine to ask before it sends, posts, buys, deletes or runs anything."""
    setup = task.get("setup") if isinstance(task.get("setup"), dict) else {}
    return applies(task) and setup.get("approval") == "ask"


def prompt_lines(task: dict) -> str:
    """What the user confirmed, for the executor's prompt (short: small models read it)."""
    setup = task.get("setup") if isinstance(task.get("setup"), dict) else {}
    if not applies(task) or not setup:
        return ""
    lines = []
    source = setup.get("source") or {}
    if source.get("kind") in {"page", "folder"}:
        lines.append(f"- Read from: {source.get('target')}")
    elif source.get("kind") == "app":
        lines.append(f"- Read from: {source.get('label')}")
    if str(setup.get("output") or "").strip():
        lines.append(f"- The user expects: {setup['output']}")
    return ("**What the user confirmed for this task:**\n" + "\n".join(lines) + "\n\n") if lines else ""


# ---------------------------------------------------------------------------- is the source there?
async def probe_page(url: str) -> str | None:
    """None when the page answers, else why not in plain words. Uses the same address rules as reading a page: an
    address Sentient never opens itself (this computer, network internals) isn't checked at all."""
    from sentient.integrations.base import IntegrationError
    from sentient.integrations.common import http_client
    from sentient.integrations.plugins import web

    label = _page_label(url)
    try:
        target = await asyncio.to_thread(web._check_url, url)
    except IntegrationError as exc:
        if "find the website" in str(exc):
            return f"the website for {label} couldn't be found"
        return None  # not an address Sentient opens: nothing to judge here
    try:
        async with http_client(timeout=PROBE_TIMEOUT_S) as http, http.stream("GET", target) as response:
            status = response.status_code
    except httpx.HTTPError:
        return f"the page {label} couldn't be reached"
    if status >= 400:
        return f"the page {label} answered with an error ({status})"
    return None


async def check_source(app: Any, task: dict) -> str | None:
    """None when the task's source is there (or it has none to check), else the plain reason it is missing."""
    setup = task.get("setup") if isinstance(task.get("setup"), dict) else {}
    source = setup.get("source") if applies(task) and isinstance(setup.get("source"), dict) else {}
    kind, target = source.get("kind"), str(source.get("target") or "")
    if kind == "page" and target:
        return await probe_page(target)
    if kind == "app" and target:
        integrations = getattr(app, "integrations", None)
        if integrations is None or not hasattr(integrations, "is_connected"):
            return None
        if not await integrations.is_connected(target):
            return f"{source.get('label') or target} isn't connected"
        return None
    if kind == "folder" and target:
        path = Path(target).expanduser()
        exists = await asyncio.to_thread(path.exists)
        return None if exists else f"the folder or file {target} isn't there"
    return None


def missing_phrase(policy: str) -> str:
    """What a real run does when the source is missing, for a try run's report and the checklist."""
    return {
        "skip": "skip that run quietly",
        "notify": "skip that run and tell you",
        "retry": "skip that run and try again a little later",
    }.get(policy, "skip that run and tell you")
