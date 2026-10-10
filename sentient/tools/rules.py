"""Lasting approval rules per tool or app (``tools.approvals.rules``, ADR 0016).

A rule key is a tool name (``gmail_send_email``, ``mcp_files_delete``) or a plugin id (``gmail``,
``mcp_files``) meaning every tool of that plugin. A tool's own rule beats its plugin's rule. Values:

- ``allow``: run without asking. Purchases still ask.
- ``ask``: always ask first, even when approvals are off, after "Allow for this chat" and for look-ups.
- ``never``: the tool is not offered to the model, and a call made anyway is refused without running.

Rules are applied in code by the approvals broker, the agent loop and the sandbox bridge; a model never
decides them. These helpers are pure so every one of those places can share them without import cycles.
They also hold the outside-content helpers (ADR 0018): which tools bring in content someone else wrote,
which calls can send data out, and the plain wording for the questions that follow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from sentient.tools.base import Risk, Tool, describe_call

RULE_CHOICES = ("allow", "ask", "never")
# describe_fn label for money leaving the user (browser: buy, pay, order, subscribe, donate, transfer ...)
PURCHASE_LABEL = "Purchase"
SETTINGS_HINT = "Change this in Settings > Approvals & safety."


def rule_for(rules: Mapping[str, str] | None, tool: Tool) -> str | None:
    """The lasting rule for ``tool``: its own rule, else its plugin's rule, else None."""
    if not rules:
        return None
    rule = rules.get(tool.name)
    if rule is None:
        rule = rules.get(tool.plugin)
    return rule


def tool_title(tool: Tool) -> str:
    """A tool's plain name without its app prefix: ``slack_post_message`` -> "Post message"."""
    prefix = f"{tool.plugin}_"
    short = tool.name[len(prefix):] if tool.name.startswith(prefix) and len(tool.name) > len(prefix) else tool.name
    words = " ".join(short.replace("-", "_").split("_")).strip() or tool.name
    return words[:1].upper() + words[1:]


def rule_label(tool: Tool, rules: Mapping[str, str] | None = None, registry: Any = None) -> str:
    """Plain name for messages: the app's name ("Slack") when the rule is the app's, else the tool's
    plain name and its app ('"Post message" in Slack')."""
    plugin = registry.plugin(tool.plugin) if registry is not None else None
    app = getattr(plugin, "display_name", None) or ""
    if rules and tool.name not in rules and tool.plugin in rules:
        return app or tool.plugin
    title = f'"{tool_title(tool)}"'
    return f"{title} in {app}" if app else title


def never_message(label: str) -> str:
    return f"You've set Sentient to never use {label}. {SETTINGS_HINT}"


def unattended_ask_message(label: str) -> str:
    """Why a task run stopped: an "ask" rule, and nobody to ask (tasks cannot pause for approval yet)."""
    return f"{label} is set to Ask, and tasks can't ask yet. Change it in Settings > Approvals & safety."


# Apps whose "internal" changes turn into outside actions later (a task runs its plan on its own), so work nobody
# asked for can't use them either (ADR 0017).
UNPROMPTED_BLOCKED_PLUGINS = frozenset({"tasks"})


def unprompted_allows(tool: Tool, risk: Risk) -> bool:
    """Work nobody asked for may look things up (effective risk ``read``) and change Sentient's own things (an
    ``internal`` tool at ``write``), never send, delete, buy, run code, change anything outside Sentient or plan
    more work. Rules and approval modes never widen this."""
    risk = Risk(risk)
    if risk == Risk.read:
        return True
    return risk == Risk.write and bool(tool.internal) and tool.plugin not in UNPROMPTED_BLOCKED_PLUGINS


def unprompted_message(label: str) -> str:
    """What the model is told when work nobody asked for tries to act."""
    return (f"Nobody asked for this work, so Sentient can only look things up. {label} was not done. "
            "If it would help, suggest it to the user instead.")


# ----------------------------------------------------------------------------- outside content (ADR 0018)
# Apps whose results are Sentient's own data or plain facts nobody else writes: reading them never counts as outside
# content. Every other app's look-ups do (email, web, browser, messages, MCP servers, apps added later).
TRUSTED_PLUGINS = frozenset({
    "memory", "files", "skills", "time", "tasks", "task_questions", "subagents", "devices", "weather", "charts",
})


def brings_untrusted(tool: Tool) -> bool:
    """True when ``tool``'s result can carry content someone else wrote (an email, a web page, a message).

    ``Tool.untrusted_output`` decides when set. Otherwise Sentient-internal tools and ``TRUSTED_PLUGINS`` are
    trusted, and any other app's look-ups (base risk ``read``) are not; writes and sends return confirmations."""
    if tool.untrusted_output is not None:
        return bool(tool.untrusted_output)
    if tool.internal or tool.plugin in TRUSTED_PLUGINS:
        return False
    return Risk(tool.risk) == Risk.read


def sends_out(tool: Tool, risk: Risk, arguments: dict | None = None, ctx: Any = None) -> bool:
    """True when the call can send data out or act for the user: effective risk ``send``/``exec``, or a tool
    marked ``exfiltrates`` (typing into a web page, writes to an MCP server, a calendar invite to other people).
    A per-call ``exfiltrates`` function that fails counts as sending out."""
    if Risk(risk) >= Risk.send:
        return True
    flag = tool.exfiltrates
    if not callable(flag):
        return bool(flag)
    try:
        return bool(flag(arguments or {}, ctx))
    except Exception:
        return True


# An address "carries data" when it has a query string, a long path or a long fragment: room to smuggle text out.
MAX_CLEAN_PATH = 100
MAX_CLEAN_FRAGMENT = 40


def call_address(tool: Tool, arguments: dict, ctx: Any) -> str:
    """The web address this call loads (``Tool.url_fn``), or "" when it loads none or it can't be worked out."""
    fn = tool.url_fn
    if fn is None:
        return ""
    try:
        url = fn(arguments or {}, ctx)
    except Exception:
        return ""
    return str(url or "").strip()


def address_host(url: str) -> str:
    """The lower-case host of a web address ("example.com"); addresses without a scheme count as https."""
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        return (urlsplit(text if "://" in text else f"https://{text}").hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def address_carries_data(url: str) -> bool:
    """True when the address could carry text to its site: a query string, a path over ``MAX_CLEAN_PATH``
    characters, a fragment over ``MAX_CLEAN_FRAGMENT``, a user name in it, or an address cut short ("…")."""
    text = str(url or "").strip()
    if "…" in text:  # a link the page snapshot shortened: the hidden part could hold anything
        return True
    try:
        parts = urlsplit(text if "://" in text else f"https://{text}")
    except ValueError:
        return True
    return bool(
        parts.query or parts.username or parts.password
        or len(parts.path) > MAX_CLEAN_PATH or len(parts.fragment) > MAX_CLEAN_FRAGMENT
    )


def untrusted_source(tool: Tool, registry: Any = None) -> str:
    """Plain name of where outside content came from: the app's name ("Gmail"), else its id."""
    plugin = registry.plugin(tool.plugin) if registry is not None else None
    return getattr(plugin, "display_name", None) or tool.plugin or tool.name


UNAVAILABLE_SOURCE = "a tool that is no longer available"


def untrusted_in(messages: list[dict] | None, registry: Any) -> str:
    """The source of the first tool result in a transcript that brought outside content in, else "".

    A result whose tool is no longer registered (an MCP server that was removed) counts as outside content, since
    its kind can't be checked any more; only the engine's own "unknown tool" refusal for a made-up name does not."""
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        name = str(m.get("name") or "")
        t = registry.get(name)
        if t is None:
            if name and not str(m.get("content") or "").startswith('{"error": "unknown tool '):
                return UNAVAILABLE_SOURCE
            continue
        if brings_untrusted(t):
            return untrusted_source(t, registry)
    return ""


SCREEN_FOLDER = "screens"  # files/screens: what the user shared from their screen (window or region hotkey)
SCREEN_SOURCE = "your screen"


def is_screen_capture(name: str) -> bool:
    """True for a Files API name under ``screens/``. A screen can show text someone else wrote (an email, a web
    page), so a chat with one attached counts as having read outside content."""
    return str(name or "").replace("\\", "/").lstrip("/").startswith(f"{SCREEN_FOLDER}/")


def untrusted_reason(source: str) -> str:
    """Why a chat asks (shown on the approval card)."""
    return f"Sentient read content from {source} in this chat, so it checks with you before sending anything."


def untrusted_address_reason(source: str, host: str) -> str:
    """Why a chat asks before loading an address on a new site that could carry data out."""
    return (f"Sentient read content from {source} in this chat, and this address could carry your data to {host}, "
            "so it checks with you first.")


def untrusted_hold_message(label: str, source: str) -> str:
    """What the model is told when a run nobody can be asked in holds a call (a task run asks the user instead)."""
    return (f"Not done: this run read content from {source}, so {label} needs the user's OK first. If it is still "
            "needed after the user answers, call it again; otherwise finish and say it is waiting for the user's OK.")


def _brief(arguments: dict) -> str:
    """Up to three short ``key: value`` pairs of a call's arguments."""
    parts: list[str] = []
    for key, value in (arguments or {}).items():
        if isinstance(value, str | int | float) and not isinstance(value, bool) and str(value).strip():
            text = " ".join(str(value).split())
            parts.append(f"{key}: {text[:60]}{'...' if len(text) > 60 else ''}")
        if len(parts) == 3:
            break
    return ", ".join(parts)


def untrusted_question(label: str, source: str, arguments: dict, target: str | None = None) -> str:
    """The question a task run asks before a held call: "This task read content from Gmail, ... OK to ...?"."""
    detail = target or _brief(arguments)
    return (f"This task read content from {source}, so it checks with you before anything leaves Sentient. "
            f"OK to use {label}{f' ({detail})' if detail else ''}?")


async def is_purchase(tool: Tool, arguments: dict, ctx: Any, risk: Risk) -> bool:
    """True when the call spends money: effective risk ``send`` or higher and approval wording "Purchase"."""
    if Risk(risk) < Risk.send:
        return False
    wording = await describe_call(tool, arguments, ctx, Risk(risk))
    return wording.get("risk_label") == PURCHASE_LABEL
