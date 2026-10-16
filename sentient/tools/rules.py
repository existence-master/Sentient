"""Lasting approval rules per tool or app (``tools.approvals.rules``, ADR 0016).

A rule key is a tool name (``gmail_send_email``, ``mcp_files_delete``) or a plugin id (``gmail``,
``mcp_files``) meaning every tool of that plugin. A tool's own rule beats its plugin's rule. Values:

- ``allow``: run without asking. Purchases still ask.
- ``if_asked`` ("Only if I asked", #140): like ``allow`` in a chat turn whose user words ask for this kind of action
  (``sentient.agent.rule_levels.asked_for``), like ``ask`` everywhere else: tasks, subagents, scripts, proactivity.
- ``ask``: always ask first, even when approvals are off, after "Allow for this chat" and for look-ups.
- ``hand_off`` ("Hand it to me", #140): a call that would change, send, delete or run something never runs; Sentient
  prepares it and hands it to the user to do themselves. Look-ups run as usual.
- ``never``: the tool is not offered to the model, and a call made anyway is refused without running.

A connection set to Read only (``integrations.read_only``, #141) works like a "never" rule on every tool of that app
that can change, send, delete or run something: ``read_only_hides`` and ``read_only_blocks``.

Rules are applied in code by the approvals broker, the agent loop and the sandbox bridge; a model never
decides them. These helpers are pure so every one of those places can share them without import cycles.
They also hold the outside-content helpers (ADR 0018): which tools bring in content someone else wrote,
which calls can send data out, and the plain wording for the questions that follow.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from pathlib import PurePath
from typing import Any
from urllib.parse import urlsplit

from sentient.tools.base import Risk, Tool, describe_call

ALLOW, IF_ASKED, ASK, HAND_OFF, NEVER = "allow", "if_asked", "ask", "hand_off", "never"
RULE_CHOICES = (ALLOW, IF_ASKED, ASK, HAND_OFF, NEVER)
# the plain name of each level, as Settings and the "Make this a rule?" card show it
RULE_TITLES = {
    ALLOW: "Allow", IF_ASKED: "Only if I asked", ASK: "Always ask", HAND_OFF: "Hand it to me", NEVER: "Never",
}
# describe_fn label for money leaving the user (browser: buy, pay, order, a priced subscribe, donate, transfer ...)
PURCHASE_LABEL = "Purchase"
# describe_fn labels for a call that opens the user's home (#175, ADR 0028): unlocking a door, turning off the alarm,
# opening a garage door, gate or door, or switching something that sounds like one. Like a purchase, it always asks.
UNLOCKS_LABEL = "Unlocks a door"
DISARMS_LABEL = "Turns off your alarm"
OPENS_GATE_LABEL = "Opens a garage door or gate"
SECURITY_SWITCH_LABEL = "Controls a door, lock or alarm"
OPENS_HOME_LABELS = frozenset({UNLOCKS_LABEL, DISARMS_LABEL, OPENS_GATE_LABEL, SECURITY_SWITCH_LABEL})
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


def applied_rule(rule: str | None, *, asked: bool = False, risk: Risk | None = None) -> str | None:
    """The rule a call is treated with: "Only if I asked" is Allow when the user's own words in this chat turn asked
    for the action (``asked``), else Ask. "Hand it to me" leaves a look-up (effective risk ``read``) to the normal
    setting, and makes anything else ask wherever it was not handed over first, so it never runs unasked. Every other
    rule is itself."""
    if rule == IF_ASKED:
        return ALLOW if asked else ASK
    if rule == HAND_OFF:
        return None if risk is not None and Risk(risk) <= Risk.read else ASK
    return rule


def hands_off(rule: str | None, risk: Risk) -> bool:
    """True when a call with effective risk ``risk`` must be handed to the user instead of run (#140): a "Hand it to
    me" rule on a call that would change, send, delete or run something. Look-ups run as usual."""
    return rule == HAND_OFF and Risk(risk) > Risk.read


def hand_off_message(label: str) -> str:
    """What the model is told when a call was handed to the user instead of run (#140)."""
    return (f'NOT DONE by Sentient: {label} is set to "Hand it to me", so Sentient did not do it. What you prepared '
            "is shown to the user in Sentient, with a copy button, for them to finish themselves. Tell the user it is "
            "ready for them there. Don't try to do it another way.")


# ----------------------------------------------------------------------------- read-only connections (#141)
READ_ONLY_HINT = "Change this in Integrations."


def read_only_hides(read_only: Collection[str] | None, tool: Tool) -> bool:
    """True when ``tool`` belongs to a connection set to Read only and can never be a look-up: its risk is above
    ``read`` and does not depend on the call. A tool whose risk depends on its arguments (``risk_fn``) stays offered
    and each call is checked with ``read_only_blocks``. Like a "never" rule on the app's other tools (ADR 0016)."""
    return bool(read_only) and tool.plugin in read_only and Risk(tool.risk) > Risk.read and tool.risk_fn is None


def read_only_blocks(read_only: Collection[str] | None, tool: Tool, risk: Risk) -> bool:
    """True when a call with effective risk ``risk`` must not run because its connection is Read only."""
    return bool(read_only) and tool.plugin in read_only and Risk(risk) > Risk.read


def read_only_message(tool: Tool, registry: Any = None) -> str:
    """What the model is told when a call to a Read only connection is refused."""
    plugin = registry.plugin(tool.plugin) if registry is not None else None
    app = getattr(plugin, "display_name", None) or tool.plugin
    return (f"{app} is set to Read only, so Sentient can look things up there but can't change, send or delete "
            f'anything. "{tool_title(tool)}" was not done. {READ_ONLY_HINT}')


# not a rule a user sets: why a call that opens the home is held (ADR 0028), for the hold and question wording
OPENS_HOME = "opens_home"


def _set_to(rule: str | None) -> str:
    """ "set to Ask", or for "Only if I asked" a sentence that says only a chat message from the user asks for it."""
    if rule == IF_ASKED:
        return "set to \"Only if I asked\", which only the user's own chat message can ask for"
    if rule == OPENS_HOME:
        return "something that opens the user's home, which always asks first"
    return "set to Ask"


def unattended_ask_message(label: str, rule: str | None = ASK) -> str:
    """Why an unattended run (a subagent, a swarm worker) stopped: an "ask" (or "only if I asked") rule, and nobody to
    ask. Single task runs pause and ask instead (#139)."""
    return f"{label} is {_set_to(rule)}, and this work can't stop to ask. Change it in Settings > Approvals & safety."


def approval_hold_message(label: str, rule: str | None = ASK) -> str:
    """What the model is told when a task run holds a call for an "ask" (or "only if I asked") rule; the run then
    pauses and asks (#139)."""
    return (f"Not done yet: {label} is {_set_to(rule)}, so the user is being asked first. If it is still needed after "
            "the user answers, it runs then; otherwise finish and say it is waiting for the user's OK.")


def approval_question(label: str, arguments: dict, target: str | None = None, rule: str | None = ASK) -> str:
    """The question a task run asks before a call held by an "ask" (or "only if I asked") rule (#139, #140)."""
    detail = target or _brief(arguments)
    if rule == IF_ASKED:
        why = "You've set it to go ahead only when you ask for it in a chat."
    elif rule == OPENS_HOME:
        why = "It opens your home, so Sentient always asks first."
    else:
        why = "You've set it to ask first."
    return f"This task wants to use {label}{f' ({detail})' if detail else ''}. {why} OK to go ahead?"


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
    its kind can't be checked any more; only the engine's own "unknown tool" refusal for a made-up name does not.
    A ``file_read`` of a saved result or a download counts too, matched to its call's arguments (#336)."""
    calls = {  # call id -> arguments, from the assistant messages that asked for them
        str(tc.get("id")): (tc.get("function") or {}).get("arguments")
        for m in messages or [] if isinstance(m, dict) and m.get("role") == "assistant"
        for tc in m.get("tool_calls") or [] if isinstance(tc, dict)
    }
    for m in messages or []:
        source = outside_source(m, registry) or _saved_read(m, calls)
        if source:
            return source
    return ""


def _saved_read(message: Any, calls: dict[str, Any]) -> str:
    """``saved_outside_source`` of a stored ``file_read`` tool message that read something, else ""."""
    if not isinstance(message, dict) or message.get("role") != "tool" or message.get("name") != "file_read":
        return ""
    if str(message.get("content") or "").startswith('{"error"'):
        return ""
    raw = calls.get(str(message.get("tool_call_id")))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except ValueError:
            return ""
    return saved_outside_source("file_read", raw)


def outside_source(message: Any, registry: Any) -> str:
    """Where a tool message came from when it brought outside content in ("Gmail", ``UNAVAILABLE_SOURCE``), else ""."""
    if not isinstance(message, dict) or message.get("role") != "tool":
        return ""
    name = str(message.get("name") or "")
    t = registry.get(name)
    if t is None:
        if name and not str(message.get("content") or "").startswith('{"error": "unknown tool '):
            return UNAVAILABLE_SOURCE
        return ""
    if brings_untrusted(t):
        return untrusted_source(t, registry)
    return result_outside_source(message.get("content"))


# A result of one of Sentient's own tools that says it holds text someone else wrote (ADR 0025): a web page clipped
# into the user's notes. The key comes first in the result, so a cut result keeps it.
OUTSIDE_KEY = "outside_content"
_OUTSIDE_PREFIX = re.compile(r'^\{\s*"outside_content"\s*:\s*"((?:[^"\\]|\\.){1,200})"')


def result_outside_source(result: Any) -> str:
    """Where a result's outside content came from when the result says so itself (``{"outside_content": source,
    ...}``, as a dict or the stored JSON text), else "". It only ever adds a mark, never removes one."""
    if isinstance(result, dict):
        value = result.get(OUTSIDE_KEY)
        return str(value)[:200] if isinstance(value, str) and value.strip() else ""
    if isinstance(result, str):
        m = _OUTSIDE_PREFIX.match(result)
        if m:
            try:
                return str(json.loads(f'"{m.group(1)}"'))
            except ValueError:
                return "outside content"
    return ""


# Folders under files/ that hold text someone else wrote (#297): the rest of a long tool result (agent/loop.py
# ``_tool_content``) and browser downloads. ``file_read`` of a file there is marked as outside content (ADR 0023).
SAVED_OUTSIDE_FOLDERS = {"outputs": "a saved tool result", "downloads": "a downloaded file"}


def saved_outside_source(name: str, arguments: Any) -> str:
    """Where a ``file_read`` result came from when it reads a saved result or a download ("a downloaded file"),
    else "". Only those folders: the user's own files and Sentient's notes are not outside content."""
    if name != "file_read" or not isinstance(arguments, dict):
        return ""
    path = PurePath(str(arguments.get("name") or ""))  # read the way file_read's own path check reads it
    if path.anchor:
        return ""  # absolute or another drive: file_read refuses it, so nothing is read
    parts: list[str] = []
    for part in path.parts:
        if part == "..":
            if not parts:
                return ""  # escapes the files folder
            parts.pop()
        elif part != ".":
            parts.append(part)
    if not parts:
        return ""
    top = PurePath(parts[0])  # case-insensitive on Windows, like the file system
    return next((source for folder, source in SAVED_OUTSIDE_FOLDERS.items() if top == PurePath(folder)), "")


SCREEN_FOLDER = "screens"  # files/screens: what the user shared from their screen (window or region hotkey)
SCREEN_SOURCE = "your screen"


def is_screen_capture(name: str) -> bool:
    """True for a Files API name under ``screens/``. A screen can show text someone else wrote (an email, a web
    page), so a chat with one attached counts as having read outside content."""
    return str(name or "").replace("\\", "/").lstrip("/").startswith(f"{SCREEN_FOLDER}/")


SELECTION_FOLDER = "selections"  # files/selections: text the user selected in another app (Ask about selected text)
SELECTION_SOURCE = "another app"


def is_selection(name: str) -> bool:
    """True for a Files API name under ``selections/``: text the user selected in another app and asked about. Someone
    else may have written it (an email, a web page), so it reaches the model marked and marks the chat (ADR 0018)."""
    return str(name or "").replace("\\", "/").lstrip("/").startswith(f"{SELECTION_FOLDER}/")


def attachment_source(name: str) -> str:
    """The outside source an attachment brings in (``SCREEN_SOURCE``, ``SELECTION_SOURCE``), else ""."""
    if is_screen_capture(name):
        return SCREEN_SOURCE
    if is_selection(name):
        return SELECTION_SOURCE
    return ""


# A message someone other than the user wrote in a paired group chat (ADR 0029) is stored with this role and the
# sender's name in ``messages.name``. The model reads it as outside content inside a user turn, never as the user's words.
OTHER_ROLE = "other"
# A chat's log of something the user did to it (``messages.role``), never sent to the model. ``trusted``: the user
# cleared the chat's outside-content mark ("I've checked it, trust this chat again", ADR 0029).
EVENT_ROLE = "event"
TRUSTED_EVENT = "trusted"


def group_source(sender: str) -> str:
    """Where a group member's message came from, as the approval card and the chat's mark name it:
    "another group member (Ravi)". The name is the member's own display name, so it is cleaned and kept short."""
    name = " ".join(re.sub(r"[\[\]()\n\r]", " ", str(sender or "")).split())[:40]
    return f"another group member ({name})" if name else "another group member"


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


async def opens_home(tool: Tool, arguments: dict, ctx: Any, risk: Risk) -> bool:
    """True when the call opens the user's home (#175, ADR 0028): effective risk ``send`` or higher and approval
    wording in ``OPENS_HOME_LABELS`` (unlock a door, turn off the alarm, open a garage door or gate)."""
    if Risk(risk) < Risk.send:
        return False
    wording = await describe_call(tool, arguments, ctx, Risk(risk))
    return wording.get("risk_label") in OPENS_HOME_LABELS


async def always_asks(tool: Tool, arguments: dict, ctx: Any, risk: Risk) -> bool:
    """A hard stop: a purchase, or a call that opens the user's home. It asks whatever the approvals mode, an Allow
    rule, "Allow for this chat" or a grant say, and a script can never make it."""
    return await is_purchase(tool, arguments, ctx, risk) or await opens_home(tool, arguments, ctx, risk)
