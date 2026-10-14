"""The two conditional rule levels (#140, ADR 0016): "Only if I asked" and "Hand it to me".

**Only if I asked** (``if_asked``): a tool runs without asking only in a chat turn whose user words ask for this kind of
action; everywhere else (tasks, subagents, scripts, proactivity, a chat turn that asked for something else) it asks
like an Ask rule. ``asked_for`` decides in code, never by a model, from the words the user typed or spoke in this turn
(``ToolContext.user_words``): never from the assistant's text, a tool result, an attachment or outside content.

- A call that changes, sends, deletes or runs something counts as asked for when one clause of the user's words names
  its action with a verb ("send", "email Priya", "delete", "post"), and either names what it acts on (a word of the
  tool's name other than the verb, or its app: "send Priya an email" for ``gmail_send_email``; an email address counts
  as naming an email) or names no other connected app ("yes, send it" after Sentient showed the draft; but "send it on
  Slack" never asks for an email). A word that can be a noun only counts as a verb where a verb goes: "summarize my
  Slack messages" asks for no message to be sent.
- A look-up counts as asked for when a clause names what it reads or its app ("what's in my inbox?").
- Clauses that start with a negation ("don't send it") or an ask-first ("ask me before you send it") never ask for
  anything. When nothing matches, the call asks: a missed match costs one question, a wrong match an unasked action.

**Hand it to me** (``hand_off``): a call that would change, send, delete or run something never runs. Its result tells
the model it was not done and carries a preview built in code from the call's own arguments (``preview.build``), which
the window shows with Copy and, for an email, Open in your email app. Look-ups run as usual.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from sentient.agent import preview
from sentient.agent.chat_rules import (
    _ASK_FIRST,
    _BENIGN,
    _DETERMINERS,
    _NEGATION,
    _STOP,
    _WORD,
    _cut,
    _match,
    _plain,
    _stem,
    _terms,
)
from sentient.tools.base import Risk, Tool
from sentient.tools.rules import hand_off_message, tool_title

# the key a handed-off call's result carries (the window shows a "Ready for you" card for it)
HANDED_OFF = "handed_off"
MAX_DETAIL = 2000

# a verb the user says -> the words a tool's name uses for that action
_FAMILIES: dict[str, frozenset[str]] = {
    "send": frozenset({"send", "reply", "forward", "post", "publish", "message"}),
    "email": frozenset({"send", "reply", "forward"}),
    "mail": frozenset({"send", "reply", "forward"}),
    "reply": frozenset({"reply", "send", "respond"}),
    "respond": frozenset({"reply", "send", "respond"}),
    "forward": frozenset({"forward", "send"}),
    "post": frozenset({"post", "publish", "send", "create", "message"}),
    "publish": frozenset({"publish", "post"}),
    "tweet": frozenset({"post", "tweet", "create"}),
    "message": frozenset({"send", "post", "message"}),
    "text": frozenset({"send", "message"}),
    "dm": frozenset({"send", "message"}),
    "comment": frozenset({"comment", "create", "post"}),
    "draft": frozenset({"draft", "create", "write", "compose"}),
    "write": frozenset({"write", "create", "draft", "compose", "append", "save"}),
    "compose": frozenset({"compose", "create", "draft"}),
    "create": frozenset({"create", "add", "insert", "new", "make"}),
    "add": frozenset({"add", "create", "insert", "append"}),
    "make": frozenset({"create", "make", "add"}),
    "save": frozenset({"save", "write", "create", "add"}),
    "schedule": frozenset({"schedule", "create", "add", "book"}),
    "book": frozenset({"book", "create", "schedule", "reserve"}),
    "invite": frozenset({"invite", "create", "add"}),
    "delete": frozenset({"delete", "trash", "remove", "erase"}),
    "remove": frozenset({"remove", "delete", "trash"}),
    "erase": frozenset({"erase", "delete", "remove", "trash"}),
    "trash": frozenset({"trash", "delete", "remove"}),
    "archive": frozenset({"archive", "modify", "label", "move"}),
    "label": frozenset({"label", "modify", "tag"}),
    "mark": frozenset({"mark", "modify", "label", "update"}),
    "move": frozenset({"move", "modify", "update", "rename"}),
    "update": frozenset({"update", "edit", "modify", "patch", "change", "set"}),
    "edit": frozenset({"edit", "update", "modify", "patch", "change"}),
    "change": frozenset({"change", "update", "edit", "modify", "set"}),
    "rename": frozenset({"rename", "update", "move"}),
    "set": frozenset({"set", "update", "change", "create"}),
    "cancel": frozenset({"cancel", "delete", "decline"}),
    "share": frozenset({"share", "send", "invite"}),
    "upload": frozenset({"upload", "create", "write", "add"}),
    "run": frozenset({"run", "execute", "exec", "start"}),
    "execute": frozenset({"execute", "run", "exec"}),
    "pay": frozenset({"pay", "payment", "purchase", "buy", "order", "checkout"}),
    "buy": frozenset({"buy", "purchase", "order", "checkout", "pay"}),
    "purchase": frozenset({"purchase", "buy", "order", "checkout", "pay"}),
    "order": frozenset({"order", "purchase", "buy", "checkout"}),
    "transfer": frozenset({"transfer", "send", "pay"}),
    "submit": frozenset({"submit", "send", "create"}),
    "close": frozenset({"close", "update", "resolve"}),
    "merge": frozenset({"merge"}),
    "install": frozenset({"install"}),
    "unsubscribe": frozenset({"unsubscribe"}),
    "fill": frozenset({"fill", "type", "submit"}),
    "type": frozenset({"type", "fill"}),
    "click": frozenset({"click", "press"}),
    "press": frozenset({"press", "click"}),
}
# every word a tool's name may use for an action: none of them names what the tool acts on
_ACTION_WORDS = frozenset().union(*_FAMILIES.values()) | frozenset(_FAMILIES)
# action words that also name a thing a tool acts on ("slack_send_message" sends a message)
_NOUNS_TOO = frozenset({
    "message", "post", "comment", "text", "dm", "draft", "label", "order", "mail", "email", "tweet", "invite",
    "payment", "event", "reply", "transfer",
})
# words in tool names and app ids that say nothing about what a tool acts on
_GENERIC = frozenset({"mcp", "api", "tool", "tools", "app", "the", "and", "for", "with", "from", "into", "via"})
# little words between requests ("draft it, then send it"): never an object that would make the next verb a noun
_FILLERS = frozenset({
    "then", "also", "now", "just", "quickly", "kindly", "pls", "go", "ahead", "after", "that", "and", "yes", "yeah",
    "yep", "sure", "okay", "alright", "great", "fine", "good", "perfect", "thanks", "looks", "sounds", "right",
})
_EMAIL_ADDRESS = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.[a-z]{2,}")


def _verb_base(word: str) -> str | None:
    """The verb ``word`` is a form of ("sending" -> send), when it names an action at all."""
    if word in _FAMILIES:
        return word
    stem = _stem(word)
    if stem in _FAMILIES:
        return stem
    if word.endswith("ed") and word[:-1] in _FAMILIES:  # "archived" -> archive
        return word[:-1]
    return None


def requested_verbs(clause: str) -> set[str]:
    """The actions a clause asks for, as the words tool names use ("email Priya" -> send, reply, forward). A word that
    can be a noun only counts where a verb goes: not right after a determiner or another noun ("my Slack messages")."""
    out: set[str] = set()
    prev_noun = False
    for word in _WORD.findall(clause):
        base = _verb_base(word)
        if base is not None and not prev_noun:
            out |= _FAMILIES[base]
            prev_noun = False
        else:
            prev_noun = word in _DETERMINERS or base is not None or (
                word not in _STOP and word not in _FILLERS and len(word) > 1
            )
    return {_stem(v) for v in out}


def clauses(words: Iterable[str]) -> list[str]:
    """The user's words cut into clauses, in plain lower case, without the ones that say not to do something ("don't
    send it") or to ask first ("ask me before you send it"): those never ask for an action."""
    out: list[str] = []
    for text in words:
        for part in _cut(str(text or "")):
            plain = " ".join(_plain(part).split())
            if not plain:
                continue
            if (_NEGATION.match(plain) and not _BENIGN.match(plain)) or _ASK_FIRST.match(plain):
                continue
            out.append(plain)
    return out


def _short_words(tool: Tool) -> list[str]:
    prefix = f"{tool.plugin}_"
    short = tool.name[len(prefix):] if tool.name.startswith(prefix) and len(tool.name) > len(prefix) else tool.name
    return [w for w in re.split(r"[^a-z0-9]+", short.lower()) if w]


def _app_tokens(tool: Tool, app_name: str) -> set[str]:
    text = f"{tool.plugin.replace('_', ' ')} {app_name}"
    return {_stem(w) for w in _WORD.findall(_plain(text)) if len(w) >= 3 and w not in _GENERIC}


def asked_for(
    tool: Tool, words: Iterable[str] | None, app_name: str = "", risk: Risk | None = None,
    other_apps: Iterable[str] = (),
) -> bool:
    """True when the user's own words in this chat turn ask for this call's kind of action (module docstring).
    ``risk`` is the call's effective risk (the tool's own risk when not given); ``other_apps`` are the names and ids of
    the other connected apps, which a bare "send it" must not name. Deterministic; no model."""
    parts = clauses(words or [])
    if not parts:
        return False
    names = _short_words(tool)
    verbs = {_stem(w) for w in names if w in _ACTION_WORDS or _stem(w) in _ACTION_WORDS}
    objects = {
        _stem(w) for w in names
        if len(w) >= 3 and w not in _GENERIC and (_stem(w) not in verbs or w in _NOUNS_TOO)
    }
    objects |= _app_tokens(tool, app_name)
    elsewhere = {_stem(w) for name in other_apps for w in _WORD.findall(_plain(str(name).replace("_", " ")))
                 if len(w) >= 3 and w not in _GENERIC} - objects
    look_up = Risk(tool.risk if risk is None else risk) == Risk.read
    for clause in parts:
        terms = _terms(clause)
        address = bool(_EMAIL_ADDRESS.search(clause))
        if address:
            terms |= {"email", "mail"}
        named = any(_match(t, o) for t in terms for o in objects)
        if look_up:
            if named:
                return True
            continue
        if not any(_match(a, v) for a in requested_verbs(clause) for v in verbs):
            continue
        if named:
            return True
        # a bare "send it": the action is named, and nothing in the clause points at another app or an email address
        if not address and not any(_match(t, o) for t in terms for o in elsewhere):
            return True
    return False


def hand_off_result(tool: Tool, arguments: dict | None, label: str, *, app_name: str = "",
                    purchase: bool = False) -> dict[str, Any]:
    """The result of a call handed to the user instead of run (#140): a plain "not done" for the model, and what the
    user needs to finish it themselves. The preview is built in code from the call's own arguments."""
    shown = preview.build(tool.name, arguments, purchase=purchase, app=app_name or None)
    out: dict[str, Any] = {
        HANDED_OFF: True,
        "message": hand_off_message(label),
        "label": label,
        "title": tool_title(tool),
        "preview": shown,
    }
    if shown is None:  # nothing a person reads in it (a command, a click): show the plain details instead
        details = [
            {"label": str(k).replace("_", " ").capitalize(), "value": str(v)[:MAX_DETAIL]}
            for k, v in (arguments or {}).items()
            if isinstance(v, str | int | float) and not isinstance(v, bool) and str(v).strip()
        ]
        out["details"] = details[:12]
    return out


def is_handed_off(result: Any) -> bool:
    return isinstance(result, dict) and result.get(HANDED_OFF) is True
