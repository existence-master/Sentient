"""Rules from chat (#130, ADR 0016): "never delete my emails" said in a chat becomes a proposed lasting rule.

Anything said once in a chat can be lost when the conversation is summarized; a rule in code cannot. When a user
message reads like a standing prohibition or a must-ask instruction about an action, Sentient proposes the matching
Never, Ask, "Only if I asked" or "Hand it to me" rule as a card in the chat. Only the user's click (``decide``) creates
the rule. The model only maps the words to rule keys, and every key is checked against the tool registry in code;
nothing a model says can create, accept or loosen a rule. The level is decided in code from the words (``level_for``):
the model only ever picks between Never and Ask.

Checks, cheapest first:

1. ``standing_text``: a deterministic pre-filter (never, don't, do not, always ask, ask me first, unless I ask, I'll
   send it myself ...). Most messages stop here without a model call. A sentence that limits itself to this request
   ("don't cc her this time", "do not open it again in part A") is no standing instruction and is left out.
2. ``candidate_tools``: tools whose names or descriptions share a word with the message (plus a few synonyms). No
   candidates, no model call ("I never eat breakfast").
3. One short ``fast``-role prompt maps the words to candidate keys and a level, parsed tolerantly. Keys outside the
   candidates, and keys an equal or stricter rule already covers, are dropped. Nothing valid, no card. When the fast
   model can't do this job (Claude Code only answers chats, ADR 0022), the chat's own model or the primary model does
   (``detect``, #311): the check is part of answering the user's chat, which waits for it.
4. ``narrow_app_keys``: a whole app is proposed only when the words name no specific action ("never use Slack").
   When they do ("never delete my emails"), an app key becomes that app's tools whose names match the action, and
   when they name a tool exactly ("don't call browser_open"), that tool. A tool key stands only when the words name
   an action, its app or the tool itself.

Until the user decides, a pending proposal makes its chat ask before the matched tools (``chat_rule``), also after a
restart or once the conversation has been summarized. While a check is still running (a slow fast model), the tools
it is weighing ask too, so a reply that stops waiting for it fails closed. Both only ever add a question. With
approvals mode "off" they ask only before calls that would change, send, delete or run something (``pending_asks``):
a proposal is a guess from the chat that the user has not confirmed, and a look-up can't do what it guards against,
so "off" keeps meaning no questions for reading (#306).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterable
from typing import Any

from sentient.config.schema import RuleOrigin
from sentient.llm import claude_code
from sentient.llm.provider import ModelRefused, parse_json_loose
from sentient.store.db import new_id, now_iso
from sentient.tools.base import Risk, Tool
from sentient.tools.rules import ALLOW, ASK, HAND_OFF, IF_ASKED, NEVER, rule_for, tool_title

log = logging.getLogger(__name__)

PENDING, ACCEPTED, DECLINED = "pending", "accepted", "declined"
LEVELS = (IF_ASKED, ASK, HAND_OFF, NEVER)
# how strict a rule is; a proposal is only made (and only applied) where it tightens. "Only if I asked" sits just above
# no rule: it never lets work nobody asked for in a chat run unasked; "Hand it to me" sits between Ask and Never: it
# never runs the action, but Sentient may still prepare it
STRICTNESS: dict[str | None, int] = {ALLOW: 0, None: 1, IF_ASKED: 2, ASK: 3, HAND_OFF: 4, NEVER: 5}
# longest a reply waits for its message's rule check (loop.RULE_CHECK_WAIT_S). Claude Code answers the check only
# inside that wait: it never works on after the reply moved on (ADR 0022)
WAIT_S = 20.0
MAX_CANDIDATES = 24
MAX_KEYS = 8
MAX_SAID = 300
DESCRIPTION_CHARS = 90

# a standing instruction: a negation, or an explicit "ask me first"
_NEGATION = re.compile(r"\b(never|don't|dont|do not|must not|mustn't|should not|shouldn't|no more)\b")
_ASK_FIRST = re.compile(
    r"\b(always (ask|check)|ask me (first|before)|check with me|without (asking|checking)|without my "
    r"(ok|okay|permission|approval|go-ahead))\b"
)
# everyday phrases with a "don't" that are no prohibition ("if you don't know, say so", "don't worry"): left out
# before the negation check, so they never reach the model
_BENIGN = re.compile(r"\b(if (you|u) (don't|dont|do not)|(don't|dont|do not) (know|worry|mind|forget|hesitate))\b")
# "don't ask me", "stop asking": the user wants fewer questions, which a proposal can never give
_LOOSEN = re.compile(r"\b(don't|dont|do not|never|no need to|stop|quit) (ask|asking|check|checking)\b")
# words that make the level "ask" whatever the model says: the user named a condition, not a ban
_ASK_HINT = re.compile(
    r"\b(ask|asking|check with me|checking with me|my (ok|okay|permission|approval|go-ahead)|unless i)\b"
)
# "unless I ask", "only when I tell you to": the rule is "Only if I asked" (#140)
_IF_ASKED_HINT = re.compile(
    r"\b(unless (i|you're|you are) (ask|asked|tell|told|say|request)|unless asked|unless told|"
    r"only\b[^.?!;]{0,40}?\b(when|if|after) (i('ve| have)? )?(ask|asked|tell|told|say|said|request)|"
    r"only (when|if) (asked|told))"
)
# "I'll send it myself", "hand it to me", "let me press send": the rule is "Hand it to me" (#140)
_HAND_OFF_HINT = re.compile(
    r"\b((i'll|i will|i'd rather|let me|i can|i want to) (\w+ ){1,3}(it |them |that |this )?myself|"
    r"hand (it|them|that|this|those|these) (over )?to me|give (it|them) to me|"
    r"let me (send|post|press|click|hit|submit|do|finish|pay|buy|book|run|reply)|"
    r"leave (it|them|that|the \w+) (for|to) me|for me to (send|post|press|click|submit|do|finish|run))\b"
)
# deciding the level only, once the words passed the pre-filter: "don't send it, just draft it" hands it over too
_HAND_OFF_LEVEL = re.compile(r"\b(just|only) (draft|prepare|write|fill|set)\b")
# a limit on this request only ("don't cc her this time", "do not open it again in part A"): a one-off instruction for
# the reply at hand, never a standing rule. Kept narrow: a missed standing rule costs more than a needless card
_ONE_OFF = re.compile(r"\b(this time|just this once|for now|in (part|step) [a-z0-9]{1,3})\b")
_SENTENCE = re.compile(r"(?<=[.!?;\n])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "a", "about", "after", "again", "all", "always", "an", "and", "any", "anything", "are", "as", "ask", "asking",
    "at", "be", "before", "but", "by", "can", "check", "checking", "could", "do", "does", "don", "dont", "ever",
    "everything", "first", "for", "from", "have", "i", "if", "in", "is", "it", "its", "let", "make", "me", "mine",
    "more", "must", "my", "never", "no", "not", "of", "ok", "okay", "on", "or", "our", "permission", "please",
    "should", "shouldn", "so", "t", "that", "the", "them", "these", "thing", "things", "this", "those", "to",
    "unless", "until", "up", "us", "we", "what", "when", "will", "with", "without", "would", "you", "your",
})
# a few everyday words that tool names and descriptions say differently
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "email": ("mail", "gmail", "inbox", "imap"),
    "mail": ("email", "gmail", "inbox", "imap"),
    "inbox": ("email", "mail", "gmail"),
    "delete": ("trash", "remove"),
    "remove": ("delete", "trash"),
    "erase": ("delete", "trash", "remove"),
    "trash": ("delete",),
    "money": ("pay", "payment", "purchase", "buy", "order", "checkout"),
    "pay": ("payment", "purchase", "buy", "order", "checkout"),
    "spend": ("pay", "payment", "purchase", "buy", "order"),
    "buy": ("purchase", "order", "checkout", "pay"),
    "purchase": ("buy", "order", "checkout", "pay"),
    "post": ("message", "send", "publish"),
    "message": ("post", "send", "chat"),
    "text": ("message", "send"),
    "dm": ("message", "send"),
    "tweet": ("post",),
    "meeting": ("calendar", "event", "invite"),
    "invite": ("calendar", "event"),
    "file": ("document", "drive"),
    "document": ("doc", "file"),
    "code": ("script", "execute", "command"),
    "command": ("terminal", "shell", "run"),
}


# verbs that name a specific action: a sentence with one gets rules for the matching tools, never a whole app
_ACTIONS = frozenset({
    "delete", "remove", "erase", "trash", "send", "post", "publish", "pay", "purchase", "buy", "order", "spend",
    "share", "archive", "reply", "forward", "move", "edit", "update", "change", "create", "cancel", "book", "invite",
    "schedule", "run", "execute", "comment", "upload", "transfer", "unsubscribe", "label", "mark", "write", "submit",
    "sign", "message", "text", "dm", "tweet", "merge", "close", "rename", "install",
})


_DETERMINERS = frozenset({
    "my", "your", "our", "their", "his", "her", "its", "the", "a", "an", "any", "all", "some", "every", "each",
    "these", "those", "this", "that",
})


# ----------------------------------------------------------------------------- pure helpers
def _plain(text: str) -> str:
    return (text or "").replace("\u2019", "'").replace("\u2018", "'").lower()


def standing_text(text: str) -> str:
    """The sentences of ``text`` that read like a standing "never" or "ask me first" instruction, joined and
    shortened, or "" when there are none (most messages). Deterministic; no model."""
    kept: list[str] = []
    for sentence in _SENTENCE.split((text or "").strip()):
        plain = _plain(sentence)
        if not plain.strip() or _LOOSEN.search(plain):
            continue
        if _ONE_OFF.search(plain):
            # drop only the instructions limited to this request: "never delete my emails, but don't cc Maya this
            # time" keeps "never delete my emails, but"
            sentence = " ".join(c for c in _cut(sentence) if not _ONE_OFF.search(_plain(c)))
            plain = _plain(sentence)
        plain = _BENIGN.sub(" ", plain)
        hints = (_NEGATION, _ASK_FIRST, _IF_ASKED_HINT, _HAND_OFF_HINT)
        if any(h.search(plain) for h in hints):
            kept.append(" ".join(sentence.split()))
    return " ".join(kept)[:MAX_SAID].strip()


def pending_asks(mode: str, risk: Risk) -> bool:
    """True when an undecided proposal (or a running check) makes a call with effective risk ``risk`` ask, in
    approvals mode ``mode``. With approvals "off" only calls that change, send, delete or run something ask: a
    proposal is not a rule the user confirmed, and a look-up can't do what it guards against (#306)."""
    return mode != "off" or Risk(risk) > Risk.read


def asks_first(text: str) -> bool:
    """True when the words name a condition ("without asking me", "always ask"): the rule is Ask, not Never."""
    return bool(_ASK_HINT.search(_plain(text)))


def level_for(text: str, model_level: str | None = None) -> str:
    """The level the words ask for, decided in code (#140): "I'll send it myself" or "just draft it" -> Hand it to
    me; "unless I ask" -> Only if I asked; "without asking me", "always ask" -> Ask; else the model's level, else
    Never. The model only ever chooses between Never and Ask."""
    plain = _plain(text)
    if _HAND_OFF_HINT.search(plain) or _HAND_OFF_LEVEL.search(plain):
        return HAND_OFF
    if _IF_ASKED_HINT.search(plain):
        return IF_ASKED
    if asks_first(plain):
        return ASK
    return model_level if model_level in {ASK, NEVER} else NEVER


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> set[str]:
    words = {w for w in _WORD.findall(_plain(text)) if len(w) >= 2 and w not in _STOP}
    out = {_stem(w) for w in words}
    for w in list(out) + list(words):
        out.update(_stem(x) for x in _SYNONYMS.get(w, ()))
    return out


def _match(term: str, token: str) -> bool:
    if term == token:
        return True
    return min(len(term), len(token)) >= 4 and (token.startswith(term) or term.startswith(token))


def _tokens(tool: Tool, app_name: str) -> set[str]:
    text = f"{tool.name.replace('_', ' ')} {tool.plugin} {app_name} {tool.description[:200]}"
    return {_stem(w) for w in _WORD.findall(_plain(text))}


def _action_base(word: str) -> str | None:
    stem = _stem(word)
    if word in _ACTIONS:
        return word
    if stem in _ACTIONS:
        return stem
    return next((a for a in sorted(_ACTIONS) if len(stem) >= 4 and a.startswith(stem)), None)


def action_terms(text: str) -> set[str]:
    """The specific actions the words name ("delete" -> delete, trash, remove), as stems; empty for "never use X".

    A word that can be a noun is only an action where a verb goes: not right after a determiner or another noun, so
    "never delete my email messages" names delete, not "message"."""
    out: set[str] = set()
    prev_noun = False
    for word in _WORD.findall(_plain(text)):
        base = _action_base(word)
        if base is not None and not prev_noun:
            out.add(_stem(base))
            out.update(_stem(x) for x in _SYNONYMS.get(base, ()))
            prev_noun = False
        else:
            # what follows a determiner ("my", "the") or a noun is an object; trigger and filler words are neither
            prev_noun = word in _DETERMINERS or base is not None or (word not in _STOP and len(word) > 1)
    return out


def _cut(text: str) -> list[str]:
    """``text`` cut before every "never", "don't", "always ask"...: one instruction per part, in its own words."""
    plain = _plain(text)
    source = text if len(text) == len(plain) else plain  # a rare letter whose lower case is longer: cut the plain text
    cuts = sorted({0, *(m.start() for m in _NEGATION.finditer(plain)), *(m.start() for m in _ASK_FIRST.finditer(plain))})
    parts = [source[a:b] for a, b in zip(cuts, [*cuts[1:], len(plain)], strict=True)]
    return [p for p in parts if p.strip()]


def _clauses(text: str) -> list[str]:
    """``text`` cut before every "never", "don't", "always ask"...: one instruction per clause, in plain words."""
    return [_plain(p) for p in _cut(text)]


def _app_named(clause: str, app_id: str, app_name: str) -> bool:
    tokens = {_stem(w) for w in _WORD.findall(_plain(f"{app_id.replace('_', ' ')} {app_name}"))}
    return any(_match(term, tok) for term in _terms(clause) for tok in tokens)


def _exact_tools(clause: str, tools: Iterable[Tool]) -> list[str]:
    """Tools named exactly by their name in ``clause`` ("don't call browser_open")."""
    words = set(re.findall(r"[a-z0-9_]+", clause))
    return [t.name for t in tools if "_" in t.name and t.name in words]


def _tool_named(clause: str, tool: Tool) -> bool:
    tokens = {_stem(w) for w in tool.name.split("_") if len(w) >= 3}
    return any(_match(term, tok) for term in _terms(clause) for tok in tokens)


def narrow_app_keys(
    keys: list[str], said: str, tools: Iterable[Tool], app_names: dict[str, str] | None = None
) -> list[str]:
    """Keep an app key only when its own instruction names no specific action. Otherwise the app key becomes that
    app's tools whose names match the action (none match: dropped). The instruction for an app is the clause that
    mentions it ("never delete my emails. never use Slack" narrows Gmail, not Slack), else all of ``said``.

    A tool key stays only when a clause names an action ("never write files"), the tool's app ("don't use my
    devices") or the tool itself ("never take photos"); "never say you don't know" names none of them, so a tool the
    model matched on a loose word is dropped. Deterministic; runs after validation."""
    by_name = {t.name: t for t in tools}
    clauses = _clauses(said) or [_plain(said)]
    names = app_names or {}
    out: list[str] = []
    for key in keys:
        tool = by_name.get(key)
        if tool is not None:
            if any(
                action_terms(c) or _app_named(c, tool.plugin, names.get(tool.plugin, "")) or _tool_named(c, tool)
                for c in clauses
            ):
                out.append(key)
            continue
        own = [c for c in clauses if _app_named(c, key, names.get(key, ""))]
        exact = [n for c in (own or clauses) for n in _exact_tools(c, by_name.values()) if by_name[n].plugin == key]
        if exact:  # the words name the app's tool itself: that tool, never the whole app
            out.extend(exact)
            continue
        actions = set().union(*(action_terms(c) for c in own)) if own else action_terms(said)
        if not actions:
            out.append(key)
            continue
        for t in by_name.values():
            if t.plugin == key and any(_match(a, _stem(w)) for a in actions for w in t.name.split("_")):
                out.append(t.name)
    return list(dict.fromkeys(out))[:MAX_KEYS]


def candidate_tools(text: str, tools: Iterable[Tool], app_names: dict[str, str]) -> list[Tool]:
    """Tools sharing a word with ``text`` (names, app, description, a few synonyms), best first, at most
    ``MAX_CANDIDATES``. A word that names the tool's app counts twice, so an app the user names ("never use Slack")
    keeps its place however many tools another app has. Actions rank above look-ups on a tie."""
    terms = _terms(text)
    if not terms:
        return []
    scored: list[tuple[int, int, str, Tool]] = []
    for t in tools:
        tokens = _tokens(t, app_names.get(t.plugin, ""))
        app = {_stem(w) for w in _WORD.findall(_plain(f"{t.plugin.replace('_', ' ')} {app_names.get(t.plugin, '')}"))}
        score = sum(1 for term in terms if any(_match(term, tok) for tok in tokens))
        score += sum(1 for term in terms if any(_match(term, tok) for tok in app))
        if score:
            scored.append((score, int(Risk(t.risk)), t.name, t))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [t for *_, t in scored[:MAX_CANDIDATES]]


def detection_messages(said: str, candidates: list[Tool], app_names: dict[str, str]) -> list[dict]:
    """The short prompt for the fast model: the user's words and the candidate tools grouped by app."""
    lines: list[str] = []
    for plugin in dict.fromkeys(t.plugin for t in candidates):
        lines.append(f"{plugin} ({app_names.get(plugin) or plugin})")
        for t in candidates:
            if t.plugin == plugin:
                desc = " ".join(t.description.split())[:DESCRIPTION_CHARS]
                lines.append(f"- {t.name}: {desc}")
    system = (
        "You turn a user's standing instruction into safety rules for an assistant's tools. Reply with JSON only: "
        '{"keys": ["tool_name"], "rule": "never"}.\n'
        "- keys: the tools from the list the instruction is about. Use an app id instead only when the message "
        'names no specific action, like "never use Slack".\n'
        '- rule: "never" when it must not happen at all, "ask" when it may happen only after asking the user.\n'
        '- If the message is not a standing instruction about these tools, reply {"keys": []}.'
    )
    user = f"Message: {json.dumps(said, ensure_ascii=False)}\n\nTools:\n" + "\n".join(lines)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_detection(data: Any, candidates: list[Tool]) -> tuple[list[str], str | None]:
    """Valid ``(keys, level)`` from the model's reply: keys must be candidate tool names or their app ids (checked
    here, never trusted), at most ``MAX_KEYS``; an app key replaces its own tools. Anything unusable gives ``[]``."""
    if isinstance(data, str):
        try:
            data = parse_json_loose(data, ("keys",))
        except ValueError:
            return [], None
    if isinstance(data, list):
        data = {"keys": data}
    if not isinstance(data, dict):
        return [], None
    raw = data.get("keys", data.get("tools"))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return [], None
    names = {t.name: t for t in candidates}
    apps = {t.plugin for t in candidates}
    lowered = {k.lower(): k for k in [*names, *apps]}
    keys: list[str] = []
    for item in raw:
        key = lowered.get(str(item).strip().lower()) if isinstance(item, str | int) else None
        if key and key not in keys:
            keys.append(key)
    app_keys = {k for k in keys if k in apps and k not in names}
    keys = [k for k in keys if k in app_keys or names[k].plugin not in app_keys][:MAX_KEYS]
    level = str(data.get("rule") or data.get("level") or "").strip().lower()
    return keys, (level if level in {ASK, NEVER} else None)


# ----------------------------------------------------------------------------- service
class ChatRules:
    """Proposals made from chat messages, the chat-scoped "ask" they cause, and the user's decision."""

    def __init__(self, app: Any):
        self.app = app
        self._pending: dict[str, dict[str, str]] = {}  # session_id -> {key: level} of undecided proposals
        self._checking: dict[str, list[frozenset[str]]] = {}  # session_id -> candidate tools of running checks
        self._deciding = asyncio.Lock()  # the desktop and a messaging app can answer at the same moment
        self.attended_s = WAIT_S  # how long Claude Code may answer a check (``detect``)

    # ------------------------------------------------------------------ lookups
    def _app_names(self) -> dict[str, str]:
        return {p.id: getattr(p, "display_name", "") or p.id for p in self.app.registry.plugins()}

    def _is_tool(self, key: str) -> bool:
        tool = self.app.registry.get(key)
        return tool is not None and tool.name == key

    def _overrides(self, rules: dict[str, str], key: str) -> list[str]:
        """Tools inside app ``key`` that have a rule of their own (which beats the app's rule); [] for a tool."""
        if self._is_tool(key):
            return []
        return [k for k in rules if self._is_tool(k) and self.app.registry.get(k).plugin == key]

    def _current(self, rules: dict[str, str], key: str) -> str | None:
        """The lasting rule that applies to ``key`` now: for a tool its own or its app's rule; for an app the loosest
        of its own rule and the rules of tools inside it, since those beat the app's rule."""
        if self._is_tool(key):
            return rule_for(rules, self.app.registry.get(key))
        values = [rules.get(key), *(rules[k] for k in self._overrides(rules, key))]
        return min(values, key=lambda r: STRICTNESS.get(r, 1))

    async def _session_pending(self, session_id: str) -> dict[str, str]:
        cached = self._pending.get(session_id)
        if cached is not None:
            return cached
        rows = await self.app.store.fetchall(
            "SELECT rule, keys FROM rule_proposals WHERE session_id = ? AND status = ?", (session_id, PENDING)
        )
        out: dict[str, str] = {}
        for r in rows:
            for key in _keys(r["keys"]):
                if STRICTNESS.get(r["rule"], 0) > STRICTNESS.get(out.get(key), 1):
                    out[key] = r["rule"]
        self._pending[session_id] = out
        return out

    async def chat_rule(self, tool: Tool, session_id: str | None) -> str | None:
        """"ask" while an undecided proposal in this chat names ``tool`` or its app, or while a check that is still
        running weighs ``tool``, else None."""
        if not session_id:
            return None
        if self.is_checking(tool, session_id):
            return "ask"
        pending = await self._session_pending(session_id)
        return "ask" if pending and (tool.name in pending or tool.plugin in pending) else None

    def is_checking(self, tool: Tool, session_id: str | None) -> bool:
        """True while a rule check that is still running weighs ``tool`` in this chat."""
        if not session_id:
            return False
        return any(tool.name in names for names in self._checking.get(session_id, ()))

    async def pending_rule(self, tool: Tool, session_id: str | None) -> str | None:
        """The strictest still-undecided rule suggestion in this chat naming ``tool`` or its app (#130), or None.
        Unlike ``chat_rule`` — which collapses every undecided proposal to "ask" — this keeps the suggestion's own
        rule, so a pending "never" can be told apart from a pending "ask"."""
        if not session_id:
            return None
        pending = await self._session_pending(session_id)
        names = {tool.name, tool.plugin}
        rules = [r for key, r in pending.items() if key in names]
        return max(rules, key=lambda r: STRICTNESS.get(r, 1), default=None)

    def _target(self, key: str, names: dict[str, str]) -> dict:
        tool = self.app.registry.get(key)
        if tool is not None and tool.name == key:
            app = names.get(tool.plugin) or tool.plugin
            title = tool_title(tool)
            return {"key": key, "app": app, "tool": title, "label": f"{app} > {title}"}
        if key in names:
            return {"key": key, "app": names[key], "tool": None, "label": names[key]}
        return {"key": key, "app": None, "tool": None, "label": key}

    def _out(self, row: Any) -> dict:
        names = self._app_names()
        data = dict(row)
        keys = _keys(data.get("keys"))
        return {
            "id": data["id"], "session_id": data["session_id"], "message_id": data.get("message_id"),
            "said": data["said"], "rule": data["rule"], "keys": keys,
            "targets": [self._target(k, names) for k in keys], "status": data["status"],
            "created_at": data["created_at"], "decided_at": data.get("decided_at"),
        }

    async def get(self, proposal_id: str) -> dict | None:
        row = await self.app.store.fetchone("SELECT * FROM rule_proposals WHERE id = ?", (proposal_id,))
        return self._out(row) if row is not None else None

    async def list(self, session_id: str, status: str | None = PENDING) -> list[dict]:
        sql, params = "SELECT * FROM rule_proposals WHERE session_id = ?", [session_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        rows = await self.app.store.fetchall(sql + " ORDER BY created_at", params)
        return [self._out(r) for r in rows]

    # ------------------------------------------------------------------ detection
    async def check(self, session_id: str, message_id: str | None, text: str) -> dict | None:
        """Propose a rule for ``text`` when it is a standing instruction about known tools. Never raises."""
        # the reply starts waiting for this check after it starts, so its wait ends after this deadline
        deadline = asyncio.get_running_loop().time() + self.attended_s
        try:
            return await self._check(session_id, message_id, text, deadline)
        except Exception as exc:  # a missed proposal must never break a chat turn
            log.warning("rule check skipped: %s", exc)
            return None

    async def _check(self, session_id: str, message_id: str | None, text: str, deadline: float) -> dict | None:
        said = standing_text(text)
        if not said:
            return None
        names = self._app_names()
        candidates = candidate_tools(said, self.app.registry.tools(include_hidden=True), names)
        if not candidates:
            return None
        weighing = frozenset(t.name for t in candidates)
        self._checking.setdefault(session_id, []).append(weighing)  # fail closed until this check ends
        try:
            return await self._propose(session_id, message_id, said, candidates, names, deadline)
        finally:
            running = self._checking.get(session_id, [])
            running.remove(weighing)
            if not running:
                self._checking.pop(session_id, None)

    async def detect(self, session_id: str, messages: list[dict], deadline: float | None = None) -> Any:
        """The model's answer to the detection prompt. The fast model gives it; when every fast model refuses this
        job (Claude Code only answers chats), the chat's own model or the primary model does (#311), since the check
        is part of answering this chat: the reply waits for it. Claude Code answers it as part of that reply, so only
        until ``deadline`` (event-loop time, before the reply stops waiting); a slower check ends with no card."""
        llm = self.app.llm
        try:
            return await llm.complete_json("fast", messages)
        except ModelRefused as exc:
            log.info("rule check: the fast model can't do it (%s), so the chat's model does", exc)
        agent = getattr(self.app, "agent", None)
        model = None
        if agent is not None:
            model, _ = await agent.chat_model(await self.app.store.get_session(session_id), "primary")
        if (model or llm.model_for("primary")).split("/", 1)[0] != claude_code.PREFIX:
            return await llm.complete_json("primary", messages, model=model)

        async def reply() -> str:
            text = ""
            with claude_code.attended():  # part of a reply the user is waiting for (ADR 0022)
                async for chunk in llm.stream("primary", messages, None, model=model):
                    text += chunk.text or ""
            return text

        # only while the reply waits for this check: a slower answer is stopped, never left running in the background
        left = (deadline - asyncio.get_running_loop().time()) if deadline is not None else self.attended_s
        if left <= 0:
            raise TimeoutError("the reply stopped waiting for this check")
        return parse_json_loose(await asyncio.wait_for(reply(), left), ("keys",))

    async def _propose(
        self, session_id: str, message_id: str | None, said: str, candidates: list[Tool], names: dict[str, str],
        deadline: float | None = None,
    ) -> dict | None:
        try:
            data = await self.detect(session_id, detection_messages(said, candidates, names), deadline)
        except Exception as exc:
            log.info("rule check: the model gave no usable answer (%s)", exc)
            return None
        keys, level = parse_detection(data, candidates)
        keys = narrow_app_keys(keys, said, self.app.registry.tools(include_hidden=True), names)
        if not keys:
            return None
        level = level_for(said, level)
        rules = dict(self.app.config.tools.approvals.rules)
        pending = await self._session_pending(session_id)
        keys = [
            k for k in keys
            if STRICTNESS[level] > STRICTNESS.get(self._current(rules, k), 1)
            and STRICTNESS[level] > STRICTNESS.get(pending.get(k), 1)
        ]
        if not keys:
            return None
        pid = new_id()
        await self.app.store.execute(
            "INSERT INTO rule_proposals(id, session_id, message_id, said, rule, keys, status, created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (pid, session_id, message_id, said, level, json.dumps(keys), PENDING, now_iso()),
        )
        for k in keys:
            pending[k] = level
        proposal = await self.get(pid)
        self.app.bus.publish("rule_proposal.updated", proposal)
        return proposal

    # ------------------------------------------------------------------ the user's decision
    async def decide(self, proposal_id: str, decision: str) -> dict:
        """The user's click: ``accept`` saves the rules (only where they tighten) with an origin note; ``decline``
        saves nothing. Raises ``LookupError`` for an unknown id and ``ValueError`` once it was answered."""
        if decision not in {"accept", "decline"}:
            raise ValueError("Choose accept or decline.")
        async with self._deciding:  # one answer wins; a second one sees it was already answered
            row = await self.app.store.fetchone("SELECT * FROM rule_proposals WHERE id = ?", (proposal_id,))
            if row is None:
                raise LookupError(proposal_id)
            if row["status"] != PENDING:
                raise ValueError("This was already answered.")
            if decision == "accept":
                self._save_rules(dict(row))
            status = ACCEPTED if decision == "accept" else DECLINED
            await self.app.store.execute(
                "UPDATE rule_proposals SET status = ?, decided_at = ? WHERE id = ?", (status, now_iso(), proposal_id)
            )
            self._pending.pop(row["session_id"], None)  # rebuilt from what is still pending
            proposal = await self.get(proposal_id)
        self.app.bus.publish("rule_proposal.updated", proposal)
        return proposal  # type: ignore[return-value]

    def _save_rules(self, row: dict) -> None:
        level = row["rule"]
        if level not in LEVELS:
            return
        cfg = self.app.config.model_copy(deep=True)
        approvals = cfg.tools.approvals
        rules = dict(approvals.rules)
        origins = dict(approvals.rule_origins)
        changed = False
        for key in _keys(row["keys"]):
            # a tool: its effective rule; an app: its own rule, plus tools inside it whose own looser rule would
            # otherwise beat the new app rule ("never use Mail" also covers a tool that was set to Allow)
            checks = [(key, self._current(rules, key) if self._is_tool(key) else rules.get(key))]
            checks += [(k, rules.get(k)) for k in self._overrides(rules, key)]
            for k, current in checks:
                if STRICTNESS[level] > STRICTNESS.get(current, 1):
                    rules[k] = level
                    origins[k] = RuleOrigin(rule=level, said=row["said"], at=now_iso(), session_id=row["session_id"])
                    changed = True
        if changed:
            approvals.rules = rules
            approvals.rule_origins = origins
            self.app.save_config(cfg)


def _keys(raw: Any) -> list[str]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    return [str(k) for k in data if isinstance(k, str) and k] if isinstance(data, list) else []
