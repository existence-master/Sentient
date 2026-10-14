"""Memories held for review (issue #137, ADR 0021, docs/API.md section 7).

A memory that came from somewhere other than the user's own words in a clean chat is saved as **pending**: a chat
or task run that read outside content (``ToolContext.untrusted``, ADR 0018), work nobody asked for
(``ToolContext.origin``, ADR 0017), an import (Hermes, a document) or a user-model insight drawn from such material.
Pending memories are never put into a prompt, returned by recall, used by proactivity or the user model, or
touched by dreaming. Only the user moves one out of pending (approve, edit and approve, discard); nothing a model
says can. Unreviewed ones are let go after ``memory.review_expire_days``.

A review note is ``{from, snippet, session_id}``: ``from`` names where it came from in plain words ("Gmail",
"Hermes", "resume.pdf", "a proactive check"), ``snippet`` is the text it was taken from (bounded) and
``session_id`` the chat, when there is one.

Follow-ups (#234): a memory from a chat that reads like an instruction for how Sentient should act ("from now on,
always cc Bob") is held with ``reason: "instruction"`` even in a clean chat (``instruction_shaped``); and in a chat
that read outside content, a fact the user's own words back completely is saved after all (``grounded``), so only
facts that depend on outside content wait. Both are decided in code, never by a model.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from sentient.tools.base import is_unprompted

log = logging.getLogger(__name__)

SNIPPET_CHARS = 400
INBOX_LIMIT = 500  # items of each kind the inbox lists; its count covers all of them
# what work nobody asked for is called on a review card
UNPROMPTED_LABELS = {
    "proactive": "a proactive check",
    "heartbeat": "a background check",
    "followups": "the follow-up scan",
    "dreaming": "nightly tidying",
    "background": "background work",
}
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def note(
    source: str,
    snippet: str | None = None,
    session_id: str | None = None,
    *,
    title: str | None = None,
    dated: str | None = None,
    export: str | None = None,
) -> dict:
    """A review note. ``source`` is the plain name of where the memory came from. An import from another assistant
    also says where in it: ``title`` (the conversation or note), ``dated`` (its day) and ``export`` (the file)."""
    text = " ".join(str(snippet or "").split())
    if len(text) > SNIPPET_CHARS:
        text = text[: SNIPPET_CHARS - 3].rstrip() + "..."
    out = {"from": str(source or "somewhere outside").strip(), "snippet": text or None, "session_id": session_id}
    extra = {"title": " ".join(str(title or "").split())[:200] or None, "dated": dated or None, "export": export or None}
    return out | {k: v for k, v in extra.items() if v}


def for_context(ctx: Any) -> dict | None:
    """The review note for a memory written in this run, or None when it may be saved directly.

    Held when the run read outside content (ADR 0018) or nobody asked for the work (ADR 0017)."""
    untrusted = str(getattr(ctx, "untrusted", "") or "")
    session_id = getattr(ctx, "session_id", None)
    if untrusted:
        return note(untrusted, session_id=session_id)
    origin = str(getattr(ctx, "origin", "") or "").strip().lower()
    if is_unprompted(origin):
        return note(UNPROMPTED_LABELS.get(origin, "background work"), session_id=session_id)
    return None


# ---------------------------------------------------------------------------- instruction-shaped memories (#234)
INSTRUCTION = "instruction"  # ``reason`` of a held memory that tells Sentient how to act
INSTRUCTION_FROM = "an instruction"  # what its review card says it came from
_ACTION_VERBS = (
    r"(send|sends|forward|forwards|reply|replies|cc|bcc|copy|email|emails|message|text|post|share|remind|tell|notify|"
    r"book|pay|transfer|delete|archive|move|add|create|schedule|cancel|accept|decline|approve|sign|buy|order|use|"
    r"include|mention|call|ask|answer|respond|write|draft|open|run|save|mark|label|unsubscribe)"
)
# any character of the same sentence: a dot only inside a word or an address ("bob@example.com"), never a full stop
_IN_SENTENCE = r"(?:[^.;!?]|\.(?=\w))"
_INSTRUCTION_RES = (
    # "from now on ...", "going forward ..."
    re.compile(r"\b(from now on|going forward|in (the )?future|from today|starting (now|today)|henceforth)\b", re.I),
    # an order to the assistant: "always cc Bob", "never reply to ...", "don't book ...", "please send ..."
    re.compile(rf"^\s*(please\s+)?(always|never|don'?t|do not|make sure to|be sure to)\s+(\w+\s+){{0,2}}?{_ACTION_VERBS}\b", re.I),
    re.compile(rf"^\s*(please\s+)?{_ACTION_VERBS}\b", re.I),
    # "whenever an invoice comes in, forward it ...", "every time Priya writes, reply ..."
    re.compile(rf"\b(whenever|every time|each time|any time)\b{_IN_SENTENCE}*?,\s*(please\s+)?(always\s+)?{_ACTION_VERBS}\b", re.I),
    re.compile(rf"\b(whenever|every time|each time|any time)\b{_IN_SENTENCE}*\b(sentient|the assistant|her assistant|his assistant|"
               rf"their assistant|you)\s+(should|must|will|can|has to|needs to|is to)?\s*{_ACTION_VERBS}\b", re.I),
    # the assistant as the one who acts: "Sentient should cc Bob", "Maya wants the assistant to ..."
    re.compile(r"\b(sentient|the assistant|her assistant|his assistant|their assistant)\s+(should|must|has to|needs to|"
               r"is to|will|always|never|can|may)\b", re.I),
    re.compile(r"\b(wants|asked|asks|told|tells|instructed|instructs|would like|prefers)\s+(sentient|the assistant|her "
               r"assistant|his assistant|their assistant|you)\s+to\b", re.I),
    # "Maya wants all invoices forwarded to accounts@...", "should always be cc'd"
    re.compile(rf"\b(wants|would like|asked for|prefers)\b{_IN_SENTENCE}{{0,80}}\b(forwarded|sent|cc'?d|copied|deleted|paid|"
               r"replied|shared|booked|archived|transferred)\b", re.I),
    re.compile(rf"\b(should|must)\s+(always|never)\s+(be\s+)?(\w+\s+){{0,1}}?{_ACTION_VERBS}", re.I),
)


def instruction_shaped(text: str) -> bool:
    """True when a memory reads like an instruction for how Sentient should act ("from now on, always cc Bob",
    "whenever an invoice comes in, forward it to ...") rather than a fact about the user. Such a memory changes what
    Sentient does in every later chat, so it is never saved without the user's yes. Deterministic; no model."""
    plain = " ".join(str(text or "").replace(chr(0x2019), "'").split())
    return any(r.search(plain) for r in _INSTRUCTION_RES)


def instruction_note(snippet: str | None = None, session_id: str | None = None) -> dict:
    """The review note of an instruction-shaped memory: it waits for the user's yes like a proposed rule (#130)."""
    return {**note(INSTRUCTION_FROM, snippet, session_id), "reason": INSTRUCTION}


# ---------------------------------------------------------------------------- the user's own words (#234)
GROUNDED_SHARE = 2 / 3  # at least this share of a fact's meaningful words must be in the user's own words


def grounded(fact: str, own_words: str, user_name: str = "") -> bool:
    """True when ``fact`` comes from ``own_words`` (what the user typed or said): every name, place, number and
    address in it is in those words, and so are two thirds of its other meaningful words. Checked in code, never by a
    model, so a fact that brings in anything the user didn't say (from an email or a web page the chat read) waits
    for review instead."""
    from sentient.memory.facts import keyword_tokens, new_details

    words = str(own_words or "")
    if not words.strip() or not str(fact or "").strip():
        return False
    names = {w.lower() for w in re.findall(r"[A-Za-z]+", user_name or "")} | {"user", "sentient", "assistant"}
    if any(d.lower() not in names for d in new_details(fact, words)):
        return False
    said = set(keyword_tokens(words, None))
    meaningful = [w for w in keyword_tokens(fact, None) if w not in names]
    if not meaningful:
        return False  # nothing of the user's backs it ("Maya is Sentient's assistant")
    hits = sum(1 for w in meaningful if any(_same_word(w, s) for s in said))
    return hits >= GROUNDED_SHARE * len(meaningful) - 1e-9


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    short = min(len(a), len(b))
    return short >= 4 and (a.startswith(b[: max(4, short - 2)]) or b.startswith(a[: max(4, short - 2)]))


def is_unprompted_note(review: dict | None) -> bool:
    """True for the note of a memory from work nobody asked for (no chat, so no words of the user's to check)."""
    return bool(review) and str(review.get("from") or "") in set(UNPROMPTED_LABELS.values())


def with_snippet(review: dict | None, text: str, fact: str) -> dict | None:
    """``review`` with the part of ``text`` that ``fact`` most likely came from as its snippet."""
    if review is None:
        return None
    return {**review, "snippet": note("", best_snippet(text, fact))["snippet"]}


def best_snippet(text: str, fact: str) -> str:
    """The sentence or line of ``text`` sharing the most words with ``fact`` (the start of ``text`` when none do)."""
    from sentient.memory.facts import word_overlap

    parts = [p.strip() for p in _SENTENCE_RE.split(text or "") if p.strip()]
    if not parts:
        return ""
    best = max(parts, key=lambda p: word_overlap(p, fact))
    return best if word_overlap(best, fact) > 0 else parts[0]


def load(raw: Any) -> dict | None:
    if not raw:
        return None
    try:
        out = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def dump(review: dict | None) -> str | None:
    return json.dumps(review) if review else None


# ---------------------------------------------------------------------------- the inbox (REST)
def _expires(created_at: str | None, days: int) -> str | None:
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt + timedelta(days=days)).isoformat()


def _item(kind: str, mid: Any, text: str, source: str, review: dict | None, created_at: str, days: int) -> dict:
    review = review or {}
    return {
        "kind": kind, "id": mid, "text": text, "source": source,
        "from": review.get("from") or source, "snippet": review.get("snippet"), "session_id": review.get("session_id"),
        "title": review.get("title"), "dated": review.get("dated"), "export": review.get("export"),
        "reason": review.get("reason"), "created_at": created_at, "expires_at": _expires(created_at, days),
    }


async def _items(app: Any, limit: int) -> list[dict]:
    days = app.config.memory.review_expire_days
    items: list[dict] = []
    if app.memory is not None:
        for f in await app.memory.pending_facts(limit):
            items.append(_item("fact", f["id"], f["content"], f["source"], f["review"], f["created_at"], days))
    um = getattr(app, "user_model", None)
    if um is not None:
        for i in await um.pending_insights(limit):
            items.append(_item("insight", i["id"], i["statement"], i["source"], i.get("review"), i["created_at"], days))
    items.sort(key=lambda i: i["created_at"] or "", reverse=True)
    return items


async def inbox(app: Any, limit: int = INBOX_LIMIT) -> dict:
    """``{items, count, expire_days}``: pending facts and insights, newest first (at most ``limit`` of each);
    ``count`` is all of them."""
    count = await app.memory.pending_count() if app.memory is not None else 0
    um = getattr(app, "user_model", None)
    if um is not None:
        count += await um.pending_count()
    return {"items": await _items(app, limit), "count": count, "expire_days": app.config.memory.review_expire_days}


async def approve(app: Any, kind: str, item_id: str, content: str | None = None) -> bool:
    """Approve one pending memory, with the user's wording when ``content`` is given. False when not pending."""
    if kind == "fact":
        if app.memory is None:
            return False
        return await app.memory.approve(_fact_id(item_id), content) is not None
    if kind == "insight":
        return await app.user_model.approve_insight(item_id, content) is not None
    raise ValueError("kind must be fact or insight")


async def discard(app: Any, kind: str, item_id: str) -> bool:
    if kind == "fact":
        return app.memory is not None and await app.memory.discard(_fact_id(item_id))
    if kind == "insight":
        return await app.user_model.discard_insight(item_id)
    raise ValueError("kind must be fact or insight")


async def approve_from(app: Any, source: str) -> int:
    """Approve every pending memory whose ``from`` is ``source`` (all of them, not one inbox page). Returns how many."""
    done = 0
    for item in await _items(app, -1):
        if item["from"] == source and await approve(app, item["kind"], str(item["id"])):
            done += 1
    return done


async def expire(app: Any, now: datetime | None = None) -> int:
    """Let go of memories that waited longer than ``memory.review_expire_days``, with a plain note. Returns how many."""
    days = app.config.memory.review_expire_days
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=days)).isoformat()
    gone = 0
    if app.memory is not None:
        gone += await app.memory.expire_pending(cutoff)
    um = getattr(app, "user_model", None)
    if um is not None:
        gone += await um.expire_pending(cutoff)
    if gone:
        noun = "memory" if gone == 1 else "memories"
        try:
            await app.notify(
                "info",
                f"{gone} {noun} waiting for your review {'was' if gone == 1 else 'were'} let go after {days} days "
                "without an answer. Nothing was saved from them.",
                title="Memory review",
            )
        except Exception as exc:
            log.debug("review expiry note failed: %s", exc)
    return gone


def _fact_id(item_id: str) -> int:
    try:
        return int(item_id)
    except (TypeError, ValueError) as exc:
        raise KeyError(item_id) from exc
