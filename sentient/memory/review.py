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


def note(source: str, snippet: str | None = None, session_id: str | None = None) -> dict:
    """A review note. ``source`` is the plain name of where the memory came from."""
    text = " ".join(str(snippet or "").split())
    if len(text) > SNIPPET_CHARS:
        text = text[: SNIPPET_CHARS - 3].rstrip() + "..."
    return {"from": str(source or "somewhere outside").strip(), "snippet": text or None, "session_id": session_id}


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
        "created_at": created_at, "expires_at": _expires(created_at, days),
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
