"""Source links, confidence and state for memories (#152).

Adds where exactly a fact came from (``facts.source_ref``, ``fact_sources.ref``), how sure Sentient is of it
(``confidence``) and whether it is still current (``state``, ``superseded_by``). Existing facts get a confidence from
their source.

It also gives facts saved before #153 from a messaging app or a device chat their real source. Those say
``conversation``; the chat they came from is found in what is kept: the review note of a memory held for review, the
``memory_remember`` call that saved it, or else the one user message in a messaging-app or device chat that says the
same thing and came shortly before it. A fact no chat explains keeps ``conversation``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

CHANNELS = ("discord", "telegram", "whatsapp")  # sentient.memory.origins.CHANNELS (imported late: no import cycle)
WINDOW = timedelta(minutes=30)  # extraction runs in the background, a little after the message it reads
CONFIDENCE = (
    "CASE WHEN source IN ('manual', 'user') THEN 1.0"
    " WHEN source IN ('conversation', 'onboarding') OR source LIKE 'channel:%' OR source LIKE 'device:%' THEN 0.9"
    " WHEN status = 'pending' THEN 0.6"
    " ELSE 0.7 END"
)


def _when(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _source_of(row: Any) -> str | None:
    channel = str(row["channel"] or "")
    if channel in CHANNELS:
        return f"channel:{channel}"
    node = str(row["node_id"] or "") if "node_id" in row.keys() else ""  # noqa: SIM118 - sqlite Row
    return f"device:{node}" if node else None


async def _session(store: Any, sid: str) -> Any:
    return await store.fetchone("SELECT * FROM sessions WHERE id = ?", (sid,))


async def _saved_by_tool(store: Any) -> dict[int, tuple[str, str]]:
    """fact id -> (session id, message id) for facts a chat's ``memory_remember`` call saved."""
    out: dict[int, tuple[str, str]] = {}
    rows = await store.fetchall(
        "SELECT id, session_id, content FROM messages WHERE role = 'tool' AND name = 'memory_remember'"
    )
    for r in rows:
        try:
            res = json.loads(r["content"] or "{}")
        except ValueError:
            continue
        if isinstance(res, dict) and isinstance(res.get("id"), int) and res.get("action") in {"ADD", "UPDATE"}:
            out.setdefault(int(res["id"]), (str(r["session_id"]), ""))
    return out


async def _backfill(store: Any) -> None:
    from sentient.memory import origins

    facts = await store.fetchall(
        "SELECT id, content, created_at, review FROM facts WHERE source = 'conversation' AND status IN ('active',"
        " 'pending', 'merged')"
    )
    if not facts:
        return
    by_tool = await _saved_by_tool(store)
    marks = ",".join("?" * len(CHANNELS))
    said = await store.fetchall(
        "SELECT m.id, m.session_id, m.content, m.created_at, s.channel, s.node_id FROM messages m"
        " JOIN sessions s ON s.id = m.session_id WHERE m.role = 'user'"
        f" AND (s.channel IN ({marks}) OR (s.node_id IS NOT NULL AND s.node_id != ''))",
        CHANNELS,
    )
    for f in facts:
        source = ref = None
        sid = None
        try:
            review = json.loads(f["review"]) if f["review"] else None
        except ValueError:
            review = None
        if isinstance(review, dict) and review.get("session_id"):
            sid = str(review["session_id"])
        elif int(f["id"]) in by_tool:
            sid = by_tool[int(f["id"])][0]
        if sid:
            row = await _session(store, sid)
            source = _source_of(row) if row is not None else None
            if source:
                ref = {"kind": "message", "session_id": sid}
        else:
            at = _when(f["created_at"])
            near = [
                m for m in said
                if at is not None and (t := _when(m["created_at"])) is not None and t <= at <= t + WINDOW
                and origins.backs(f["content"], m["content"] or "")
            ]
            if len({m["session_id"] for m in near}) == 1:  # one chat says it: that's where it came from
                m = max(near, key=lambda m: m["created_at"])
                source = _source_of(m)
                ref = {"kind": "message", "session_id": m["session_id"], "message_id": m["id"]}
        if source:
            await store.execute(
                "UPDATE facts SET source = ?, source_ref = COALESCE(source_ref, ?) WHERE id = ?",
                (source, json.dumps(ref) if ref else None, f["id"]),
            )


async def migrate(store: Any) -> None:
    await store.ensure_column("facts", "confidence", "REAL NOT NULL DEFAULT 0.8")
    await store.ensure_column("facts", "state", "TEXT NOT NULL DEFAULT 'current'")
    await store.ensure_column("facts", "superseded_by", "INTEGER")
    await store.ensure_column("facts", "source_ref", "TEXT")
    await store.ensure_column("fact_sources", "ref", "TEXT")
    cols = await store.columns("facts")
    if "confidence" not in cols:  # no facts table (a very old database): the schema files make it
        return
    if await store.columns("sessions") and await store.columns("messages"):
        await _backfill(store)
    await store.execute(f"UPDATE facts SET confidence = {CONFIDENCE} WHERE confidence = 0.8")
