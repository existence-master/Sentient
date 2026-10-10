"""Episodic memory: first-person summaries of older conversation chunks.

Port of v2 ``summarize_old_conversations`` (Celery beat + ChromaDB) and the v2
history MCP (``semantic_search`` over summaries, ``time_based_search`` over
messages). Summaries live in the core ``summaries`` table with vectors in a
``summaries_vec`` sqlite-vec table keyed by the summary's rowid; summarized
messages get ``messages.summarized = 1``.

A summary of a chat that read outside content carries that chat's mark (``summaries.untrusted``, ADR 0021): it is
listed for the user, but ``search`` (what other chats, proactivity and profile upkeep see) leaves it out.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sentient.config.schema import SentientConfig
from sentient.llm.provider import LLMProvider
from sentient.memory import prompts
from sentient.memory.vectors import VecTable, pack
from sentient.store.db import Store, new_id, now_iso

log = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 1500
# summaries of chats that never read outside content: the summary's own mark and its chat's current one (a chat from
# before the mark is classified later, ADR 0018). Self-contained, for queries on ``summaries`` without an alias.
CLEAN = (
    "COALESCE(summaries.untrusted, '') = '' AND NOT EXISTS (SELECT 1 FROM sessions WHERE sessions.id ="
    " summaries.session_id AND COALESCE(sessions.untrusted, '') != '')"
)


def parse_when(value: str, *, end: bool = False) -> datetime:
    """ISO date or datetime -> aware UTC datetime. A bare date as ``end`` means end of that day."""
    raw = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if end and len(raw) == 10:
        dt = dt + timedelta(days=1) - timedelta(microseconds=1)
    return dt.astimezone(UTC)


class EpisodicMemory:
    def __init__(self, store: Store, llm: LLMProvider, config: SentientConfig):
        self.store = store
        self.llm = llm
        self.config = config
        self.vec = VecTable(store, "summaries_vec")

    # ------------------------------------------------------------------ vectors
    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vecs = await self.llm.embed(texts)
        if vecs:
            await self.vec.ensure(len(vecs[0]), self.reindex)
        return vecs

    async def reindex(self) -> int:
        rows = await self.store.fetchall("SELECT rowid AS rid, content FROM summaries ORDER BY rowid")
        for i in range(0, len(rows), 16):
            batch = rows[i : i + 16]
            vecs = await self.llm.embed([r["content"] for r in batch])
            for r, v in zip(batch, vecs, strict=False):
                await self.vec.upsert(int(r["rid"]), v)
        return len(rows)

    # ------------------------------------------------------------------ summarization job
    async def pending_chunks(self, now: datetime | None = None) -> list[tuple[str, list[dict]]]:
        """Chunks of unsummarized user/assistant messages older than ``summarize_after_minutes``.

        Full chunks are always ready. A trailing partial chunk (>= 2 messages) is ready only
        when its conversation has been idle since the cutoff.
        """
        now = now or datetime.now(UTC)
        cutoff = (now - timedelta(minutes=self.config.memory.summarize_after_minutes)).isoformat()
        size = self.config.memory.summary_chunk_messages
        rows = await self.store.fetchall(
            "SELECT id, session_id, role, content, attachments, created_at FROM messages"
            " WHERE summarized = 0 AND role IN ('user', 'assistant') AND content IS NOT NULL AND content != ''"
            " AND created_at < ? ORDER BY session_id, created_at, rowid",
            (cutoff,),
        )
        by_session: dict[str, list[dict]] = {}
        for r in rows:
            by_session.setdefault(r["session_id"], []).append(dict(r))
        ready: list[tuple[str, list[dict]]] = []
        for sid, msgs in by_session.items():
            last = await self.store.fetchone("SELECT MAX(created_at) AS t FROM messages WHERE session_id = ?", (sid,))
            idle = bool(last and last["t"] and last["t"] < cutoff)
            for i in range(0, len(msgs), size):
                chunk = msgs[i : i + size]
                if len(chunk) == size or (idle and len(chunk) >= 2):
                    ready.append((sid, chunk))
        ready.sort(key=lambda item: item[1][0]["created_at"])
        return ready

    @staticmethod
    def transcript(chunk: list[dict]) -> str:
        lines = []
        for m in chunk:
            text = (m.get("content") or "")[:MAX_MESSAGE_CHARS]
            attachments = json.loads(m["attachments"]) if m.get("attachments") else []
            if attachments:
                text = f"(Attached file for context: {', '.join(attachments)}) {text}"
            lines.append(f"{m['role']}: {text}")
        return "\n".join(lines)

    async def summarize_chunk(self, session_id: str, chunk: list[dict], user_name: str = "") -> dict | None:
        summary = await self.llm.complete_text(
            "fast",
            [
                {"role": "system", "content": prompts.summarize_system(user_name)},
                {"role": "user", "content": self.transcript(chunk)},
            ],
        )
        summary = (summary or "").strip()
        if not summary:
            log.warning("empty summary for session %s; chunk skipped", session_id)
            return None
        return await self.add_summary(session_id, summary, chunk)

    async def add_summary(self, session_id: str | None, content: str, chunk: list[dict]) -> dict:
        sid = new_id()
        ids = [m["id"] for m in chunk]
        start_at, end_at = chunk[0]["created_at"], chunk[-1]["created_at"]
        session = await self.store.fetchone("SELECT untrusted FROM sessions WHERE id = ?", (session_id,))
        untrusted = (session["untrusted"] or "") if session else ""
        cur = await self.store.execute(
            "INSERT INTO summaries(id, session_id, content, start_at, end_at, message_ids, created_at, untrusted)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (sid, session_id, content, start_at, end_at, json.dumps(ids), now_iso(), untrusted),
        )
        try:
            [vec] = await self._embed([content])
            await self.vec.upsert(int(cur.lastrowid), vec)
        except Exception as exc:
            log.warning("summary embedding failed (kept without vector): %s", exc)
        marks = ",".join("?" * len(ids))
        await self.store.execute(f"UPDATE messages SET summarized = 1 WHERE id IN ({marks})", ids)
        return {
            "id": sid, "content": content, "start_at": start_at, "end_at": end_at, "session_id": session_id,
            "untrusted": untrusted or None,
        }

    async def summarize_pending(
        self, *, user_name: str = "", now: datetime | None = None, max_chunks: int = 5
    ) -> list[dict]:
        created = []
        for sid, chunk in (await self.pending_chunks(now))[:max_chunks]:
            try:
                s = await self.summarize_chunk(sid, chunk, user_name)
            except Exception as exc:
                log.warning("summarizing session %s failed: %s", sid, exc)
                continue
            if s:
                created.append(s)
        return created

    # ------------------------------------------------------------------ reads
    async def list(self, limit: int = 50) -> list[dict]:
        """Every summary for the user, with ``untrusted`` (the app whose content that chat read, else null)."""
        rows = await self.store.fetchall(
            "SELECT id, content, start_at, end_at, session_id, COALESCE(NULLIF(untrusted, ''),"
            " (SELECT NULLIF(s.untrusted, '') FROM sessions s WHERE s.id = summaries.session_id)) AS untrusted"
            " FROM summaries ORDER BY end_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """v2 history semantic_search over conversation summaries. Summaries of chats that read outside content are
        left out (ADR 0021): this is what other chats, proactivity and profile upkeep see."""
        if not query.strip():
            return []
        rows = None
        try:
            if self.vec.ready or await self.vec.exists():
                # nearest clean summaries, filtered before the limit so marked ones can't crowd them out
                [qvec] = await self._embed([query])
                rows = await self.store.fetchall(
                    "SELECT summaries.id, summaries.content, summaries.start_at, summaries.end_at, summaries.session_id,"
                    " vec_distance_cosine(v.embedding, ?) AS distance FROM summaries"
                    f" JOIN {self.vec.name} v ON v.rowid = summaries.rowid WHERE {CLEAN} ORDER BY distance LIMIT ?",
                    (pack(qvec), limit),
                )
        except Exception as exc:
            log.debug("summary vector search failed: %s", exc)
            rows = None
        if rows is not None:
            return [
                {k: r[k] for k in ("id", "content", "start_at", "end_at", "session_id")}
                | {"similarity": round(1.0 - float(r["distance"]), 4)}
                for r in rows
            ]
        # no vector index (or embeddings failing): keyword fallback
        words = [w for w in query.split() if len(w) > 2][:6] or [query]
        clause = " OR ".join("content LIKE ?" for _ in words)
        rows = await self.store.fetchall(
            f"SELECT id, content, start_at, end_at, session_id FROM summaries WHERE ({clause}) AND {CLEAN}"
            " ORDER BY end_at DESC LIMIT ?",
            [*[f"%{w}%" for w in words], limit],
        )
        return [dict(r) for r in rows]

    async def messages_between(self, start: str, end: str, limit: int = 200) -> list[dict[str, Any]]:
        """v2 history time_based_search: every user/assistant message in a date range."""
        s = parse_when(start).isoformat()
        e = parse_when(end, end=True).isoformat()
        rows = await self.store.fetchall(
            "SELECT m.session_id, m.role, m.content, m.created_at, s.title FROM messages m"
            " LEFT JOIN sessions s ON s.id = m.session_id"
            " WHERE m.created_at >= ? AND m.created_at <= ? AND m.role IN ('user', 'assistant')"
            " AND m.content IS NOT NULL AND m.content != '' ORDER BY m.created_at LIMIT ?",
            (s, e, limit),
        )
        return [dict(r) for r in rows]
