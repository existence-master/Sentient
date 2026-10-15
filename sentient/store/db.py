"""Async SQLite store with sqlite-vec loaded.

One connection per process is plenty for a single-user desktop app; WAL mode
lets the UI read while a task run writes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from sentient import paths

log = logging.getLogger(__name__)

_SCHEMA = Path(__file__).with_name("schema.sql")
_PKG_ROOT = Path(__file__).resolve().parents[1]
# Feature packages own their tables. Each file must be idempotent. Order matters for foreign keys.
# Bump when a change to a schema.sql or an ensure_column needs existing databases updated. Opening a database
# with another version takes a snapshot first (issue #151); so does the first column added to an existing one.
SCHEMA_VERSION = 10  # 2: sessions.incognito (#154); 3: messages.sources (#290); 4: tasks.notepad (#292); 5: sessions.model (#312)
# 6: approval_grants (#139); 7: usage.cost_usd (#148); 8: evolution_reviews.activity_mark (#297)
# 9: messages.approval and the activity timeline's indexes (#147); 10: try runs and file snapshots (#142)
PACKAGE_SCHEMAS = [
    "tasks", "integrations", "proactivity", "evolution", "voice", "memory", "nodes", "channels", "browser", "sandbox",
    "folders", "reminders", "snapshots",
]


# SQL condition: the chat ``col`` names is not incognito (#154). Learning, summaries and history search use it so an
# incognito chat is never read back. ``.format(col="m.session_id")``.
NOT_INCOGNITO = "NOT EXISTS (SELECT 1 FROM sessions inc WHERE inc.id = {col} AND inc.incognito = 1)"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------- watermarks (#236)
# Rows saved in the same clock tick share a timestamp (Windows' clock moves about every 15.6 ms), so "newer than the
# last timestamp seen" skips a row saved in that tick after the read. A watermark keeps the newest timestamp read and
# the ids already read at it, as JSON: {"at": "<iso>", "ids": [...]}. A plain timestamp (from an older build, or a
# starting point) means "strictly newer than this".


def _read_mark(mark: str | None) -> tuple[str, list]:
    if not mark:
        return "", []
    try:
        data = json.loads(mark)
    except ValueError:
        return mark, []
    if isinstance(data, dict):
        return str(data.get("at") or ""), list(data.get("ids") or [])
    return mark, []


def after_watermark(mark: str | None, ts_col: str, id_col: str) -> tuple[str, list]:
    """SQL condition and parameters for rows after ``mark``: newer than its timestamp, or at it and not read yet."""
    at, ids = _read_mark(mark)
    if not ids:
        return f"{ts_col} > ?", [at]
    marks = ",".join("?" * len(ids))
    return f"({ts_col} > ? OR ({ts_col} = ? AND {id_col} NOT IN ({marks})))", [at, at, *ids]


def next_watermark(mark: str | None, rows: Iterable[Any], ts_key: str, id_key: str) -> str | None:
    """The watermark after reading ``rows`` (each with ``ts_key`` and ``id_key``): their newest timestamp and the ids
    read at it, kept together with ``mark``'s ids when the timestamp did not move. ``mark`` itself when no rows."""
    rows = list(rows)
    if not rows:
        return mark
    at = max(str(r[ts_key]) for r in rows)
    ids = [r[id_key] for r in rows if str(r[ts_key]) == at]
    old_at, old_ids = _read_mark(mark)
    if old_at == at:
        ids = [*old_ids, *(i for i in ids if i not in old_ids)]
    return json.dumps({"at": at, "ids": ids})


async def _load_vec(db: aiosqlite.Connection) -> bool:
    try:
        import sqlite_vec

        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
        return True
    except Exception as exc:  # pragma: no cover - platform dependent
        import logging

        logging.getLogger(__name__).warning("sqlite-vec failed to load: %s", exc)
        return False


def _peek_version(path: Path) -> tuple[bool, int]:
    """(has tables, user_version) of a database file, before anything changes it."""
    if not path.exists():
        return False, 0
    con = sqlite3.connect(path, timeout=30)
    try:
        tables = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
        return bool(tables), int(con.execute("PRAGMA user_version").fetchone()[0])
    finally:
        con.close()


class Store:
    def __init__(self, path: Path | None = None, *, before_migration: Callable[[str], Awaitable[None]] | None = None):
        self.path = path or paths.db_file()
        self._db: aiosqlite.Connection | None = None
        self.vec_available = False
        # called once, before the first change to an existing database (a snapshot, issue #151)
        self.before_migration = before_migration
        self._existing = False
        self._snapshot_taken = False

    async def _before_migration(self, reason: str) -> None:
        if not self._existing or self._snapshot_taken or self.before_migration is None:
            return
        self._snapshot_taken = True
        try:
            await self.before_migration(reason)
        except Exception:  # a failed snapshot never keeps Sentient from starting
            log.exception("could not take a snapshot before updating the database")

    # ------------------------------------------------------------------ lifecycle
    async def open(self) -> Store:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._existing, version = await asyncio.to_thread(_peek_version, self.path)
        if self._existing and version != SCHEMA_VERSION:
            await self._before_migration(f"before updating the database (version {version} to {SCHEMA_VERSION})")
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        self.vec_available = await _load_vec(self._db)
        await self._db.executescript(_SCHEMA.read_text(encoding="utf-8"))
        # additive migrations for databases created by older builds
        await self.ensure_column("sessions", "context_summary", "TEXT")
        await self.ensure_column("sessions", "context_upto", "TEXT")
        await self.ensure_column("sessions", "untrusted", "TEXT")
        await self.ensure_column("sessions", "visited_hosts", "TEXT")
        await self.ensure_column("sessions", "incognito", "INTEGER NOT NULL DEFAULT 0")
        await self.ensure_column("sessions", "model", "TEXT")
        await self.ensure_column("messages", "attachments", "TEXT")
        await self.ensure_column("messages", "interjection", "INTEGER NOT NULL DEFAULT 0")
        await self.ensure_column("messages", "memory_sources", "TEXT")
        await self.ensure_column("messages", "sources", "TEXT")
        await self.ensure_column("messages", "approval", "TEXT")
        await self.ensure_column("messages", "checks", "TEXT")
        await self.ensure_column("facts", "status", "TEXT NOT NULL DEFAULT 'active'")
        await self.ensure_column("facts", "review", "TEXT")
        await self.ensure_column("summaries", "untrusted", "TEXT")
        await self.ensure_column("usage", "cost_usd", "REAL")
        # the activity timeline's spending lines (#147); after ensure_column: older databases get cost_usd above
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_usage_paid ON usage(created_at) WHERE cost_usd > 0")
        # summaries written before the mark existed take it from their chat (ADR 0021)
        await self.db.execute(
            "UPDATE summaries SET untrusted = (SELECT s.untrusted FROM sessions s WHERE s.id = summaries.session_id)"
            " WHERE untrusted IS NULL AND session_id IS NOT NULL"
        )
        for pkg in PACKAGE_SCHEMAS:
            extra = _PKG_ROOT / pkg / "schema.sql"
            if extra.exists():
                await self._db.executescript(extra.read_text(encoding="utf-8"))
        await self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Store is not open"
        return self._db

    # ------------------------------------------------------------------ helpers
    async def execute(self, sql: str, params: Iterable[Any] = ()) -> aiosqlite.Cursor:
        cur = await self.db.execute(sql, tuple(params))
        await self.db.commit()
        return cur

    async def fetchone(self, sql: str, params: Iterable[Any] = ()) -> aiosqlite.Row | None:
        async with self.db.execute(sql, tuple(params)) as cur:
            return await cur.fetchone()

    async def fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[aiosqlite.Row]:
        async with self.db.execute(sql, tuple(params)) as cur:
            return list(await cur.fetchall())

    async def ensure_column(self, table: str, column: str, decl: str) -> None:
        """Add a column if it is missing. Packages use this for additive schema changes."""
        async with self.db.execute(f"PRAGMA table_info({table})") as cur:
            cols = {r[1] for r in await cur.fetchall()}
        if cols and column not in cols:
            await self._before_migration(f"before adding {column} to {table}")
            await self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    async def get_meta(self, key: str) -> str | None:
        row = await self.fetchone("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    async def set_meta(self, key: str, value: str) -> None:
        await self.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------ sessions & messages
    async def create_session(
        self, channel: str = "web", title: str | None = None, *, incognito: bool = False, model: str | None = None
    ) -> str:
        sid = new_id()
        ts = now_iso()
        await self.execute(
            "INSERT INTO sessions(id, title, channel, created_at, updated_at, incognito, model) VALUES(?,?,?,?,?,?,?)",
            (sid, title, channel, ts, ts, int(bool(incognito)), model or None),
        )
        return sid

    async def set_session_model(self, session_id: str, model: str | None) -> bool:
        """Pick the model for one chat (#312); None (or "") makes it follow the default again. False: no such chat."""
        cur = await self.execute("UPDATE sessions SET model = ? WHERE id = ?", (model or None, session_id))
        return cur.rowcount > 0

    async def is_incognito(self, session_id: str | None) -> bool:
        """True for an incognito chat (#154): nothing is learned from it and it is deleted when closed."""
        if not session_id:
            return False
        row = await self.fetchone("SELECT incognito FROM sessions WHERE id = ?", (session_id,))
        return bool(row and row["incognito"])

    async def incognito_sessions(self, idle_before: str | None = None) -> list[str]:
        """Ids of incognito chats, or only those last used before ``idle_before`` (ISO time)."""
        if idle_before is None:
            rows = await self.fetchall("SELECT id FROM sessions WHERE incognito = 1")
        else:
            rows = await self.fetchall(
                "SELECT id FROM sessions WHERE incognito = 1 AND updated_at < ?", (idle_before,)
            )
        return [r["id"] for r in rows]

    async def touch_session(self, session_id: str, title: str | None = None) -> None:
        if title:
            await self.execute(
                "UPDATE sessions SET updated_at = ?, title = COALESCE(title, ?) WHERE id = ?",
                (now_iso(), title, session_id),
            )
        else:
            await self.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now_iso(), session_id))

    async def list_sessions(self, limit: int = 50) -> list[dict]:
        rows = await self.fetchall(
            "SELECT * FROM sessions WHERE archived = 0 ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str | None,
        *,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        thinking: str | None = None,
        attachments: list[str] | None = None,
        interjection: bool = False,
        memory_sources: list[dict] | None = None,
        sources: list[dict] | None = None,
        approval: dict | None = None,
        checks: list[dict] | None = None,
    ) -> str:
        mid = new_id()
        await self.execute(
            "INSERT INTO messages(id, session_id, role, content, tool_calls, tool_call_id, name, thinking, attachments,"
            " interjection, memory_sources, sources, approval, checks, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                mid,
                session_id,
                role,
                content,
                json.dumps(tool_calls) if tool_calls else None,
                tool_call_id,
                name,
                thinking,
                json.dumps(attachments) if attachments else None,
                int(bool(interjection)),
                json.dumps(memory_sources, ensure_ascii=False) if memory_sources else None,
                json.dumps(sources, ensure_ascii=False) if sources else None,
                json.dumps(approval, ensure_ascii=False) if approval else None,
                json.dumps(checks, ensure_ascii=False) if checks else None,
                now_iso(),
            ),
        )
        await self.touch_session(session_id)
        return mid

    async def recent_messages(self, session_id: str, limit: int) -> list[dict]:
        rows = await self.fetchall(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (session_id, limit),
        )
        out = []
        for r in reversed(rows):
            d = dict(r)
            d["tool_calls"] = json.loads(d["tool_calls"]) if d["tool_calls"] else None
            d["attachments"] = json.loads(d["attachments"]) if d.get("attachments") else []
            d["interjection"] = bool(d.get("interjection"))
            d["memory_sources"] = json.loads(d["memory_sources"]) if d.get("memory_sources") else []
            d["sources"] = json.loads(d["sources"]) if d.get("sources") else []
            d["approval"] = json.loads(d["approval"]) if d.get("approval") else None
            d["checks"] = json.loads(d["checks"]) if d.get("checks") else []
            out.append(d)
        return out

    async def get_session(self, session_id: str) -> dict | None:
        row = await self.fetchone("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return dict(row) if row else None

    async def count_messages(self, session_id: str) -> int:
        row = await self.fetchone("SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,))
        return int(row["n"]) if row else 0

    async def search_messages(self, query: str, limit: int = 10) -> list[dict]:
        """Keyword search over past chats. Incognito chats are never found (#154)."""
        rows = await self.fetchall(
            "SELECT m.* FROM messages_fts f JOIN messages m ON m.rowid = f.rowid"
            f" WHERE messages_fts MATCH ? AND {NOT_INCOGNITO.format(col='m.session_id')} ORDER BY rank LIMIT ?",
            (query, limit),
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ notifications
    async def add_notification(self, kind: str, body: str, title: str | None = None, payload: dict | None = None) -> str:
        """Low-level insert. Services should call ``app.notify(...)`` instead, which also
        publishes ``notification.new`` to the desktop UI."""
        nid = new_id()
        await self.execute(
            "INSERT INTO notifications(id, kind, title, body, payload, created_at) VALUES(?,?,?,?,?,?)",
            (nid, kind, title, body, json.dumps(payload) if payload else None, now_iso()),
        )
        return nid

    async def record_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int, *, role: str | None = None, source: str = "chat",
        cost_usd: float | None = None,
    ) -> None:
        await self.execute(
            "INSERT INTO usage(model, role, source, prompt_tokens, completion_tokens, cost_usd, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (model, role, source, prompt_tokens, completion_tokens, cost_usd, now_iso()),
        )

    async def delete_session(self, session_id: str) -> None:
        await self.execute("DELETE FROM subagents WHERE session_id = ?", (session_id,))
        await self.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    async def rename_session(self, session_id: str, title: str) -> None:
        await self.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, now_iso(), session_id))
