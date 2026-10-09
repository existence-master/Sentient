"""Async SQLite store with sqlite-vec loaded.

One connection per process is plenty for a single-user desktop app; WAL mode
lets the UI read while a task run writes.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from sentient import paths

_SCHEMA = Path(__file__).with_name("schema.sql")
_PKG_ROOT = Path(__file__).resolve().parents[1]
# Feature packages own their tables. Each file must be idempotent. Order matters for foreign keys.
PACKAGE_SCHEMAS = ["tasks", "integrations", "proactivity", "evolution", "voice", "memory", "nodes", "channels", "browser", "sandbox"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


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


class Store:
    def __init__(self, path: Path | None = None):
        self.path = path or paths.db_file()
        self._db: aiosqlite.Connection | None = None
        self.vec_available = False

    # ------------------------------------------------------------------ lifecycle
    async def open(self) -> Store:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        self.vec_available = await _load_vec(self._db)
        await self._db.executescript(_SCHEMA.read_text(encoding="utf-8"))
        # additive migrations for databases created by older builds
        await self.ensure_column("sessions", "context_summary", "TEXT")
        await self.ensure_column("sessions", "context_upto", "TEXT")
        await self.ensure_column("sessions", "untrusted", "TEXT")
        await self.ensure_column("messages", "attachments", "TEXT")
        await self.ensure_column("messages", "interjection", "INTEGER NOT NULL DEFAULT 0")
        await self.ensure_column("messages", "memory_sources", "TEXT")
        for pkg in PACKAGE_SCHEMAS:
            extra = _PKG_ROOT / pkg / "schema.sql"
            if extra.exists():
                await self._db.executescript(extra.read_text(encoding="utf-8"))
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
    async def create_session(self, channel: str = "web", title: str | None = None) -> str:
        sid = new_id()
        ts = now_iso()
        await self.execute(
            "INSERT INTO sessions(id, title, channel, created_at, updated_at) VALUES(?,?,?,?,?)",
            (sid, title, channel, ts, ts),
        )
        return sid

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
    ) -> str:
        mid = new_id()
        await self.execute(
            "INSERT INTO messages(id, session_id, role, content, tool_calls, tool_call_id, name, thinking, attachments,"
            " interjection, memory_sources, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
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
            out.append(d)
        return out

    async def get_session(self, session_id: str) -> dict | None:
        row = await self.fetchone("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return dict(row) if row else None

    async def count_messages(self, session_id: str) -> int:
        row = await self.fetchone("SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,))
        return int(row["n"]) if row else 0

    async def search_messages(self, query: str, limit: int = 10) -> list[dict]:
        rows = await self.fetchall(
            "SELECT m.* FROM messages_fts f JOIN messages m ON m.rowid = f.rowid"
            " WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
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

    async def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int, *, role: str | None = None, source: str = "chat") -> None:
        await self.execute(
            "INSERT INTO usage(model, role, source, prompt_tokens, completion_tokens, created_at) VALUES(?,?,?,?,?,?)",
            (model, role, source, prompt_tokens, completion_tokens, now_iso()),
        )

    async def delete_session(self, session_id: str) -> None:
        await self.execute("DELETE FROM subagents WHERE session_id = ?", (session_id,))
        await self.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    async def rename_session(self, session_id: str, title: str) -> None:
        await self.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, now_iso(), session_id))
