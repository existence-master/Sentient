"""SQLite persistence for channels: connection state, paired chats, pairing codes, refusals, delivered task questions."""

from __future__ import annotations

import hashlib
import hmac
import secrets as _pysecrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sentient.store.db import Store, now_iso

SCHEMA = Path(__file__).with_name("schema.sql")


def _hash(channel: str, code: str) -> str:
    return hashlib.sha256(f"{channel}:{code}".encode()).hexdigest()


class PairingResult:
    OK = "ok"
    INVALID = "invalid"  # wrong code (or none active)
    EXPIRED = "expired"
    BURNED = "burned"  # too many wrong attempts: code cancelled


class ChannelStore:
    def __init__(self, store: Store):
        self.store = store

    async def migrate(self) -> None:
        await self.store.db.executescript(SCHEMA.read_text(encoding="utf-8"))
        await self.store.db.commit()

    # ------------------------------------------------------------------ state
    async def state(self, channel: str) -> dict[str, Any]:
        row = await self.store.fetchone("SELECT * FROM channel_state WHERE channel = ?", (channel,))
        if row is None:
            return {"channel": channel, "enabled": 0, "status": "disconnected", "account_label": None,
                    "error": None, "cursor": None}
        return dict(row)

    async def set_state(self, channel: str, **fields: Any) -> None:
        await self.store.execute(
            "INSERT INTO channel_state(channel, updated_at) VALUES(?, ?) ON CONFLICT(channel) DO NOTHING",
            (channel, now_iso()),
        )
        if fields:
            cols = ", ".join(f"{k} = ?" for k in fields)
            await self.store.execute(
                f"UPDATE channel_state SET {cols}, updated_at = ? WHERE channel = ?",
                [*fields.values(), now_iso(), channel],
            )

    # ------------------------------------------------------------------ paired chats
    async def chats(self, channel: str) -> list[dict]:
        rows = await self.store.fetchall(
            "SELECT * FROM channel_chats WHERE channel = ? ORDER BY paired_at", (channel,)
        )
        return [self._chat(r) for r in rows]

    async def chat(self, channel: str, chat_id: str) -> dict | None:
        row = await self.store.fetchone(
            "SELECT * FROM channel_chats WHERE channel = ? AND chat_id = ?", (channel, str(chat_id))
        )
        return self._chat(row) if row else None

    @staticmethod
    def _chat(row: Any) -> dict:
        return {
            "chat_id": row["chat_id"],
            "label": row["label"],
            "paired_at": row["paired_at"],
            "deliver": bool(row["deliver"]),
            "session_id": row["session_id"],
        }

    async def add_chat(self, channel: str, chat_id: str, label: str, *, deliver: bool, session_id: str | None) -> dict:
        await self.store.execute(
            "INSERT INTO channel_chats(channel, chat_id, label, paired_at, deliver, session_id) VALUES(?,?,?,?,?,?)"
            " ON CONFLICT(channel, chat_id) DO UPDATE SET label = excluded.label, paired_at = excluded.paired_at",
            (channel, str(chat_id), label, now_iso(), int(deliver), session_id),
        )
        await self.store.execute(
            "DELETE FROM channel_refusals WHERE channel = ? AND chat_id = ?", (channel, str(chat_id))
        )
        chat = await self.chat(channel, chat_id)
        assert chat is not None
        return chat

    async def update_chat(self, channel: str, chat_id: str, **fields: Any) -> bool:
        if "deliver" in fields:
            fields["deliver"] = int(bool(fields["deliver"]))
        cols = ", ".join(f"{k} = ?" for k in fields)
        cur = await self.store.execute(
            f"UPDATE channel_chats SET {cols} WHERE channel = ? AND chat_id = ?", [*fields.values(), channel, str(chat_id)]
        )
        return cur.rowcount > 0

    async def remove_chat(self, channel: str, chat_id: str) -> bool:
        cur = await self.store.execute(
            "DELETE FROM channel_chats WHERE channel = ? AND chat_id = ?", (channel, str(chat_id))
        )
        return cur.rowcount > 0

    async def chat_for_session(self, session_id: str) -> tuple[str, dict] | None:
        row = await self.store.fetchone("SELECT * FROM channel_chats WHERE session_id = ?", (session_id,))
        return (row["channel"], self._chat(row)) if row else None

    # ------------------------------------------------------------------ task questions
    async def add_question(self, channel: str, chat_id: str, message_ids: list[str], *, task_id: str, run_id: str) -> None:
        """Remember the messages that delivered a task's question, so a reply to one of them answers it."""
        now = now_iso()
        for mid in message_ids:
            await self.store.execute(
                "INSERT OR REPLACE INTO channel_questions(channel, chat_id, message_id, task_id, run_id, created_at)"
                " VALUES(?,?,?,?,?,?)",
                (channel, str(chat_id), str(mid), task_id, run_id, now),
            )
        # bounded: questions older than 90 days are no longer worth matching
        cutoff = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        await self.store.execute("DELETE FROM channel_questions WHERE created_at < ?", (cutoff,))

    async def question_for(self, channel: str, chat_id: str, message_id: str) -> dict | None:
        row = await self.store.fetchone(
            "SELECT task_id, run_id FROM channel_questions WHERE channel = ? AND chat_id = ? AND message_id = ?",
            (channel, str(chat_id), str(message_id)),
        )
        return {"task_id": row["task_id"], "run_id": row["run_id"]} if row else None

    # ------------------------------------------------------------------ refusals
    async def refuse_once(self, channel: str, chat_id: str) -> bool:
        """True the first time a given unpaired chat is refused."""
        cur = await self.store.execute(
            "INSERT OR IGNORE INTO channel_refusals(channel, chat_id, refused_at) VALUES(?,?,?)",
            (channel, str(chat_id), now_iso()),
        )
        return cur.rowcount > 0

    # ------------------------------------------------------------------ pairing codes
    async def new_code(self, channel: str, minutes: int) -> tuple[str, str]:
        code = f"{_pysecrets.randbelow(1_000_000):06d}"
        expires = (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()
        await self.store.execute(
            "INSERT INTO channel_pairing_codes(channel, code_hash, expires_at, attempts, created_at) VALUES(?,?,?,0,?)"
            " ON CONFLICT(channel) DO UPDATE SET code_hash = excluded.code_hash, expires_at = excluded.expires_at,"
            " attempts = 0, created_at = excluded.created_at",
            (channel, _hash(channel, code), expires, now_iso()),
        )
        return code, expires

    async def redeem_code(self, channel: str, code: str, max_attempts: int) -> str:
        row = await self.store.fetchone("SELECT * FROM channel_pairing_codes WHERE channel = ?", (channel,))
        if row is None:
            return PairingResult.INVALID
        if datetime.fromisoformat(row["expires_at"]) <= datetime.now(UTC):
            await self.clear_code(channel)
            return PairingResult.EXPIRED
        if hmac.compare_digest(row["code_hash"], _hash(channel, code.strip())):
            await self.clear_code(channel)  # single use
            return PairingResult.OK
        attempts = int(row["attempts"]) + 1
        if attempts >= max_attempts:
            await self.clear_code(channel)
            return PairingResult.BURNED
        await self.store.execute(
            "UPDATE channel_pairing_codes SET attempts = ? WHERE channel = ?", (attempts, channel)
        )
        return PairingResult.INVALID

    async def clear_code(self, channel: str) -> None:
        await self.store.execute("DELETE FROM channel_pairing_codes WHERE channel = ?", (channel,))
