"""Notifications: persisted in SQLite and pushed live to the desktop window.

Shape returned to the UI (see docs/API.md)::

    {id, kind, title, message, payload, task_id, read, created_at}

``kind``: info | task | approval | proactive | skill | error
"""

from __future__ import annotations

import json
from typing import Any

from sentient.services import Service
from sentient.store.db import new_id, now_iso


def _row_to_dict(r: Any) -> dict:
    payload = json.loads(r["payload"]) if r["payload"] else {}
    return {
        "id": r["id"],
        "kind": r["kind"],
        "title": r["title"],
        "message": r["body"],
        "payload": payload,
        "task_id": payload.get("task_id"),
        "read": bool(r["read"]),
        "created_at": r["created_at"],
    }


class NotificationService(Service):
    name = "notifications"
    MAX_KEPT = 500

    async def create(
        self,
        kind: str,
        message: str,
        *,
        title: str | None = None,
        payload: dict | None = None,
    ) -> dict:
        nid = new_id()
        ts = now_iso()
        await self.app.store.execute(
            "INSERT INTO notifications(id, kind, title, body, payload, created_at) VALUES(?,?,?,?,?,?)",
            (nid, kind, title, message, json.dumps(payload) if payload else None, ts),
        )
        # keep the table bounded (v2 capped at 50; desktop keeps more history)
        await self.app.store.execute(
            "DELETE FROM notifications WHERE id NOT IN (SELECT id FROM notifications ORDER BY created_at DESC LIMIT ?)",
            (self.MAX_KEPT,),
        )
        note = {
            "id": nid,
            "kind": kind,
            "title": title,
            "message": message,
            "payload": payload or {},
            "task_id": (payload or {}).get("task_id"),
            "read": False,
            "created_at": ts,
        }
        self.app.bus.publish("notification.new", note)
        return note

    async def get(self, notification_id: str) -> dict | None:
        r = await self.app.store.fetchone("SELECT * FROM notifications WHERE id = ?", (notification_id,))
        return _row_to_dict(r) if r else None

    async def list(self, limit: int = 100, unread_only: bool = False) -> list[dict]:
        where = "WHERE read = 0" if unread_only else ""
        rows = await self.app.store.fetchall(
            f"SELECT * FROM notifications {where} ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [_row_to_dict(r) for r in rows]

    async def unread_count(self) -> int:
        r = await self.app.store.fetchone("SELECT COUNT(*) AS n FROM notifications WHERE read = 0")
        return int(r["n"]) if r else 0

    async def mark_read(self, notification_id: str | None = None) -> None:
        if notification_id:
            await self.app.store.execute("UPDATE notifications SET read = 1 WHERE id = ?", (notification_id,))
        else:
            await self.app.store.execute("UPDATE notifications SET read = 1")
        self.app.bus.publish("notification.read", {"id": notification_id})

    async def update_payload(self, notification_id: str, payload: dict) -> None:
        await self.app.store.execute(
            "UPDATE notifications SET payload = ? WHERE id = ?", (json.dumps(payload), notification_id)
        )
        note = await self.get(notification_id)
        if note is not None:
            # every window (and the tray) sees suggestion/approval status changes immediately
            self.app.bus.publish("notification.updated", note)

    async def delete(self, notification_id: str | None = None) -> None:
        if notification_id:
            await self.app.store.execute("DELETE FROM notifications WHERE id = ?", (notification_id,))
        else:
            await self.app.store.execute("DELETE FROM notifications")
        self.app.bus.publish("notification.deleted", {"id": notification_id})
