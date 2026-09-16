"""Inbound webhooks: let any app or script trigger Sentient (docs/API.md section 16).

A hook has an id (part of its URL) and a random secret shown once at creation. Only a
SHA-256 hash of the secret is stored. A valid call publishes ``source.items``
``{source: "webhook", event: <hook id>, origin: "webhook", items: [{id, name, body, received_at, ...}]}``.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets as pysecrets
from typing import TYPE_CHECKING, Any

from sentient.store.db import new_id, now_iso

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

log = logging.getLogger(__name__)

SOURCE = "webhook"
SECRET_HEADER = "X-Sentient-Secret"
MAX_NAME_CHARS = 80


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def hook_url(hook_id: str, base_url: str | None) -> str:
    # TODO(nodes): when hooks are exposed on the LAN listener, return that address too.
    base = (base_url or "").rstrip("/")
    return f"{base}/hooks/{hook_id}"


class HookStore:
    def __init__(self, mgr: IntegrationManager):
        self.mgr = mgr

    def _public(self, row: dict, base_url: str | None) -> dict:
        return {"id": row["id"], "name": row["name"], "url": hook_url(row["id"], base_url),
                "created_at": row["created_at"], "last_called_at": row["last_called_at"], "calls": int(row["calls"] or 0)}

    async def list(self, base_url: str | None = None) -> list[dict]:
        rows = await self.mgr.app.store.fetchall("SELECT * FROM hooks ORDER BY created_at")
        return [self._public(dict(r), base_url) for r in rows]

    async def get(self, hook_id: str) -> dict | None:
        row = await self.mgr.app.store.fetchone("SELECT * FROM hooks WHERE id = ?", (hook_id,))
        return dict(row) if row else None

    async def create(self, name: str, base_url: str | None = None) -> dict:
        name = " ".join(str(name or "").split())
        if not name:
            raise ValueError("Give the webhook a name, e.g. 'Door sensor' or 'Build finished'.")
        if len(name) > MAX_NAME_CHARS:
            raise ValueError(f"Keep the name under {MAX_NAME_CHARS} characters.")
        hook_id = new_id()[:16]
        secret = pysecrets.token_urlsafe(32)
        row = {"id": hook_id, "name": name, "secret_hash": hash_secret(secret), "created_at": now_iso(),
               "last_called_at": None, "calls": 0}
        await self.mgr.app.store.execute(
            "INSERT INTO hooks(id, name, secret_hash, created_at, last_called_at, calls) VALUES(?,?,?,?,?,?)",
            (row["id"], row["name"], row["secret_hash"], row["created_at"], None, 0),
        )
        await self.mgr.publish(SOURCE)  # the webhook pseudo-integration's triggers changed
        return {**self._public(row, base_url), "secret": secret}

    async def delete(self, hook_id: str) -> bool:
        cur = await self.mgr.app.store.execute("DELETE FROM hooks WHERE id = ?", (hook_id,))
        if cur.rowcount <= 0:
            return False
        await self._disable_tasks_for(hook_id)
        await self.mgr.publish(SOURCE)
        return True

    async def _disable_tasks_for(self, hook_id: str) -> None:
        """Tasks triggered by a deleted hook can never fire again: switch them off and tell the user."""
        app = self.mgr.app
        tasks = getattr(app, "tasks", None)
        if tasks is None:
            return
        names: list[str] = []
        try:
            for task in await tasks.repo.list_tasks():
                schedule = task.get("schedule") or {}
                if (
                    schedule.get("type") == "triggered"
                    and schedule.get("source") == SOURCE
                    and schedule.get("event") == hook_id
                    and task.get("enabled")
                ):
                    await tasks.update(task["id"], {"enabled": False})
                    names.append(task.get("name") or "Untitled task")
            if names:
                listed = ", ".join(f"'{n}'" for n in names[:5])
                await app.notify(
                    "task",
                    f"The webhook was deleted, so I switched off {listed}. Pick another trigger to use it again.",
                    title="Tasks paused",
                    payload={"event": "webhook_removed", "hook_id": hook_id},
                )
        except Exception:
            log.exception("could not disable tasks for deleted webhook %s", hook_id)

    @staticmethod
    def secret_ok(hook: dict, given: str | None) -> bool:
        if not given:
            return False
        return hmac.compare_digest(hash_secret(given), str(hook["secret_hash"]))

    async def receive(self, hook: dict, body: Any, *, content_type: str | None = None,
                      query: dict[str, Any] | None = None) -> dict:
        """Record a verified call and publish it. Returns the item."""
        ts = now_iso()
        await self.mgr.app.store.execute(
            "UPDATE hooks SET calls = calls + 1, last_called_at = ? WHERE id = ?", (ts, hook["id"])
        )
        item = {"id": new_id(), "name": hook["name"], "body": body, "received_at": ts,
                "content_type": content_type or None, "query": dict(query or {})}
        await self.mgr.emit_items(SOURCE, "webhook", [item], event=hook["id"])
        return item
