"""Incognito chats (#154): nothing is learned from them and they are deleted when closed."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("sessions", "incognito", "INTEGER NOT NULL DEFAULT 0")
