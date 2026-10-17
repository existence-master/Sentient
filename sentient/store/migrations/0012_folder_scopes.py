"""Folder scopes (#392), plus columns that arrived at version 11 without a step of their own."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    # folders added before scopes are recalled in every chat
    await store.ensure_column("folders", "scope", "TEXT NOT NULL DEFAULT 'always'")
    # group chats (ADR 0029): only the person who paired one talks to Sentient there
    await store.ensure_column("channel_chats", "is_group", "INTEGER NOT NULL DEFAULT 0")
    await store.ensure_column("channel_chats", "owner_id", "TEXT")
    # stop or snooze a ringing alarm (#390)
    await store.ensure_column("reminders", "stopped_at", "TEXT")
    await store.ensure_column("reminders", "snoozed_from", "TEXT")
