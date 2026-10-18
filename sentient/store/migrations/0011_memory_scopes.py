"""Memory scopes (#155), and outcome checks (#165), which arrived at version 10 without a step of their own."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    # every memory saved before scopes is personal
    await store.ensure_column("facts", "scope", "TEXT NOT NULL DEFAULT 'personal'")
    await store.ensure_column("sessions", "memory_scope", "TEXT")
    await store.ensure_column("tasks", "memory_scope", "TEXT")
    # the chat's memory scope when the summary was written
    await store.ensure_column("summaries", "scope", "TEXT")
    # outcome checks (#165)
    await store.ensure_column("messages", "checks", "TEXT")
    await store.ensure_column("task_runs", "checks", "TEXT")
