"""Scoped approval grants (#139), plus columns that arrived at version 5 without a step of their own.

The ``approval_grants`` table itself comes from ``sentient/store/schema.sql``, like every new table.
"""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    # edit protection for skills (#164)
    await store.ensure_column("skill_stats", "origin_hash", "TEXT")
    await store.ensure_column("skill_stats", "origin_text", "TEXT")
    await store.ensure_column("skill_stats", "proposal_base_hash", "TEXT")
    # a task never paused while the user is away (#163)
    await store.ensure_column("tasks", "keep_running", "INTEGER NOT NULL DEFAULT 0")
