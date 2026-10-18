"""A routine's setup checklist (#161) and the time a run skipped for a missing source is tried again."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("tasks", "setup", "TEXT")
    await store.ensure_column("task_runs", "retry_at", "TEXT")
