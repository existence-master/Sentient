"""How far self-evolution's review has read the activity (#297), plus a column that arrived at version 7 without a
step of its own."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("evolution_reviews", "activity_mark", "TEXT")
    # the pages, emails, files and apps a task run used (#333)
    await store.ensure_column("task_runs", "sources", "TEXT")
