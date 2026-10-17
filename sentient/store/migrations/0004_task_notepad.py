"""Recurring tasks remember their last runs in a notepad (#292)."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("tasks", "notepad", "TEXT")
