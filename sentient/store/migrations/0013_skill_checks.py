"""A skill's own checks (#386), which arrived at version 12 without a step of their own."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("skill_stats", "check_report", "TEXT")
    await store.ensure_column("skill_stats", "pending_check_report", "TEXT")
