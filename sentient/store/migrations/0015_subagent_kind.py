"""Subagents say what kind they are: a worker with tools, or the tool-less reader (#144)."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("subagents", "kind", "TEXT NOT NULL DEFAULT 'worker'")
