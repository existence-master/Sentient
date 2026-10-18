"""The pages, emails, files and apps an answer used (#290)."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("messages", "sources", "TEXT")
