"""The price of each model call, for the cost meter (#148)."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("usage", "cost_usd", "REAL")
