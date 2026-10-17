"""A chat started on a paired device remembers which one (#153), so removing the device can forget it."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("sessions", "node_id", "TEXT")
