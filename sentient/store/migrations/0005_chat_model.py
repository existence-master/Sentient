"""The model chosen for one chat (#312)."""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("sessions", "model", "TEXT")
