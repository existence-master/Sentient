"""The activity timeline (#147): how the user answered each approval.

The timeline's indexes come from the ``schema.sql`` files, which run after the steps.
"""

from __future__ import annotations

from typing import Any


async def migrate(store: Any) -> None:
    await store.ensure_column("messages", "approval", "TEXT")
