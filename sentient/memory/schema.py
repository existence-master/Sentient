"""Idempotent creation of the memory package's own tables and columns.

``sentient/store/db.py`` only loads the schemas listed in ``PACKAGE_SCHEMAS``; memory
creates its tables itself (lazily, once per store) so it works whether or not core
adds it to that list.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from sentient.store.db import Store

log = logging.getLogger(__name__)

_SQL = Path(__file__).with_name("schema.sql")


async def ensure_memory_schema(store: Store) -> None:
    if getattr(store, "_memory_schema_ready", False):
        return
    lock = getattr(store, "_memory_schema_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        store._memory_schema_lock = lock  # type: ignore[attr-defined]
    async with lock:
        if getattr(store, "_memory_schema_ready", False):
            return
        await store.db.executescript(_SQL.read_text(encoding="utf-8"))
        await store.ensure_column("facts", "recall_count", "INTEGER NOT NULL DEFAULT 0")
        await store.ensure_column("facts", "last_recalled_at", "TEXT")
        await store.ensure_column("user_insights", "review", "TEXT")
        await store.db.commit()
        if await store.get_meta("facts_fts_built") != "1":
            try:
                await store.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
                await store.set_meta("facts_fts_built", "1")
            except Exception as exc:  # pragma: no cover - fts5 missing
                log.warning("facts keyword index unavailable: %s", exc)
        store._memory_schema_ready = True  # type: ignore[attr-defined]
