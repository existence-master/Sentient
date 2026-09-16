"""sqlite-vec helpers shared by facts and episodic summaries.

vec0 tables default to L2 distance. Every table here is created with
``distance_metric=cosine`` so ``similarity = 1 - distance``. Tables created by
older builds (L2, no metric recorded in ``meta``) or with a different embedding
dimension are dropped and rebuilt by re-embedding the source rows.
"""

from __future__ import annotations

import asyncio
import logging
import math
import struct
from collections.abc import Awaitable, Callable

from sentient.store.db import Store

log = logging.getLogger(__name__)

METRIC = "cosine"


def pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VecTable:
    """A lazily created vec0 table whose rowids mirror an integer key of a source table.

    ``rebuild`` is awaited (with the new dimension) when the table had to be dropped;
    it should re-embed and insert every row.
    """

    def __init__(self, store: Store, name: str):
        self.store = store
        self.name = name
        self.dim: int | None = None
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self.dim is not None

    async def exists(self) -> bool:
        row = await self.store.fetchone("SELECT name FROM sqlite_master WHERE name = ?", (self.name,))
        return row is not None

    async def ensure(self, dim: int, rebuild: Callable[[], Awaitable[None]] | None = None) -> bool:
        """Create the table for ``dim``. Returns True when an existing index was rebuilt."""
        if self.dim == dim:
            return False
        async with self._lock:
            if self.dim == dim:
                return False
            if not self.store.vec_available:
                raise RuntimeError("sqlite-vec extension is not available")
            stored_dim = await self.store.get_meta(f"{self.name}_dim")
            stored_metric = await self.store.get_meta(f"{self.name}_metric")
            had_table = await self.exists()
            stale = had_table and (stored_metric != METRIC or (stored_dim is not None and int(stored_dim) != dim))
            if stale:
                log.warning(
                    "rebuilding vector index %s (dim %s->%s, metric %s->%s)",
                    self.name, stored_dim, dim, stored_metric, METRIC,
                )
                await self.store.execute(f"DROP TABLE IF EXISTS {self.name}")
            await self.store.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.name} USING vec0(embedding float[{dim}] distance_metric={METRIC})"
            )
            await self.store.set_meta(f"{self.name}_dim", str(dim))
            await self.store.set_meta(f"{self.name}_metric", METRIC)
            self.dim = dim
        if stale and rebuild is not None:
            try:
                await rebuild()
            except Exception as exc:  # keep working with a partial index
                log.warning("re-embedding for %s failed: %s", self.name, exc)
        return stale

    async def upsert(self, rowid: int, vec: list[float]) -> None:
        await self.store.execute(f"DELETE FROM {self.name} WHERE rowid = ?", (rowid,))
        await self.store.execute(f"INSERT INTO {self.name}(rowid, embedding) VALUES(?, ?)", (rowid, pack(vec)))

    async def delete(self, rowid: int) -> None:
        if self.ready or await self.exists():
            await self.store.execute(f"DELETE FROM {self.name} WHERE rowid = ?", (rowid,))

    async def knn(self, vec: list[float], k: int) -> list[tuple[int, float]]:
        """[(rowid, similarity)] nearest first."""
        rows = await self.store.fetchall(
            f"SELECT rowid, distance FROM {self.name} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (pack(vec), max(1, k)),
        )
        return [(int(r["rowid"]), 1.0 - float(r["distance"])) for r in rows]

    async def all_vectors(self) -> dict[int, list[float]]:
        rows = await self.store.fetchall(f"SELECT rowid, embedding FROM {self.name}")
        return {int(r["rowid"]): unpack(r["embedding"]) for r in rows}
