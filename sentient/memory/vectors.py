"""sqlite-vec helpers shared by facts and episodic summaries.

vec0 tables default to L2 distance. Every table here is created with
``distance_metric=cosine`` so ``similarity = 1 - distance``. Tables created by
older builds (L2, no metric recorded in ``meta``), with a different embedding
dimension or by another embedding model are dropped and rebuilt by re-embedding
the source rows. Two models can share a dimension (nomic-embed-text and
paraphrase-multilingual are both 768, mxbai-embed-large and bge-m3 both 1024),
and their vectors don't compare, so the model is recorded too (#339).
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
        self.model: str | None = None
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self.dim is not None

    async def exists(self) -> bool:
        row = await self.store.fetchone("SELECT name FROM sqlite_master WHERE name = ?", (self.name,))
        return row is not None

    async def ensure(
        self, dim: int, rebuild: Callable[[], Awaitable[None]] | None = None, model: str | None = None
    ) -> bool:
        """Create the table for ``dim`` vectors from embedding ``model``. Returns True when an existing index was
        rebuilt: another dimension or metric, or another model when both the stored and the new model are known."""
        if self.dim == dim and (model is None or self.model == model):
            return False
        async with self._lock:
            if self.dim == dim and (model is None or self.model == model):
                return False
            if not self.store.vec_available:
                raise RuntimeError("sqlite-vec extension is not available")
            stored_dim = await self.store.get_meta(f"{self.name}_dim")
            stored_metric = await self.store.get_meta(f"{self.name}_metric")
            stored_model = await self.store.get_meta(f"{self.name}_model")
            had_table = await self.exists()
            other_model = bool(model and stored_model and stored_model != model)
            stale = had_table and (
                stored_metric != METRIC or (stored_dim is not None and int(stored_dim) != dim) or other_model
            )
            if stale:
                log.warning(
                    "rebuilding vector index %s (dim %s->%s, metric %s->%s, model %s->%s)",
                    self.name, stored_dim, dim, stored_metric, METRIC, stored_model, model,
                )
                await self.store.execute(f"DROP TABLE IF EXISTS {self.name}")
            await self.store.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.name} USING vec0(embedding float[{dim}] distance_metric={METRIC})"
            )
            if not stale:
                await self._record(dim, model)
            self.dim = dim
            self.model = model or self.model
        if stale:
            try:
                if rebuild is not None:
                    await rebuild()
            except Exception as exc:
                # keep working with a partial index; the old dimension and model stay recorded, so the next start
                # rebuilds it again instead of trusting it
                log.warning("re-embedding for %s failed: %s", self.name, exc)
            else:
                await self._record(dim, model)
        return stale

    async def _record(self, dim: int, model: str | None) -> None:
        await self.store.set_meta(f"{self.name}_dim", str(dim))
        await self.store.set_meta(f"{self.name}_metric", METRIC)
        if model:
            await self.store.set_meta(f"{self.name}_model", model)

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
