from __future__ import annotations


async def add_fact(
    mem,
    content: str,
    *,
    source: str = "conversation",
    topics: list[str] | None = None,
    memory_type: str = "long-term",
    updated_at: str | None = None,
) -> int:
    """Insert a fact directly, bypassing dedup and the CUD model (for consolidation tests)."""
    [vec] = await mem._embed([content])
    fid = await mem._insert(
        content, vec, source=source, topics=topics or ["Miscellaneous"], memory_type=memory_type, duration=None
    )
    if updated_at:
        await mem.store.execute("UPDATE facts SET updated_at = ?, created_at = ? WHERE id = ?", (updated_at, updated_at, fid))
    return fid


def drain(q) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out
