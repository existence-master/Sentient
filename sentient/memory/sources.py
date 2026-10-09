"""Memory sources: which memories a reply had in mind.

Deterministic attribution (issue #136). A chat turn records every fact and user-model insight that
was put into its system prompt, plus every fact a memory tool returned while it ran. The model is
never asked which ones it used: small models answer that unreliably, so the UI shows what it was given.

A source is ``{kind: "fact"|"insight", id, text, source, via: "prompt"|"tool"}``: ``id`` is the fact id
(int) or insight id (str), ``source`` is where the memory came from (a fact's ``source`` such as
``conversation`` or ``file:resume.pdf``; an insight's ``user`` or ``inferred``).
"""

from __future__ import annotations

import contextlib
from typing import Any

# tools whose results are lists of facts ({"id", "fact", ...})
FACT_TOOLS = frozenset({"memory_recall", "memory_search_by_source"})
MAX_SOURCES = 40


class MemorySources:
    """Collects the memories one turn had in mind, first mention wins, in order."""

    def __init__(self) -> None:
        self._items: list[dict] = []
        self._seen: set[tuple[str, str]] = set()

    def _add(self, kind: str, mid: Any, text: Any, source: Any, via: str) -> None:
        if mid is None or not isinstance(text, str) or not text.strip() or len(self._items) >= MAX_SOURCES:
            return
        key = (kind, str(mid))
        if key in self._seen:
            return
        self._seen.add(key)
        self._items.append({"kind": kind, "id": mid, "text": text.strip(), "source": str(source or ""), "via": via})

    def add_facts(self, facts: list[dict], via: str = "prompt") -> None:
        for f in facts or []:
            if isinstance(f, dict):
                self._add("fact", f.get("id"), f.get("content"), f.get("source"), via)

    def add_insights(self, insights: list[dict]) -> None:
        for i in insights or []:
            if isinstance(i, dict):
                self._add("insight", i.get("id"), i.get("statement"), i.get("source"), "prompt")

    def add_tool_result(self, name: str, result: Any) -> None:
        """Facts returned by ``memory_recall`` / ``memory_search_by_source``. Their source is filled in by
        :meth:`resolve`, since the tools do not return it."""
        if name not in FACT_TOOLS or not isinstance(result, list):
            return
        for r in result:
            if isinstance(r, dict) and isinstance(r.get("id"), int):
                self._add("fact", r["id"], r.get("fact"), r.get("source"), "tool")

    async def resolve(self, store: Any) -> list[dict]:
        """The collected sources, with missing fact sources looked up (one query)."""
        missing = [s["id"] for s in self._items if s["kind"] == "fact" and not s["source"]]
        if missing and store is not None:
            with contextlib.suppress(Exception):
                rows = await store.fetchall(
                    f"SELECT id, source FROM facts WHERE id IN ({','.join('?' * len(missing))})", missing
                )
                found = {int(r["id"]): r["source"] for r in rows}
                for s in self._items:
                    if s["kind"] == "fact" and not s["source"]:
                        s["source"] = found.get(s["id"], "")
        return [dict(s) for s in self._items]

    def __len__(self) -> int:
        return len(self._items)
