"""Text the user selected in another app goes with its chat (#332).

Ask about selected text (#173) uploads the selection to ``files/selections/`` and attaches it to the chat it starts.
When that chat is deleted, the file is deleted too: always for an incognito chat (#154), and for any other chat unless
another chat still has the same file attached. A selection no chat has (its chat was deleted before this rule, or the
question was never sent) is removed at the next start once it is ``STRAY_AFTER_S`` old.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from sentient import paths
from sentient.tools.rules import SELECTION_FOLDER, is_selection

log = logging.getLogger(__name__)

STRAY_AFTER_S = 3600  # a selection uploaded just now may be waiting for its chat to start


def _norm(name: str) -> str:
    return str(name or "").replace("\\", "/").lstrip("/")


def _names(raw: str | None) -> set[str]:
    try:
        items = json.loads(raw or "[]")
    except ValueError:
        return set()
    return {_norm(n) for n in items if isinstance(n, str) and is_selection(n)} if isinstance(items, list) else set()


async def attached(store: Any, session_id: str | None = None) -> set[str]:
    """The selection files attached to ``session_id``'s messages, or to any message when it is None."""
    sql = "SELECT attachments FROM messages WHERE attachments LIKE ?"
    params: tuple[Any, ...] = (f"%{SELECTION_FOLDER}/%",)
    if session_id is not None:
        sql += " AND session_id = ?"
        params += (session_id,)
    out: set[str] = set()
    for row in await store.fetchall(sql, params):
        out |= _names(row["attachments"])
    return out


def _remove(names: set[str]) -> list[str]:
    root = paths.files_dir().resolve()
    folder = (root / SELECTION_FOLDER).resolve()
    removed = []
    for name in sorted(names):
        p = (root / name).resolve()
        if folder not in p.parents or not p.is_file():
            continue
        p.unlink(missing_ok=True)
        removed.append(name)
    return removed


async def delete_with_chat(store: Any, names: set[str], *, incognito: bool) -> list[str]:
    """Delete ``names`` (the selections a deleted chat had). Another chat's attachment keeps a file, except after an
    incognito chat. Call it after the chat's messages are gone. Returns the removed names."""
    if not names:
        return []
    keep = set() if incognito else await attached(store)
    return await asyncio.to_thread(_remove, names - keep)


def _strays(used: set[str], older_than: float) -> set[str]:
    folder = paths.files_dir() / SELECTION_FOLDER
    if not folder.is_dir():
        return set()
    out = set()
    for p in folder.rglob("*"):
        name = p.relative_to(paths.files_dir()).as_posix()
        if p.is_file() and name not in used and p.stat().st_mtime < older_than:
            out.add(name)
    return out


async def remove_strays(store: Any, *, now: float | None = None) -> list[str]:
    """Remove selections no chat has, once they are ``STRAY_AFTER_S`` old. Returns the removed names."""
    used = await attached(store)
    strays = await asyncio.to_thread(_strays, used, (now or time.time()) - STRAY_AFTER_S)
    removed = await asyncio.to_thread(_remove, strays)
    if removed:
        log.info("removed %d selected text file(s) no chat has", len(removed))
    return removed

