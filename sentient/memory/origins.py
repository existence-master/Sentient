"""Where a memory came from, as a source you can disconnect (#153).

A fact's ``source`` names where it was learned. A chat on a desktop window is ``conversation``; a chat in a paired
messaging app is ``channel:<telegram|discord|whatsapp>``; a voice chat from a paired device is ``device:<node id>``.
Besides that first source, a fact learned in a chat is also linked (``fact_sources``) to every connected app
(``app:<plugin id>``, an MCP server's plugin id included) and folder of notes (``folder:<folder id>``) whose content
that chat read, taken from the answer sources the chat already keeps (``messages.sources``, #174). Disconnecting one of
them with "Also forget what Sentient learned from it" deletes the facts no other source backs.

Deterministic: nothing here asks a model.
"""

from __future__ import annotations

import json
import os
from typing import Any

CONVERSATION = "conversation"
CHANNELS = frozenset({"telegram", "discord", "whatsapp"})
KINDS = ("app", "channel", "device", "folder")
# answer sources that are not an app you connect (Sentient's own web tools, files and notes)
NOT_APPS = frozenset({"web", "browser", "internet_search", "folders", "files", "memory"})


def key(kind: str, ident: str) -> str:
    """``app:gmail``, ``channel:telegram``, ``device:<id>``, ``folder:<id>``. ValueError for an unknown kind."""
    kind, ident = str(kind or "").strip(), str(ident or "").strip()
    if kind not in KINDS or not ident:
        raise ValueError(f"unknown source {kind}:{ident}")
    return f"{kind}:{ident}"


def split(source: str) -> tuple[str, str]:
    """``("channel", "telegram")`` for ``channel:telegram``; ``("", source)`` for anything else."""
    kind, sep, ident = str(source or "").partition(":")
    return (kind, ident) if sep and kind in KINDS and ident else ("", str(source or ""))


def is_chat(source: str) -> bool:
    """A source that is a chat with the user (the desktop, a messaging app, a device): what they said themselves."""
    return source == CONVERSATION or split(source)[0] in {"channel", "device"}


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(str(path or "")))


async def _folders(store: Any) -> list[tuple[str, str, str]]:
    try:
        rows = await store.fetchall("SELECT id, path, name FROM folders")
    except Exception:  # the folders package isn't there (an older test store)
        return []
    return [(str(r["id"]), _norm(r["path"]), str(r["name"] or "")) for r in rows]


def _folder_of(item: dict, folders: list[tuple[str, str, str]]) -> str | None:
    path = item.get("path")
    if isinstance(path, str) and path:
        p = _norm(path)
        best = max(
            (f for f in folders if p == f[1] or p.startswith(f[1].rstrip(os.sep) + os.sep)),
            key=lambda f: len(f[1]), default=None,
        )
        if best is not None:
            return best[0]
    name = item.get("folder")
    same = [f for f in folders if name and f[2] == name]
    return same[0][0] if len(same) == 1 else None


def from_answer_sources(items: list, folders: list[tuple[str, str, str]]) -> list[str]:
    """The disconnectable sources among one reply's answer sources (``AnswerSources.items()``)."""
    out: list[str] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        kind, app = item.get("kind"), str(item.get("app") or "")
        found: str | None = None
        if kind == "file" and (item.get("folder") or app == "folders"):
            fid = _folder_of(item, folders)
            found = f"folder:{fid}" if fid else None
        elif kind in {"email", "app"} and app and app not in NOT_APPS:
            found = f"app:{app}"
        if found and found not in out:
            out.append(found)
    return out


async def of_session(store: Any, session_id: str | None) -> tuple[str, list[str]]:
    """``(first source, other sources)`` of what is learned in chat ``session_id``: the chat's own app (desktop,
    messaging app or device) and every app and folder its replies read."""
    if not session_id:
        return CONVERSATION, []
    row = await store.fetchone("SELECT * FROM sessions WHERE id = ?", (session_id,))
    if row is None:
        return CONVERSATION, []
    keys = row.keys()
    channel = str(row["channel"] or "")
    node = str(row["node_id"] or "") if "node_id" in keys else ""
    if channel in CHANNELS:
        first = f"channel:{channel}"
    elif node:
        first = f"device:{node}"
    else:
        first = CONVERSATION
    rows = await store.fetchall(
        "SELECT sources FROM messages WHERE session_id = ? AND sources IS NOT NULL AND sources != '[]'"
        " ORDER BY created_at",
        (session_id,),
    )
    others: list[str] = []
    if rows:
        folders = await _folders(store)
        for r in rows:
            try:
                items = json.loads(r["sources"] or "[]")
            except ValueError:
                continue
            for s in from_answer_sources(items, folders):
                if s != first and s not in others:
                    others.append(s)
    return first, others
