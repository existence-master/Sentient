"""Where a memory came from, as a source you can disconnect (#153).

A fact's ``source`` names where it was learned. A chat on a desktop window is ``conversation``; a chat in a paired
messaging app is ``channel:<telegram|discord|whatsapp>``; a voice chat from a paired device is ``device:<node id>``.
What a chat read comes from the answer sources it already keeps (``messages.sources``, #174): connected apps
(``app:<plugin id>``, an MCP server's plugin id included) and folders of notes (``folder:<folder id>``), each with the
excerpt the model read. A fact learned in a chat is also linked (``fact_sources``) to a folder whose excerpt says what
the fact says (:func:`backs`); a fact held for review because the chat read outside content belongs to what it read
instead of the chat. Disconnecting one of them with "Also forget what Sentient learned from it" deletes the facts no
other source backs.

Deterministic: nothing here asks a model.
"""

from __future__ import annotations

import json
import os
import re
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


_WORD = re.compile(r"[a-z0-9]+")
_COMMON = frozenset({
    "the", "and", "for", "are", "was", "were", "been", "you", "your", "its", "this", "that", "these", "those", "what",
    "how", "can", "could", "with", "from", "about", "has", "have", "had", "does", "did", "not", "than", "then",
    "there", "their", "they", "them", "she", "his", "her", "him", "our", "who", "whom", "which", "when", "where",
    "why", "will", "would", "should", "also", "very", "just", "now", "into", "over", "under", "more", "most", "some",
    "any", "all", "each", "every", "user", "users", "note", "notes", "says",
})


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(str(text or "").lower()) if len(w) > 2 and w not in _COMMON}


def backs(fact: str, text: str, user_name: str = "") -> bool:
    """Whether ``text`` (an excerpt a chat read) says what ``fact`` says: two of the fact's words, or one of its
    numbers, besides the user's own name. Deterministic, like ``agent.sources.note_used``."""
    shared = _words(fact) & _words(text) - _words(user_name)
    return len(shared) >= 2 or any(any(c.isdigit() for c in w) for w in shared)


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


def from_answer_sources(items: list, folders: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """The disconnectable sources among one reply's answer sources (``AnswerSources.items()``), each with what was
    read from it (title and excerpt)."""
    out: list[tuple[str, str]] = []
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
        if found:
            out.append((found, f"{item.get('title') or ''} {item.get('snippet') or ''}"))
    return out


async def of_session(store: Any, session_id: str | None) -> tuple[str, list[tuple[str, str]]]:
    """``(the chat's own source, what it read)`` for chat ``session_id``: its app (desktop, messaging app or device),
    and every app and folder its replies read, each ``(source, excerpt)`` in order."""
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
    read: list[tuple[str, str]] = []
    if rows:
        folders = await _folders(store)
        for r in rows:
            try:
                items = json.loads(r["sources"] or "[]")
            except ValueError:
                continue
            read += from_answer_sources(items, folders)
    return first, read


def sources_for(fact: str, first: str, read: list[tuple[str, str]], *, held: bool, user_name: str = "") -> list[str]:
    """The sources of ``fact`` learned in a chat (:func:`of_session`), its own first. Held for review because the chat
    read outside content: what it read that says it (else everything it read). Otherwise the chat's, plus every
    folder whose notes say it."""
    said = list(dict.fromkeys(k for k, text in read if backs(fact, text, user_name)))
    if held:
        chosen = said or list(dict.fromkeys(k for k, _ in read))
        if chosen:
            return chosen
    return [first, *(k for k in said if k.startswith("folder:") and k != first)]
