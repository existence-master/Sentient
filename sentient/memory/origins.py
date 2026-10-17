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


Read = tuple[str, str, dict]  # (source, what was read: title and excerpt, the answer source item)


def from_answer_sources(items: list, folders: list[tuple[str, str, str]]) -> list[Read]:
    """The disconnectable sources among one reply's answer sources (``AnswerSources.items()``), each with what was
    read from it (title and excerpt) and the item itself."""
    out: list[Read] = []
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
            out.append((found, f"{item.get('title') or ''} {item.get('snippet') or ''}", item))
    return out


async def of_session(store: Any, session_id: str | None) -> tuple[str, list[Read]]:
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
    read: list[Read] = []
    if rows:
        folders = await _folders(store)
        for r in rows:
            try:
                items = json.loads(r["sources"] or "[]")
            except ValueError:
                continue
            read += from_answer_sources(items, folders)
    return first, read


def sources_for(fact: str, first: str, read: list[Read], *, held: bool, user_name: str = "") -> list[str]:
    """The sources of ``fact`` learned in a chat (:func:`of_session`), its own first. Held for review because the chat
    read outside content: what it read that says it (else everything it read). Otherwise the chat's, plus every
    folder whose notes say it."""
    said = list(dict.fromkeys(r[0] for r in read if backs(fact, r[1], user_name)))
    if held:
        chosen = said or list(dict.fromkeys(r[0] for r in read))
        if chosen:
            return chosen
    return [first, *(k for k in said if k.startswith("folder:") and k != first)]


# ---------------------------------------------------------------------- spans and confidence (#152)
QUOTE_CHARS = 200
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_ABBREVIATION = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|vs|etc|No|e\.g|i\.e)|\b[A-Z])\.$")


def quote(text: str, fact: str) -> str:
    """The sentence of ``text`` that says most of ``fact`` (its start when none does), short."""
    parts: list[str] = []
    for p in (p.strip() for p in _SENTENCE.split(str(text or ""))):
        if p and parts and _ABBREVIATION.search(parts[-1]):  # "Dr. Mehta" is one sentence
            parts[-1] = f"{parts[-1]} {p}"
        elif p:
            parts.append(p)
    if not parts:
        return ""
    want = _words(fact)
    best = max(parts, key=lambda p: len(want & _words(p)))
    best = " ".join(best.split())
    return best if len(best) <= QUOTE_CHARS else best[: QUOTE_CHARS - 1].rstrip() + "…"


def says(fact: str, text: str, user_name: str = "") -> bool:
    """Whether ``text`` repeats ``fact``: most of its words (three in five, at least two), besides the user's own
    name. Stricter than :func:`backs`: it decides what is scrubbed when a memory is deleted."""
    want = _words(fact) - _words(user_name)
    if len(want) < 2:
        return False
    have = _words(text)
    return len(want & have) >= max(2, -(-len(want) * 3 // 5))


def confidence(source: str, *, held: bool = False) -> float:
    """How sure Sentient is of a fact when it is first learned, from where it came: the user's own words most, what
    it read in an app least."""
    if held:
        return 0.6
    kind, _ = split(source)
    if source in {"manual", "user"}:
        return 1.0
    if is_chat(source) or source == "onboarding":
        return 0.9
    if kind == "folder":
        return 0.8
    if kind == "app":
        return 0.6
    return 0.7  # files, imports, tasks, the user model


def _item_ref(item: dict, session_id: str, fact: str, folder_id: str | None = None) -> dict:
    keep = {k: item.get(k) for k in ("kind", "title", "url", "id", "app", "path") if item.get(k)}
    if keep.get("kind") == "file":
        keep["kind"] = "note"
    if folder_id:
        keep["folder_id"] = folder_id
    text = f"{item.get('snippet') or ''}"
    return {**keep, "session_id": session_id, "quote": quote(text, fact) or None}


async def spans(
    store: Any, fact: str, sources: list[str], session_id: str, read: list[Read], user_name: str = ""
) -> dict[str, dict]:
    """Exactly where each of ``sources`` said ``fact`` in chat ``session_id``: the user's message for the chat
    itself (the newest of its last 40 that says it, else the newest), the email, app result or note for what the chat
    read. ``{source: ref}``; a source with nothing to point at is left out."""
    out: dict[str, dict] = {}
    for src in sources:
        if is_chat(src):
            rows = await store.fetchall(
                "SELECT id, content FROM messages WHERE session_id = ? AND role = 'user'"
                " ORDER BY created_at DESC, rowid DESC LIMIT 40",
                (session_id,),
            )
            if not rows:
                continue
            best = next((r for r in rows if backs(fact, r["content"] or "", user_name)), rows[0])
            out[src] = {"kind": "message", "session_id": session_id, "message_id": best["id"],
                        "quote": quote(best["content"] or "", fact) or None}
            continue
        items = [r for r in read if r[0] == src]
        if not items:
            continue
        item = next((r for r in reversed(items) if backs(fact, r[1], user_name)), items[-1])
        kind, ident = split(src)
        out[src] = _item_ref(item[2], session_id, fact, ident if kind == "folder" else None)
    return out
