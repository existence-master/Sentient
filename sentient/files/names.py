"""Names for files people share with Sentient (#458).

A chat keeps its attachments by name (``screens/Inbox - Mail.png``) and shows whatever file has that name, so a name
must never point at a different file later. A file can be deleted while a chat still has it attached (an incognito
chat's capture goes with that chat, #332; the user deletes a file), and the next file must not take the freed name.

- Screen captures and selected text always get a short random part (``screens/Inbox - Mail 3f9a2c.png``): they are
  deleted on their own and name what was on screen, so even a name a memory or summary still mentions is never
  handed out again.
- Other uploads keep their own name, adding `` (1)``, `` (2)``... when a file has it or any message has it attached.
"""

from __future__ import annotations

import json
import secrets
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sentient import paths


def _norm(name: str) -> str:
    # one form for the same name: composed accents (macOS may store them decomposed) and any letter case, like
    # Windows and macOS file systems
    return unicodedata.normalize("NFC", str(name or "")).replace("\\", "/").lstrip("/").casefold()


def _variants(name: str, *, unique: bool) -> Iterator[str]:
    p = Path(name)
    stem, suffix = p.stem, p.suffix
    if unique:
        while True:
            yield f"{stem} {secrets.token_hex(3)}{suffix}"
    yield name
    i = 1
    while True:
        yield f"{stem} ({i}){suffix}"
        i += 1


async def attached(store: Any, folder: str) -> set[str]:
    """Every name under ``folder`` (``uploads``) that a message has attached, also when its file is gone, normalized:
    forward slashes and case-folded."""
    # the folder name is plain ASCII, so it matches however JSON stored the rest of the name (non-ASCII is escaped)
    rows = await store.fetchall("SELECT attachments FROM messages WHERE attachments LIKE ?", (f"%{folder}%",))
    prefix = _norm(folder) + "/"
    out: set[str] = set()
    for row in rows:
        try:
            items = json.loads(row["attachments"] or "[]")
        except ValueError:
            continue
        if isinstance(items, list):
            out |= {n for n in (_norm(i) for i in items if isinstance(i, str)) if n.startswith(prefix)}
    return out


async def referenced(store: Any, name: str) -> bool:
    """True when any message has ``name`` (a Files API name) attached, also when its file is gone."""
    norm = _norm(name)
    return norm in await attached(store, norm.split("/", 1)[0])


async def create(folder: Path, name: str, *, store: Any = None, unique: bool = False) -> Path:
    """Create an empty file in ``folder`` (under ``files/``) for ``name`` and return its path. The name is one no file
    there has and, with ``store``, no message has attached. ``unique`` always adds a short random part."""
    folder.mkdir(parents=True, exist_ok=True)
    rel = folder.resolve().relative_to(paths.files_dir().resolve()).as_posix()
    taken = await attached(store, rel) if store is not None else set()
    for candidate in _variants(name, unique=unique):
        target = folder / candidate
        if _norm(f"{rel}/{candidate}") in taken:
            continue
        try:
            with target.open("xb"):  # claims the name at once, so two uploads at the same moment can't share it
                pass
        except FileExistsError:
            continue
        return target
    raise AssertionError("unreachable")  # _variants never ends
