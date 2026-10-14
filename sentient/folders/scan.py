"""Walk a folder the user added and list the files worth reading (#176).

Pure functions that run in a worker thread. Nothing here writes to the folder: Sentient only reads it.

Ignore rules, checked against the path inside the folder (forward slashes):

- hidden files and folders (a name starting with ".": ``.obsidian``, ``.git``, ``.trash``) are always left out;
- a pattern ending in ``/`` is a folder: ``node_modules/`` leaves out every folder with that name, and
  ``Archive/2019/`` that one folder;
- any other pattern is matched (``fnmatch``, case-insensitive) against the file name and the whole path:
  ``*.tmp``, ``Templates/*``, ``drafts/*.md``;
- an Obsidian vault's own "Excluded files" (``userIgnoreFilters`` in ``.obsidian/app.json``) are added, except
  regular expressions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

SUFFIXES = frozenset({".md", ".markdown", ".txt", ".text", ".pdf", ".docx"})
MARKDOWN = frozenset({".md", ".markdown"})


@dataclass
class Listing:
    """What a walk found: ``files`` maps the path inside the folder to ``(size, mtime_ns)``."""

    files: dict[str, tuple[int, int]] = field(default_factory=dict)
    skipped: int = 0  # supported files left out: too big, or over the cap
    capped: bool = False  # stopped at ``max_files``


def is_obsidian(root: Path) -> bool:
    return (root / ".obsidian").is_dir()


def obsidian_excluded(root: Path) -> list[str]:
    """The vault's "Excluded files" as ignore patterns (plain paths only; ``/regex/`` filters are skipped)."""
    try:
        data = json.loads((root / ".obsidian" / "app.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for f in data.get("userIgnoreFilters") or [] if isinstance(data, dict) else []:
        if not isinstance(f, str) or not f.strip() or (f.startswith("/") and f.endswith("/") and len(f) > 1):
            continue
        f = f.strip().replace("\\", "/")
        # Obsidian matches a filter as a path prefix: "Templates/" or "Templates" both leave the folder out
        out.append(f if f.endswith("/") or "*" in f or "." in f.rsplit("/", 1)[-1] else f + "/")
    return out


class Ignore:
    """Compiled ignore patterns (see the module docs)."""

    def __init__(self, patterns: list[str] | None = None):
        self.dir_names: set[str] = set()
        self.dir_paths: list[str] = []
        self.globs: list[str] = []
        for raw in patterns or []:
            p = str(raw or "").strip().replace("\\", "/").lstrip("/")
            if not p or p.startswith("#"):
                continue
            if p.endswith("/"):
                p = p.rstrip("/").lower()
                if "/" in p:
                    self.dir_paths.append(p)
                elif p:
                    self.dir_names.add(p)
            else:
                self.globs.append(p.lower())

    def skip_dir(self, rel: str, name: str) -> bool:
        if name.startswith("."):
            return True
        low = rel.lower()
        if name.lower() in self.dir_names or any(low == d or low.startswith(d + "/") for d in self.dir_paths):
            return True
        return any(fnmatchcase(low, g) or fnmatchcase(low + "/", g) for g in self.globs if g.endswith("*"))

    def skip_file(self, rel: str, name: str) -> bool:
        if name.startswith("."):
            return True
        low, low_name = rel.lower(), name.lower()
        return any(fnmatchcase(low_name, g) or fnmatchcase(low, g) for g in self.globs)

    def skip_path(self, rel: str) -> bool:
        """True when a path inside the folder (a file or a folder) is left out by any rule."""
        parts = [p for p in rel.replace("\\", "/").split("/") if p]
        for i, name in enumerate(parts[:-1]):
            if self.skip_dir("/".join(parts[: i + 1]), name):
                return True
        return bool(parts) and self.skip_file("/".join(parts), parts[-1])


def supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUFFIXES


def walk(root: Path, ignore: Ignore, *, max_bytes: int, max_files: int) -> Listing:
    """Every supported file under ``root`` that the rules keep. Symbolic links are not followed (no loops, nothing
    outside the folder). Folders that can't be read are skipped quietly."""
    out = Listing()
    stack = [("", str(root))]
    while stack:
        rel_dir, path = stack.pop()
        try:
            entries = list(os.scandir(path))
        except OSError:
            continue
        entries.sort(key=lambda e: e.name)
        for e in entries:
            rel = f"{rel_dir}/{e.name}" if rel_dir else e.name
            try:
                if e.is_dir(follow_symlinks=False):
                    if not ignore.skip_dir(rel, e.name):
                        stack.append((rel, e.path))
                    continue
                if not e.is_file(follow_symlinks=False) or not supported(e.name) or ignore.skip_file(rel, e.name):
                    continue
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            if st.st_size > max_bytes or len(out.files) >= max_files:
                out.skipped += 1
                out.capped = out.capped or len(out.files) >= max_files
                continue
            out.files[rel] = (int(st.st_size), int(st.st_mtime_ns))
    return out
