"""File snapshots before code writes (#142, ADR 0026).

Code Sentient runs can change files: a command on this computer (``terminal_run``), a script run as a local process
(``execute_code``) and ``file_write`` in Sentient's own files. Before one of them runs, ``guard`` copies the files it
could change into ``<home>/snapshots/<id>/``: for a command or a script, every file in the user's allowed folders
(Settings > Terminal) and Sentient's files folder; for ``file_write``, the one file. When it is done, the copies of
files that didn't change are dropped at once. If something changed, a row in ``file_snapshots`` lists what (changed,
deleted, made) and the Activity timeline offers **Restore**, which puts the kept copies back (``restore``). Restore
itself keeps copies of what it replaces first, so it can be undone the same way.

**Bounded, and cheap to decide.** First the folders are listed with ``stat`` alone (no file is opened): files over
``tools.snapshots.max_file_mb`` and files whose name says they aren't text (``BINARY_SUFFIXES``) are left out with a
note. If what is left is more than ``max_files`` files or ``max_total_mb``, or the folders hold more than
``WALK_LIMIT`` files, nothing is copied for that run: copying part of a folder would look safe and isn't. If it
then changes files, Activity still lists them, with "Too many files to keep a copy" and no Restore. Folders such as
``.git`` and ``node_modules`` are never looked at. Copies older than ``keep_days`` go, and all of them together stay
under ``max_kept_mb`` (oldest first).

**Crash-safe restore.** Restore marks the snapshot ``restoring_at`` before it writes anything and ``restored_at`` only
after every file is back. A restore cut short by a crash leaves ``restoring_at`` set; ``repair`` (at start) clears
it, so the snapshot can be restored again. Nothing is copied for an incognito chat (#154),
for a script in Docker (it can only touch its own folder) or for a command that never needs asking (it only reads).

All file work runs in worker threads. Nothing here calls a model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import stat
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sentient import paths
from sentient.services import Service
from sentient.store.db import new_id, now_iso

log = logging.getLogger(__name__)

SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".next", ".cache",
})
WALK_LIMIT = 10_000  # files listed for one run, across all its folders; more than this and nothing is copied
SNIFF_BYTES = 8192  # a file with a zero byte in its start isn't text
RECENT_S = 2.0  # a file changed this close to the copy is compared by content, not only by size and time
MAX_CREATED = 50
ORPHAN_AGE_S = 3600
PRUNE_EVERY_S = 6 * 3600
SOURCES = ("terminal", "code", "file_write", "restore")

# names that say a file isn't text: left out by name, so deciding what to copy never opens a file
BINARY_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tif", ".tiff", ".heic", ".psd", ".pdf", ".zip",
    ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".tar", ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib",
    ".class", ".jar", ".pyc", ".whl", ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".mov", ".avi", ".mkv",
    ".webm", ".woff", ".woff2", ".ttf", ".otf", ".db", ".sqlite", ".sqlite3", ".docx", ".xlsx", ".pptx", ".iso",
    ".dmg", ".msi",
})
TOO_MANY = "Too many files to keep a copy."
NOTES = {
    "big": "Too big to keep a copy (over {mb} MB).",
    "binary": "Not a text file, so no copy was kept.",
    "too_many": TOO_MANY,
    "unreadable": "Couldn't be read, so no copy was kept.",
}
TOO_MANY_NOTE = (
    "Too many files to keep a copy: the folders hold more than {files} text files or {mb} MB of text, so nothing "
    "was copied. Raise the limits in Settings > Approvals & safety, or allow a smaller folder in Settings > Terminal."
)


def store_dir() -> Path:
    return paths.home() / "snapshots"


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _never_entered() -> set[str]:
    """Sentient's own working folders: never copied, even when an allowed folder holds them."""
    return {_key(str(store_dir())), _key(str(paths.home() / "sandbox"))}


def _walk(roots: Iterable[Path], limit: int) -> tuple[list[tuple[str, os.stat_result]], bool]:
    """``([(path, stat)], complete)``: regular files under ``roots`` (a root may be one file), at most ``limit``.
    Links are not followed; ``SKIP_DIRS`` and Sentient's own working folders are left out."""
    skip = _never_entered()
    out: list[tuple[str, os.stat_result]] = []
    seen: set[str] = set()

    def add(path: str) -> bool:
        key = _key(path)
        if key in seen:
            return True
        try:
            st = os.lstat(path)
        except OSError:
            return True
        if not stat.S_ISREG(st.st_mode):
            return True
        if len(out) >= limit:
            return False
        seen.add(key)
        out.append((path, st))
        return True

    for root in roots:
        root_s = str(root)
        if os.path.isfile(root_s) and not os.path.islink(root_s):
            if not add(root_s):
                return out, False
            continue
        if not os.path.isdir(root_s) or _key(root_s) in skip:
            continue
        for folder, dirs, files in os.walk(root_s, followlinks=False):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and _key(os.path.join(folder, d)) not in skip]
            for name in files:
                if not add(os.path.join(folder, name)):
                    return out, False
    return out, True


def _is_text(path: str) -> bool:
    with open(path, "rb") as f:
        return b"\0" not in f.read(SNIFF_BYTES)


def _same(a: str, b: Path) -> bool:
    try:
        with open(a, "rb") as fa, open(b, "rb") as fb:
            while True:
                x, y = fa.read(1 << 16), fb.read(1 << 16)
                if x != y:
                    return False
                if not x:
                    return True
    except OSError:
        return False


@dataclass
class _Entry:
    path: str
    size: int
    mtime_ns: int
    copy: str | None = None
    note: str | None = None  # why no copy was kept (a NOTES key)


@dataclass
class Capture:
    """The copies taken before one run. ``snapshot_id`` is set once it is settled and something changed."""

    id: str
    source: str
    what: str
    origin: dict
    roots: list[Path] = field(default_factory=list)
    started: float = 0.0
    entries: dict[str, _Entry] = field(default_factory=dict)
    complete: bool = True
    too_many: bool = False  # over the limits: nothing was copied (decided from stat alone)
    snapshot_id: str | None = None


def _copy_all(cap: Capture, cfg: Any) -> None:
    """Copy what the run could change into ``store_dir()/cap.id`` (worker thread). Decided from ``stat`` alone first:
    over the limits, nothing is copied (``cap.too_many``) and only the listing is kept, to see what changed."""
    folder = store_dir() / cap.id
    max_file = int(float(cfg.max_file_mb) * 1024 * 1024)
    max_total = int(cfg.max_total_mb) * 1024 * 1024
    found, cap.complete = _walk(cap.roots, WALK_LIMIT)
    wanted: list[_Entry] = []
    for path, st in found:
        entry = _Entry(path=path, size=st.st_size, mtime_ns=st.st_mtime_ns)
        cap.entries[_key(path)] = entry
        if st.st_size > max_file:
            entry.note = "big"
        elif os.path.splitext(path)[1].lower() in BINARY_SUFFIXES:
            entry.note = "binary"
        else:
            wanted.append(entry)
    cap.too_many = (not cap.complete or len(wanted) > int(cfg.max_files)
                    or sum(e.size for e in wanted) > max_total)
    if cap.too_many:
        for entry in wanted:
            entry.note = "too_many"
        return
    copies = 0
    for entry in wanted:
        path = entry.path
        try:
            if not _is_text(path):
                entry.note = "binary"
                continue
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{copies:06d}"
            shutil.copy2(path, folder / name)
        except OSError:
            entry.note = "unreadable"
            continue
        entry.copy = name
        copies += 1


def _settle(cap: Capture, cfg: Any) -> dict | None:
    """What changed since ``_copy_all``, with the copies of unchanged files dropped; None when nothing changed (its
    folder is removed then). Worker thread."""
    folder = store_dir() / cap.id
    changed: list[dict] = []
    for entry in cap.entries.values():
        change = None
        try:
            st = os.lstat(entry.path)
        except FileNotFoundError:
            change = "deleted"
        except OSError:
            st = None
        else:
            if not stat.S_ISREG(st.st_mode):
                change = "deleted"
            elif st.st_size != entry.size or st.st_mtime_ns != entry.mtime_ns:
                change = "changed"
            elif entry.copy and st.st_mtime_ns / 1e9 >= cap.started - RECENT_S and not _same(entry.path, folder / entry.copy):
                change = "changed"  # rewritten within the clock's resolution: compare what it holds
        if change is None:
            if entry.copy:
                with contextlib.suppress(OSError):
                    (folder / entry.copy).unlink()
            continue
        note = NOTES.get(entry.note or "", "").format(mb=f"{float(cfg.max_file_mb):g}") or None
        changed.append({"path": entry.path, "change": change, "bytes": entry.size, "copy": entry.copy, "note": note})
    created: list[str] = []
    if cap.complete:
        after, _ = _walk(cap.roots, WALK_LIMIT)
        created = [p for p, _st in after if _key(p) not in cap.entries][:MAX_CREATED]
    if not changed:
        shutil.rmtree(folder, ignore_errors=True)
        return None
    changed.sort(key=lambda f: f["path"].lower())
    notes = [TOO_MANY_NOTE.format(files=int(cfg.max_files), mb=int(cfg.max_total_mb))] if cap.too_many else []
    kept = sum(int(f["bytes"]) for f in changed if f["copy"])
    return {"files": changed, "created": created, "notes": notes, "bytes": kept}


def _put_back(folder: Path, files: list[dict]) -> int:
    """Copy kept files back to where they were (worker thread). Returns how many."""
    done = 0
    for f in files:
        target = Path(f["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / f["copy"], target)
        done += 1
    return done


def _loads(raw: Any) -> Any:
    try:
        return json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None


class FileSnapshots(Service):
    name = "snapshots"

    def __init__(self, app: Any):
        super().__init__(app)
        self._active: set[str] = set()  # captures not settled yet: their folders have no row yet

    @property
    def config(self) -> Any:
        return self.app.config.tools.snapshots

    async def start(self) -> None:
        await self.repair()
        if self.app.enable_background:
            self.run_every(PRUNE_EVERY_S, self.prune, name="prune", initial_delay=60)

    # ------------------------------------------------------------------ taking copies
    async def origin_of(self, ctx: Any) -> dict | None:
        """Where a run happened ({session_id, task_id, run_id, call_id}); None for an incognito chat (#154), which
        keeps nothing."""
        if ctx is None:
            return {}
        with contextlib.suppress(Exception):
            if await ctx.is_incognito():
                return None
        extra = getattr(ctx, "extra", None) or {}
        return {
            "session_id": getattr(ctx, "session_id", None), "task_id": extra.get("task_id"),
            "run_id": extra.get("run_id"), "call_id": getattr(ctx, "call_id", None),
        }

    def code_roots(self, first: Path | None = None) -> list[Path]:
        """The folders code can change: the allowed folders (``first``'s own folder first) and Sentient's files."""
        from sentient.terminal.guard import allowed_roots

        roots = allowed_roots(self.app.config.terminal.allowed_folders)
        if first is not None:
            roots.sort(key=lambda r: 0 if first == r or r in first.parents else 1)
        return [*roots, paths.files_dir()]

    @contextlib.asynccontextmanager
    async def guard(
        self, roots: Iterable[Path | str], ctx: Any = None, *, source: str, what: str = "", origin: dict | None = None,
    ) -> AsyncIterator[Capture | None]:
        """Copy what ``roots`` hold before the body runs and keep the copies of what changed after it, even when the
        body fails or is cancelled. Yields None when nothing is copied (turned off, an incognito chat)."""
        if origin is None:
            origin = await self.origin_of(ctx)
        if not self.config.enabled or origin is None:
            yield None
            return
        cap = Capture(id=new_id(), source=source, what=str(what or "")[:500], origin=origin,
                      roots=[Path(r) for r in roots], started=time.time())
        self._active.add(cap.id)
        copied = True
        try:
            await asyncio.to_thread(_copy_all, cap, self.config)
        except Exception:  # never stop the run itself: it just goes without copies
            log.exception("snapshots: copying before %s failed", source)
            copied = False
            self._active.discard(cap.id)
            shutil.rmtree(store_dir() / cap.id, ignore_errors=True)
        if not copied:
            yield None
            return
        try:
            yield cap
        finally:
            settle = asyncio.ensure_future(self._settle(cap))
            try:
                await asyncio.shield(settle)
            except asyncio.CancelledError:
                await settle  # the run was stopped: what it changed is still kept
                raise

    async def _settle(self, cap: Capture) -> None:
        try:
            data = await asyncio.to_thread(_settle, cap, self.config)
            if data is None:
                return
            cap.snapshot_id = cap.id
            o = cap.origin
            await self.app.store.execute(
                "INSERT INTO file_snapshots(id, source, what, session_id, task_id, run_id, call_id, files, created,"
                " notes, bytes, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (cap.id, cap.source, cap.what, o.get("session_id"), o.get("task_id"), o.get("run_id"), o.get("call_id"),
                 json.dumps(data["files"], ensure_ascii=False), json.dumps(data["created"], ensure_ascii=False),
                 json.dumps(data["notes"], ensure_ascii=False), data["bytes"], now_iso()),
            )
            log.info("snapshots: kept %d changed file(s) from %s", len(data["files"]), cap.source)
        except Exception:
            log.exception("snapshots: settling after %s failed", cap.source)
        finally:
            self._active.discard(cap.id)

    # ------------------------------------------------------------------ reading and restoring
    async def get(self, snapshot_id: str) -> dict | None:
        row = await self.app.store.fetchone("SELECT * FROM file_snapshots WHERE id = ?", (snapshot_id,))
        if row is None:
            return None
        out = dict(row)
        for key in ("files", "created", "notes"):
            out[key] = _loads(out.get(key)) or []
        return out

    def restorable(self, snap: dict) -> bool:
        """True while Restore can put files back: not done or going yet and the copies are still there."""
        return (not snap.get("restored_at") and not snap.get("restoring_at")
                and any(f.get("copy") for f in snap.get("files") or [])
                and (store_dir() / str(snap["id"])).is_dir())

    async def restore(self, snapshot_id: str) -> dict:
        """Put back the files kept for ``snapshot_id``. What they are replaced with is copied first, so the restore
        can be undone. ``{"ok": True, "restored": n, "message"}`` or ``{"ok": False, "error"}`` in plain words."""
        snap = await self.get(snapshot_id)
        if snap is None:
            return {"ok": False, "error": "There is nothing to put back for this item."}
        if snap.get("restored_at"):
            return {"ok": False, "error": "These files were already put back."}
        if snap.get("restoring_at"):
            return {"ok": False, "error": "These files are being put back right now."}
        files = [f for f in snap["files"] if f.get("copy")]
        if not files:
            return {"ok": False, "error": "No copies were kept for this, so it can't be put back."}
        folder = store_dir() / snapshot_id
        if not all((folder / f["copy"]).is_file() for f in files):
            return {"ok": False, "error": f"The copies are gone. They are kept for {self.config.keep_days} days."}
        # claim it in one write first, so two presses at once can't both put files back; it counts as put back only
        # once every file is written, and a crash in between is repaired at start (``repair``)
        claim = await self.app.store.execute(
            "UPDATE file_snapshots SET restoring_at = ? WHERE id = ? AND restored_at IS NULL AND restoring_at IS NULL",
            (now_iso(), snapshot_id),
        )
        if not (claim.rowcount or 0):
            return {"ok": False, "error": "These files are being put back already."}
        origin = {k: snap.get(k) for k in ("session_id", "task_id", "run_id")}
        count = len(files)
        what = f"{count} file{'s' if count != 1 else ''}"
        try:
            async with self.guard([f["path"] for f in files], source="restore", what=what, origin=origin) as cap:
                done = await asyncio.to_thread(_put_back, folder, files)
        except BaseException as exc:
            await self.app.store.execute("UPDATE file_snapshots SET restoring_at = NULL WHERE id = ?", (snapshot_id,))
            if isinstance(exc, OSError):
                return {"ok": False, "error": f"Some files couldn't be put back: {exc}"}
            raise
        await self.app.store.execute(
            "UPDATE file_snapshots SET restored_at = ?, restored_by = ?, restoring_at = NULL WHERE id = ?",
            (now_iso(), cap.snapshot_id if cap is not None else None, snapshot_id),
        )
        log.info("snapshots: put back %d file(s) from %s", done, snapshot_id)
        return {"ok": True, "restored": done, "message": f"Put back {done} file{'s' if done != 1 else ''}."}

    async def repair(self) -> int:
        """Restores cut short by a crash or a stop (``restoring_at`` set, ``restored_at`` not): they can be restored
        again. Nothing is restoring when this runs (at start). Returns how many."""
        cur = await self.app.store.execute(
            "UPDATE file_snapshots SET restoring_at = NULL WHERE restoring_at IS NOT NULL AND restored_at IS NULL"
        )
        count = cur.rowcount or 0
        if count:
            log.warning("snapshots: %d restore(s) didn't finish; they can be restored again", count)
        return count

    # ------------------------------------------------------------------ retention
    async def prune(self) -> int:
        """Remove copies older than ``keep_days``, then the oldest while all of them are over ``max_kept_mb``, and
        folders left without a row (a crash mid-run). Returns how many snapshots were removed."""
        cfg = self.config
        cutoff = (datetime.now(UTC) - timedelta(days=int(cfg.keep_days))).isoformat()
        rows = await self.app.store.fetchall("SELECT id, bytes, created_at FROM file_snapshots ORDER BY created_at")
        drop = [r["id"] for r in rows if str(r["created_at"]) < cutoff]
        left = [r for r in rows if str(r["created_at"]) >= cutoff]
        total = sum(int(r["bytes"] or 0) for r in left)
        limit = int(cfg.max_kept_mb) * 1024 * 1024
        for r in left:
            if total <= limit:
                break
            drop.append(r["id"])
            total -= int(r["bytes"] or 0)
        for sid in drop:
            await self.app.store.execute("DELETE FROM file_snapshots WHERE id = ?", (sid,))
        keep = {r["id"] for r in rows} - set(drop)
        await asyncio.to_thread(self._remove_folders, set(drop), keep | self._active)
        return len(drop)

    @staticmethod
    def _remove_folders(drop: set[str], keep: set[str]) -> None:
        root = store_dir()
        if not root.is_dir():
            return
        now = time.time()
        for child in _children(root):
            name = child.name
            if name in drop:
                shutil.rmtree(child, ignore_errors=True)
            elif name not in keep:
                with contextlib.suppress(OSError):
                    if now - child.stat().st_mtime > ORPHAN_AGE_S:
                        shutil.rmtree(child, ignore_errors=True)


def _children(root: Path) -> Iterator[Path]:
    with contextlib.suppress(OSError):
        yield from (p for p in root.iterdir() if p.is_dir())
