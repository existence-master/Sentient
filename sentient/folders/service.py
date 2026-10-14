"""Folders and Obsidian vaults as searchable notes (#176).

The user adds folders (``POST /api/folders``). For each one Sentient keeps an index in its own database: the
files it read (``folder_files``), their text cut into pieces (``folder_chunks``), a keyword index (``folder_fts``)
and the pieces' embeddings from the memory embedding model (``folder_vec``). ``folder_search`` searches both and
gives the model the best pieces with where they came from.

**Read only, always.** Nothing here writes to, moves or deletes anything in the user's folders; removing a folder only
forgets Sentient's index of it.

**Kept up to date, cheaply.** A look through a folder (a *scan*) lists its files with their size and change time in
a worker thread and reads only files that are new or changed; files whose text did not change are not embedded
again; files that are gone are forgotten. Scans run one at a time in one worker, after a folder is added, when a file
changes (a watcher, ``folders.watch``), every ``folders.rescan_minutes`` and when the user presses Check again. All
file reading and text extraction runs in threads, so a folder of 10,000 notes never holds up the engine.
Embeddings go through the local model queue (``sentient.llm.jobs``) as background work, so chats go first. When the
embedding model can't be used, keyword search still works and the pieces are embedded at a later scan.

**Clipped web pages.** A folder's notes are the user's own files, so reading them is not outside content
(ADR 0018). A note that is a web page clipped into the vault (a source link in its front matter, a clippings tag
or a Clippings folder) was written by someone else: with ``folders.clippings_are_outside`` (default on) a result
that includes one carries ``outside_content`` first, and the agent loop treats it like a web page (ADR 0025).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sentient import paths
from sentient.folders import notes as notes_mod
from sentient.folders.scan import MARKDOWN, Ignore, is_obsidian, obsidian_excluded, supported, walk
from sentient.memory.vectors import VecTable, pack
from sentient.services import Service
from sentient.store.db import new_id, now_iso

log = logging.getLogger(__name__)

PLUGIN_ID = "folders"
EMBED_BATCH = 32
EMBED_EVERY_FILES = 25  # embed what is waiting after this many files, so search by meaning improves during a long scan
PUBLISH_EVERY_S = 1.0
CANDIDATES = 60  # pieces each search method proposes before they are merged
RRF_K = 60
PER_FILE = 2  # most pieces of one file in one search's results
MIN_SIMILARITY = 0.3
SIMILARITY_WINDOW = 0.15  # a piece found by meaning only counts within this much of the best one
CLIPPED_SOURCE = "a web page clipped into your notes"
_WORDS = re.compile(r"[\w']+", re.U)
_STOP = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "was", "were", "my", "me", "i", "you",
    "it", "this", "that", "what", "whats", "how", "when", "where", "who", "which", "can", "please", "with", "at", "be",
    "do", "does", "did", "about", "tell", "from", "any", "notes", "note", "have", "has", "had",
})


STRONG_MARGIN = 0.15  # a piece this much above folders.prompt_min_similarity goes in without sharing the words


def content_words(text: str) -> list[str]:
    """The words of ``text`` that say something (no stop words, no single letters), lower case, at most 16."""
    return list(dict.fromkeys(w for w in _WORDS.findall(str(text or "").lower()) if w not in _STOP and len(w) > 1))[:16]


def excerpt(text: str, words: list[str], chars: int) -> str:
    """``text`` on one line, cut to about ``chars`` characters around the first of ``words`` it holds."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= chars:
        return flat
    low = flat.lower()
    at = min((i for w in words if (i := low.find(w)) >= 0), default=0)
    start = max(0, at - chars // 3)
    if start:  # begin at a word
        space = flat.find(" ", start)
        start = space + 1 if 0 <= space < at else start
    end = start + chars
    if end < len(flat):
        space = flat.rfind(" ", start, end)
        end = space if space > start + chars // 2 else end
    return ("…" if start else "") + flat[start:end].strip() + ("…" if end < len(flat) else "")


class FolderError(Exception):
    """A plain sentence for the user (a folder that can't be added)."""


def _norm(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _inside(child: str, parent: str) -> bool:
    child, parent = _norm(child), _norm(parent)
    return child == parent or child.startswith(parent.rstrip("\\/") + os.sep)


class FolderService(Service):
    name = "folders"
    pause_on_stop = True
    model_kind = "background"

    def __init__(self, app: Any):
        super().__init__(app)
        self.vec = VecTable(app.store, "folder_vec")
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._queued: set[str] = set()
        self._scanning: str | None = None
        self._scan_task: asyncio.Task | None = None
        self._progress: dict[str, dict] = {}
        self._published: dict[str, float] = {}
        self._embed_error = ""
        self._seen: set[str] = set()  # folders scanned since the engine started
        self._watch_changed = asyncio.Event()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        from sentient.folders.tools import FoldersPlugin

        if self.app.registry.plugin(PLUGIN_ID) is None:
            self.app.registry.register(FoldersPlugin())
        await self._apply_visibility()
        self._loops.append(asyncio.create_task(self._work(), name="folders:worker"))
        if self.app.enable_background:
            self.run_every(60, self._periodic, name="rescan", initial_delay=5)
            self._loops.append(asyncio.create_task(self._watch(), name="folders:watch"))
            self._loops.append(asyncio.create_task(self._watch_config(), name="folders:config"))

    async def halt(self) -> int:
        n = await super().halt()
        task = self._scan_task
        if task is not None and not task.done():
            task.cancel()
            n += 1
        return n

    async def idle(self) -> None:
        """Wait until every queued scan has finished (tests, and the real-test script)."""
        await self._queue.join()

    async def _apply_visibility(self) -> None:
        row = await self.app.store.fetchone("SELECT COUNT(*) AS n FROM folders")
        if self.app.registry.plugin(PLUGIN_ID) is not None:
            self.app.registry.set_hidden(PLUGIN_ID, not (row and row["n"]))

    # ------------------------------------------------------------------ folders
    async def _rows(self) -> list[dict]:
        return [dict(r) for r in await self.app.store.fetchall("SELECT * FROM folders ORDER BY added_at")]

    async def _row(self, folder_id: str) -> dict | None:
        row = await self.app.store.fetchone("SELECT * FROM folders WHERE id = ?", (folder_id,))
        return dict(row) if row else None

    async def overview(self) -> dict:
        rows = await self._rows()
        stats: dict[str, dict] = {r["id"]: {"files": 0, "unreadable": 0, "chunks": 0, "embedded": 0, "clippings": 0}
                                  for r in rows}
        for r in await self.app.store.fetchall(
            "SELECT folder_id, SUM(error IS NULL) AS files, SUM(error IS NOT NULL) AS bad,"
            " SUM(clipped IS NOT NULL) AS clips FROM folder_files GROUP BY folder_id"
        ):
            if r["folder_id"] in stats:
                stats[r["folder_id"]].update(files=int(r["files"] or 0), unreadable=int(r["bad"] or 0),
                                             clippings=int(r["clips"] or 0))
        for r in await self.app.store.fetchall(
            "SELECT folder_id, COUNT(*) AS n, SUM(embedded) AS e FROM folder_chunks GROUP BY folder_id"
        ):
            if r["folder_id"] in stats:
                stats[r["folder_id"]].update(chunks=int(r["n"] or 0), embedded=int(r["e"] or 0))
        return {
            "folders": [self._public(r, stats[r["id"]]) for r in rows],
            "searching_by_meaning": not self._embed_error,
            "embedding_error": self._embed_error or None,
        }

    async def get(self, folder_id: str) -> dict | None:
        return next((f for f in (await self.overview())["folders"] if f["id"] == folder_id), None)

    def _public(self, row: dict, stats: dict) -> dict:
        fid = row["id"]
        if self._scanning == fid:
            status = "scanning"
        elif fid in self._queued:
            status = "waiting"
        elif not Path(row["path"]).is_dir():
            status = "missing"
        elif row.get("last_scan_at"):
            status = "ready"
        else:
            status = "waiting"
        return {
            "id": fid, "path": row["path"], "name": row["name"], "kind": row["kind"],
            "ignore": json.loads(row.get("ignore") or "[]"), "status": status,
            "progress": self._progress.get(fid), "added_at": row["added_at"],
            "last_scan_at": row.get("last_scan_at"), "last_scan_ms": row.get("last_scan_ms"),
            "skipped": int(row.get("skipped") or 0), "error": row.get("error"), **stats,
        }

    def _publish(self, folder_id: str, *, force: bool = False, removed: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._published.get(folder_id, 0.0) < PUBLISH_EVERY_S:
            return
        self._published[folder_id] = now
        data: dict = {"id": folder_id}
        if removed:
            data["removed"] = True
        elif folder_id in self._progress:
            data["progress"] = self._progress[folder_id]
        self.app.bus.publish("folders.updated", data)

    def _check_new(self, path: str, existing: list[dict]) -> Path:
        raw = str(path or "").strip().strip('"')
        if not raw:
            raise FolderError("Pick a folder to add.")
        p = Path(os.path.expanduser(raw))
        if not p.is_absolute():
            raise FolderError("Use the folder's full path.")
        if not p.is_dir():
            raise FolderError("That folder can't be found.")
        p = p.resolve()
        if p.parent == p:
            raise FolderError("A whole drive is too much to read. Pick the folder with your notes in it.")
        if _norm(p) == _norm(Path.home()):
            raise FolderError("Your whole home folder is too much to read. Pick the folder with your notes in it.")
        home = paths.home()
        if _inside(str(p), str(home)) or _inside(str(home), str(p)):
            raise FolderError("That folder holds Sentient's own data. Pick a folder with your notes in it.")
        for f in existing:
            if _norm(f["path"]) == _norm(p):
                raise FolderError(f"{f['name']} is already added.")
            if _inside(str(p), f["path"]):
                raise FolderError(f"That folder is already searched as part of {f['name']}.")
            if _inside(f["path"], str(p)):
                raise FolderError(f"That folder holds {f['name']}, which is already added. Remove it first.")
        return p

    async def add(self, path: str, ignore: list[str] | None = None) -> dict:
        p = self._check_new(path, await self._rows())
        fid = new_id()
        kind = "obsidian" if is_obsidian(p) else "folder"
        await self.app.store.execute(
            "INSERT INTO folders(id, path, name, kind, ignore, added_at) VALUES(?,?,?,?,?,?)",
            (fid, str(p), p.name or str(p), kind, json.dumps(_clean_patterns(ignore)), now_iso()),
        )
        await self._apply_visibility()
        self._watch_changed.set()  # watch it too
        self.enqueue(fid)
        log.info("folder added: %s (%s)", p, kind)
        return await self.get(fid) or {"id": fid}

    async def update(self, folder_id: str, ignore: list[str] | None) -> dict | None:
        if await self._row(folder_id) is None:
            return None
        if ignore is not None:
            await self.app.store.execute(
                "UPDATE folders SET ignore = ? WHERE id = ?", (json.dumps(_clean_patterns(ignore)), folder_id)
            )
            self._watch_changed.set()
            self.enqueue(folder_id)
        return await self.get(folder_id)

    async def remove(self, folder_id: str) -> bool:
        """Forget a folder and its whole index. The folder itself is not touched."""
        row = await self._row(folder_id)
        if row is None:
            return False
        await self.app.store.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        self._queued.discard(folder_id)
        if self._scanning == folder_id and self._scan_task is not None:
            self._scan_task.cancel()
        await self._forget_where("folder_id = ?", (folder_id,))
        await self._apply_visibility()
        self._watch_changed.set()
        self._publish(folder_id, force=True, removed=True)
        log.info("folder removed: %s", row["path"])
        return True

    def enqueue(self, folder_id: str) -> bool:
        """Look through a folder again soon. False when it is already waiting."""
        if folder_id in self._queued:
            return False
        self._queued.add(folder_id)
        self._queue.put_nowait(folder_id)
        self._publish(folder_id, force=True)
        return True

    async def rescan(self, folder_id: str) -> bool:
        if await self._row(folder_id) is None:
            return False
        self.enqueue(folder_id)
        return True

    # ------------------------------------------------------------------ the worker
    async def _work(self) -> None:
        while True:
            fid = await self._queue.get()
            try:
                while self.paused():
                    await asyncio.sleep(2)
                self._queued.discard(fid)
                row = await self._row(fid)
                if row is None:
                    continue
                self._scanning = fid
                self._publish(fid, force=True)
                self._scan_task = asyncio.create_task(self._scan(row), name=f"folders:scan:{fid}")
                try:
                    await asyncio.wait({self._scan_task})
                except asyncio.CancelledError:
                    self._scan_task.cancel()
                    raise
                if self._scan_task.cancelled():
                    self._seen.discard(fid)  # stopped part way (Stop everything): look again at the next check
                elif self._scan_task.exception() is not None:
                    log.error("scanning %s failed", row["path"], exc_info=self._scan_task.exception())
                    with contextlib.suppress(Exception):
                        await self.app.store.execute(
                            "UPDATE folders SET error = ? WHERE id = ?",
                            ("Something went wrong while reading this folder. Try Check again.", fid),
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("folder worker failed on %s", fid)
            finally:
                self._scanning = None
                self._scan_task = None
                self._progress.pop(fid, None)
                self._queue.task_done()
                self._publish(fid, force=True)

    async def _periodic(self) -> None:
        every = self.app.config.folders.rescan_minutes * 60
        for row in await self._rows():
            last = row.get("last_scan_at")
            stale = True
            if last and row["id"] in self._seen:
                with contextlib.suppress(ValueError):
                    stale = (time.time() - datetime.fromisoformat(last).timestamp()) >= every
            if stale:
                self.enqueue(row["id"])

    def _ignore(self, row: dict, root: Path) -> Ignore:
        patterns = list(self.app.config.folders.ignore) + json.loads(row.get("ignore") or "[]")
        if row.get("kind") == "obsidian":
            patterns += obsidian_excluded(root)
        return Ignore(patterns)

    async def _scan(self, row: dict) -> None:
        fid, root = row["id"], Path(row["path"])
        cfg = self.app.config.folders
        started = time.monotonic()
        self._seen.add(fid)
        if not root.is_dir():
            await self.app.store.execute(
                "UPDATE folders SET error = ? WHERE id = ?",
                ("The folder can't be found. If it is on a drive that isn't connected, connect it and press Check "
                 "again. What was found before is kept until then.", fid),
            )
            return
        kind = "obsidian" if is_obsidian(root) else "folder"
        await self._check_embedding_model()
        listing = await asyncio.to_thread(
            walk, root, self._ignore(row, root), max_bytes=int(cfg.max_file_mb * 1024 * 1024), max_files=cfg.max_files
        )
        known = {
            r["rel"]: dict(r) for r in await self.app.store.fetchall(
                "SELECT id, rel, size, mtime_ns, digest FROM folder_files WHERE folder_id = ?", (fid,)
            )
        }
        # a listing that missed part of the folder (a folder it couldn't open, the file cap) forgets nothing
        complete = not listing.partial and not listing.capped
        gone = [k["id"] for rel, k in known.items() if rel not in listing.files] if complete else []
        if gone:
            await self._forget_files(gone)
        todo = [
            (rel, size, mtime) for rel, (size, mtime) in listing.files.items()
            if rel not in known or (known[rel]["size"], known[rel]["mtime_ns"]) != (size, mtime)
        ]
        self._progress[fid] = {"done": 0, "total": len(todo)}
        self._publish(fid, force=True)
        changed = 0
        for i, (rel, size, mtime) in enumerate(todo, 1):
            if await self._index_file(fid, root, rel, size, mtime, known.get(rel)):
                changed += 1
            self._progress[fid] = {"done": i, "total": len(todo)}
            self._publish(fid)
            if i % EMBED_EVERY_FILES == 0:
                await self.app.store.db.commit()
                await self._embed_pending(fid)
        await self.app.store.db.commit()
        await self._embed_pending(fid)
        ms = int((time.monotonic() - started) * 1000)
        error = None
        if listing.capped:
            error = f"This folder has more than {cfg.max_files} files, so only part of it is searched."
        await self.app.store.execute(
            "UPDATE folders SET last_scan_at = ?, last_scan_ms = ?, skipped = ?, error = ?, kind = ? WHERE id = ?",
            (now_iso(), ms, listing.skipped, error, kind, fid),
        )
        log.info(
            "folder scan %s: %d files, %d new or changed (%d re-read), %d gone, %d left out, %d ms",
            row["path"], len(listing.files), len(todo), changed, len(gone), listing.skipped, ms,
        )

    async def _index_file(self, fid: str, root: Path, rel: str, size: int, mtime: int, old: dict | None) -> bool:
        """Read one new or changed file. True when its pieces changed."""
        cfg = self.app.config.folders
        path = root / rel
        markdown = Path(rel).suffix.lower() in MARKDOWN

        def read() -> tuple[notes_mod.Note, list[notes_mod.Piece]]:
            note = notes_mod.read_note(real_file(root, rel), rel)  # checked again: it may have changed since the walk
            return note, notes_mod.pieces(note, markdown=markdown, size=cfg.chunk_chars)

        db = self.app.store.db
        try:
            note, pieces = await asyncio.to_thread(read)
        except Exception as exc:  # unreadable: remember why, keep nothing of it
            reason = str(exc)[:200] or exc.__class__.__name__
            log.debug("could not read %s: %s", path, reason)
            if old:
                await self._forget_chunks_of([old["id"]], commit=False)
            await db.execute(
                "INSERT INTO folder_files(folder_id, rel, size, mtime_ns, digest, title, error, indexed_at)"
                " VALUES(?,?,?,?,NULL,?,?,?) ON CONFLICT(folder_id, rel) DO UPDATE SET size = excluded.size,"
                " mtime_ns = excluded.mtime_ns, digest = NULL, error = excluded.error, indexed_at = excluded.indexed_at",
                (fid, rel, size, mtime, Path(rel).stem, reason, now_iso()),
            )
            return False
        dg = notes_mod.digest(json.dumps([note.title, note.tags, note.clipped, note.text], ensure_ascii=False))
        if old and old.get("digest") == dg:
            await db.execute("UPDATE folder_files SET size = ?, mtime_ns = ? WHERE id = ?", (size, mtime, old["id"]))
            return False
        if old:
            await self._forget_chunks_of([old["id"]], commit=False)
        rows = await db.execute_fetchall(  # one step, so no statement stays open while other work commits
            "INSERT INTO folder_files(folder_id, rel, size, mtime_ns, digest, title, tags, clipped, error, indexed_at)"
            " VALUES(?,?,?,?,?,?,?,?,NULL,?) ON CONFLICT(folder_id, rel) DO UPDATE SET size = excluded.size,"
            " mtime_ns = excluded.mtime_ns, digest = excluded.digest, title = excluded.title, tags = excluded.tags,"
            " clipped = excluded.clipped, error = NULL, indexed_at = excluded.indexed_at RETURNING id",
            (fid, rel, size, mtime, dg, note.title, json.dumps(note.tags, ensure_ascii=False), note.clipped or None,
             now_iso()),
        )
        file_id = next(iter(rows))[0]
        if pieces:  # few round trips per file; the scan commits every few files
            await db.executemany(
                "INSERT INTO folder_chunks(file_id, folder_id, ord, heading, text) VALUES(?,?,?,?,?)",
                [(file_id, fid, i, p.heading, p.text) for i, p in enumerate(pieces)],
            )
            ids = await db.execute_fetchall("SELECT id FROM folder_chunks WHERE file_id = ? ORDER BY ord", (file_id,))
            title = " ".join([note.title, *note.aliases, *note.tags])
            await db.executemany(
                "INSERT INTO folder_fts(rowid, title, heading, text) VALUES(?,?,?,?)",
                [(r[0], title, p.heading, p.text) for r, p in zip(ids, pieces, strict=True)],
            )
        return True

    # ------------------------------------------------------------------ forgetting
    async def _forget_where(self, where: str, params: tuple) -> None:
        """Forget every file (and its pieces) of ``folder_files`` matching ``where``."""
        ids = [r["id"] for r in await self.app.store.fetchall(f"SELECT id FROM folder_files WHERE {where}", params)]
        await self._forget_files(ids)

    async def _forget_files(self, file_ids: list[int]) -> None:
        if not file_ids:
            return
        await self._forget_chunks_of(file_ids)
        db = self.app.store.db
        for i in range(0, len(file_ids), 500):
            part = file_ids[i: i + 500]
            await db.execute(f"DELETE FROM folder_files WHERE id IN ({','.join('?' * len(part))})", part)
        await db.commit()

    async def _forget_chunks_of(self, file_ids: list[int], *, commit: bool = True) -> None:
        db = self.app.store.db
        vec = self.vec.ready or await self.vec.exists()
        for i in range(0, len(file_ids), 500):
            part = file_ids[i: i + 500]
            marks = ",".join("?" * len(part))
            chunk_ids = [r[0] for r in await self.app.store.fetchall(
                f"SELECT id FROM folder_chunks WHERE file_id IN ({marks})", part
            )]
            for j in range(0, len(chunk_ids), 500):
                ids = chunk_ids[j: j + 500]
                cm = ",".join("?" * len(ids))
                await db.execute(f"DELETE FROM folder_fts WHERE rowid IN ({cm})", ids)
                if vec:
                    await db.execute(f"DELETE FROM folder_vec WHERE rowid IN ({cm})", ids)
            await db.execute(f"DELETE FROM folder_chunks WHERE file_id IN ({marks})", part)
        if commit:
            await db.commit()

    # ------------------------------------------------------------------ embeddings
    def _embed_model(self) -> str:
        try:
            return str(self.app.llm.model_for("embedding"))
        except Exception:
            return ""

    async def _check_embedding_model(self) -> None:
        """A different embedding model makes the stored vectors meaningless: forget them so they are made again."""
        model = self._embed_model()
        stored = await self.app.store.get_meta("folder_vec_model")
        if stored is not None and stored != model:
            log.info("embedding model changed (%s -> %s): folder pieces are embedded again", stored, model)
            await self.app.store.execute("DROP TABLE IF EXISTS folder_vec")
            await self.app.store.execute("UPDATE folder_chunks SET embedded = 0")
            self.vec.dim = None
            await self.app.store.execute("DELETE FROM meta WHERE key IN ('folder_vec_dim', 'folder_vec_metric')")
            self._embed_error = ""
        if stored != model:
            await self.app.store.set_meta("folder_vec_model", model)

    async def _ensure_vec(self, dim: int) -> None:
        async def rebuild() -> None:  # the table was dropped for a new dimension: embed everything again
            await self.app.store.execute("UPDATE folder_chunks SET embedded = 0")

        await self.vec.ensure(dim, rebuild)

    async def _vec_ready(self) -> bool:
        if self.vec.ready:
            return True
        dim = await self.app.store.get_meta("folder_vec_dim")
        if dim is None or not self.app.store.vec_available or not await self.vec.exists():
            return False
        with contextlib.suppress(Exception):
            await self.vec.ensure(int(dim))
        return self.vec.ready

    async def _embed_pending(self, fid: str) -> None:
        if not self.app.store.vec_available:
            return
        db = self.app.store.db
        while True:
            rows = await self.app.store.fetchall(
                "SELECT c.id, c.heading, c.text, f.title, f.tags FROM folder_chunks c JOIN folder_files f"
                " ON f.id = c.file_id WHERE c.folder_id = ? AND c.embedded = 0 ORDER BY c.id LIMIT ?",
                (fid, EMBED_BATCH),
            )
            if not rows:
                return
            texts = [
                notes_mod.embed_text(r["title"] or "", notes_mod.Piece(r["heading"], r["text"]),
                                     json.loads(r["tags"] or "[]")[:8])
                for r in rows
            ]
            try:
                vecs = await self.app.llm.embed(texts)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._embed_error = (
                    f"Searching by meaning is off until the embedding model works ({str(exc)[:160]}). "
                    "Searching by words still works."
                )
                log.warning("embedding folder pieces failed: %s", exc)
                return
            if not vecs or len(vecs) != len(rows):
                return
            self._embed_error = ""
            await self._ensure_vec(len(vecs[0]))
            ids = [int(r["id"]) for r in rows]
            marks = ",".join("?" * len(ids))
            await db.execute(f"DELETE FROM folder_vec WHERE rowid IN ({marks})", ids)
            await db.executemany(
                "INSERT INTO folder_vec(rowid, embedding) VALUES(?, ?)", [(i, pack(v)) for i, v in zip(ids, vecs, strict=True)]
            )
            await db.execute(f"UPDATE folder_chunks SET embedded = 1 WHERE id IN ({marks})", ids)
            await db.commit()

    # ------------------------------------------------------------------ search
    async def search(self, query: str, *, limit: int | None = None, folder: str = "") -> dict:
        """The pieces of the user's notes that best match ``query``, best first, at most ``PER_FILE`` per file."""
        return (await self._search(query, limit=limit, folder=folder))[0]

    async def _search(
        self, query: str, *, limit: int | None = None, folder: str = "", window: bool = True
    ) -> tuple[dict, bool]:
        """``(search result, whether search by meaning ran)``. Without ``window`` every piece close enough in meaning
        counts, not only those near the best one, and carries its similarity."""
        query = str(query or "").strip()
        limit = max(1, min(int(limit or self.app.config.folders.results), 20))
        rows = await self._rows()
        if folder:
            want = folder.strip().lower()
            rows = [r for r in rows if r["id"] == folder or r["name"].lower() == want] or rows
        if not rows:
            return {"results": [], "note": "No folders are added yet. The user can add them in Settings > Folders."}, False
        if not query:
            return {"error": "Say what to look for."}, False
        allowed = {r["id"]: r for r in rows}
        scores: dict[int, float] = {}
        similarity: dict[int, float] = {}
        keyword_rank: dict[int, int] = {}
        words = content_words(query)  # the same words prompt_pieces checks, each once
        if words:
            match = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
            try:
                fts = await self.app.store.fetchall(
                    "SELECT rowid, bm25(folder_fts, 4.0, 2.0, 1.0) AS b FROM folder_fts WHERE folder_fts MATCH ?"
                    " ORDER BY b LIMIT ?", (match, CANDIDATES),
                )
            except Exception as exc:
                log.debug("keyword search failed: %s", exc)
                fts = []
            best = float(fts[0][1]) if fts else 0.0
            for rank, r in enumerate(fts):
                # rank fusion, scaled by how good the keyword match is next to the best one (bm25 is negative)
                strength = min(1.0, max(0.0, float(r[1]) / best)) if best < 0 else 1.0
                scores[int(r[0])] = scores.get(int(r[0]), 0.0) + (0.5 + 0.5 * strength) / (RRF_K + rank)
                keyword_rank.setdefault(int(r[0]), rank)
        by_meaning = False
        if await self._vec_ready():
            try:
                qv = (await self.app.llm.embed([query]))[0]
                hits = await self.vec.knn(qv, CANDIDATES)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.info("search by meaning failed, using words only: %s", exc)
                hits = []
            by_meaning = bool(hits)
            # only pieces close to the best match: the rest is noise (and could bring a clipped page in for nothing)
            floor = max(MIN_SIMILARITY, (hits[0][1] if hits else 0.0) - (SIMILARITY_WINDOW if window else 1.0))
            for rank, (cid, sim) in enumerate(h for h in hits if h[1] >= floor):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                similarity[cid] = sim
        if not scores:
            return {"results": [], "note": "Nothing in the user's folders matches that."}, by_meaning
        # best fused score first; on a tie the better keyword match wins (names and exact words are precise)
        ranked = sorted(scores, key=lambda c: (-round(scores[c], 9), keyword_rank.get(c, CANDIDATES)))
        marks = ",".join("?" * len(ranked))
        info = {
            int(r["id"]): dict(r) for r in await self.app.store.fetchall(
                "SELECT c.id, c.heading, c.text, c.folder_id, f.rel, f.title, f.tags, f.clipped FROM folder_chunks c"
                f" JOIN folder_files f ON f.id = c.file_id WHERE c.id IN ({marks})", ranked,
            )
        }
        out: list[dict] = []
        per_file: dict[tuple[str, str], int] = {}
        for cid in ranked:
            c = info.get(cid)
            if c is None or c["folder_id"] not in allowed:
                continue
            key = (c["folder_id"], c["rel"])
            if per_file.get(key, 0) >= PER_FILE:
                continue
            per_file[key] = per_file.get(key, 0) + 1
            out.append(self._result(allowed[c["folder_id"]], c, similarity.get(cid)))
            if len(out) >= limit:
                break
        result: dict = {}
        clipped = [r for r in out if r.get("clipped_from")]
        if clipped and self.app.config.folders.clippings_are_outside:
            result["outside_content"] = CLIPPED_SOURCE  # first, so it survives a cut (ADR 0025)
        result["results"] = out
        return result, by_meaning

    async def prompt_pieces(self, query: str) -> list[dict]:
        """Up to ``folders.prompt_notes`` short pieces of the user's notes that clearly match ``query``, one per note,
        best first, to go with a chat message (#354): small models rarely search notes on their own. A piece counts
        when it is close in meaning (``folders.prompt_min_similarity``) and holds at least half of the message's
        words, or is very close in meaning (``STRONG_MARGIN`` more); without search by meaning, when it holds most of
        the message's words. Clipped web pages never go in (they are outside content, ADR 0025: ``folder_search``
        still finds them). Each is ``{title, folder, path, file, link?, heading?, text}`` with ``text`` cut to
        ``folders.prompt_note_chars`` around the words that matched."""
        cfg = self.app.config.folders
        query = str(query or "").strip()
        if cfg.prompt_notes <= 0 or len(query) < 3 or not await self._rows():
            return []
        # every close piece with its similarity: a clipped page that matches best must not hide the user's own note
        # the most results a search gives, so clipped pages and pieces that don't qualify can't crowd out one that does
        found, by_meaning = await self._search(query, limit=20, window=False)
        words = content_words(query)
        out: list[dict] = []
        files: set[str] = set()
        for r in found.get("results") or []:
            if r.get("clipped_from") or r["file"] in files:
                continue
            hay = set(_WORDS.findall(" ".join((r.get("title") or "", r.get("heading") or "", r.get("text") or "")).lower()))
            hits = [w for w in words if any(h.startswith(w) for h in hay)]  # "renew" in "renews", not "rent" in "current"
            most = bool(hits) and len(hits) * 2 >= len(words)  # at least half of the message's words
            sim = r.get("similarity")
            if by_meaning:
                ok = sim is not None and (
                    (sim >= cfg.prompt_min_similarity and most) or sim >= cfg.prompt_min_similarity + STRONG_MARGIN
                )
            else:  # words only: most of the message's words, and at least two of them when it has two
                ok = most and len(hits) >= min(2, len(words))
            if not ok:
                continue
            files.add(r["file"])
            piece = {k: r[k] for k in ("title", "folder", "path", "file", "link", "heading") if r.get(k)}
            piece["text"] = excerpt(r.get("text") or "", hits, cfg.prompt_note_chars)
            out.append(piece)
            if len(out) >= cfg.prompt_notes:
                break
        return out

    def _result(self, folder: dict, c: dict, sim: float | None) -> dict:
        abs_path = str(Path(folder["path"]) / c["rel"])
        item: dict = {
            "title": c["title"] or Path(c["rel"]).stem,
            "folder": folder["name"],
            "path": c["rel"],
            "heading": c["heading"] or None,
            "text": c["text"],
            "tags": json.loads(c["tags"] or "[]")[:10] or None,
            "file": abs_path,
        }
        if folder.get("kind") == "obsidian":
            item["link"] = notes_mod.obsidian_url(folder["name"], c["rel"])
        if c.get("clipped"):
            item["clipped_from"] = c["clipped"]
        if sim is not None:
            item["similarity"] = round(sim, 3)
        return {k: v for k, v in item.items() if v is not None}

    # ------------------------------------------------------------------ reading one file
    async def resolve(self, path: str) -> tuple[dict, dict] | None:
        """``(folder, file row)`` of an indexed file named by its full path, its path inside a folder, or
        "<folder name>/<path inside>". Only files Sentient indexed can be read (never hidden or ignored ones)."""
        raw = str(path or "").strip().strip('"').replace("\\", "/")
        if not raw:
            return None
        rows = await self._rows()
        candidates: list[tuple[dict, str]] = []
        for f in rows:
            if os.path.isabs(raw.replace("/", os.sep)) or raw.startswith("/"):
                if _inside(raw, f["path"]):
                    candidates.append((f, os.path.relpath(_norm(raw), _norm(f["path"])).replace("\\", "/")))
                continue
            rel = raw.lstrip("./")
            candidates.append((f, rel))
            if rel.lower().startswith(f["name"].lower() + "/"):
                candidates.append((f, rel[len(f["name"]) + 1:]))
        for f, rel in candidates:
            for name in (rel, rel + ".md") if not supported(rel) else (rel,):
                row = await self.app.store.fetchone(
                    "SELECT * FROM folder_files WHERE folder_id = ? AND rel = ? COLLATE NOCASE", (f["id"], name)
                )
                if row is not None and not self._ignore(f, Path(f["path"])).skip_path(row["rel"]):
                    return f, dict(row)  # a file the current rules leave out can't be read, even before a rescan
        return None

    async def read(self, path: str) -> tuple[dict, dict, str] | None:
        """``(folder, file row, text)`` of an indexed file, read now from disk. None when it isn't one."""
        found = await self.resolve(path)
        if found is None:
            return None
        folder, row = found
        root, rel = Path(folder["path"]), row["rel"]
        from sentient.files.extract import extract_text

        def load() -> tuple[str, str]:
            path = real_file(root, rel)  # never through a link that leads out of the folder
            text = extract_text(path, notes_mod.MAX_TEXT_CHARS)
            try:  # whether it is a clipped web page now, not at the last scan
                clipped = notes_mod.read_note(path, rel).clipped
            except (OSError, ValueError):
                clipped = row.get("clipped") or ""
            return text, clipped

        text, clipped = await asyncio.to_thread(load)
        return folder, {**row, "clipped": clipped or None}, text

    # ------------------------------------------------------------------ watching
    async def _watch_config(self) -> None:
        async with self.app.bus.subscribe() as queue:
            while True:
                event = await queue.get()
                if event.get("type") == "config.updated" and "folders" in ((event.get("data") or {}).get("sections")
                                                                           or ["folders"]):
                    self._watch_changed.set()

    async def _watch(self) -> None:
        try:
            from watchfiles import awatch
        except ImportError:  # periodic rescans still keep the index up to date
            log.info("watchfiles is not installed: folders are checked every few minutes instead")
            return
        while True:
            self._watch_changed.clear()
            rows = [r for r in await self._rows() if Path(r["path"]).is_dir()]
            if not rows or not self.app.config.folders.watch:
                await self._watch_changed.wait()
                continue
            roots = {_norm(r["path"]): (r, self._ignore(r, Path(r["path"]))) for r in rows}
            stop = asyncio.Event()

            async def restart(stop: asyncio.Event = stop) -> None:
                await self._watch_changed.wait()
                stop.set()

            waiter = asyncio.create_task(restart())

            def keep(_change: Any, path: str, roots: dict = roots) -> bool:
                return self._owner(path, roots) is not None

            try:
                async for changes in awatch(
                    *[r["path"] for r in rows], watch_filter=keep, debounce=1500, stop_event=stop,
                    recursive=True, ignore_permission_denied=True,
                ):
                    hit = {self._owner(p, roots) for _, p in changes} - {None}
                    for fid in hit:
                        if self.enqueue(fid):  # type: ignore[arg-type]
                            log.info("a folder changed on disk: %s", fid)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("watching folders stopped (%s); checking every few minutes instead", exc)
                await asyncio.sleep(60)
            finally:
                waiter.cancel()

    @staticmethod
    def _owner(path: str, roots: dict[str, tuple[dict, Ignore]]) -> str | None:
        """The folder id ``path`` belongs to, when it could be a file worth reading (not hidden, not ignored)."""
        p = _norm(path)
        for root, (row, ignore) in roots.items():
            if p == root or p.startswith(root.rstrip("\\/") + os.sep):
                rel = p[len(root):].lstrip("\\/").replace("\\", "/")
                if not rel:
                    return None
                name = rel.rsplit("/", 1)[-1]
                if "." in name and not supported(name):
                    return None  # another kind of file (a folder has no dot, usually: a deleted one counts)
                if ignore.skip_path(rel):
                    return None
                return row["id"]
        return None


def real_file(root: Path, rel: str) -> Path:
    """The file ``rel`` inside ``root``, checked now: a regular file that is not a link and really lies inside the
    folder. Raises ``ValueError`` otherwise, so nothing outside the folder is ever read."""
    path = root / rel
    if path.is_symlink() or not path.is_file():
        raise ValueError("not a regular file in the folder")
    real, base = os.path.realpath(path), os.path.realpath(root)
    if not _inside(real, base) or _norm(real) == _norm(base):
        raise ValueError("this file leads outside the folder")
    return path


def _clean_patterns(patterns: list[str] | None) -> list[str]:
    out = []
    for p in patterns or []:
        s = str(p or "").strip()
        if s and s not in out:
            out.append(s[:200])
    return out[:100]
