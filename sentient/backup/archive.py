"""Backup archives of the Sentient home folder (issue #151, docs/API.md section 20).

A backup is one zip file:

- ``manifest.json``: what is inside, when and by which version, counts for the confirmation, and the sign-ins
  that live in the keychain (so the window can say which ones to redo).
- ``sentient.db``: a consistent copy made with SQLite's online backup API, never a raw copy of a live WAL file.
- ``config.yaml``, ``workspace/``, ``skills/``, ``files/`` and ``tls/`` (the devices certificate).

Never inside: keychain secrets (API keys, tokens, sign-ins), ``gateway.token``, logs, the browser profile,
WhatsApp's link, downloaded voice models, scratch folders, the backups folder itself, and open incognito chats.

Backups go to ``Documents/Sentient Backups`` by default (:func:`default_folder`), outside the data folder; any
data folder other than ``~/.sentient`` keeps them in its own ``backups`` folder instead.

A snapshot is the same format without ``files/`` (nothing that takes one changes files). Sentient takes one before
it updates its database and before an import; the last N are kept.

A restore is staged into ``<home>/backups/restore-pending`` and applied by :func:`apply_pending_restore` the next
time the engine starts, before anything opens the database. What it replaces is saved as a snapshot first, so a
restore can be undone. Only the items the backup's manifest lists are replaced; a snapshot leaves ``files/`` alone.

Everything here is synchronous file work; the service runs it in a thread.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from sentient import __version__, paths, secrets

log = logging.getLogger(__name__)

FORMAT = 1
MANIFEST = "manifest.json"
DB_NAME = "sentient.db"
CONFIG_NAME = "config.yaml"
DIRS = ("workspace", "skills", "files", "tls")
ITEMS = frozenset({DB_NAME, CONFIG_NAME, *DIRS})
DB_SIDECARS = ("-wal", "-shm", "-journal")
BACKUP_PREFIX = "sentient-backup-"
SNAPSHOT_PREFIX = "sentient-snapshot-"
PENDING = "restore-pending"
LAST_RESTORE = "last-restore.json"
KINDS = ("manual", "scheduled", "snapshot")


class BackupError(Exception):
    """A plain sentence the window can show as is."""


# ----------------------------------------------------------------------------- where things go
def backups_root(home: Path | None = None) -> Path:
    return (home or paths.home()) / "backups"


BACKUP_FOLDER_NAME = "Sentient Backups"


_ONEDRIVE_ENV = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")


def cloud_service(path: Path) -> str | None:
    """The sync service a folder is inside (OneDrive, Dropbox, Google Drive or iCloud), from its path; else None.

    A backup there is copied to the cloud too, so Settings says so."""
    for var in _ONEDRIVE_ENV:
        root = os.environ.get(var, "").strip()
        if root and is_inside(path, Path(root)):
            return "OneDrive"
    for seg in (s.lower() for s in Path(path).parts):
        if seg.startswith("onedrive"):  # "OneDrive", "OneDrive - Contoso", macOS CloudStorage "OneDrive-Personal"
            return "OneDrive"
        if seg.startswith("dropbox"):
            return "Dropbox"
        if seg in {"google drive", "googledrive", "my drive"} or seg.startswith("googledrive-"):
            return "Google Drive"
        if seg in {"icloud drive", "iclouddrive", "mobile documents", "com~apple~clouddocs"}:
            return "iCloud"
    return None


def default_folder(home: Path | None = None) -> Path:
    """Where backups go unless the user picked a folder: ``Documents/Sentient Backups``, outside the data folder so a
    backup survives losing it. ``<home>/backups`` when Documents is missing or can't be written to, and always for a
    data folder other than the default ``~/.sentient`` (a test or demo home, another profile), so its backups never
    land next to the real ones."""
    if not secrets.is_default_home(home):
        return backups_root(home)
    docs = paths.documents_dir()
    target = docs / BACKUP_FOLDER_NAME
    try:
        if target.is_dir():
            return target if os.access(target, os.W_OK) else backups_root(home)
        if docs.is_dir() and os.access(docs, os.W_OK):
            return target
    except OSError:
        pass
    return backups_root(home)


def snapshots_dir(home: Path | None = None) -> Path:
    return backups_root(home) / "snapshots"


def pending_dir(home: Path | None = None) -> Path:
    return backups_root(home) / PENDING


def _stamp() -> str:
    return datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S")  # local time, for people


def _unique(folder: Path, prefix: str, suffix: str = "") -> Path:
    base = f"{prefix}{_stamp()}{suffix}"
    path = folder / f"{base}.zip"
    n = 2
    while path.exists():
        path = folder / f"{base}-{n}.zip"
        n += 1
    return path


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


# ----------------------------------------------------------------------------- database
def copy_database(src: Path, dst: Path) -> None:
    """Copy a database that may be open and in WAL mode, with SQLite's online backup API.

    The copy is switched to a plain rollback journal so it is one self-contained file (the engine turns WAL back on
    when it opens it)."""
    if not src.exists():
        raise BackupError("There is no database to back up yet.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", *DB_SIDECARS):
        Path(f"{dst}{suffix}").unlink(missing_ok=True)
    source = sqlite3.connect(src, timeout=30)
    try:
        target = sqlite3.connect(dst)
        try:
            source.backup(target)
            target.execute("PRAGMA journal_mode = DELETE")
            target.commit()
        finally:
            target.close()
    finally:
        source.close()


def check_database(db: Path) -> None:
    """Raise BackupError unless SQLite's quick check passes."""
    try:
        con = sqlite3.connect(db)
        try:
            row = con.execute("PRAGMA quick_check").fetchone()
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise BackupError("The database in this backup can't be opened, so nothing was restored.") from exc
    if not row or row[0] != "ok":
        raise BackupError("The database in this backup is damaged, so nothing was restored.")


def database_version(db: Path) -> int:
    con = sqlite3.connect(db)
    try:
        return int(con.execute("PRAGMA user_version").fetchone()[0])
    finally:
        con.close()


def _drop_incognito(db: Path) -> None:
    """Incognito chats (#154) are never kept: remove them, and their messages, from the copy with secure delete on."""
    con = sqlite3.connect(db)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(sessions)")}
        if "incognito" not in cols:
            return
        con.execute("PRAGMA secure_delete = ON")
        con.execute("PRAGMA foreign_keys = ON")
        gone = "SELECT id FROM sessions WHERE incognito = 1"
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "subagents" in tables:
            con.execute(f"DELETE FROM subagents WHERE session_id IN ({gone})")
        con.execute(f"DELETE FROM messages WHERE session_id IN ({gone})")
        con.execute("DELETE FROM sessions WHERE incognito = 1")
        con.commit()
    finally:
        con.close()


_COUNTS = {
    "chats": "SELECT COUNT(*) FROM sessions",
    "memories": "SELECT COUNT(*) FROM facts WHERE status = 'active'",
    "tasks": "SELECT COUNT(*) FROM tasks",
}


def _counts(db: Path, home: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    con = sqlite3.connect(db)
    try:
        for key, sql in _COUNTS.items():
            with contextlib.suppress(sqlite3.Error):
                out[key] = int(con.execute(sql).fetchone()[0])
    finally:
        con.close()
    skills = home / "skills"
    if skills.is_dir():
        out["skills"] = sum(1 for _ in skills.rglob("SKILL.md"))
    return out


# ----------------------------------------------------------------------------- writing
def create_archive(
    dest: Path,
    *,
    kind: str,
    reason: str | None = None,
    home: Path | None = None,
    db_path: Path | None = None,
    include_files: bool = True,
    sign_ins: list[dict] | None = None,
) -> dict:
    """Write one backup zip to ``dest`` and return its entry (see :func:`entry`)."""
    home = home or paths.home()
    db_path = db_path or (home / DB_NAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    skip_roots = [backups_root(home), dest.parent]
    dirs = [d for d in DIRS if include_files or d != "files"]
    part = dest.with_name(dest.name + ".part")
    skipped: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="sentient-backup-") as work:
            contents: list[str] = []
            db_copy = Path(work) / DB_NAME
            counts: dict[str, int] = {}
            schema = None
            if db_path.exists():
                copy_database(db_path, db_copy)
                _drop_incognito(db_copy)
                contents.append(DB_NAME)
                counts = _counts(db_copy, home)
                schema = database_version(db_copy)
            config = home / CONFIG_NAME
            if config.is_file():
                contents.append(CONFIG_NAME)
            contents += dirs
            manifest = {
                "format": FORMAT,
                "kind": kind,
                "reason": reason,
                "created_at": datetime.now(UTC).isoformat(),
                "version": __version__,
                "schema_version": schema,
                "contents": contents,
                "counts": counts,
                "sign_ins": sign_ins or [],
            }
            with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                if DB_NAME in contents:
                    zf.write(db_copy, DB_NAME)
                if CONFIG_NAME in contents:
                    zf.write(config, CONFIG_NAME)
                for d in dirs:
                    root = home / d
                    if not root.is_dir():
                        continue
                    for p in sorted(root.rglob("*")):
                        if p.is_symlink() or not p.is_file() or any(is_inside(p, s) for s in skip_roots):
                            continue
                        rel = p.relative_to(home).as_posix()
                        try:
                            zf.write(p, rel)
                        except OSError as exc:  # a file another program holds open
                            log.warning("backup could not read %s: %s", rel, exc)
                            skipped.append(rel)
                if skipped:
                    # a backup is complete or not made: a restore replaces whole folders, so a missing file would
                    # be lost from the restored data
                    raise BackupError(_unreadable(skipped))
                zf.writestr(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False))
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    return entry(dest, manifest)


def _unreadable(names: list[str]) -> str:
    shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
    return (
        f"The backup wasn't made, because Sentient couldn't read {shown}. It may be open in another program: close it "
        "and try again."
    )


def create_snapshot(reason: str, *, keep: int, home: Path | None = None, db_path: Path | None = None) -> dict:
    """A backup without files/ in ``<home>/backups/snapshots``; only the newest ``keep`` are kept."""
    folder = snapshots_dir(home)
    made = create_archive(
        _unique(folder, SNAPSHOT_PREFIX), kind="snapshot", reason=reason, home=home, db_path=db_path,
        include_files=False,
    )
    prune(folder, keep, kind="snapshot")
    return made


def new_backup_path(folder: Path, kind: str = "manual") -> Path:
    return _unique(folder, BACKUP_PREFIX, "-auto" if kind == "scheduled" else "")


# ----------------------------------------------------------------------------- reading
def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise BackupError("That backup file isn't there.")
    try:
        with zipfile.ZipFile(path) as zf:
            raw = zf.read(MANIFEST)
    except KeyError as exc:
        raise BackupError("That file isn't a Sentient backup.") from exc
    except (zipfile.BadZipFile, OSError) as exc:
        raise BackupError("That file isn't a Sentient backup, or it is damaged.") from exc
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise BackupError("That backup is damaged.") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("contents"), list):
        raise BackupError("That file isn't a Sentient backup.")
    if manifest.get("format") != FORMAT:
        raise BackupError("That backup was made by a different version of Sentient that this one can't read.")
    return manifest


def entry(path: Path, manifest: dict) -> dict:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {
        "name": path.name,
        "path": str(path),
        "kind": manifest.get("kind") if manifest.get("kind") in KINDS else "manual",
        "reason": manifest.get("reason"),
        "created_at": manifest.get("created_at"),
        "version": manifest.get("version"),
        "schema_version": manifest.get("schema_version"),
        "size": size,
        "includes_files": "files" in manifest.get("contents", []),
        "counts": manifest.get("counts") or {},
        "sign_ins": manifest.get("sign_ins") or [],
    }


def list_archives(folder: Path) -> list[dict]:
    """Backups and snapshots in ``folder``, newest first. Files that aren't Sentient backups are left out."""
    out = []
    if not folder.is_dir():
        return out
    for p in folder.glob("sentient-*.zip"):
        if not p.name.startswith((BACKUP_PREFIX, SNAPSHOT_PREFIX)):
            continue
        try:
            out.append(entry(p, read_manifest(p)))
        except BackupError:
            continue
    out.sort(key=lambda e: e.get("created_at") or "", reverse=True)
    return out


def prune(folder: Path, keep: int, *, kind: str) -> list[str]:
    """Delete the oldest archives of one kind beyond ``keep``. Returns the names deleted."""
    gone = []
    for e in [e for e in list_archives(folder) if e["kind"] == kind][max(keep, 1):]:
        with contextlib.suppress(OSError):
            Path(e["path"]).unlink()
            gone.append(e["name"])
    return gone


# ----------------------------------------------------------------------------- restoring
def stage_restore(archive: Path, home: Path | None = None) -> dict:
    """Unpack and check a backup into the pending folder. Nothing in use changes until the engine restarts."""
    home = home or paths.home()
    manifest = read_manifest(archive)
    contents = [c for c in manifest["contents"] if c in ITEMS]
    if DB_NAME not in contents:
        raise BackupError("That backup has no database in it, so there is nothing to restore.")
    pending = pending_dir(home)
    work = pending.with_name(PENDING + ".part")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        _unpack(archive, work, contents)
        check_database(work / DB_NAME)
        staged = {**manifest, "contents": contents, "source": str(archive), "staged_at": datetime.now(UTC).isoformat()}
        (work / MANIFEST).write_text(json.dumps(staged, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.rmtree(pending, ignore_errors=True)
        os.replace(work, pending)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return staged


_ODD_PLACE = "This backup has a file in an unexpected place, so nothing was restored."


def _unpack(archive: Path, work: Path, contents: list[str]) -> None:
    """Extract only the listed items, refusing any name that could land outside ``work``."""
    try:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                name = info.filename
                if name == MANIFEST:
                    continue
                parts = PurePosixPath(name).parts
                if (
                    not parts or "\\" in name or name.startswith("/") or ".." in parts
                    or parts[0] not in contents or (len(parts) > 1 and parts[0] in (DB_NAME, CONFIG_NAME))
                ):
                    raise BackupError(_ODD_PLACE)
                target = work.joinpath(*parts)
                if not is_inside(target, work):
                    raise BackupError(_ODD_PLACE)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile as exc:
        raise BackupError("That backup is damaged, so nothing was restored.") from exc
    if not (work / DB_NAME).is_file():
        raise BackupError("That backup has no database in it, so there is nothing to restore.")


def pending_restore(home: Path | None = None) -> dict | None:
    f = pending_dir(home) / MANIFEST
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def cancel_restore(home: Path | None = None) -> bool:
    pending = pending_dir(home)
    if not pending.exists():
        return False
    shutil.rmtree(pending, ignore_errors=True)
    return True


def _item_names(item: str) -> list[str]:
    return [DB_NAME, *(DB_NAME + s for s in DB_SIDECARS)] if item == DB_NAME else [item]


def _move(src: Path, dst: Path, tries: int = 6) -> None:
    """os.replace, retried briefly: on Windows a just-stopped engine or a virus scanner can hold a file a moment."""
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(0.5)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _check_staged(contents: list[str], pending: Path) -> None:
    """A staged restore damaged after it was checked (a partial write) is refused, never half applied."""
    if DB_NAME not in contents or not (pending / DB_NAME).is_file():
        raise BackupError(
            "The restore didn't happen, because the prepared backup was damaged. Your data was left as it was. Try "
            "restoring the backup again."
        )


def apply_pending_restore(home: Path | None = None, *, snapshot_keep: int = 5) -> dict | None:
    """Swap a staged restore into place. Call before anything opens the database or reads config.yaml.

    Returns the result written to ``backups/last-restore.json`` (also on failure, with ``error``), or None when
    nothing was staged."""
    home = home or paths.home()
    pending = pending_dir(home)
    staged = pending_restore(home)
    if staged is None:
        if pending.exists():  # half-written: never apply it
            shutil.rmtree(pending, ignore_errors=True)
        return None
    contents = [c for c in staged.get("contents", []) if c in ITEMS]
    result: dict[str, Any] = {
        "restored_at": datetime.now(UTC).isoformat(),
        "source": staged.get("source"),
        "kind": staged.get("kind"),
        "backup_created_at": staged.get("created_at"),
        "contents": contents,
        "sign_ins": staged.get("sign_ins") or [],
        "snapshot": None,
        "error": None,
    }
    keep_pending = False
    try:
        _check_staged(contents, pending)
        if (home / DB_NAME).exists():
            try:
                result["snapshot"] = create_snapshot("before restoring a backup", keep=snapshot_keep, home=home)["name"]
            except Exception as exc:
                log.exception("could not take a snapshot before restoring")
                keep_pending = True  # nothing changed: it is tried again at the next start, or cancelled in Settings
                raise BackupError(
                    "The restore didn't happen, because Sentient couldn't save a copy of your current data first. "
                    "It tries again the next time Sentient starts, or you can cancel it in Settings > Backups."
                ) from exc
        # a fresh folder each time: one left by a failed rollback may hold the only copy of something
        old = backups_root(home) / f"restore-old-{_stamp()}"
        n = 2
        while old.exists():
            old = backups_root(home) / f"restore-old-{_stamp()}-{n}"
            n += 1
        try:
            old.mkdir(parents=True)
        except OSError as exc:
            raise BackupError(
                "The restore didn't happen, because Sentient couldn't make room for it in its data folder. Your data "
                "was left as it was."
            ) from exc
        moved: list[str] = []
        placed: list[str] = []
        try:
            for item in contents:
                for name in _item_names(item):
                    cur = home / name
                    if cur.exists():
                        _move(cur, old / name)
                        moved.append(name)
            for item in contents:
                src = pending / item
                if src.exists():
                    _move(src, home / item)
                    placed.append(item)
        except OSError as exc:
            log.exception("restore failed, putting the current data back")
            for item in placed:
                with contextlib.suppress(OSError):
                    _remove(home / item)
            lost = []
            for name in moved:
                try:
                    _move(old / name, home / name)
                except OSError:
                    lost.append(name)
            if lost:
                raise BackupError(
                    "The restore didn't finish and some of your data couldn't be put back in place. Nothing is lost: "
                    f"it is in {old}. Close any program using Sentient's folder and move it back."
                ) from exc
            shutil.rmtree(old, ignore_errors=True)
            raise BackupError(
                "The restore didn't finish, so your data was left as it was. Close any program using Sentient's "
                "folder and try again."
            ) from exc
        shutil.rmtree(old, ignore_errors=True)
        log.warning("restored a backup made %s from %s", staged.get("created_at"), staged.get("source"))
    except BackupError as exc:
        result["error"] = str(exc)
    finally:
        if not keep_pending:
            shutil.rmtree(pending, ignore_errors=True)
    out = backups_root(home) / LAST_RESTORE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def last_restore(home: Path | None = None) -> dict | None:
    f = backups_root(home) / LAST_RESTORE
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def dismiss_last_restore(home: Path | None = None) -> None:
    (backups_root(home) / LAST_RESTORE).unlink(missing_ok=True)
