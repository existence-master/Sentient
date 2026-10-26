"""Back up now, restore, snapshots and scheduled backups (issue #151, docs/API.md section 20).

The file work is in ``archive.py`` and runs in a thread. This service adds what needs the running app: the folder
and schedule from config, the sign-ins to list (they live in the keychain, never in a backup), events for the window,
and one backup or restore at a time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sentient import paths, secrets
from sentient.backup import archive
from sentient.backup.archive import BackupError
from sentient.services import Service

log = logging.getLogger(__name__)

SCHEDULE_META = "backup.last_scheduled"
FAILURE_META = "backup.failure_notified"
SCHEDULE_EVERY = {"daily": timedelta(days=1), "weekly": timedelta(days=7)}
CHECK_EVERY_S = 3600
BROWSER_SIGN_IN = {"kind": "browser", "id": "browser", "label": "Websites you signed in to in Sentient's browser"}


def _older_than(iso: str, now: datetime, age: timedelta) -> bool:
    try:
        return now - datetime.fromisoformat(iso) >= age
    except (ValueError, TypeError):
        return True


class BackupService(Service):
    name = "backups"

    def __init__(self, app):
        super().__init__(app)
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self.app.enable_background:
            self.run_every(CHECK_EVERY_S, self._scheduled, name="scheduled", initial_delay=300)

    # ------------------------------------------------------------------ settings
    def folder(self) -> Path:
        raw = (self.app.config.backups.folder or "").strip()
        return Path(raw).expanduser() if raw else archive.default_folder()

    def _publish(self, what: str, **extra: Any) -> None:
        self.app.bus.publish("backup.updated", {"what": what, **extra})

    # ------------------------------------------------------------------ sign-ins (never in a backup)
    async def sign_ins(self) -> list[dict]:
        """Everything signed in through the keychain right now: the list a restore elsewhere has to redo."""
        from sentient.gateway.routes.models import PROVIDERS

        out: list[dict] = []
        for p in PROVIDERS:
            if p.get("key_required") and secrets.get_secret(p["id"]):
                out.append({"kind": "model", "id": p["id"], "label": p["label"]})
        try:
            for i in await self.app.integrations.list_integrations():
                plugin = self.app.integrations.plugin(i["id"])
                if i.get("connected") and plugin is not None and not plugin.is_builtin:
                    out.append({"kind": "integration", "id": i["id"], "label": i["display_name"]})
        except Exception:
            log.exception("could not list integrations for the backup")
        try:
            for c in await self.app.channels.list():
                if await self.app.channels.is_connected(c["id"]):
                    out.append({"kind": "channel", "id": c["id"], "label": c["display_name"]})
        except Exception:
            log.exception("could not list messaging apps for the backup")
        browser = paths.home() / "browser"
        if browser.is_dir() and any(browser.iterdir()):
            out.append(dict(BROWSER_SIGN_IN))
        return out

    async def _signed_in(self, item: dict) -> bool | None:
        """Whether a listed sign-in works on this computer now; None when Sentient can't tell."""
        kind, ident = item.get("kind"), str(item.get("id") or "")
        try:
            if kind == "model":
                return bool(secrets.get_secret(ident))
            if kind == "integration":
                return await self.app.integrations.is_connected(ident)
            if kind == "channel":
                return await self.app.channels.is_connected(ident)
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------ overview
    async def overview(self) -> dict:
        folder = self.folder()
        backups = await asyncio.to_thread(archive.list_archives, folder)
        snapshots = await asyncio.to_thread(archive.list_archives, archive.snapshots_dir())
        cfg = self.app.config.backups
        return {
            "folder": str(folder),
            "default_folder": str(archive.default_folder()),
            "home": str(paths.home()),
            "folder_in_home": archive.is_inside(folder, paths.home()),
            "home_is_default": secrets.is_default_home(),
            "folder_cloud": archive.cloud_service(folder),
            "backups": [b for b in backups if b["kind"] != "snapshot"],
            "snapshots": [s for s in snapshots if s["kind"] == "snapshot"],
            "schedule": cfg.schedule,
            "keep": cfg.keep,
            "snapshots_keep": cfg.snapshots_keep,
            "last_scheduled_at": await self.app.store.get_meta(SCHEDULE_META),
            "pending_restore": self._pending(),
            "last_restore": await self.last_restore(),
        }

    def _pending(self) -> dict | None:
        staged = archive.pending_restore()
        if staged is None:
            return None
        return {"source": staged.get("source"), "created_at": staged.get("created_at"),
                "staged_at": staged.get("staged_at"), "kind": staged.get("kind")}

    async def last_restore(self) -> dict | None:
        result = archive.last_restore()
        if result is None:
            return None
        items = []
        for item in result.get("sign_ins") or []:
            if isinstance(item, dict):
                items.append({**item, "signed_in": await self._signed_in(item)})
        return {**result, "sign_ins": items}

    # ------------------------------------------------------------------ backing up
    async def backup_now(self, folder: str | None = None, *, kind: str = "manual") -> dict:
        target = Path(folder).expanduser() if folder and folder.strip() else self.folder()
        if not target.is_absolute():
            raise BackupError("Pick a full folder path for the backup.")
        sign_ins = await self.sign_ins()
        async with self._lock:
            try:
                made = await asyncio.to_thread(
                    archive.create_archive, archive.new_backup_path(target, kind), kind=kind,
                    db_path=self.app.store.path, sign_ins=sign_ins,
                )
            except PermissionError as exc:
                raise BackupError("Sentient can't write to that folder. Pick another one.") from exc
            except OSError as exc:
                raise BackupError(f"The backup couldn't be written: {exc.strerror or exc}.") from exc
            if kind == "scheduled":
                await asyncio.to_thread(archive.prune, target, self.app.config.backups.keep, kind="scheduled")
        self._publish("backup", name=made["name"])
        return made

    async def snapshot(self, reason: str, *, db_path: Path | None = None) -> dict | None:
        """Take a snapshot now (before a database update or an import). Never raises; None when it failed."""
        try:
            async with self._lock:
                made = await asyncio.to_thread(
                    archive.create_snapshot, reason, keep=self.app.config.backups.snapshots_keep,
                    db_path=db_path or self.app.store.path,
                )
        except Exception:
            log.exception("could not take a snapshot (%s)", reason)
            return None
        log.info("snapshot %s taken %s", made["name"], reason)
        self._publish("snapshot", name=made["name"])
        return made

    async def _scheduled(self) -> None:
        every = SCHEDULE_EVERY.get(self.app.config.backups.schedule)
        if every is None:
            return
        last = await self.app.store.get_meta(SCHEDULE_META)
        now = datetime.now(UTC)
        if last and not _older_than(last, now, every):
            return
        try:
            await self.backup_now(kind="scheduled")
        except BackupError as exc:
            # tried again next hour (an unplugged drive may be back); the user hears about it once per interval
            log.warning("scheduled backup failed: %s", exc)
            told = await self.app.store.get_meta(FAILURE_META)
            if not told or _older_than(told, now, every):
                await self.app.notify("backup", f"The automatic backup didn't work: {exc}", title="Backup")
                await self.app.store.set_meta(FAILURE_META, now.isoformat())
            return
        await self.app.store.set_meta(SCHEDULE_META, now.isoformat())
        await self.app.store.set_meta(FAILURE_META, "")

    # ------------------------------------------------------------------ restoring
    async def inspect(self, path: str) -> dict:
        """What a backup holds, for the confirmation before restoring."""
        p = Path(path).expanduser()
        manifest = await asyncio.to_thread(archive.read_manifest, p)
        from sentient.store.db import SCHEMA_VERSION

        made = archive.entry(p, manifest)
        newer = (manifest.get("schema_version") or 0) > SCHEMA_VERSION
        return {**made, "newer": newer, "contents": manifest.get("contents") or []}

    async def restore(self, path: str) -> dict:
        """Check and stage a backup. It replaces the current data when the engine restarts."""
        p = Path(path).expanduser()
        async with self._lock:
            staged = await asyncio.to_thread(archive.stage_restore, p)
        log.warning("restore staged from %s", p)
        self._publish("restore_staged", source=str(p))
        return {
            "staged": True,
            "restart_required": True,
            "source": str(p),
            "created_at": staged.get("created_at"),
            "contents": staged.get("contents"),
            "sign_ins": staged.get("sign_ins") or [],
        }

    async def cancel_restore(self) -> bool:
        async with self._lock:
            done = await asyncio.to_thread(archive.cancel_restore)
        if done:
            self._publish("restore_cancelled")
        return done

    def dismiss_last_restore(self) -> None:
        archive.dismiss_last_restore()
        self._publish("restore_dismissed")


def apply_pending_restore() -> dict | None:
    """Run at engine start, before the config and database are opened (gateway lifespan and ``sentient restore``)."""
    from sentient.config import load_config

    try:
        keep = load_config().backups.snapshots_keep
    except Exception:
        keep = 5
    result = archive.apply_pending_restore(snapshot_keep=keep)
    if result is not None:
        log.warning("restore result: %s", json.dumps({k: result[k] for k in ("source", "error", "snapshot")}))
    return result
