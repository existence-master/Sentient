"""Backups, restore and snapshots of Sentient's data folder (issue #151, docs/API.md section 20)."""

from sentient.backup.archive import BackupError
from sentient.backup.service import BackupService, apply_pending_restore

__all__ = ["BackupError", "BackupService", "apply_pending_restore"]
