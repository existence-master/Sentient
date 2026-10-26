"""The Skills hub lockfile (#210): which version of each skill installed from the hub is in use.

``skills/.hub/lock.json`` holds one entry per installed skill, keyed by its name in Sentient::

    {"version": 1, "skills": {"internal-comms": {
        "source": "anthropic", "source_name": "Anthropic", "source_url": "https://github.com/anthropics/skills",
        "trust": "trusted", "path": "internal-comms", "repo_path": "skills/internal-comms",
        "commit": "dbd4588f...",           # the exact commit it was fetched at (null for an index source)
        "version": "1.0",                  # the skill's own version, when it declares one (or the index's)
        "fingerprint": "9c1f...",          # SHA-256 over its files' git blob hashes: what "same version" means
        "files": {"SKILL.md": "<git blob sha>", ...},
        "made_by": "Anthropic", "url": "https://github.com/.../tree/<commit>/skills/internal-comms",
        "installed_at": "...", "status": "pending" | "active",
        "active": {...the same fields for the approved version, or null...},
        "changes": {"added": [...], "removed": [...], "changed": [...]} | null}}}

The top-level fields describe the newest version fetched (the one waiting for review while ``status`` is
``pending``); ``active`` is the version the user approved. Approving copies the top level into ``active``; declining
an update puts ``active`` back on top, and declining a first install removes the entry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sentient.store.db import now_iso

LOCK_VERSION = 1
PINNED = ("commit", "version", "fingerprint", "files", "url", "made_by")


class HubLock:
    def __init__(self, path: Path):
        self.path = path

    def read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": LOCK_VERSION, "skills": {}}
        if not isinstance(data, dict) or not isinstance(data.get("skills"), dict):
            return {"version": LOCK_VERSION, "skills": {}}
        return data

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**data, "version": LOCK_VERSION}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def all(self) -> dict[str, dict]:
        return {k: v for k, v in self.read()["skills"].items() if isinstance(v, dict)}

    def get(self, name: str) -> dict | None:
        return self.all().get(name.strip().lower())

    def put(self, name: str, entry: dict) -> None:
        data = self.read()
        data["skills"][name.strip().lower()] = entry
        self._write(data)

    def remove(self, name: str) -> bool:
        data = self.read()
        found = data["skills"].pop(name.strip().lower(), None) is not None
        if found:
            self._write(data)
        return found

    def find(self, source: str, path: str) -> tuple[str, dict] | None:
        """The installed skill that came from ``path`` in ``source``."""
        for name, entry in self.all().items():
            if entry.get("source") == source and entry.get("path") == path:
                return name, entry
        return None

    # ------------------------------------------------------------------ review decisions
    def approved(self, name: str) -> dict | None:
        entry = self.get(name)
        if entry is None or entry.get("status") != "pending":
            return entry
        entry["active"] = {**{k: entry.get(k) for k in PINNED}, "approved_at": now_iso()}
        entry.update(status="active", changes=None)
        self.put(name, entry)
        return entry

    def declined(self, name: str) -> dict | None:
        """The pending version was declined: an update falls back to the approved version, a first install is
        forgotten. Returns the entry that is left, or None."""
        entry = self.get(name)
        if entry is None or entry.get("status") != "pending":
            return entry
        active = entry.get("active")
        if not isinstance(active, dict):
            self.remove(name)
            return None
        entry.update({k: active.get(k) for k in PINNED}, status="active", changes=None)
        self.put(name, entry)
        return entry


def file_changes(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, list[str]]:
    """Which files an update adds, removes or changes (by git blob hash)."""
    old = old or {}
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(k for k in set(old) & set(new) if old[k] != new[k]),
    }
