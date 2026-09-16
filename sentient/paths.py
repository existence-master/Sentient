"""Where Sentient keeps its state on disk.

Everything lives under one home directory (default ``~/.sentient``) so that
backing up, moving, or wiping the assistant is a single folder operation.
Override with the ``SENTIENT_HOME`` environment variable (useful for tests
and for running several isolated profiles).
"""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    raw = os.environ.get("SENTIENT_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".sentient"


def config_file() -> Path:
    return home() / "config.yaml"


def db_file() -> Path:
    return home() / "sentient.db"


def token_file() -> Path:
    return home() / "gateway.token"


def workspace_dir() -> Path:
    """Human-readable markdown the user can open and edit: SOUL.md, USER.md, MEMORY.md, notes/."""
    return home() / "workspace"


def notes_dir() -> Path:
    return workspace_dir() / "notes"


def skills_dir() -> Path:
    return home() / "skills"


def files_dir() -> Path:
    """Scratch space the agent may read/write on the user's behalf."""
    return home() / "files"


def logs_dir() -> Path:
    return home() / "logs"


def ensure_layout() -> Path:
    root = home()
    for d in (root, workspace_dir(), notes_dir(), skills_dir(), files_dir(), logs_dir()):
        d.mkdir(parents=True, exist_ok=True)
    return root
