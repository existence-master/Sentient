"""Where Sentient keeps its state on disk.

Everything lives under one home directory (default ``~/.sentient``) so that
backing up, moving, or wiping the assistant is a single folder operation.
Override with the ``SENTIENT_HOME`` environment variable (useful for tests
and for running several isolated profiles).
"""

from __future__ import annotations

import os
import sys
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


# FOLDERID_Documents: Windows' own Documents folder, which may be redirected (to OneDrive, another drive...)
_FOLDERID_DOCUMENTS = "FDD39AD0-238F-46AF-ADB4-6C85480369C7"


def _windows_documents() -> Path | None:
    """Documents from the Known Folder API (SHGetKnownFolderPath), so a folder redirected to OneDrive is found where
    the user sees it. None when the call fails."""
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        u = uuid.UUID(_FOLDERID_DOCUMENTS)
        guid = GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8).from_buffer_copy(u.bytes[8:]))
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]
        shell32.SHGetKnownFolderPath.argtypes = [
            ctypes.POINTER(GUID), wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p),
        ]
        shell32.SHGetKnownFolderPath.restype = ctypes.c_long
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        out = ctypes.c_void_p()
        hr = shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out))
        try:
            value = ctypes.wstring_at(out.value) if hr == 0 and out.value else ""
        finally:
            if out.value:
                ole32.CoTaskMemFree(out)
        return Path(value) if value else None
    except Exception:
        return None


def documents_dir() -> Path:
    """The user's Documents folder (it may not exist).

    ``SENTIENT_DOCUMENTS_DIR`` overrides it (tests always set it). Windows: the Known Folder API's Documents (it
    follows a redirect to OneDrive), else ``~/Documents``. macOS: ``~/Documents``. Linux: ``XDG_DOCUMENTS_DIR`` when
    set, else ``~/Documents``."""
    raw = os.environ.get("SENTIENT_DOCUMENTS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    if sys.platform == "win32":
        known = _windows_documents()
        if known is not None:
            return known
    if sys.platform.startswith("linux"):
        xdg = os.environ.get("XDG_DOCUMENTS_DIR", "").strip()
        if xdg:
            return Path(os.path.expandvars(xdg)).expanduser()
    return Path.home() / "Documents"


def ensure_layout() -> Path:
    root = home()
    for d in (root, workspace_dir(), notes_dir(), skills_dir(), files_dir(), logs_dir()):
        d.mkdir(parents=True, exist_ok=True)
    return root
