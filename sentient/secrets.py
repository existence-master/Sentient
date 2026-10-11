"""API keys and tokens.

Order of lookup for a provider key:
1. OS keychain entry ``<service>/<provider>`` (Windows Credential Manager, macOS Keychain, libsecret).
2. The environment variable named in ``models.providers.<provider>.api_key_env``.

Nothing is ever written to config.yaml.

The keychain is per data folder (issue #302). The default data folder (``~/.sentient``, used when ``SENTIENT_HOME``
is unset or points at it) uses the service ``sentient``, the names Sentient always used, so existing sign-ins keep
working. Any other ``SENTIENT_HOME`` (a test or demo profile) uses its own service ``sentient-<12 hex>``, a hash of
the resolved folder, so it never reads, overwrites or deletes the default folder's sign-ins. Entry names (and the
``<name>:1``, ``<name>:2``... chunks of long values) are the same in every namespace.
``SENTIENT_KEYCHAIN_NAMESPACE`` overrides the choice: ``default`` shares the default folder's entries (for a
deliberate real test), any other word gives the service ``sentient-<word>``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from sentient import paths

SERVICE = "sentient"  # the default data folder's service (also every entry before #302)
ENV_NAMESPACE = "SENTIENT_KEYCHAIN_NAMESPACE"
DEFAULT_NAMESPACE = "default"
# Windows Credential Manager holds about 1280 characters per entry.
CHUNK = 1000
MAX_CHUNKS = 32


def _norm(path: Path) -> str:
    # realpath also resolves the existing part of a missing path, so the result is the same before and after the
    # folder is created (and Windows 8.3 short names are expanded); normcase makes C:\Users and c:\users equal.
    return os.path.normcase(os.path.realpath(path.expanduser()))


def is_default_home(home: Path | None = None) -> bool:
    """Whether ``home`` (default: the current data folder) is the default one, ``~/.sentient``."""
    return _norm(home or paths.home()) == _norm(Path.home() / ".sentient")


def namespace(home: Path | None = None) -> str:
    """``default`` for the default data folder (or the override ``default``), else the namespace's id."""
    raw = os.environ.get(ENV_NAMESPACE, "").strip()
    if raw:
        if raw.lower() == DEFAULT_NAMESPACE:
            return DEFAULT_NAMESPACE  # only the exact word shares the default folder's entries
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-.")[:40].lower()
        if not cleaned or cleaned == DEFAULT_NAMESPACE:  # e.g. "---" or "default!": still a namespace of its own
            cleaned = "x" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return cleaned
    if is_default_home(home):
        return DEFAULT_NAMESPACE
    return hashlib.sha256(_norm(home or paths.home()).encode("utf-8")).hexdigest()[:12]


def service(home: Path | None = None) -> str:
    """The keychain service this data folder's entries live under: ``sentient`` or ``sentient-<namespace>``."""
    ns = namespace(home)
    return SERVICE if ns == DEFAULT_NAMESPACE else f"{SERVICE}-{ns}"


def describe() -> str:
    """One log line saying where this data folder's secrets live (never a value)."""
    svc = service()
    if svc == SERVICE:
        how = "set by " + ENV_NAMESPACE if os.environ.get(ENV_NAMESPACE, "").strip() else "the default data folder"
        return f"keychain service '{svc}' ({how}: shared with ~/.sentient)"
    return f"keychain service '{svc}' (only this data folder's secrets)"


def _keyring():
    try:
        import keyring

        return keyring
    except Exception:  # pragma: no cover - keyring missing/unusable backend
        return None


def get_secret(name: str, env_var: str | None = None) -> str | None:
    kr = _keyring()
    if kr is not None:
        try:
            value = kr.get_password(service(), name)
            if value:
                return value
        except Exception:
            pass
    if env_var:
        return os.environ.get(env_var) or None
    return None


def set_secret(name: str, value: str) -> bool:
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.set_password(service(), name, value)
        return True
    except Exception:
        return False


def delete_secret(name: str) -> bool:
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.delete_password(service(), name)
        return True
    except Exception:
        return False


# JSON values too long for one keychain entry (OAuth tokens) are split over ``<name>:1``, ``<name>:2``...
def save_json(name: str, data: dict) -> bool:
    raw = json.dumps(data, separators=(",", ":"))
    parts = [raw[i : i + CHUNK] for i in range(0, len(raw), CHUNK)] or [""]
    if len(parts) > MAX_CHUNKS:
        return False
    ok = set_secret(name, f"{len(parts)}\n{parts[0]}")
    for i, part in enumerate(parts[1:], 1):
        ok = set_secret(f"{name}:{i}", part) and ok
    for i in range(len(parts), MAX_CHUNKS):
        if not get_secret(f"{name}:{i}"):
            break
        delete_secret(f"{name}:{i}")
    return ok


def load_json(name: str) -> dict | None:
    head = get_secret(name)
    if not head:
        return None
    count, _, first = head.partition("\n")
    if not count.isdigit():
        return None
    parts = [first]
    for i in range(1, min(int(count), MAX_CHUNKS)):
        part = get_secret(f"{name}:{i}")
        if part is None:
            return None
        parts.append(part)
    try:
        data = json.loads("".join(parts))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def delete_json(name: str) -> None:
    head = get_secret(name) or ""
    count = head.partition("\n")[0]
    delete_secret(name)
    for i in range(1, min(int(count) if count.isdigit() else 1, MAX_CHUNKS)):
        delete_secret(f"{name}:{i}")
