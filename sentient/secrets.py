"""API keys and tokens.

Order of lookup for a provider key:
1. OS keychain entry ``sentient/<provider>`` (Windows Credential Manager, macOS Keychain, libsecret).
2. The environment variable named in ``models.providers.<provider>.api_key_env``.

Nothing is ever written to config.yaml.
"""

from __future__ import annotations

import json
import os

SERVICE = "sentient"
# Windows Credential Manager holds about 1280 characters per entry.
CHUNK = 1000
MAX_CHUNKS = 32


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
            value = kr.get_password(SERVICE, name)
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
        kr.set_password(SERVICE, name, value)
        return True
    except Exception:
        return False


def delete_secret(name: str) -> bool:
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.delete_password(SERVICE, name)
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
