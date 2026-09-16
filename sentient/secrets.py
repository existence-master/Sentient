"""API keys and tokens.

Order of lookup for a provider key:
1. OS keychain entry ``sentient/<provider>`` (Windows Credential Manager, macOS Keychain, libsecret).
2. The environment variable named in ``models.providers.<provider>.api_key_env``.

Nothing is ever written to config.yaml.
"""

from __future__ import annotations

import os

SERVICE = "sentient"


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
