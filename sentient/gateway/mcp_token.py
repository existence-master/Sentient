"""The MCP-only token (issue #443, ADR 0034).

When the user turns on "Let other apps use Sentient", the engine issues a token of its own for that
feature. It lives in the keychain (``sentient.secrets``), never in a config file, and it is not the
gateway token: it opens only the handful of endpoints the MCP tools need (see ``auth.py``), never
approvals, config or anything else. The user can replace it from Settings > Integrations, which
cuts off every app still holding the old one.
"""

from __future__ import annotations

import secrets as _secrets

from sentient import secrets

SECRET_NAME = "mcp-server-token"


def current_token() -> str | None:
    """The token apps must present today, or None when the feature never issued one."""
    return secrets.get_secret(SECRET_NAME)


def ensure_token() -> str | None:
    """The current token, creating and storing one first when there is none.

    None when the keychain is unavailable: the feature then stays unusable rather than falling
    back to a file anyone on the machine could read.
    """
    token = current_token()
    if token:
        return token
    return rotate_token()


def rotate_token() -> str | None:
    """Replace the token with a fresh one (the old one stops working at once). None when the
    keychain is unavailable."""
    token = _secrets.token_urlsafe(32)
    return token if secrets.set_secret(SECRET_NAME, token) else None


def token_matches(candidate: str | None) -> bool:
    """Constant-time check of a presented token against the stored one."""
    current = current_token()
    return bool(candidate) and current is not None and _secrets.compare_digest(candidate, current)
