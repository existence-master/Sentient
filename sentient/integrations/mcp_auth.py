"""Sign-in for remote MCP servers: keychain storage for headers and OAuth tokens.

The OAuth flow itself (protected resource and authorization server discovery, dynamic
client registration, PKCE, token exchange and refresh) is the MCP SDK's
``OAuthClientProvider``. This module gives it keychain storage and remembers when a stored
token expires, so a token loaded after a restart is refreshed instead of sent stale.

Keychain entries per server (never in config): ``mcp:<name>:headers`` (header values),
``mcp:<name>:oauth`` (tokens) and ``mcp:<name>:client`` (the registered OAuth client).
Values longer than one keychain entry allows are split over ``<entry>:1``, ``<entry>:2``... (``sentient.secrets``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from sentient.secrets import delete_json, load_json, save_json

# Refresh a little before the server's expiry so a token never dies mid-request.
EXPIRY_MARGIN_S = 60


class NeedsSignIn(Exception):
    """The server wants a sign-in the user has not done (or that expired)."""


class SignInFailed(Exception):
    """A browser sign-in did not finish. The message is shown to the user."""


class _QuietExpected(logging.Filter):
    """The SDK logs every stopped OAuth flow as an error; ours stop on purpose."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, NeedsSignIn | SignInFailed)


logging.getLogger("mcp.client.auth.oauth2").addFilter(_QuietExpected())


def headers_secret(server: str) -> str:
    return f"mcp:{server}:headers"


def tokens_secret(server: str) -> str:
    return f"mcp:{server}:oauth"


def client_secret(server: str) -> str:
    return f"mcp:{server}:client"


# Chunked keychain JSON lives in sentient.secrets; these names stay for existing callers.


def forget_server(server: str) -> None:
    for name in (headers_secret(server), tokens_secret(server), client_secret(server)):
        delete_json(name)


# ---------------------------------------------------------------------------- OAuth storage
class KeychainTokenStorage:
    """The SDK's ``TokenStorage`` backed by the keychain.

    ``fresh=True`` (used while signing in) ignores stored tokens, so the flow starts over
    without deleting a working sign-in until a new one has replaced it.
    """

    def __init__(self, server: str, *, fresh: bool = False):
        self.server = server
        self.fresh = fresh
        self.saved = False

    def record(self) -> dict | None:
        return load_json(tokens_secret(self.server))

    def has_tokens(self) -> bool:
        rec = self.record()
        return bool(rec and (rec.get("tokens") or {}).get("access_token"))

    def expires_at(self) -> float | None:
        if self.fresh and not self.saved:
            return None
        rec = self.record() or {}
        value = rec.get("expires_at")
        return float(value) if value is not None else None

    def mark_expired(self) -> bool:
        """Make the next connection refresh first. False when there is nothing to refresh with."""
        rec = self.record()
        if not rec or not (rec.get("tokens") or {}).get("refresh_token"):
            return False
        rec["expires_at"] = 1.0  # long past; the SDK treats 0 as "never expires"
        return save_json(tokens_secret(self.server), rec)

    async def get_tokens(self) -> OAuthToken | None:
        if self.fresh and not self.saved:
            return None
        rec = self.record()
        if not rec or not rec.get("tokens"):
            return None
        try:
            return OAuthToken.model_validate(rec["tokens"])
        except Exception:
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        expires_at = time.time() + int(tokens.expires_in) - EXPIRY_MARGIN_S if tokens.expires_in is not None else None
        rec = {"tokens": tokens.model_dump(mode="json", exclude_none=True), "expires_at": expires_at}
        if not save_json(tokens_secret(self.server), rec):
            raise SignInFailed("The system keychain is unavailable, so the sign-in can't be saved.")
        self.saved = True

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = load_json(client_secret(self.server))
        if not data:
            return None
        try:
            return OAuthClientInformationFull.model_validate(data)
        except Exception:
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        if not save_json(client_secret(self.server), client_info.model_dump(mode="json", exclude_none=True)):
            raise SignInFailed("The system keychain is unavailable, so the sign-in can't be saved.")


class SentientOAuthProvider(OAuthClientProvider):
    """The SDK provider, plus the stored expiry time so tokens loaded at startup refresh on time."""

    async def _initialize(self) -> None:
        await super()._initialize()
        storage: Any = self.context.storage
        if isinstance(storage, KeychainTokenStorage) and self.context.current_tokens is not None:
            self.context.token_expiry_time = storage.expires_at()


def client_metadata(redirect_uri: str) -> OAuthClientMetadata:
    return OAuthClientMetadata(
        client_name="Sentient",
        redirect_uris=[redirect_uri],  # type: ignore[list-item]
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )
