"""Sign-in for remote MCP servers: keychain storage for headers and OAuth tokens.

The OAuth flow itself (protected resource and authorization server discovery, dynamic
client registration, PKCE, token exchange and refresh) is the MCP SDK's
``OAuthClientProvider``. This module gives it keychain storage and remembers when a stored
token expires, so a token loaded after a restart is refreshed instead of sent stale.

The SDK keeps the discovered authorization server metadata only in memory, and without it a
refresh goes to ``<server origin>/token``, which is wrong for most servers. So the token record
also keeps that metadata (and the server URL it belongs to), and a provider puts it back when it
loads the tokens. Records saved before that are upgraded by discovering the metadata once.

Keychain entries per server (never in config): ``mcp:<name>:headers`` (header values),
``mcp:<name>:oauth`` (tokens) and ``mcp:<name>:client`` (the registered OAuth client).
Values longer than one keychain entry allows are split over ``<entry>:1``, ``<entry>:2``... (``sentient.secrets``).

The keychain is shared by every Sentient setup on the computer, so the token and client records also carry the
server address they belong to (``server_url``). A setup that adds a server with the same name only clears a sign-in
made for another address (``stale_sign_in``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anyio
import httpx2
from mcp.client.auth import OAuthClientProvider
from mcp.client.auth.oauth2 import _origin_issuer  # same legacy issuer as the SDK
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_oauth_metadata_request,
    credentials_match_issuer,
    handle_auth_metadata_response,
    handle_protected_resource_response,
    issuers_match,
)
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)

from sentient.secrets import delete_json, load_json, save_json

# Refresh a little before the server's expiry so a token never dies mid-request.
EXPIRY_MARGIN_S = 60
# Looking up the metadata for a record saved without it: short, since a failure only means the SDK's fallback.
DISCOVERY_TIMEOUT_S = 8

log = logging.getLogger(__name__)


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

SERVER_URL_KEY = "server_url"


def stale_sign_in(server: str, url: str | None, *, changed_here: bool) -> list[str]:
    """The sign-in entries (tokens, registered client) that belong to another address than ``url``.

    A record names its address and is stale when ``url`` is another one (a local server, ``url=None``, leaves it to
    the setup that uses it). An older record without an address counts as stale only when this setup had the server
    at another address (``changed_here``), since another setup may be using it."""
    stale = []
    for name in (tokens_secret(server), client_secret(server)):
        rec = load_json(name)
        if not isinstance(rec, dict):
            continue
        saved = rec.get(SERVER_URL_KEY)
        if (url is not None and saved != url) if saved else changed_here:
            stale.append(name)
    return stale


def forget_server(server: str) -> None:
    for name in (headers_secret(server), tokens_secret(server), client_secret(server)):
        delete_json(name)


# ---------------------------------------------------------------------------- OAuth storage
class KeychainTokenStorage:
    """The SDK's ``TokenStorage`` backed by the keychain.

    ``fresh=True`` (used while signing in) ignores stored tokens, so the flow starts over
    without deleting a working sign-in until a new one has replaced it.
    """

    def __init__(self, server: str, *, fresh: bool = False, url: str | None = None):
        self.server = server
        self.fresh = fresh
        self.url = url
        self.saved = False
        self.context: Any = None  # the provider's OAuthContext: its discovered metadata is saved with the tokens

    def _tag(self, rec: dict) -> dict:
        return {**rec, SERVER_URL_KEY: self.url} if self.url else rec

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
        rec = self._tag({"tokens": tokens.model_dump(mode="json", exclude_none=True), "expires_at": expires_at,
                         **_metadata_of(self.context)})
        if not save_json(tokens_secret(self.server), rec):
            raise SignInFailed("The system keychain is unavailable, so the sign-in can't be saved.")
        self.saved = True

    def save_metadata(self) -> bool:
        """Add the context's discovered metadata to the stored record, keeping its tokens."""
        rec = self.record()
        meta = _metadata_of(self.context)
        if not rec or not meta:
            return False
        return save_json(tokens_secret(self.server), {**rec, **meta})

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = load_json(client_secret(self.server))
        if not data:
            return None
        try:
            return OAuthClientInformationFull.model_validate({k: v for k, v in data.items() if k != SERVER_URL_KEY})
        except Exception:
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        if not save_json(client_secret(self.server), self._tag(client_info.model_dump(mode="json", exclude_none=True))):
            raise SignInFailed("The system keychain is unavailable, so the sign-in can't be saved.")


def _metadata_of(context: Any) -> dict:
    """What a token record keeps about where its tokens came from, so a refresh after a restart finds them."""
    meta = getattr(context, "oauth_metadata", None)
    if meta is None:
        return {}
    prm = context.protected_resource_metadata
    return {
        "server_url": context.server_url,
        "auth_server_url": context.auth_server_url,
        "oauth_metadata": meta.model_dump(mode="json", exclude_none=True),
        "protected_resource_metadata": prm.model_dump(mode="json", exclude_none=True) if prm is not None else None,
    }


class SentientOAuthProvider(OAuthClientProvider):
    """The SDK provider, plus what the keychain record remembers, so tokens loaded at startup refresh on time
    and at the right address."""

    def __init__(self, *args: Any, discovery_transport: Any = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._discovery_transport = discovery_transport  # tests: an httpx2 transport for the metadata lookup
        if isinstance(self.context.storage, KeychainTokenStorage):
            self.context.storage.context = self.context

    async def _initialize(self) -> None:
        await super()._initialize()
        storage: Any = self.context.storage
        if not isinstance(storage, KeychainTokenStorage) or self.context.current_tokens is None:
            return
        self.context.token_expiry_time = storage.expires_at()
        if self.context.oauth_metadata is not None or not self.context.current_tokens.refresh_token:
            return
        if not self._restore_metadata(storage.record() or {}) and await self._discover_metadata():
            storage.save_metadata()

    def _issuer_usable(self, issuer: str) -> bool:
        """SEP-2352: a client registered with one authorization server is never sent to another."""
        info = self.context.client_info
        return info is None or credentials_match_issuer(info, issuer, self.context.client_metadata_url)

    def _restore_metadata(self, rec: dict) -> bool:
        """Put back the metadata saved with these tokens, when it is for this server URL and this client."""
        if not rec.get("oauth_metadata") or rec.get("server_url") != self.context.server_url:
            return False
        try:
            meta = OAuthMetadata.model_validate(rec["oauth_metadata"])
            prm_data = rec.get("protected_resource_metadata")
            prm = ProtectedResourceMetadata.model_validate(prm_data) if prm_data else None
        except Exception:
            return False
        auth_server_url = rec.get("auth_server_url") or None
        if auth_server_url is not None and not issuers_match(str(meta.issuer), auth_server_url):
            return False
        if not self._issuer_usable(str(meta.issuer)):
            return False
        self._use_metadata(prm, auth_server_url, meta)
        return True

    async def _discover_metadata(self) -> bool:
        """Look up the authorization server metadata for a record saved without it, the way the SDK does on a
        401. Any failure leaves the context as it was, so the SDK behaves as before."""
        server_url = self.context.server_url
        prm: ProtectedResourceMetadata | None = None
        auth_server_url: str | None = None
        meta: OAuthMetadata | None = None
        try:
            with anyio.fail_after(DISCOVERY_TIMEOUT_S):
                async with httpx2.AsyncClient(timeout=DISCOVERY_TIMEOUT_S, transport=self._discovery_transport) as http:
                    for url in build_protected_resource_metadata_discovery_urls(None, server_url):
                        found = await handle_protected_resource_response(
                            await http.send(create_oauth_metadata_request(url)))
                        if found is not None:
                            await self._validate_resource_match(found)
                            prm = found
                            auth_server_url = self._select_authorization_server(
                                [str(u) for u in found.authorization_servers])
                            break
                    expected = auth_server_url or _origin_issuer(server_url)
                    if not self._issuer_usable(expected):
                        return False
                    for url in build_oauth_authorization_server_metadata_discovery_urls(auth_server_url, server_url):
                        ok, asm = await handle_auth_metadata_response(
                            await http.send(create_oauth_metadata_request(url)))
                        if not ok:
                            break
                        if asm is not None:
                            meta = asm if issuers_match(str(asm.issuer), expected) else None
                            break
        except Exception as exc:
            log.info("mcp sign-in details for %s could not be looked up (%s)", self.context.storage.server,
                     type(exc).__name__)
            return False
        if meta is None:
            return False
        self._use_metadata(prm, auth_server_url, meta)
        return True

    def _use_metadata(self, prm: ProtectedResourceMetadata | None, auth_server_url: str | None,
                      meta: OAuthMetadata) -> None:
        self.context.protected_resource_metadata = prm
        self.context.auth_server_url = auth_server_url
        self.context.oauth_metadata = meta


def client_metadata(redirect_uri: str) -> OAuthClientMetadata:
    return OAuthClientMetadata(
        client_name="Sentient",
        redirect_uris=[redirect_uri],  # type: ignore[list-item]
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )
