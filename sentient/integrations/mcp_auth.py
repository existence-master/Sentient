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

import hashlib
import logging
import time
from typing import Any
from urllib.parse import urlsplit

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


SERVER_URL_KEY = "server_url"


def canonical_url(url: str | None) -> str:
    """Normalize a server address for deterministic key hashing, safely handling IPv6 literals."""
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except Exception:
        return raw.rstrip("/")

    scheme = parts.scheme.lower()
    if not scheme or not parts.netloc:
        return raw.rstrip("/")

    try:
        port = parts.port
        hostname = parts.hostname or ""
    except ValueError:
        # Invalid port syntax (e.g. non-numeric port): preserve lowercased netloc safely
        path = parts.path.rstrip("/")
        canon = f"{scheme}://{parts.netloc.lower()}{path}"
        return f"{canon}?{parts.query}" if parts.query else canon

    host = f"[{hostname}]" if ":" in hostname else hostname
    is_default = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    userinfo = ""
    if parts.username:
        userinfo = f"{parts.username}:{parts.password}@" if parts.password else f"{parts.username}@"
    host_port = f"{host}:{port}" if (port is not None and not is_default) else host
    netloc = f"{userinfo}{host_port}"
    path = parts.path.rstrip("/")
    canonical = f"{scheme}://{netloc}{path}"
    if parts.query:
        canonical = f"{canonical}?{parts.query}"
    return canonical


def url_hash(url: str | None) -> str:
    """A deterministic 12-character hex hash of the canonical server URL."""
    canon = canonical_url(url)
    if not canon:
        return ""
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def headers_secret(server: str) -> str:
    return f"mcp:{server}:headers"


def tokens_secret(server: str, url: str | None = None) -> str:
    h = url_hash(url)
    return f"mcp:{server}:{h}:oauth" if h else f"mcp:{server}:oauth"


def client_secret(server: str, url: str | None = None) -> str:
    h = url_hash(url)
    return f"mcp:{server}:{h}:client" if h else f"mcp:{server}:client"


def _has_valid_tokens(data: Any) -> bool:
    """Whether a stored OAuth token record has valid access or refresh tokens."""
    if not isinstance(data, dict):
        return False
    toks = data.get("tokens")
    return isinstance(toks, dict) and bool(toks.get("access_token") or toks.get("refresh_token"))


def _has_valid_client_info(data: Any) -> bool:
    """Whether a stored OAuth client info record has a valid client_id."""
    return isinstance(data, dict) and bool(data.get("client_id"))


def migrate_server_credentials(server: str, url: str | None) -> None:
    """Migrate legacy unhashed keychain entries to the new hashed format for this server URL.

    Compatibility rules (#269):
    - Never assume an untagged legacy credential (saved is None) belongs to this URL.
      Preserve untagged legacy entries so other setups sharing the keychain are not starved.
    - Legacy entries are copied to URL-specific keys but preserved in place for backward
      compatibility with older installations sharing the keychain.
    - Untagged legacy client info is copied only when the legacy token record is tagged for
      this same server URL.
    - Never overwrite a destination record with conflicting or malformed data.
    """
    if not url:
        return

    canon = canonical_url(url)

    # 1. Tokens
    legacy_tok = tokens_secret(server, None)
    new_tok = tokens_secret(server, url)
    leg_data = None
    if legacy_tok != new_tok:
        dest_data = load_json(new_tok)
        leg_data = load_json(legacy_tok)
        if isinstance(leg_data, dict):
            saved = leg_data.get(SERVER_URL_KEY)
            if saved is not None and canonical_url(saved) == canon:
                if dest_data is not None:
                    if not _has_valid_tokens(dest_data):
                        new_rec = {**leg_data, SERVER_URL_KEY: url}
                        save_json(new_tok, new_rec)
                    elif dest_data.get("tokens") != leg_data.get("tokens"):
                        log.warning("Preserving legacy tokens for %s: destination key contains conflicting credentials", server)
                else:
                    new_rec = {**leg_data, SERVER_URL_KEY: url}
                    if not save_json(new_tok, new_rec):
                        log.warning("Could not save migrated tokens for %s", server)
            # If saved is None: NEVER migrate untagged legacy bearer tokens across server URLs (CWE-522).
            # The untagged legacy record is preserved without modification so other setups are not starved.
            # Require explicit OAuth sign-in for each URL whose token ownership cannot be verified.

    # 2. Client info
    legacy_cli = client_secret(server, None)
    new_cli = client_secret(server, url)
    if legacy_cli != new_cli:
        dest_cli = load_json(new_cli)
        leg_cli = load_json(legacy_cli)
        if isinstance(leg_cli, dict):
            saved = leg_cli.get(SERVER_URL_KEY)
            should_copy = False
            if saved is not None and canonical_url(saved) == canon:
                should_copy = True
            elif saved is None and _has_valid_client_info(leg_cli) and isinstance(leg_data, dict):
                # Copy untagged client info ONLY when legacy tokens are tagged for this same address
                tok_saved = leg_data.get(SERVER_URL_KEY)
                if tok_saved is not None and canonical_url(tok_saved) == canon:
                    should_copy = True

            if should_copy:
                if dest_cli is not None:
                    if not _has_valid_client_info(dest_cli):
                        new_rec = {**leg_cli, SERVER_URL_KEY: url}
                        save_json(new_cli, new_rec)
                    elif dest_cli.get("client_id") != leg_cli.get("client_id"):
                        log.warning("Preserving legacy client info for %s: destination contains conflicting client info", server)
                else:
                    new_rec = {**leg_cli, SERVER_URL_KEY: url}
                    if not save_json(new_cli, new_rec):
                        log.warning("Could not save migrated client info for %s", server)


def stale_sign_in(
    server: str,
    url: str | None,
    *,
    changed_here: bool,
    previous_url: str | None = None,
) -> list[str]:
    """The sign-in entries that belong to another address and can safely be cleared.

    When a server's address changes on this setup (changed_here=True and previous_url is set),
    the previous URL's URL-specific tokens and client info secrets are cleared so live tokens
    are not left behind in the keychain.
    A legacy entry tagged with another server URL must NEVER be deleted, as another setup
    sharing the OS keychain may own it (#269).
    An untagged legacy entry is only considered stale if this setup previously moved the server
    elsewhere (changed_here=True)."""
    stale = []
    canon = canonical_url(url) if url else None
    prev_canon = canonical_url(previous_url) if previous_url else None

    # URL-specific entries from previous_url when address changes to another address
    if changed_here and prev_canon and prev_canon != canon:
        for s in (tokens_secret(server, previous_url), client_secret(server, previous_url)):
            if s not in stale:
                stale.append(s)

    for name in (tokens_secret(server, None), client_secret(server, None)):
        rec = load_json(name)
        if not isinstance(rec, dict):
            continue
        saved = rec.get(SERVER_URL_KEY)
        if saved is not None:
            # If explicitly tagged with previous_url and changed here to another address,
            # this setup owned it and moved away:
            if (
                changed_here
                and prev_canon
                and prev_canon != canon
                and canonical_url(saved) == prev_canon
                and name not in stale
            ):
                stale.append(name)
            continue
        # Untagged record: only clear if this setup previously configured this server elsewhere
        if changed_here and name not in stale:
            stale.append(name)
    return stale


def forget_server(server: str, url: str | None = None) -> None:
    delete_json(headers_secret(server))
    for name in (tokens_secret(server, url), client_secret(server, url)):
        delete_json(name)
    if url:
        canon = canonical_url(url)
        for fn in (tokens_secret, client_secret):
            leg_key = fn(server, None)
            rec = load_json(leg_key)
            if isinstance(rec, dict):
                saved = rec.get(SERVER_URL_KEY)
                # Only delete legacy entry if it was explicitly tagged for THIS url.
                # Never delete an untagged entry (saved is None) or another URL's entry!
                if saved is not None and canonical_url(saved) == canon:
                    delete_json(leg_key)


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

    @property
    def effective_url(self) -> str | None:
        if self.url:
            return self.url
        if self.context is not None:
            return getattr(self.context, "server_url", None)
        return None

    def _tag(self, rec: dict) -> dict:
        url = self.effective_url
        return {**rec, SERVER_URL_KEY: url} if url else rec

    def record(self) -> dict | None:
        url = self.effective_url
        if url:
            rec = load_json(tokens_secret(self.server, url))
            if rec is not None:
                return rec
            # Only fall back to legacy key if it is explicitly tagged for this URL.
            # Never fall back if saved is None!
            leg = load_json(tokens_secret(self.server, None))
            if isinstance(leg, dict):
                saved = leg.get(SERVER_URL_KEY)
                if saved is not None and canonical_url(saved) == canonical_url(url):
                    return leg
            return None
        rec = load_json(tokens_secret(self.server, None))
        return rec if isinstance(rec, dict) else None

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
        return save_json(tokens_secret(self.server, self.effective_url), rec)

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
        if not save_json(tokens_secret(self.server, self.effective_url), rec):
            raise SignInFailed("The system keychain is unavailable, so the sign-in can't be saved.")
        self.saved = True

    def save_metadata(self) -> bool:
        """Add the context's discovered metadata to the stored record, keeping its tokens. If the larger record
        can't be written completely, the old one is put back, so a working sign-in is never lost to this."""
        rec = self.record()
        meta = _metadata_of(self.context)
        if not rec or not meta:
            return False
        key = tokens_secret(self.server, self.effective_url)
        if save_json(key, {**rec, **meta}):
            return True
        save_json(key, rec)
        return False

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        url = self.effective_url
        if url:
            data = load_json(client_secret(self.server, url))
            if data is None:
                leg = load_json(client_secret(self.server, None))
                if isinstance(leg, dict):
                    saved = leg.get(SERVER_URL_KEY)
                    if saved is not None and canonical_url(saved) == canonical_url(url):
                        data = leg
        else:
            data = load_json(client_secret(self.server, None))
        if not data:
            return None
        try:
            return OAuthClientInformationFull.model_validate({k: v for k, v in data.items() if k != SERVER_URL_KEY})
        except Exception:
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        if not save_json(client_secret(self.server, self.effective_url), self._tag(client_info.model_dump(mode="json", exclude_none=True))):
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
        if (not self._restore_metadata(storage.record() or {}) and await self._discover_metadata()
                and not storage.save_metadata()):
            log.info("mcp sign-in details for %s could not be saved; they are looked up again next time", storage.server)

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
