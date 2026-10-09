"""Connecting the AI plans people already pay for. Contract: docs/API.md section 3.

- OpenRouter: its browser sign-in for apps on the user's computer (OAuth with PKCE). The browser comes back to
  the shared loopback listener of the integration manager with a code, which is swapped for an API key.
- Claude (Max and Team plans include monthly API credits) and Nous Portal: an ordinary API key, checked here
  with a free request that lists the account's models.

Every key lands in the keychain under the provider's id (``openrouter``, ``anthropic``, ``nous``), the same entry
a pasted key uses, so removing the key disconnects.
"""

from __future__ import annotations

import base64
import hashlib
import secrets as pysecrets
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import httpx

from sentient import secrets
from sentient.llm.provider import provider_config

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
OPENROUTER_KEYS_URL = "https://openrouter.ai/api/v1/auth/keys"
OPENROUTER_KEY_INFO_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
ANTHROPIC_API = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
FLOW_TTL_S = 600  # OpenRouter codes expire after 10 minutes
CATALOG_TTL_S = 600
TIMEOUT_S = 15

CHECKABLE = {"anthropic", "openrouter", "nous"}
CATALOGS = {"anthropic", "openrouter", "nous"}
LABELS = {"anthropic": "Anthropic", "openrouter": "OpenRouter", "nous": "Nous Portal"}


class ConnectError(Exception):
    """A plain sentence for the user."""


def pkce_pair() -> tuple[str, str]:
    """A random verifier and its S256 challenge (RFC 7636)."""
    verifier = base64.urlsafe_b64encode(pysecrets.token_bytes(48)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def openrouter_auth_url(callback_url: str, challenge: str, state: str) -> str:
    q = {"callback_url": callback_url, "code_challenge": challenge, "code_challenge_method": "S256",
         "state": state, "key_label": "Sentient"}
    return f"{OPENROUTER_AUTH_URL}?{urlencode(q)}"


def _http() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT_S)


def _error_text(r: httpx.Response) -> str:
    try:
        body = r.json()
    except Exception:
        return r.text[:200]
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)[:200]
    return str(err or body)[:200]


async def exchange_openrouter_code(code: str, verifier: str) -> str:
    """Swap the sign-in code for an OpenRouter API key."""
    try:
        async with _http() as http:
            r = await http.post(OPENROUTER_KEYS_URL, json={"code": code, "code_verifier": verifier,
                                                           "code_challenge_method": "S256"})
    except httpx.HTTPError as exc:
        raise ConnectError("Couldn't reach OpenRouter. Check your internet connection and try again.") from exc
    if r.status_code >= 400:
        raise ConnectError(f"OpenRouter turned down the sign-in ({_error_text(r)}). Please try again.")
    key = (r.json() or {}).get("key")
    if not isinstance(key, str) or not key:
        raise ConnectError("OpenRouter didn't send a key back. Please try again.")
    return key


def _key(config, provider: str) -> str | None:
    pc = provider_config(config, provider)
    return secrets.get_secret(provider, pc.api_key_env if pc else None)


def _base(config, provider: str, default: str) -> str:
    pc = provider_config(config, provider)
    return ((pc.api_base if pc else None) or default).rstrip("/")


def _key_problem(provider: str, r: httpx.Response) -> str:
    label = LABELS[provider]
    if r.status_code == 401:
        return f"{label} didn't accept this key. Copy it again and paste it in full."
    if r.status_code == 403:
        return f"This key isn't allowed to use the {label} API. Check it in your {label} account."
    return f"{label} answered with an error ({r.status_code}): {_error_text(r)}"


async def _get(url: str, headers: dict[str, str], provider: str) -> Any:
    try:
        async with _http() as http:
            r = await http.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise ConnectError(f"Couldn't reach {LABELS[provider]}. Check your internet connection.") from exc
    if r.status_code >= 400:
        raise ConnectError(_key_problem(provider, r))
    return r.json()


async def list_models(config, provider: str) -> list[dict]:
    """The provider's models as ``[{id, label, free, tools, context_length}]``, ``id`` with Sentient's prefix."""
    if provider == "openrouter":  # a public list; no key needed
        data = await _get(OPENROUTER_MODELS_URL, {}, provider)
        out = []
        for m in data.get("data") or []:
            mid = m.get("id")
            if not mid:
                continue
            pricing = m.get("pricing") or {}
            free = mid.endswith(":free") or (str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0")
            out.append({"id": f"openrouter/{mid}", "label": m.get("name") or mid, "free": free,
                        "tools": "tools" in (m.get("supported_parameters") or []),
                        "context_length": m.get("context_length")})
        return out
    key = _key(config, provider)
    if not key:
        raise ConnectError(f"Add your {LABELS[provider]} key first.")
    if provider == "anthropic":
        base = _base(config, provider, ANTHROPIC_API)
        data = await _get(f"{base}/v1/models?limit=100", {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION},
                          provider)
        return [{"id": f"anthropic/{m['id']}", "label": m.get("display_name") or m["id"], "free": False, "tools": True,
                 "context_length": None} for m in data.get("data") or [] if m.get("id")]
    base = _base(config, provider, "")
    data = await _get(f"{base}/models", {"Authorization": f"Bearer {key}"}, provider)
    return [{"id": f"{provider}/{m['id']}", "label": m["id"], "free": False, "tools": None, "context_length": None}
            for m in data.get("data") or [] if m.get("id")]


async def check_key(config, provider: str) -> dict:
    """A free request with the stored key. ``{ok, detail}`` or ``{ok: False, error}``; never spends credits."""
    label = LABELS[provider]
    if provider == "openrouter" and not _key(config, provider):
        return {"ok": False, "error": "Connect OpenRouter or add a key first."}
    try:
        if provider == "openrouter":
            key = _key(config, provider) or ""
            info = (await _get(OPENROUTER_KEY_INFO_URL, {"Authorization": f"Bearer {key}"}, provider)).get("data") or {}
            limit = info.get("limit_remaining")
            detail = "Your OpenRouter key works."
            if isinstance(limit, int | float):
                detail += f" ${limit:.2f} left on this key's spending limit."
            return {"ok": True, "detail": detail}
        models = await list_models(config, provider)
    except ConnectError as exc:
        return {"ok": False, "error": str(exc)}
    n = len(models)
    return {"ok": True, "detail": f"Your {label} key works. {n} model{'' if n == 1 else 's'} available."}


class ProviderConnections:
    """OpenRouter sign-ins in flight, and a short-lived cache of model lists."""

    def __init__(self, app: SentientApp):
        self.app = app
        self._flows: dict[str, dict] = {}
        self._catalog: dict[str, tuple[float, list[dict]]] = {}

    # ------------------------------------------------------------------ OpenRouter sign-in
    async def start_openrouter(self) -> dict:
        now = time.time()
        for state, flow in list(self._flows.items()):
            if now - flow["created"] > FLOW_TTL_S:
                self._flows.pop(state, None)
        listener = self.app.integrations.listener
        await listener.start(self.app.config.integrations.oauth_redirect_port)
        verifier, challenge = pkce_pair()
        state = pysecrets.token_urlsafe(24)
        self._flows[state] = {"verifier": verifier, "created": now, "status": "waiting", "error": None}
        return {"auth_url": openrouter_auth_url(listener.redirect_uri(), challenge, state), "state": state}

    def flow_status(self, state: str) -> dict | None:
        flow = self._flows.get(state)
        if flow is None:
            return None
        if flow["status"] == "waiting" and time.time() - flow["created"] > FLOW_TTL_S:
            return {"status": "failed", "error": "The sign-in took too long. Please try again."}
        return {"status": flow["status"], "error": flow["error"]}

    def owns_state(self, state: str) -> bool:
        return bool(state) and state in self._flows

    async def oauth_callback(self, params: dict[str, str]) -> tuple[bool, str]:
        """The browser came back to the loopback listener from OpenRouter."""
        flow = self._flows.get(params.get("state", ""))
        if flow is None or time.time() - flow["created"] > FLOW_TTL_S:
            return False, "This sign-in link has expired."
        if flow["status"] != "waiting":
            return False, "This sign-in link was already used."
        if params.get("error") or not params.get("code"):
            return self._fail(flow, "You cancelled the sign-in." if params.get("error") == "access_denied"
                              else "OpenRouter didn't send a sign-in code.")
        flow["status"] = "exchanging"
        try:
            key = await exchange_openrouter_code(params["code"], flow["verifier"])
        except ConnectError as exc:
            return self._fail(flow, str(exc))
        if not secrets.set_secret("openrouter", key):
            return self._fail(flow, "Your system keychain is unavailable, so the key can't be saved.")
        flow["status"] = "connected"
        self._catalog.pop("openrouter", None)
        self.app.bus.publish("config.updated", {"sections": ["secrets"]})
        return True, "OpenRouter is connected."

    @staticmethod
    def _fail(flow: dict, reason: str) -> tuple[bool, str]:
        flow["status"], flow["error"] = "failed", reason
        return False, reason

    # ------------------------------------------------------------------ model lists
    async def catalog(self, provider: str) -> list[dict]:
        hit = self._catalog.get(provider)
        if hit and time.time() - hit[0] < CATALOG_TTL_S:
            return hit[1]
        models = await list_models(self.app.config, provider)
        self._catalog[provider] = (time.time(), models)
        return models

    def forget(self, provider: str) -> None:
        self._catalog.pop(provider, None)
