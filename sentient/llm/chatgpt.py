"""Sign in with ChatGPT: use a ChatGPT Plus or Pro plan's models (issue #205). Contract: docs/API.md section 3.

OpenAI offers plan usage to open-source and locally run apps (https://developers.openai.com/siwc/quickstart).
Each install registers itself on its first sign-in by sending ``client_id=dynamic_agent_client``; the browser comes
back with an issued client id that later sign-ins reuse. Sentient never borrows another app's client id.

- Sign-in: OAuth with PKCE (S256), ``state`` and ``nonce``, through the shared loopback listener
  (``http://127.0.0.1:<port>/oauth/callback``; only the port may change between sign-ins).
- Tokens: one keychain entry, ``chatgpt`` (chunked JSON). The access token lasts an hour and is refreshed shortly
  before it runs out; the refresh token rotates, so refreshes are serialized. Sign-out revokes the refresh token and
  removes the entry.
- Models: ``chatgpt/<slug>`` from ``GET /v1/models``, sent to ``POST /v1/responses`` by ``sentient.llm.responses``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
import weakref
from typing import Any
from urllib.parse import urlencode

import httpx

from sentient.config.schema import SentientConfig
from sentient.secrets import delete_json, load_json, save_json

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
REVOKE_URL = f"{ISSUER}/api/accounts/oauth/revoke"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
API_BASE = "https://api.openai.com/v1"
RESOURCE = API_BASE
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
DYNAMIC_CLIENT = "dynamic_agent_client"
APP_NAME = "Sentient"
MANAGE_USAGE_URL = "https://chatgpt.com/settings/usage"
SECRET = "chatgpt"
PREFIX = "chatgpt"
CLIENT_META = "chatgpt.client_id"  # the issued client id (not a secret), kept after sign-out for the next sign-in
HOST_META = "chatgpt.host_id"  # a stable random id for this computer (ext_agent_host_id), not a credential
REFRESH_MARGIN_S = 300
TIMEOUT_S = 15
# Refresh errors that mean the sign-in is gone for good (https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery)
DEAD_REFRESH = {"invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
                "refresh_token_invalidated", "refresh_token_reused"}

_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()


class ChatGPTError(Exception):
    """A plain sentence for the user."""


def _lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _locks.get(loop)
    if lock is None:
        lock = _locks[loop] = asyncio.Lock()
    return lock


# ---------------------------------------------------------------------------- settings
def configured_client(config: SentientConfig) -> str:
    """``dynamic_agent_client`` (each install registers itself), a client id from OpenAI, or "" when turned off."""
    return (config.models.chatgpt_client_id or "").strip()


def unavailable_reason(config: SentientConfig) -> str | None:
    if configured_client(config):
        return None
    return ("Sign in with ChatGPT is turned off on this computer because no ChatGPT client id is set. To turn it on, "
            f"set models.chatgpt_client_id in config.yaml back to {DYNAMIC_CLIENT}.")


def responses_url(config: SentientConfig) -> str:
    pc = config.models.providers.get(PREFIX)
    return f"{((pc.api_base if pc else None) or API_BASE).rstrip('/')}/responses"


def models_url(config: SentientConfig) -> str:
    pc = config.models.providers.get(PREFIX)
    return f"{((pc.api_base if pc else None) or API_BASE).rstrip('/')}/models"


def request_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "text/event-stream"}


def auth_url(*, client_id: str, redirect_uri: str, challenge: str, state: str, nonce: str, host_id: str,
             registering: bool) -> str:
    q = {"client_id": client_id, "response_type": "code", "redirect_uri": redirect_uri, "scope": SCOPES,
         "resource": RESOURCE, "state": state, "nonce": nonce, "code_challenge": challenge,
         "code_challenge_method": "S256", "ext_agent_host_id": host_id}
    if registering:
        q["agent_name_hint"] = APP_NAME  # only on the first, registering sign-in
    return f"{AUTHORIZE_URL}?{urlencode(q)}"


# ---------------------------------------------------------------------------- tokens in the keychain
def load_tokens() -> dict | None:
    data = load_json(SECRET)
    return data if data and data.get("access_token") and data.get("client_id") else None


def signed_in() -> bool:
    return load_tokens() is not None


def _record(client_id: str, tok: dict, previous: dict | None = None) -> dict:
    previous = previous or {}
    expires_in = tok.get("expires_in")
    return {
        "client_id": client_id,
        "access_token": tok["access_token"],
        "refresh_token": tok.get("refresh_token") or previous.get("refresh_token"),
        "expires_at": time.time() + (float(expires_in) if isinstance(expires_in, int | float) else 3600.0),
        "scope": tok.get("scope") or previous.get("scope") or "",
        "email": previous.get("email"),
    }


def _oauth_error(r: httpx.Response) -> tuple[str, str]:
    try:
        body = r.json()
    except ValueError:
        return "", r.text[:200]
    if not isinstance(body, dict):
        return "", str(body)[:200]
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("code") or err.get("type") or ""), str(err.get("message") or "")[:200]
    return str(err or ""), str(body.get("error_description") or "")[:200]


async def _post_form(url: str, data: dict[str, str]) -> httpx.Response:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as http:
            return await http.post(url, data=data, headers={"Accept": "application/json"})
    except httpx.HTTPError as exc:
        raise ChatGPTError("Couldn't reach ChatGPT. Check your internet connection and try again.") from exc


async def exchange_code(*, client_id: str, code: str, verifier: str, redirect_uri: str) -> dict:
    r = await _post_form(TOKEN_URL, {"grant_type": "authorization_code", "client_id": client_id, "code": code,
                                     "code_verifier": verifier, "redirect_uri": redirect_uri, "resource": RESOURCE})
    if r.status_code >= 400:
        code_, detail = _oauth_error(r)
        err = ChatGPTError(f"ChatGPT turned down the sign-in ({detail or code_ or r.status_code}). Please try again.")
        err.code = code_  # type: ignore[attr-defined]
        raise err
    tok = r.json()
    if not isinstance(tok, dict) or not tok.get("access_token"):
        raise ChatGPTError("ChatGPT didn't send a sign-in back. Please try again.")
    return tok


def has_plan_scope(scope: str | None) -> bool:
    return PLAN_SCOPE in (scope or "").split()


async def access_token(config: SentientConfig, *, force_refresh: bool = False) -> str:
    """A current access token, refreshed when it is about to run out (or when ``force_refresh``)."""
    tokens = load_tokens()
    if tokens is None:
        raise ChatGPTError("Sign in with ChatGPT first, in Settings > Models.")
    if not force_refresh and tokens["expires_at"] - time.time() > REFRESH_MARGIN_S:
        return tokens["access_token"]
    async with _lock():
        current = load_tokens()  # another request may have refreshed while this one waited
        if current is None:
            raise ChatGPTError("Sign in with ChatGPT first, in Settings > Models.")
        if current["access_token"] != tokens["access_token"] and current["expires_at"] - time.time() > REFRESH_MARGIN_S:
            return current["access_token"]
        return (await _refresh(current))["access_token"]


async def _refresh(tokens: dict) -> dict:
    if not tokens.get("refresh_token"):
        delete_json(SECRET)
        raise ChatGPTError("Your ChatGPT sign-in ran out. Sign in with ChatGPT again in Settings > Models.")
    r = await _post_form(TOKEN_URL, {"grant_type": "refresh_token", "client_id": tokens["client_id"],
                                     "refresh_token": tokens["refresh_token"], "resource": RESOURCE})
    if r.status_code >= 400:
        code, _detail = _oauth_error(r)
        if code in DEAD_REFRESH or r.status_code in {400, 401}:
            delete_json(SECRET)
            raise ChatGPTError("ChatGPT signed you out. Sign in with ChatGPT again in Settings > Models.")
        raise ChatGPTError(f"ChatGPT couldn't renew the sign-in right now ({r.status_code}). Try again in a moment.")
    record = _record(tokens["client_id"], r.json(), tokens)
    if not save_json(SECRET, record):
        raise ChatGPTError("Your system keychain is unavailable, so the ChatGPT sign-in can't be kept.")
    return record


async def sign_out() -> None:
    """Revoke the refresh token (best effort) and remove the sign-in from the keychain."""
    tokens = load_json(SECRET)
    if tokens and tokens.get("refresh_token") and tokens.get("client_id"):
        for _ in range(2):  # an empty 200 means done, even for a token that was already invalid
            try:
                r = await _post_form(REVOKE_URL, {"token": tokens["refresh_token"], "token_type_hint": "refresh_token",
                                                  "client_id": tokens["client_id"]})
            except ChatGPTError:
                continue
            if r.status_code < 500:
                break
    delete_json(SECRET)


# ---------------------------------------------------------------------------- ID token
def _b64(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


async def verify_id_token(id_token: str, *, client_id: str, nonce: str) -> dict:
    """Check the ID token's RS256 signature against OpenAI's published keys, then issuer, audience, expiry and nonce."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    bad = ChatGPTError("ChatGPT's sign-in reply couldn't be verified. Please try again.")
    try:
        head_b64, body_b64, sig_b64 = id_token.split(".")
        header = json.loads(_b64(head_b64))
        claims = json.loads(_b64(body_b64))
    except (ValueError, TypeError) as exc:
        raise bad from exc
    if header.get("alg") != "RS256":
        raise bad
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as http:
            keys = (await http.get(JWKS_URL)).json().get("keys") or []
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        raise ChatGPTError("Couldn't reach ChatGPT to check the sign-in. Please try again.") from exc
    jwk = next((k for k in keys if k.get("kid") == header.get("kid") and k.get("kty") == "RSA"), None)
    if jwk is None:
        raise bad
    key = rsa.RSAPublicNumbers(int.from_bytes(_b64(jwk["e"]), "big"), int.from_bytes(_b64(jwk["n"]), "big")).public_key()
    try:
        key.verify(_b64(sig_b64), f"{head_b64}.{body_b64}".encode(), padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature as exc:
        raise bad from exc
    aud = claims.get("aud")
    auds = aud if isinstance(aud, list) else [aud]
    if (claims.get("iss") != ISSUER or client_id not in auds or claims.get("nonce") != nonce
            or not isinstance(claims.get("exp"), int | float) or claims["exp"] < time.time() - 60):
        raise bad
    return claims


# ---------------------------------------------------------------------------- the browser sign-in
async def host_id(store: Any) -> str:
    """``ext_agent_host_id``: a random id for this computer, made once before the first sign-in."""
    value = await store.get_meta(HOST_META)
    if not value:
        value = f"urn:uuid:{uuid.uuid4()}"
        await store.set_meta(HOST_META, value)
    return value


async def finish_sign_in(store: Any, flow: dict, params: dict[str, str]) -> str:
    """The browser came back with a code: swap it for tokens, check them and keep them. Returns the account email."""
    issued = params.get("client_id") or ""
    if flow["registering"]:
        if not issued or issued == DYNAMIC_CLIENT:
            raise ChatGPTError("ChatGPT didn't finish registering Sentient. Please try again.")
        client_id = issued
    else:
        client_id = flow["client_id"]
        if issued and issued != client_id:
            raise ChatGPTError("ChatGPT answered for a different app. Please try again.")
    try:
        tok = await exchange_code(client_id=client_id, code=params["code"], verifier=flow["verifier"],
                                  redirect_uri=flow["redirect_uri"])
    except ChatGPTError as exc:
        if getattr(exc, "code", "") == "invalid_client" and not flow["registering"]:
            await store.set_meta(CLIENT_META, "")  # the saved registration is gone: register again next time
        raise
    if not has_plan_scope(tok.get("scope") or params.get("scope")):
        raise ChatGPTError("ChatGPT didn't allow Sentient to use your plan. Plan usage needs ChatGPT Plus or Pro; "
                           "try again and allow it.")
    claims: dict = {}
    if tok.get("id_token"):
        claims = await verify_id_token(tok["id_token"], client_id=client_id, nonce=flow["nonce"])
    if flow["registering"]:
        await store.set_meta(CLIENT_META, client_id)
    record = _record(client_id, {**tok, "scope": tok.get("scope") or params.get("scope")})
    record["email"] = claims.get("email") if isinstance(claims.get("email"), str) else None
    if not save_json(SECRET, record):
        raise ChatGPTError("Your system keychain is unavailable, so the sign-in can't be saved.")
    return record["email"] or ""


# ---------------------------------------------------------------------------- the plan's models
async def list_models(config: SentientConfig) -> list[dict]:
    """The plan's models in OpenAI's order as ``[{id, label, free, tools, context_length}]`` (listed ones only)."""
    token = await access_token(config)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as http:
            r = await http.get(models_url(config), headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        raise ChatGPTError("Couldn't reach ChatGPT. Check your internet connection.") from exc
    if r.status_code == 401:
        raise ChatGPTError("ChatGPT signed you out. Sign in with ChatGPT again in Settings > Models.")
    if r.status_code >= 400:
        raise ChatGPTError(f"ChatGPT answered with an error ({r.status_code}): {_oauth_error(r)[1]}")
    try:
        data = r.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        data = {}
    out = []
    for m in data.get("models") or data.get("data") or []:
        slug = m.get("slug") or m.get("id")
        if not slug or m.get("visibility", "list") != "list":
            continue
        out.append({"id": f"{PREFIX}/{slug}", "label": m.get("display_name") or slug, "free": False, "tools": True,
                    "context_length": m.get("context_window") or m.get("context_length")})
    return out


def pick_models(models: list[str]) -> dict[str, str] | None:
    """A main and a fast model from the plan's list: the first listed, and the first mini or nano one (or the same)."""
    if not models:
        return None
    fast = next((m for m in models if any(w in m.lower() for w in ("mini", "nano"))), models[0])
    return {"main": models[0], "fast": fast}
