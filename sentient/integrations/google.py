"""Google installed-app OAuth 2.0 (PKCE + loopback redirect) and authenticated API calls.

The user creates one OAuth "Desktop app" client in Google Cloud Console; its id and
secret are stored once in the keychain as ``google_oauth_client`` and reused for every
Google integration. Each integration gets its own token with only its own scopes.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import html
import logging
import secrets as pysecrets
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from sentient.integrations.base import IntegrationError, NotConnected, SetupField, manager_from
from sentient.integrations.common import http_client, load_secret_json, store_secret_json
from sentient.tools.base import ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

log = logging.getLogger(__name__)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
CLIENT_SECRET_NAME = "google_oauth_client"
CALLBACK_PATH = "/oauth/callback"

BASE_SCOPES = ["openid", "email"]
SCOPES: dict[str, list[str]] = {
    "gmail": ["https://www.googleapis.com/auth/gmail.modify", "https://www.googleapis.com/auth/gmail.send"],
    "gcalendar": ["https://www.googleapis.com/auth/calendar"],
    "gdrive": ["https://www.googleapis.com/auth/drive"],
    "gdocs": ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive.file"],
    "gsheets": ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"],
    "gslides": ["https://www.googleapis.com/auth/presentations", "https://www.googleapis.com/auth/drive.file"],
    "gpeople": ["https://www.googleapis.com/auth/contacts"],
}

GOOGLE_DOCS_URL = "https://developers.google.com/workspace/guides/create-credentials#desktop-app"


def google_instructions(api_name: str, api_slug: str) -> str:
    return (
        "Sentient talks to Google directly from your computer. Google needs a free \"OAuth client\" that "
        "belongs to you. You only do this once; every Google integration reuses it.\n\n"
        "1. Open https://console.cloud.google.com/ and sign in with your Google account.\n"
        "2. At the top, click the project picker, then **New project**. Name it `Sentient` and click **Create**. "
        "Make sure the new project is selected.\n"
        f"3. Open https://console.cloud.google.com/apis/library/{api_slug} and click **Enable** to turn on the "
        f"**{api_name}**.\n"
        "4. Open **Google Auth Platform → Overview** (https://console.cloud.google.com/auth/overview). Click "
        "**Get started**, enter `Sentient` as the app name and your email, choose **External** as the audience, "
        "and finish.\n"
        "5. Open **Audience**, click **Add users** and add your own Google address as a test user.\n"
        "6. Open **Clients** (https://console.cloud.google.com/auth/clients), click **Create client**, choose "
        "**Desktop app**, name it `Sentient`, and click **Create**.\n"
        "7. Copy the **Client ID** and **Client secret** into the fields here and click **Connect**. If you "
        "already connected another Google integration, you can leave them empty.\n"
        "8. Your browser opens a Google sign-in page. Pick your account. Google may warn that the app isn't "
        "verified: click **Continue**, because the app is yours. Allow the permissions.\n"
        "9. When the page says **You can close this tab**, come back to Sentient. Done.\n"
    )


def google_setup_fields() -> list[SetupField]:
    return [
        SetupField("client_id", "OAuth Client ID", secret=False, required=True,
                   help="From Google Cloud Console → Clients → your Desktop app client.",
                   placeholder="1234-abc.apps.googleusercontent.com"),
        SetupField("client_secret", "OAuth Client secret", secret=True, required=True,
                   help="Shown next to the Client ID. Stored in your system keychain."),
    ]


def get_client() -> dict | None:
    c = load_secret_json(CLIENT_SECRET_NAME)
    if c and c.get("client_id") and c.get("client_secret"):
        return c
    return None


def save_client(client_id: str, client_secret: str) -> None:
    data = {"client_id": client_id.strip(), "client_secret": client_secret.strip()}
    if not store_secret_json(CLIENT_SECRET_NAME, data):
        raise IntegrationError("Your system keychain is unavailable, so the Google client can't be saved.")


def pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(pysecrets.token_bytes(48)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def build_auth_url(client_id: str, redirect_uri: str, scopes: list[str], state: str, challenge: str) -> str:
    q = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(dict.fromkeys([*BASE_SCOPES, *scopes])),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{AUTH_URL}?{urlencode(q)}"


def _token_record(data: dict, previous: dict | None = None) -> dict:
    rec = dict(previous or {})
    rec["access_token"] = data["access_token"]
    rec["expires_at"] = time.time() + float(data.get("expires_in", 3600))
    rec["token_type"] = data.get("token_type", "Bearer")
    if data.get("refresh_token"):
        rec["refresh_token"] = data["refresh_token"]
    if data.get("scope"):
        rec["scope"] = data["scope"]
    return rec


def _oauth_error(r: httpx.Response) -> str:
    try:
        body = r.json()
        return str(body.get("error_description") or body.get("error") or r.text)
    except Exception:
        return r.text[:300]


async def exchange_code(code: str, verifier: str, redirect_uri: str) -> dict:
    client = get_client()
    if client is None:
        raise IntegrationError("The Google OAuth client is missing. Enter the Client ID and secret again.")
    async with http_client() as http:
        r = await http.post(TOKEN_URL, data={
            "client_id": client["client_id"], "client_secret": client["client_secret"], "code": code,
            "code_verifier": verifier, "grant_type": "authorization_code", "redirect_uri": redirect_uri,
        })
    if r.status_code >= 400:
        raise IntegrationError(f"Google rejected the sign-in: {_oauth_error(r)}")
    return _token_record(r.json())


async def fetch_email(token: str) -> str | None:
    try:
        async with http_client() as http:
            r = await http.get(USERINFO_URL, headers={"Authorization": f"Bearer {token}"})
        if r.status_code >= 400:
            return None
        return r.json().get("email")
    except httpx.HTTPError:
        return None


async def access_token(mgr: IntegrationManager, plugin_id: str, *, force_refresh: bool = False) -> str:
    rec = await mgr.get_credentials(plugin_id)
    if not rec or not rec.get("access_token"):
        raise NotConnected(plugin_id)
    if not force_refresh and float(rec.get("expires_at", 0)) > time.time() + 60:
        return rec["access_token"]
    async with mgr.lock(f"google-refresh:{plugin_id}"):
        rec = await mgr.get_credentials(plugin_id) or rec
        if not force_refresh and float(rec.get("expires_at", 0)) > time.time() + 60:
            return rec["access_token"]
        client = get_client()
        if client is None or not rec.get("refresh_token"):
            await mgr.set_error(plugin_id, "Google sign-in expired. Reconnect this integration.")
            raise IntegrationError("Google sign-in expired. Reconnect it from Integrations.")
        async with http_client() as http:
            r = await http.post(TOKEN_URL, data={
                "client_id": client["client_id"], "client_secret": client["client_secret"],
                "refresh_token": rec["refresh_token"], "grant_type": "refresh_token",
            })
        if r.status_code in (400, 401):
            await mgr.set_error(plugin_id, "Google sign-in expired or was revoked. Reconnect this integration.")
            raise IntegrationError("Google sign-in expired or was revoked. Reconnect it from Integrations.")
        r.raise_for_status()
        new = _token_record(r.json(), rec)
        await mgr.store_credentials(plugin_id, new)
        return new["access_token"]


async def gapi(ctx: ToolContext | None, plugin_id: str, method: str, url: str, *, params: dict | None = None,
               json: Any = None, content: bytes | None = None, headers: dict | None = None, raw: bool = False,
               mgr: IntegrationManager | None = None) -> Any:
    """Authenticated Google API call with one automatic refresh-and-retry on 401."""
    mgr = mgr or manager_from(ctx)
    if not await mgr.is_connected(plugin_id):
        raise NotConnected(plugin_id)
    resp: httpx.Response | None = None
    for attempt in range(2):
        token = await access_token(mgr, plugin_id, force_refresh=attempt == 1)
        h = {"Authorization": f"Bearer {token}", **(headers or {})}
        async with http_client(timeout=60) as http:
            resp = await http.request(method, url, params=params, json=json, content=content, headers=h)
        if resp.status_code != 401:
            break
    assert resp is not None
    resp.raise_for_status()
    if raw:
        return resp
    if resp.status_code == 204 or not resp.content:
        return {}
    return resp.json()


# ---------------------------------------------------------------------------- loopback listener
CallbackFn = Callable[[dict[str, str]], Awaitable[tuple[bool, str]]]

_PAGE = (
    "<!doctype html><html><head><meta charset=\"utf-8\"><title>Sentient</title><style>"
    "body{{font-family:system-ui,sans-serif;background:#0f0f14;color:#eee;display:flex;align-items:center;"
    "justify-content:center;height:100vh;margin:0}}.c{{text-align:center;max-width:28rem;padding:2rem}}"
    "h1{{font-size:1.4rem}}p{{color:#aaa}}</style></head><body><div class=\"c\"><h1>{title}</h1>"
    "<p>{body}</p></div></body></html>"
)
_REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}


class LoopbackListener:
    """Tiny asyncio HTTP server on 127.0.0.1 that receives OAuth redirects."""

    def __init__(self, callback: CallbackFn):
        self._callback = callback
        self._server: asyncio.Server | None = None
        self.port: int = 0

    @property
    def running(self) -> bool:
        return self._server is not None

    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}{CALLBACK_PATH}"

    async def start(self, port: int = 0) -> int:
        if self._server is not None:
            return self.port
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", port)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 2)
            self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            line = head.split(b"\r\n", 1)[0].decode("latin-1")
            parts = line.split(" ")
            split = urlsplit(parts[1] if len(parts) > 1 else "/")
            if split.path != CALLBACK_PATH:
                await self._respond(writer, 404, "Not found", "Nothing here.")
                return
            params = {k: v[0] for k, v in parse_qs(split.query).items()}
            ok, message = await self._callback(params)
            if ok:
                await self._respond(writer, 200, "You're connected", html.escape(message) + " You can close this tab.")
            else:
                await self._respond(writer, 400, "Connection failed",
                                    html.escape(message) + " You can close this tab and try again in Sentient.")
        except Exception as exc:  # never crash the listener
            log.warning("oauth callback handling failed: %s", exc)
            with contextlib.suppress(Exception):
                await self._respond(writer, 500, "Something went wrong", "Please try connecting again from Sentient.")
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    @staticmethod
    async def _respond(writer: asyncio.StreamWriter, status: int, title: str, body: str) -> None:
        payload = _PAGE.format(title=html.escape(title), body=body).encode()
        writer.write(
            f"HTTP/1.1 {status} {_REASONS[status]}\r\nContent-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode() + payload
        )
        await writer.drain()
