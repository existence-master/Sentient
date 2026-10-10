"""Remote MCP servers with headers and OAuth sign-in, against in-process mock servers (no network)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets as pysecrets
from urllib.parse import parse_qs, urlsplit

import httpx
import httpx2
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from sentient import paths
from sentient.integrations import mcp as mcp_mod
from sentient.integrations import mcp_auth

BASE = "http://mock.test"
URL = f"{BASE}/mcp"


class MockServer:
    """A tiny MCP server (JSON responses) that requires a bearer token, plus an OAuth authorization server."""

    def __init__(self, *, oauth: bool, static_token: str = "static-secret"):
        self.oauth = oauth
        self.valid = set() if oauth else {static_token}
        self.refresh_tokens: set[str] = set()
        self.codes: dict[str, dict] = {}
        self.clients: dict[str, dict] = {}
        self.expires_in = 3600
        self.token_calls: list[dict] = []
        self.issued: list[str] = []
        self.seen_tokens: list[str] = []
        self.app = Starlette(routes=[
            Route("/mcp", self.mcp, methods=["GET", "POST", "DELETE"]),
            Route("/.well-known/oauth-protected-resource/mcp", self.prm),
            Route("/.well-known/oauth-authorization-server", self.asm),
            Route("/register", self.register, methods=["POST"]),
            Route("/token", self.token, methods=["POST"]),
        ])

    def transport(self) -> httpx2.ASGITransport:
        return httpx2.ASGITransport(app=self.app)

    # ---------------------------------------------------------------- MCP endpoint
    async def mcp(self, request: Request) -> Response:
        if request.method == "GET":
            return Response(status_code=405)
        if request.method == "DELETE":
            return Response(status_code=200)
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if token not in self.valid:
            headers = {}
            if self.oauth:
                headers["WWW-Authenticate"] = f'Bearer resource_metadata="{BASE}/.well-known/oauth-protected-resource/mcp"'
            return Response(status_code=401, headers=headers)
        self.seen_tokens.append(token)
        msg = json.loads(await request.body())
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:
            return Response(status_code=202)
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "mock", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "hello", "description": "Say hello.", "inputSchema": {"type": "object"},
                                 "annotations": {"readOnlyHint": True}}]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": "hi there"}]}
        else:
            return JSONResponse({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "unknown"}})
        return JSONResponse({"jsonrpc": "2.0", "id": mid, "result": result})

    # ---------------------------------------------------------------- OAuth
    async def prm(self, request: Request) -> Response:
        if not self.oauth:
            return Response(status_code=404)
        return JSONResponse({"resource": URL, "authorization_servers": [BASE]})

    async def asm(self, request: Request) -> Response:
        if not self.oauth:
            return Response(status_code=404)
        return JSONResponse({
            "issuer": BASE, "authorization_endpoint": f"{BASE}/authorize", "token_endpoint": f"{BASE}/token",
            "registration_endpoint": f"{BASE}/register", "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
        })

    async def register(self, request: Request) -> Response:
        if not self.oauth:
            return Response(status_code=404)
        body = await request.json()
        client_id = f"client-{len(self.clients) + 1}"
        self.clients[client_id] = body
        return JSONResponse({**body, "client_id": client_id, "token_endpoint_auth_method": "none"}, status_code=201)

    def browser_approves(self, auth_url: str) -> tuple[str, str, str]:
        """What the provider's sign-in page does: remember the PKCE challenge and hand out a code."""
        q = {k: v[0] for k, v in parse_qs(urlsplit(auth_url).query).items()}
        assert q["client_id"] in self.clients and q["code_challenge_method"] == "S256"
        assert q["redirect_uri"] in self.clients[q["client_id"]]["redirect_uris"]
        code = pysecrets.token_urlsafe(8)
        self.codes[code] = q
        return q["redirect_uri"], code, q["state"]

    def _issue(self) -> dict:
        access, refresh = f"at-{pysecrets.token_hex(4)}", f"rt-{pysecrets.token_hex(4)}"
        self.valid.add(access)
        self.issued.append(access)
        self.refresh_tokens.add(refresh)
        return {"access_token": access, "token_type": "bearer", "expires_in": self.expires_in, "refresh_token": refresh}

    async def token(self, request: Request) -> Response:
        form = dict(await request.form())
        self.token_calls.append(form)
        if form.get("grant_type") == "authorization_code":
            req = self.codes.pop(form.get("code", ""), None)
            if req is None or req["client_id"] != form.get("client_id") or req["redirect_uri"] != form.get("redirect_uri"):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            digest = hashlib.sha256(form.get("code_verifier", "").encode()).digest()
            if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != req["code_challenge"]:
                return JSONResponse({"error": "invalid_grant", "error_description": "PKCE check failed"}, status_code=400)
            return JSONResponse(self._issue())
        if form.get("grant_type") == "refresh_token":
            if form.get("refresh_token") not in self.refresh_tokens:
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            self.refresh_tokens.discard(form["refresh_token"])
            return JSONResponse(self._issue())
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)


async def _wait(app, name: str, status: str, timeout: float = 20) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    server: dict = {}
    while loop.time() < deadline:
        server = next(s for s in app.integrations.mcp.list() if s["name"] == name)
        if server["status"] == status:
            return server
        await asyncio.sleep(0.05)
    raise AssertionError(f"{name} never reached {status}: {server}")


def _tokens(keychain: dict, name: str) -> dict:
    return mcp_auth.load_json(f"mcp:{name}:oauth") or {}


async def _sign_in(app, mock: MockServer, name: str) -> httpx.Response:
    started = await app.integrations.mcp.sign_in(name)
    assert set(started) == {"auth_url", "state"}
    redirect, code, state = mock.browser_approves(started["auth_url"])
    assert state == started["state"] and redirect.startswith("http://127.0.0.1:")
    async with httpx.AsyncClient() as http:
        return await http.get(redirect, params={"code": code, "state": state}, timeout=30)


# ---------------------------------------------------------------------------- headers
async def test_header_auth_server_uses_keychain_header(app, ctx, keychain):
    mock = MockServer(oauth=False)
    app.integrations.mcp.http_transport = mock.transport()

    added = await app.integrations.mcp.add(
        "Static", {"transport": "http", "url": URL, "headers": {"Authorization": "Bearer static-secret"}})
    assert added["status"] == "connected" and added["auth"] == "headers"
    assert added["header_keys"] == ["Authorization"] and added["signed_in"] is False
    assert [t["mcp_name"] for t in added["tools"]] == ["hello"]
    assert (await app.registry.get("mcp_static_hello").call(ctx, {}))["content"] == "hi there"
    assert set(mock.seen_tokens) == {"static-secret"}

    stored = app.config.integrations.mcp_servers["Static"]
    assert stored["header_keys"] == ["Authorization"] and "headers" not in stored
    assert "static-secret" not in paths.config_file().read_text(encoding="utf-8")
    assert "static-secret" in keychain["mcp:Static:headers"]

    assert (await app.integrations.mcp.test("Static"))["ok"] is True
    await app.integrations.mcp.remove("Static")
    assert not any(k.startswith("mcp:Static") for k in keychain)


async def test_401_shows_needs_sign_in(app, keychain):
    mock = MockServer(oauth=False)
    app.integrations.mcp.http_transport = mock.transport()

    wrong = await app.integrations.mcp.add(
        "Wrong", {"transport": "http", "url": URL, "headers": {"Authorization": "Bearer nope"}})
    assert wrong["status"] == "needs_sign_in" and "headers" in wrong["error"]
    bare = await app.integrations.mcp.add("Bare", {"transport": "http", "url": URL})
    assert bare["status"] == "needs_sign_in" and bare["auth"] == "none"
    assert bare["error"] == "This server asks you to sign in."
    tested = await app.integrations.mcp.test("Bare")
    assert tested == {"ok": False, "tools": [], "error": "This server asks you to sign in."}
    # a server without the OAuth endpoints can't be signed in to: a plain explanation, no browser
    with pytest.raises(ValueError, match="sign in on their own"):
        await app.integrations.mcp.sign_in("Bare")


async def test_header_validation(app):
    with pytest.raises(ValueError, match="at least one header"):
        await app.integrations.mcp.add("x", {"transport": "http", "url": URL, "auth": "headers"})
    with pytest.raises(ValueError, match="valid header name"):
        await app.integrations.mcp.add("x", {"transport": "http", "url": URL, "headers": {"Bad Name": "v"}})
    with pytest.raises(ValueError, match="only for remote"):
        await app.integrations.mcp.add("x", {"transport": "stdio", "command": "x", "headers": {"A": "b"}})


async def test_missing_values_are_filled_in_and_used_on_reconnect(app, ctx, keychain):
    """An imported server lists header and environment names without values; "Add values" fills them in."""
    mock = MockServer(oauth=False)
    app.integrations.mcp.http_transport = mock.transport()
    mcp = app.integrations.mcp
    notes = await mcp.import_server("Notes", {"transport": "http", "url": URL.replace("http://", "https://"),
                                              "auth": "headers", "header_keys": ["Authorization"]})
    assert notes["missing_values"] == ["Authorization"] and notes["status"] == "disabled"
    local = await mcp.import_server("Local", {"transport": "stdio", "command": "npx", "args": ["-y", "some-mcp"],
                                              "env_keys": ["API_TOKEN", "REGION"]})
    assert local["missing_values"] == ["API_TOKEN", "REGION"]

    with pytest.raises(ValueError, match="isn't one of this server's headers"):
        await mcp.set_values("Notes", {"X-Other": "v"})
    with pytest.raises(ValueError, match="single-line"):
        await mcp.set_values("Notes", {"Authorization": "Bearer a\nb"})
    with pytest.raises(KeyError):
        await mcp.set_values("Missing", {})
    # a key is never sent to a plain http:// address on the network (this computer is fine)
    await mcp.import_server("Plain", {"transport": "http", "url": URL, "auth": "headers", "header_keys": ["Authorization"]})
    with pytest.raises(ValueError, match="https://"):
        await mcp.set_values("Plain", {"Authorization": "Bearer static-secret"})
    assert "mcp:Plain:headers" not in keychain
    await mcp.import_server("Here", {"transport": "http", "url": "http://127.0.0.1:9/mcp", "header_keys": ["X-Key"]})
    assert (await mcp.set_values("Here", {"X-Key": "local-only"}))["missing_values"] == []

    filled = await mcp.set_values("Notes", {"Authorization": "Bearer static-secret"}, enable=True)
    assert filled["status"] == "connected" and filled["missing_values"] == [] and filled["enabled"] is True
    assert set(mock.seen_tokens) == {"static-secret"}
    assert (await app.registry.get("mcp_notes_hello").call(ctx, {}))["content"] == "hi there"
    assert "static-secret" in keychain["mcp:Notes:headers"]
    # a blank value keeps the saved one; the server reconnects with it
    mock.seen_tokens.clear()
    again = await mcp.set_values("Notes", {"Authorization": "  "})
    assert again["status"] == "connected" and set(mock.seen_tokens) == {"static-secret"}

    partly = await mcp.set_values("Local", {"API_TOKEN": "tok-123"})
    assert partly["missing_values"] == ["REGION"] and partly["status"] == "disabled"
    assert "tok-123" in keychain["mcp:Local"]
    saved = paths.config_file().read_text(encoding="utf-8")
    assert "static-secret" not in saved and "tok-123" not in saved
    assert "headers" not in app.config.integrations.mcp_servers["Notes"]
    assert "env" not in app.config.integrations.mcp_servers["Local"]


# ---------------------------------------------------------------------------- OAuth
async def test_oauth_sign_in_with_pkce_and_dynamic_registration(app, ctx, keychain):
    mock = MockServer(oauth=True)
    app.integrations.mcp.http_transport = mock.transport()

    added = await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL})
    assert added["status"] == "needs_sign_in"

    async with httpx.AsyncClient() as http:
        started = await app.integrations.mcp.sign_in("Notes")
        assert next(s for s in app.integrations.mcp.list() if s["name"] == "Notes")["signing_in"] is True
        redirect, code, state = mock.browser_approves(started["auth_url"])
        bad = await http.get(redirect, params={"code": code, "state": "forged"})
        ok = await http.get(redirect, params={"code": code, "state": state}, timeout=30)
    assert bad.status_code == 400 and "expired" in bad.text
    assert ok.status_code == 200 and "Notes is connected" in ok.text

    exchange = next(c for c in mock.token_calls if c["grant_type"] == "authorization_code")
    assert exchange["resource"] == URL and len(exchange["code_verifier"]) >= 43
    assert list(mock.clients) == ["client-1"] and mock.clients["client-1"]["client_name"] == "Sentient"

    server = await _wait(app, "Notes", "connected")
    assert server["auth"] == "oauth" and server["signed_in"] is True and server["signing_in"] is False
    assert (await app.registry.get("mcp_notes_hello").call(ctx, {}))["content"] == "hi there"

    access = _tokens(keychain, "Notes")["tokens"]["access_token"]
    assert access in mock.valid and mock.seen_tokens[-1] == access
    assert app.config.integrations.mcp_servers["Notes"]["auth"] == "oauth"
    config_text = paths.config_file().read_text(encoding="utf-8")
    assert access not in config_text and "rt-" not in config_text and "client-1" not in config_text


async def test_oauth_refreshes_expired_and_rejected_tokens(app, keychain, monkeypatch):
    monkeypatch.setattr(mcp_mod, "MIN_BACKOFF_S", 0.05)
    mock = MockServer(oauth=True)
    mock.expires_in = 30  # inside the refresh margin: the stored token counts as expired at once
    app.integrations.mcp.http_transport = mock.transport()
    await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL, "auth": "oauth"})
    assert (await _wait(app, "Notes", "needs_sign_in"))["error"] == "Sign in to use this server."

    assert (await _sign_in(app, mock, "Notes")).status_code == 200
    conn = app.integrations.mcp.servers["Notes"]

    async def settle(calls: int) -> list[str]:
        for _ in range(200):
            if len(mock.token_calls) == calls and conn.status == "connected":
                break
            await asyncio.sleep(0.05)
        return [c["grant_type"] for c in mock.token_calls]

    # the runner reconnects with the stored token, which counts as expired, so it refreshes first
    assert await settle(2) == ["authorization_code", "refresh_token"]
    assert _tokens(keychain, "Notes")["tokens"]["access_token"] == mock.issued[1] == mock.seen_tokens[-1]

    mock.expires_in = 3600
    conn.broken.set()
    assert (await settle(3))[-1] == "refresh_token"
    assert _tokens(keychain, "Notes")["tokens"]["access_token"] == mock.issued[2] == mock.seen_tokens[-1]

    # a fresh token the server drops early: one refresh after the 401, then connected again without a browser
    conn.broken.set()
    await asyncio.sleep(0.1)
    await _wait(app, "Notes", "connected")
    assert len(mock.token_calls) == 3  # still valid: no refresh
    mock.valid.discard(mock.issued[2])
    conn.broken.set()
    assert (await settle(4))[-1] == "refresh_token" and conn.status == "connected"
    assert mock.seen_tokens[-1] == mock.issued[3]

    # refresh token revoked too: the server needs a new sign-in
    mock.valid.clear()
    mock.refresh_tokens.clear()
    conn.refresh_tried = False
    conn.broken.set()
    await _wait(app, "Notes", "needs_sign_in")


async def test_sign_out_clears_tokens(app, keychain):
    mock = MockServer(oauth=True)
    app.integrations.mcp.http_transport = mock.transport()
    await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL})
    await _sign_in(app, mock, "Notes")
    await _wait(app, "Notes", "connected")
    assert "mcp:Notes:oauth" in keychain

    out = await app.integrations.mcp.sign_out("Notes")
    assert out["signed_in"] is False and out["status"] == "needs_sign_in" and out["auth"] == "oauth"
    assert not any(k.startswith("mcp:Notes:oauth") for k in keychain)
    assert app.registry.get("mcp_notes_hello") is None

    await app.integrations.mcp.remove("Notes")
    assert not any(k.startswith("mcp:Notes") for k in keychain)


async def test_sign_in_cancelled_in_browser(app, keychain):
    mock = MockServer(oauth=True)
    app.integrations.mcp.http_transport = mock.transport()
    await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL})
    started = await app.integrations.mcp.sign_in("Notes")
    redirect, _code, state = mock.browser_approves(started["auth_url"])
    async with httpx.AsyncClient() as http:
        r = await http.get(redirect, params={"error": "access_denied", "state": state}, timeout=30)
    assert r.status_code == 400 and "You cancelled the sign-in." in r.text
    server = next(s for s in app.integrations.mcp.list() if s["name"] == "Notes")
    assert server["signing_in"] is False and server["signed_in"] is False and server["status"] == "needs_sign_in"
    assert server["error"] == "You cancelled the sign-in."
    assert "mcp:Notes:oauth" not in keychain


async def test_sign_in_only_for_remote_servers(app):
    await app.integrations.mcp.add("local", {"transport": "stdio", "command": "x", "enabled": False})
    with pytest.raises(ValueError, match="Only remote"):
        await app.integrations.mcp.sign_in("local")
    with pytest.raises(KeyError):
        await app.integrations.mcp.sign_in("missing")


def test_long_keychain_values_are_split(keychain):
    data = {"access_token": "x" * 4500}
    assert mcp_auth.save_json("mcp:big:oauth", data)
    assert {"mcp:big:oauth:1", "mcp:big:oauth:4"} <= set(keychain) and all(len(v) <= 1003 for v in keychain.values())
    assert mcp_auth.load_json("mcp:big:oauth") == data
    assert mcp_auth.save_json("mcp:big:oauth", {"a": 1}) and "mcp:big:oauth:1" not in keychain
    mcp_auth.save_json("mcp:big:oauth", data)
    mcp_auth.delete_json("mcp:big:oauth")
    assert keychain == {}
