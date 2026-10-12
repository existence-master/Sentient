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
from sentient.app import SentientApp
from sentient.config.schema import SentientConfig
from sentient.integrations import mcp as mcp_mod
from sentient.integrations import mcp_auth
from tests.conftest import FakeProvider

BASE = "http://mock.test"
URL = f"{BASE}/mcp"


class MockServer:
    """A tiny MCP server (JSON responses) that requires a bearer token, plus an OAuth authorization server."""

    def __init__(self, *, oauth: bool, static_token: str = "static-secret", token_path: str = "/token",
                 base: str = BASE):
        self.oauth = oauth
        self.token_path = token_path
        self.base = base
        self.url = f"{base}/mcp"
        self.metadata_status = 200  # what the well-known metadata addresses answer (500: a broken lookup)
        self.token_posts: list[str] = []  # the path of every token request
        self.metadata_calls = 0
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
            Route("/oauth2/token", self.token, methods=["POST"]),
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
                headers["WWW-Authenticate"] = f'Bearer resource_metadata="{self.base}/.well-known/oauth-protected-resource/mcp"'
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
        self.metadata_calls += 1
        if not self.oauth:
            return Response(status_code=404)
        if self.metadata_status != 200:
            return Response(status_code=self.metadata_status)
        return JSONResponse({"resource": self.url, "authorization_servers": [self.base]})

    async def asm(self, request: Request) -> Response:
        self.metadata_calls += 1
        if not self.oauth:
            return Response(status_code=404)
        if self.metadata_status != 200:
            return Response(status_code=self.metadata_status)
        return JSONResponse({
            "issuer": self.base, "authorization_endpoint": f"{self.base}/authorize", "token_endpoint": self.base + self.token_path,
            "registration_endpoint": f"{self.base}/register", "response_types_supported": ["code"],
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
        self.token_posts.append(request.url.path)
        if request.url.path != self.token_path:
            return Response(status_code=404)
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


def _tokens(keychain: dict, name: str, url: str | None = None) -> dict:
    target_url = url or URL
    return (mcp_auth.load_json(mcp_auth.tokens_secret(name, target_url))
            or mcp_auth.load_json(mcp_auth.tokens_secret(name, None)) or {})


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
    for url, ok in (("https://a.example/mcp", True), ("http://localhost:8/mcp", True), ("http://[::1]:8/mcp", True),
                    ("http://127.attacker.example/mcp", False), ("http://10.0.0.5/mcp", False), ("ftp://127.0.0.1/", False)):
        assert mcp_mod._protected_url(url) is ok, url

    # changes to one server run one at a time: a removal never leaves a relaunched server behind
    await mcp.import_server("Racy", {"transport": "stdio", "command": "npx", "env_keys": ["API_TOKEN"]})
    await asyncio.gather(mcp.set_values("Racy", {"API_TOKEN": "t"}), mcp.remove("Racy"), return_exceptions=True)
    assert "Racy" not in mcp.servers and "Racy" not in app.config.integrations.mcp_servers

    filled = await mcp.set_values("Notes", {"Authorization": "Bearer static-secret"}, enable=True)
    assert filled["status"] == "connected" and filled["missing_values"] == [] and filled["enabled"] is True
    assert set(mock.seen_tokens) == {"static-secret"}
    assert (await app.registry.get("mcp_notes_hello").call(ctx, {}))["content"] == "hi there"
    assert "static-secret" in keychain["mcp:Notes:headers"]
    # a blank value keeps the saved one; the server reconnects with it
    mock.seen_tokens.clear()
    again = await mcp.set_values("Notes", {"Authorization": "  "})
    assert again["status"] == "connected" and set(mock.seen_tokens) == {"static-secret"}

    partly = await mcp.set_values("Local", {"API_TOKEN": "tok-123"}, enable=True)  # REGION is still missing: stays off
    assert partly["missing_values"] == ["REGION"] and partly["status"] == "disabled" and partly["enabled"] is False
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
    seen = len(mock.seen_tokens)
    conn.broken.set()
    for _ in range(400):  # until it has reconnected (it used the token again), not just still connected from before
        if len(mock.seen_tokens) > seen and conn.status == "connected":
            break
        await asyncio.sleep(0.05)
    assert len(mock.seen_tokens) > seen and conn.status == "connected" and mock.seen_tokens[-1] == mock.issued[2]
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


META_KEYS = ("server_url", "auth_server_url", "oauth_metadata", "protected_resource_metadata")


async def _reconnect_expired(app, mock: MockServer, name: str) -> None:
    """Connect again with the stored token expired. Every connection builds its provider from the keychain alone,
    so this is what the first connection after an engine restart does."""
    conn = app.integrations.mcp.servers[name]
    posts, seen = len(mock.token_posts), len(mock.seen_tokens)
    assert mcp_auth.KeychainTokenStorage(name, url=conn.spec.get("url")).mark_expired()
    conn.broken.set()
    for _ in range(400):
        if len(mock.token_posts) > posts and len(mock.seen_tokens) > seen and conn.status == "connected":
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"{name} did not reconnect: {conn.status} {conn.error} {mock.token_posts}")


async def _signed_in(app, mock: MockServer, name: str = "Notes") -> None:
    app.integrations.mcp.http_transport = mock.transport()
    await app.integrations.mcp.add(name, {"transport": "http", "url": URL})
    assert (await _sign_in(app, mock, name)).status_code == 200
    await _wait(app, name, "connected")


async def test_refresh_after_restart_uses_the_saved_token_endpoint(app, keychain, monkeypatch):
    """Issue #267: the SDK keeps the authorization server's details in memory only, so after a restart a refresh
    went to <server>/token. The token record now keeps them, and the refresh goes to the real token endpoint."""
    monkeypatch.setattr(mcp_mod, "MIN_BACKOFF_S", 0.05)
    mock = MockServer(oauth=True, token_path="/oauth2/token")
    await _signed_in(app, mock)
    rec = _tokens(keychain, "Notes")
    assert rec["server_url"] == URL and rec["oauth_metadata"]["token_endpoint"] == f"{BASE}/oauth2/token"
    assert rec["auth_server_url"].rstrip("/") == BASE and rec["protected_resource_metadata"]["resource"] == URL

    looked = mock.metadata_calls
    await _reconnect_expired(app, mock, "Notes")
    assert mock.token_posts == ["/oauth2/token", "/oauth2/token"]
    assert [c["grant_type"] for c in mock.token_calls] == ["authorization_code", "refresh_token"]
    assert mock.token_calls[-1]["resource"] == URL  # as at sign-in
    assert mock.metadata_calls == looked  # the saved copy was used, nothing looked up
    rec = _tokens(keychain, "Notes")
    assert rec["tokens"]["access_token"] == mock.issued[1] == mock.seen_tokens[-1]
    assert rec["oauth_metadata"]["token_endpoint"] == f"{BASE}/oauth2/token"  # still kept after the refresh

    # a second restart works the same way
    await _reconnect_expired(app, mock, "Notes")
    assert mock.token_posts[-1] == "/oauth2/token" and mock.seen_tokens[-1] == mock.issued[2]


async def test_refresh_looks_up_the_token_endpoint_for_an_older_record(app, keychain, monkeypatch):
    """A sign-in saved before the record kept the server's details: they are looked up once, then saved."""
    monkeypatch.setattr(mcp_mod, "MIN_BACKOFF_S", 0.05)
    mock = MockServer(oauth=True, token_path="/oauth2/token")
    await _signed_in(app, mock)
    rec = _tokens(keychain, "Notes")
    mcp_auth.save_json(mcp_auth.tokens_secret("Notes", URL), {k: v for k, v in rec.items() if k not in META_KEYS})

    looked = mock.metadata_calls
    await _reconnect_expired(app, mock, "Notes")
    assert mock.token_posts == ["/oauth2/token", "/oauth2/token"] and mock.metadata_calls > looked
    assert _tokens(keychain, "Notes")["oauth_metadata"]["token_endpoint"] == f"{BASE}/oauth2/token"
    assert mock.seen_tokens[-1] == mock.issued[1]

    looked = mock.metadata_calls
    await _reconnect_expired(app, mock, "Notes")
    assert mock.token_posts[-1] == "/oauth2/token" and mock.metadata_calls == looked


async def test_failed_lookup_keeps_the_old_behaviour(app, keychain, monkeypatch):
    """When the details can't be looked up, the SDK's own fallback address is used, as before."""
    monkeypatch.setattr(mcp_mod, "MIN_BACKOFF_S", 0.05)
    mock = MockServer(oauth=True)  # the token endpoint is the fallback address
    await _signed_in(app, mock)
    rec = _tokens(keychain, "Notes")
    mcp_auth.save_json(mcp_auth.tokens_secret("Notes", URL), {k: v for k, v in rec.items() if k not in META_KEYS})
    mock.metadata_status = 500

    await _reconnect_expired(app, mock, "Notes")
    assert mock.token_posts == ["/token", "/token"] and mock.seen_tokens[-1] == mock.issued[1]
    assert "oauth_metadata" not in _tokens(keychain, "Notes")

    # an address that can't be reached at all is ignored too
    def unreachable(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("offline", request=request)

    store = mcp_auth.KeychainTokenStorage("Notes")
    provider = mcp_auth.SentientOAuthProvider(URL, mcp_auth.client_metadata("http://127.0.0.1:1/cb"), store,
                                              discovery_transport=httpx2.MockTransport(unreachable))
    await provider._initialize()
    assert provider.context.current_tokens is not None and provider.context.oauth_metadata is None


async def test_saved_details_are_only_used_for_the_same_server_and_client(keychain):
    meta = {"issuer": BASE, "authorization_endpoint": f"{BASE}/authorize", "token_endpoint": f"{BASE}/oauth2/token",
            "response_types_supported": ["code"]}
    tokens = {"access_token": "at", "token_type": "Bearer", "refresh_token": "rt"}

    async def restored(rec: dict, client: dict | None = None) -> str | None:
        keychain.clear()
        target_url = rec.get("server_url") or URL
        tok_key = mcp_auth.tokens_secret("T", target_url)
        cli_key = mcp_auth.client_secret("T", target_url)
        mcp_auth.save_json(tok_key, {"tokens": tokens, "expires_at": 1.0, **rec})
        mcp_auth.save_json(cli_key, client or {"client_id": "c1", "redirect_uris": ["http://127.0.0.1:1/cb"]})

        def offline(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(404)

        provider = mcp_auth.SentientOAuthProvider(URL, mcp_auth.client_metadata("http://127.0.0.1:1/cb"),
                                                  mcp_auth.KeychainTokenStorage("T", url=URL),
                                                  discovery_transport=httpx2.MockTransport(offline))
        await provider._initialize()
        md = provider.context.oauth_metadata
        return str(md.token_endpoint) if md else None

    good = {"server_url": URL, "auth_server_url": BASE, "oauth_metadata": meta}
    assert await restored(good) == f"{BASE}/oauth2/token"
    # the server address changed: those details belong to the old one
    assert await restored({**good, "server_url": "http://other.test/mcp"}) is None
    # SEP-2352: a client registered with another authorization server is never sent to this one
    other = {"client_id": "c1", "redirect_uris": ["http://127.0.0.1:1/cb"], "issuer": "http://other.test"}
    assert await restored(good, other) is None
    # details that don't match the authorization server they were saved for
    assert await restored({**good, "auth_server_url": "http://other.test"}) is None


def test_saving_looked_up_details_never_loses_the_sign_in(keychain, monkeypatch):
    """The record grows with the details; if the keychain fails partway, the old record is put back."""
    from sentient import secrets

    tok_key = mcp_auth.tokens_secret("T", URL)
    rec = {"tokens": {"access_token": "at", "token_type": "Bearer", "refresh_token": "rt"}, "expires_at": 1.0}
    mcp_auth.save_json(tok_key, rec)
    meta = {"issuer": BASE, "authorization_endpoint": f"{BASE}/authorize", "token_endpoint": f"{BASE}/oauth2/token",
            "response_types_supported": ["code"], "scopes_supported": [f"scope-{i}" for i in range(200)]}
    store = mcp_auth.KeychainTokenStorage("T", url=URL)
    provider = mcp_auth.SentientOAuthProvider(URL, mcp_auth.client_metadata("http://127.0.0.1:1/cb"), store)
    provider._use_metadata(None, None, mcp_auth.OAuthMetadata.model_validate(meta))

    real_set = secrets.set_secret

    def flaky(name: str, value: str) -> bool:
        return False if name == f"{tok_key}:1" else real_set(name, value)

    monkeypatch.setattr(secrets, "set_secret", flaky)
    assert store.save_metadata() is False
    assert mcp_auth.load_json(tok_key) == rec
    monkeypatch.setattr(secrets, "set_secret", real_set)
    assert store.save_metadata() is True
    assert mcp_auth.load_json(tok_key)["oauth_metadata"]["token_endpoint"] == f"{BASE}/oauth2/token"


async def test_sign_out_clears_tokens(app, keychain):
    mock = MockServer(oauth=True)
    app.integrations.mcp.http_transport = mock.transport()
    await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL})
    await _sign_in(app, mock, "Notes")
    await _wait(app, "Notes", "connected")
    assert mcp_auth.tokens_secret("Notes", URL) in keychain

    out = await app.integrations.mcp.sign_out("Notes")
    assert out["signed_in"] is False and out["status"] == "needs_sign_in" and out["auth"] == "oauth"
    assert not any(k.startswith(f"mcp:Notes:{mcp_auth.url_hash(URL)}:oauth") for k in keychain)
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
    assert mcp_auth.tokens_secret("Notes", URL) not in keychain


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


async def test_another_setup_adding_the_same_server_keeps_the_sign_in(app, keychain):
    """The keychain is shared by every setup on the computer (#248): adding the same server elsewhere reuses it."""
    mock = MockServer(oauth=True)
    app.integrations.mcp.http_transport = mock.transport()
    await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL})
    await _sign_in(app, mock, "Notes")
    await _wait(app, "Notes", "connected")
    assert _tokens(keychain, "Notes")["server_url"] == URL
    assert mcp_auth.load_json(mcp_auth.client_secret("Notes", URL))["server_url"] == URL
    access = _tokens(keychain, "Notes")["tokens"]["access_token"]

    # a second setup has never seen "Notes": same name and address, so the sign-in stays and is used
    app.config.integrations.mcp_servers = {}
    added = await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL, "auth": "oauth"})
    assert added["status"] == "connected" and added["signed_in"] is True
    assert _tokens(keychain, "Notes")["tokens"]["access_token"] == access and len(mock.token_calls) == 1

    # a local server with the same name leaves the sign-in to the setup that uses it
    app.config.integrations.mcp_servers = {}
    await app.integrations.mcp.add("Notes", {"transport": "stdio", "command": "x", "enabled": False})
    assert mcp_auth.tokens_secret("Notes", URL) in keychain and mcp_auth.client_secret("Notes", URL) in keychain

    # another address under the same name: keyed by its own urlhash, so the existing address sign-in is kept (#269)
    await app.integrations.mcp.add("Notes", {"transport": "http", "url": f"{BASE}/other", "enabled": False})
    assert mcp_auth.tokens_secret("Notes", URL) in keychain


async def test_older_sign_in_without_an_address(app, keychain):
    """Untagged legacy tokens are never migrated across server URLs (CWE-522), while client registration is kept."""
    legacy_tokens = {"tokens": {"access_token": "old", "token_type": "bearer"}, "expires_at": None}
    legacy_client = {"client_id": "c-1", "redirect_uris": ["http://127.0.0.1:1/oauth/callback"]}
    mcp_auth.save_json("mcp:Notes:oauth", legacy_tokens)
    mcp_auth.save_json("mcp:Notes:client", legacy_client)

    added = await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL, "enabled": False})  # new to this setup
    assert added["signed_in"] is False and added["status"] == "disabled"
    # Legacy token is NOT copied to the URL-scoped key
    assert mcp_auth.tokens_secret("Notes", URL) not in keychain
    assert mcp_auth.KeychainTokenStorage("Notes", url=URL).has_tokens() is False
    # Legacy token is preserved in keychain for other setups
    assert mcp_auth.load_json("mcp:Notes:oauth") == legacy_tokens
    # Client registration metadata is copied
    cli = mcp_auth.load_json(mcp_auth.client_secret("Notes", URL))
    assert cli["client_id"] == legacy_client["client_id"]


async def test_tagged_client_record_still_loads():
    store = mcp_auth.KeychainTokenStorage("Tagged", url=URL)
    mcp_auth.save_json("mcp:Tagged:client", {"client_id": "c-9", "redirect_uris": ["http://127.0.0.1:1/cb"],
                                             "server_url": URL})
    info = await store.get_client_info()
    assert info is not None and info.client_id == "c-9"


# ---------------------------------------------------------------------------- Issue #269 tests
def test_canonical_url_and_url_hash():
    # default ports 80/443 stripped, trailing slashes removed, scheme/host lowercased
    assert mcp_auth.canonical_url("HTTP://EXAMPLE.COM:80/mcp/") == "http://example.com/mcp"
    assert mcp_auth.canonical_url("HTTPS://Example.COM:443/mcp/") == "https://example.com/mcp"
    assert mcp_auth.canonical_url("http://example.com:8080/mcp/") == "http://example.com:8080/mcp"
    assert mcp_auth.canonical_url("http://example.com/mcp?foo=1&bar=2") == "http://example.com/mcp?foo=1&bar=2"
    assert mcp_auth.canonical_url(None) == ""
    assert mcp_auth.canonical_url("") == ""

    # IPv6 bracketed literals: default and non-default ports
    assert mcp_auth.canonical_url("http://[::1]:80/mcp/") == "http://[::1]/mcp"
    assert mcp_auth.canonical_url("http://[::1]:8080/mcp/") == "http://[::1]:8080/mcp"
    assert mcp_auth.canonical_url("https://[2001:db8::1]:443/mcp/") == "https://[2001:db8::1]/mcp"
    assert mcp_auth.canonical_url("https://[2001:db8::1]:8443/mcp") == "https://[2001:db8::1]:8443/mcp"
    assert mcp_auth.canonical_url("http://user:pass@[::1]:80/mcp") == "http://user:pass@[::1]/mcp"

    # Malformed / non-numeric port is preserved safely without merging distinct endpoints
    assert mcp_auth.canonical_url("http://example.com:notaport/mcp") == "http://example.com:notaport/mcp"

    # url_hash is 12 hex chars and deterministic
    h1 = mcp_auth.url_hash("http://example.com/mcp")
    h2 = mcp_auth.url_hash("HTTP://example.com:80/mcp/")
    assert len(h1) == 12 and h1 == h2
    assert mcp_auth.url_hash(None) == ""
    assert mcp_auth.url_hash("") == ""

    # distinct URLs produce distinct hashes (including IPv6)
    h3 = mcp_auth.url_hash("http://example.com/other")
    h4 = mcp_auth.url_hash("http://other.com/mcp")
    h5 = mcp_auth.url_hash("http://[::1]:8080/mcp")
    h6 = mcp_auth.url_hash("http://[::1]:8081/mcp")
    assert len({h1, h3, h4, h5, h6}) == 5

    # secret key naming
    assert mcp_auth.tokens_secret("srv", "http://example.com/mcp") == f"mcp:srv:{h1}:oauth"
    assert mcp_auth.tokens_secret("srv", None) == "mcp:srv:oauth"
    assert mcp_auth.client_secret("srv", "http://example.com/mcp") == f"mcp:srv:{h1}:client"
    assert mcp_auth.client_secret("srv", None) == "mcp:srv:client"
    assert mcp_auth.headers_secret("srv") == "mcp:srv:headers"


def test_migration_legacy_credentials_verified_before_deletion(keychain):
    legacy_tok = {"tokens": {"access_token": "a1", "token_type": "Bearer"}, "expires_at": 100.0, "server_url": URL}
    legacy_cli = {"client_id": "c1", "redirect_uris": ["http://127.0.0.1:1/cb"], "server_url": URL}
    mcp_auth.save_json("mcp:Notes:oauth", legacy_tok)
    mcp_auth.save_json("mcp:Notes:client", legacy_cli)

    mcp_auth.migrate_server_credentials("Notes", URL)

    new_tok_key = mcp_auth.tokens_secret("Notes", URL)
    new_cli_key = mcp_auth.client_secret("Notes", URL)
    assert keychain.get(new_tok_key) is not None
    assert keychain.get(new_cli_key) is not None
    assert mcp_auth.load_json(new_tok_key) == legacy_tok
    assert mcp_auth.load_json(new_cli_key) == legacy_cli
    # Legacy keys are deleted only after verified write
    assert "mcp:Notes:oauth" not in keychain
    assert "mcp:Notes:client" not in keychain


def test_migration_failure_safety_and_retry(keychain, monkeypatch):
    from sentient import secrets

    legacy_tok = {"tokens": {"access_token": "a1", "token_type": "Bearer"}, "server_url": URL}
    mcp_auth.save_json("mcp:Notes:oauth", legacy_tok)

    new_tok_key = mcp_auth.tokens_secret("Notes", URL)
    real_set = secrets.set_secret

    # 1. Write failure: set_secret returns False
    monkeypatch.setattr(secrets, "set_secret", lambda name, val: False if name == new_tok_key else real_set(name, val))
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert "mcp:Notes:oauth" in keychain  # legacy key preserved on failure
    assert new_tok_key not in keychain

    # 2. Verification failure: data mismatch
    def corrupt_set(name: str, val: str) -> bool:
        if name == new_tok_key:
            keychain[name] = "corrupted"
            return True
        return real_set(name, val)

    monkeypatch.setattr(secrets, "set_secret", corrupt_set)
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert "mcp:Notes:oauth" in keychain  # legacy key still preserved

    # 3. Successful retry: writing and verification succeed
    monkeypatch.setattr(secrets, "set_secret", real_set)
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert "mcp:Notes:oauth" not in keychain  # deleted only upon verified destination
    assert mcp_auth.load_json(new_tok_key) == legacy_tok


def test_migration_existing_destination_conflicts_and_malformed_records(keychain):
    legacy_tok = {"tokens": {"access_token": "legacy_val", "token_type": "Bearer"}, "server_url": URL}
    legacy_cli = {"client_id": "legacy_client", "server_url": URL}

    # 1. Destination is malformed / missing tokens: legacy entry must NOT be deleted
    mcp_auth.save_json("mcp:Notes:oauth", legacy_tok)
    mcp_auth.save_json(mcp_auth.tokens_secret("Notes", URL), {"tokens": {}})
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert "mcp:Notes:oauth" in keychain

    # 2. Destination has conflicting valid tokens: legacy entry must NOT be deleted
    conflicting = {"tokens": {"access_token": "different_val", "token_type": "Bearer"}, "server_url": URL}
    mcp_auth.save_json(mcp_auth.tokens_secret("Notes", URL), conflicting)
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert "mcp:Notes:oauth" in keychain

    # 3. Destination matches legacy tokens: legacy entry is safely deleted
    matching = {**legacy_tok}
    mcp_auth.save_json(mcp_auth.tokens_secret("Notes", URL), matching)
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert "mcp:Notes:oauth" not in keychain

    # 4. Client info: destination is malformed (no client_id)
    mcp_auth.save_json("mcp:Notes:client", legacy_cli)
    mcp_auth.save_json(mcp_auth.client_secret("Notes", URL), {})
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert "mcp:Notes:client" in keychain

    # 5. Client info: destination has conflicting client_id
    mcp_auth.save_json(mcp_auth.client_secret("Notes", URL), {"client_id": "different_client"})
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert "mcp:Notes:client" in keychain

    # 6. Client info: destination matches
    mcp_auth.save_json(mcp_auth.client_secret("Notes", URL), legacy_cli)
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert "mcp:Notes:client" not in keychain


def test_migration_untagged_legacy_ambiguity(keychain):
    """Untagged legacy tokens have unverified ownership and are NEVER copied to URL keys (CWE-522)."""
    untagged_tok = {"tokens": {"access_token": "shared_tok", "token_type": "Bearer"}}
    untagged_cli = {"client_id": "shared_client"}
    mcp_auth.save_json("mcp:Notes:oauth", untagged_tok)
    mcp_auth.save_json("mcp:Notes:client", untagged_cli)

    mcp_auth.migrate_server_credentials("Notes", URL)
    # Legacy keys survive untouched without modification
    assert "mcp:Notes:oauth" in keychain
    assert "mcp:Notes:client" in keychain
    assert mcp_auth.load_json("mcp:Notes:oauth") == untagged_tok
    assert mcp_auth.load_json("mcp:Notes:client")["client_id"] == untagged_cli["client_id"]
    # Bearer tokens are NEVER copied to destination
    assert mcp_auth.tokens_secret("Notes", URL) not in keychain
    # Client registration metadata is copied to destination
    dest_cli = mcp_auth.load_json(mcp_auth.client_secret("Notes", URL))
    assert dest_cli["client_id"] == untagged_cli["client_id"]
    assert dest_cli["server_url"] == URL


async def test_sign_out_with_untagged_legacy_credentials_never_restores_token(app, keychain):
    """Untagged legacy token is not migrated; sign-out and relaunch never resurrect it (#269)."""
    mock = MockServer(oauth=True)
    app.integrations.mcp.http_transport = mock.transport()

    untagged_tok = {"tokens": {"access_token": "legacy_tok", "token_type": "Bearer"}}
    untagged_cli = {"client_id": "legacy_cli"}
    mcp_auth.save_json("mcp:Notes:oauth", untagged_tok)
    mcp_auth.save_json("mcp:Notes:client", untagged_cli)

    # 1. Add server: untagged legacy token is NOT migrated; server requires sign-in
    added = await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL})
    tok_key = mcp_auth.tokens_secret("Notes", URL)
    assert tok_key not in keychain
    assert added["signed_in"] is False and added["status"] == "needs_sign_in"

    # 2. User signs in via OAuth flow
    await _sign_in(app, mock, "Notes")
    await _wait(app, "Notes", "connected")
    assert tok_key in keychain
    fresh_tok = mcp_auth.load_json(tok_key)["tokens"]
    assert fresh_tok["access_token"] != "legacy_tok"

    # Legacy entry survives untouched in keychain
    assert mcp_auth.load_json("mcp:Notes:oauth") == untagged_tok

    # 3. Sign out: removes the URL-scoped token
    out = await app.integrations.mcp.sign_out("Notes")
    assert out["signed_in"] is False and out["status"] == "needs_sign_in"
    assert tok_key not in keychain
    assert mcp_auth.load_json("mcp:Notes:oauth") == untagged_tok

    # 4. Connection relaunch / describe must NOT resurrect any token
    server = next(s for s in app.integrations.mcp.list() if s["name"] == "Notes")
    assert server["signed_in"] is False
    assert server["status"] == "needs_sign_in"
    assert mcp_auth.KeychainTokenStorage("Notes", url=URL).has_tokens() is False
    assert tok_key not in keychain

    # 5. Sentient start / re-migration must NOT re-import the legacy token
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert tok_key not in keychain
    assert mcp_auth.KeychainTokenStorage("Notes", url=URL).has_tokens() is False

    # 6. Remove and re-add: still must NOT re-import the legacy token
    await app.integrations.mcp.remove("Notes")
    added2 = await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL})
    assert added2["signed_in"] is False and added2["status"] == "needs_sign_in"
    assert tok_key not in keychain

    # 7. Another setup with a different URL also will NOT migrate the untagged legacy token
    url2 = "http://setup2.example.com/mcp"
    tok2_key = mcp_auth.tokens_secret("Notes", url2)
    mcp_auth.migrate_server_credentials("Notes", url2)
    assert tok2_key not in keychain


def test_stale_sign_in_cross_setup_protection(keychain):
    url1 = "http://setup1.example.com/mcp"
    url2 = "http://setup2.example.com/mcp"

    # Setup 1 has legacy entry tagged with URL1
    tok1 = {"tokens": {"access_token": "setup1_tok"}, "server_url": url1}
    mcp_auth.save_json("mcp:Notes:oauth", tok1)

    # Setup 2 adds Notes at URL2 before Setup 1 migrates:
    # Setup 1's legacy entry must NOT be deleted
    assert mcp_auth.stale_sign_in("Notes", url2, changed_here=False) == []
    assert mcp_auth.stale_sign_in("Notes", url2, changed_here=True) == []
    assert "mcp:Notes:oauth" in keychain

    # Untagged legacy record with changed_here=True is cleaned up
    mcp_auth.save_json("mcp:Notes:oauth", {"tokens": {"access_token": "untagged"}})
    assert mcp_auth.stale_sign_in("Notes", url2, changed_here=False) == []
    assert mcp_auth.stale_sign_in("Notes", url2, changed_here=True) == ["mcp:Notes:oauth"]


def test_cross_setup_isolation_same_name_different_urls(keychain):
    name = "SharedService"
    url1 = "http://srv1.test/mcp"
    url2 = "http://srv2.test/mcp"

    tok1 = {"tokens": {"access_token": "t1"}, "server_url": url1}
    tok2 = {"tokens": {"access_token": "t2"}, "server_url": url2}
    cli1 = {"client_id": "c1", "server_url": url1}
    cli2 = {"client_id": "c2", "server_url": url2}

    mcp_auth.save_json(mcp_auth.tokens_secret(name, url1), tok1)
    mcp_auth.save_json(mcp_auth.client_secret(name, url1), cli1)
    mcp_auth.save_json(mcp_auth.tokens_secret(name, url2), tok2)
    mcp_auth.save_json(mcp_auth.client_secret(name, url2), cli2)

    assert mcp_auth.tokens_secret(name, url1) in keychain
    assert mcp_auth.tokens_secret(name, url2) in keychain

    # Forgetting server for url1 leaves url2 completely intact
    mcp_auth.forget_server(name, url=url1)
    assert mcp_auth.tokens_secret(name, url1) not in keychain
    assert mcp_auth.client_secret(name, url1) not in keychain
    assert mcp_auth.tokens_secret(name, url2) in keychain
    assert mcp_auth.client_secret(name, url2) in keychain
    assert mcp_auth.load_json(mcp_auth.tokens_secret(name, url2)) == tok2
    assert mcp_auth.load_json(mcp_auth.client_secret(name, url2)) == cli2


async def test_two_engines_separate_homes_synthetic_server_isolation(keychain, tmp_path, monkeypatch):
    """Test two engines with separate SENTIENT_HOME directories and a made-up MCP server name.
    Signing out in one setup must leave the other setup working (#269)."""
    home1 = tmp_path / "engine1_home"
    home2 = tmp_path / "engine2_home"
    home1.mkdir()
    home2.mkdir()

    server_name = "synthetic_shared_service"
    url1 = "http://engine1-service.test/mcp"
    url2 = "http://engine2-service.test/mcp"

    mock1 = MockServer(oauth=True, base="http://engine1-service.test")
    mock2 = MockServer(oauth=True, base="http://engine2-service.test")

    # Engine 1
    monkeypatch.setenv("SENTIENT_HOME", str(home1))
    cfg1 = SentientConfig()
    app1 = SentientApp(cfg1, llm=FakeProvider(), db_path=home1 / "sentient.db", enable_background=False)
    app1.integrations.mcp.http_transport = mock1.transport()
    await app1.start()

    # Engine 2
    monkeypatch.setenv("SENTIENT_HOME", str(home2))
    cfg2 = SentientConfig()
    app2 = SentientApp(cfg2, llm=FakeProvider(), db_path=home2 / "sentient.db", enable_background=False)
    app2.integrations.mcp.http_transport = mock2.transport()
    await app2.start()

    try:
        # Add server with the same synthetic name to both engines
        await app1.integrations.mcp.add(server_name, {"transport": "http", "url": url1})
        await _sign_in(app1, mock1, server_name)
        await _wait(app1, server_name, "connected")

        await app2.integrations.mcp.add(server_name, {"transport": "http", "url": url2})
        await _sign_in(app2, mock2, server_name)
        await _wait(app2, server_name, "connected")

        tok1_key = mcp_auth.tokens_secret(server_name, url1)
        tok2_key = mcp_auth.tokens_secret(server_name, url2)
        assert tok1_key != tok2_key
        assert tok1_key in keychain
        assert tok2_key in keychain
        assert app1.integrations.mcp.servers[server_name].status == "connected"
        assert app2.integrations.mcp.servers[server_name].status == "connected"

        # Sign out in Engine 1
        out1 = await app1.integrations.mcp.sign_out(server_name)
        assert out1["signed_in"] is False and out1["status"] == "needs_sign_in"
        assert tok1_key not in keychain

        # Verify Engine 2 is STILL connected, signed in, and its credentials remain in keychain
        assert tok2_key in keychain
        conn2 = app2.integrations.mcp.servers[server_name]
        assert conn2.status == "connected"
        descr2 = app2.integrations.mcp.describe(conn2)
        assert descr2["signed_in"] is True
        assert descr2["status"] == "connected"

        # Verify Engine 2 can still execute tools
        ctx2 = app2.agent.tool_context(None, "test")
        tool = app2.registry.get(f"mcp_{server_name.lower()}_hello")
        assert tool is not None
        res = await tool.call(ctx2, {})
        assert res["content"] == "hi there"

    finally:
        await app1.stop()
        await app2.stop()


def test_untagged_token_not_copied_across_urls(keychain):
    """Untagged legacy bearer tokens are never migrated across server URLs (Finding A regression test)."""
    untagged_tok = {"tokens": {"access_token": "secret_corp_token", "token_type": "Bearer"}}
    untagged_cli = {"client_id": "legacy_client"}
    mcp_auth.save_json("mcp:Notes:oauth", untagged_tok)
    mcp_auth.save_json("mcp:Notes:client", untagged_cli)

    url1 = "http://srv1.test/mcp"
    url2 = "http://attacker.test/mcp"

    mcp_auth.migrate_server_credentials("Notes", url1)
    mcp_auth.migrate_server_credentials("Notes", url2)

    # Neither URL receives the untagged bearer token
    assert mcp_auth.tokens_secret("Notes", url1) not in keychain
    assert mcp_auth.tokens_secret("Notes", url2) not in keychain

    # Client registration is copied
    assert mcp_auth.client_secret("Notes", url1) in keychain
    assert mcp_auth.client_secret("Notes", url2) in keychain

    # Legacy records are preserved untouched
    assert mcp_auth.load_json("mcp:Notes:oauth") == untagged_tok
    assert mcp_auth.load_json("mcp:Notes:client")["client_id"] == untagged_cli["client_id"]


async def test_reauthentication_after_sign_out_uses_new_token(app, keychain):
    """After sign-out, explicit re-authentication writes fresh tokens without resurrecting the legacy token."""
    mock = MockServer(oauth=True)
    app.integrations.mcp.http_transport = mock.transport()

    untagged_tok = {"tokens": {"access_token": "old_legacy_tok", "token_type": "Bearer"}}
    untagged_cli = {"client_id": "old_client"}
    mcp_auth.save_json("mcp:Notes:oauth", untagged_tok)
    mcp_auth.save_json("mcp:Notes:client", untagged_cli)

    # 1. Add server: untagged legacy token is NOT migrated; requires sign-in
    added = await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL})
    tok_key = mcp_auth.tokens_secret("Notes", URL)
    assert tok_key not in keychain
    assert added["signed_in"] is False and added["status"] == "needs_sign_in"

    # 2. Explicit re-authentication: sign in via OAuth flow
    await _sign_in(app, mock, "Notes")
    await _wait(app, "Notes", "connected")

    # 3. Active tokens are now the fresh tokens issued by MockServer, NOT old_legacy_tok
    fresh_rec = mcp_auth.load_json(tok_key)
    assert fresh_rec is not None
    assert fresh_rec["tokens"]["access_token"] != "old_legacy_tok"

    # 4. Old legacy record remains untouched
    assert mcp_auth.load_json("mcp:Notes:oauth") == untagged_tok

    # 5. Sign out: removes fresh tokens, does NOT fall back to old_legacy_tok
    out = await app.integrations.mcp.sign_out("Notes")
    assert out["signed_in"] is False and out["status"] == "needs_sign_in"
    assert tok_key not in keychain
    assert mcp_auth.KeychainTokenStorage("Notes", url=URL).has_tokens() is False
    assert mcp_auth.load_json("mcp:Notes:oauth") == untagged_tok


async def test_two_engines_separate_homes_untagged_legacy_credential_isolation(keychain, tmp_path, monkeypatch):
    """Two engines with separate SENTIENT_HOME directories sharing the keychain with untagged legacy tokens.
    Neither engine migrates the untagged token; each engine requires its own explicit sign-in."""
    home1 = tmp_path / "eng1_home"
    home2 = tmp_path / "eng2_home"
    home1.mkdir()
    home2.mkdir()

    server_name = "shared_legacy_srv"
    url1 = "http://eng1-srv.test/mcp"
    url2 = "http://eng2-srv.test/mcp"

    untagged_tok = {"tokens": {"access_token": "shared_legacy_tok", "token_type": "Bearer"}}
    untagged_cli = {"client_id": "shared_client_id"}
    mcp_auth.save_json(f"mcp:{server_name}:oauth", untagged_tok)
    mcp_auth.save_json(f"mcp:{server_name}:client", untagged_cli)

    mock1 = MockServer(oauth=True, base="http://eng1-srv.test")
    mock2 = MockServer(oauth=True, base="http://eng2-srv.test")

    # Engine 1
    monkeypatch.setenv("SENTIENT_HOME", str(home1))
    cfg1 = SentientConfig()
    app1 = SentientApp(cfg1, llm=FakeProvider(), db_path=home1 / "sentient.db", enable_background=False)
    app1.integrations.mcp.http_transport = mock1.transport()
    await app1.start()

    # Engine 2
    monkeypatch.setenv("SENTIENT_HOME", str(home2))
    cfg2 = SentientConfig()
    app2 = SentientApp(cfg2, llm=FakeProvider(), db_path=home2 / "sentient.db", enable_background=False)
    app2.integrations.mcp.http_transport = mock2.transport()
    await app2.start()

    try:
        # Both engines add the server with separate URLs
        add1 = await app1.integrations.mcp.add(server_name, {"transport": "http", "url": url1})
        add2 = await app2.integrations.mcp.add(server_name, {"transport": "http", "url": url2})

        tok1_key = mcp_auth.tokens_secret(server_name, url1)
        tok2_key = mcp_auth.tokens_secret(server_name, url2)
        assert tok1_key != tok2_key
        # Neither engine receives the untagged legacy token
        assert tok1_key not in keychain
        assert tok2_key not in keychain
        assert add1["signed_in"] is False and add1["status"] == "needs_sign_in"
        assert add2["signed_in"] is False and add2["status"] == "needs_sign_in"

        # Engine 1 signs in
        await _sign_in(app1, mock1, server_name)
        await _wait(app1, server_name, "connected")
        assert tok1_key in keychain
        assert tok2_key not in keychain

        # Engine 1 signs out
        out1 = await app1.integrations.mcp.sign_out(server_name)
        assert out1["signed_in"] is False and out1["status"] == "needs_sign_in"
        assert tok1_key not in keychain

        # Legacy key survived intact in keychain
        assert f"mcp:{server_name}:oauth" in keychain
        assert mcp_auth.load_json(f"mcp:{server_name}:oauth") == untagged_tok

    finally:
        await app1.stop()
        await app2.stop()


async def test_sign_out_reports_deletion_failure(app, keychain, monkeypatch):
    """When credential deletion fails during sign-out, sign_out raises ValueError."""
    tok_key = mcp_auth.tokens_secret("Notes", URL)
    mcp_auth.save_json(tok_key, {"tokens": {"access_token": "active_token"}})
    await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL, "enabled": False})

    # Simulate deletion failure where delete_json cannot remove the key
    monkeypatch.setattr(mcp_auth, "delete_json", lambda _key: None)
    import sentient.integrations.mcp as mcp_mod
    monkeypatch.setattr(mcp_mod, "delete_json", lambda _key: None)

    with pytest.raises(ValueError, match="Could not remove credentials for 'Notes'"):
        await app.integrations.mcp.sign_out("Notes")


def test_url_tagged_legacy_credentials_migrate_and_delete_legacy(keychain):
    """When legacy credentials are explicitly tagged with matching server_url, they migrate safely."""
    tagged_tok = {"tokens": {"access_token": "owned_token"}, "server_url": URL}
    tagged_cli = {"client_id": "owned_client", "server_url": URL}
    mcp_auth.save_json("mcp:Notes:oauth", tagged_tok)
    mcp_auth.save_json("mcp:Notes:client", tagged_cli)

    mcp_auth.migrate_server_credentials("Notes", URL)

    # Destination keys exist with verified credentials
    tok_key = mcp_auth.tokens_secret("Notes", URL)
    cli_key = mcp_auth.client_secret("Notes", URL)
    assert mcp_auth.load_json(tok_key)["tokens"]["access_token"] == "owned_token"
    assert mcp_auth.load_json(cli_key)["client_id"] == "owned_client"

    # Legacy keys are safely deleted upon verification
    assert "mcp:Notes:oauth" not in keychain
    assert "mcp:Notes:client" not in keychain


async def test_one_url_cannot_access_or_delete_another_url_credentials(app, keychain):
    """Credentials belonging to URL1 cannot be accessed or deleted by URL2 operations."""
    url1 = "http://srv1.test/mcp"
    url2 = "http://srv2.test/mcp"

    tok1_key = mcp_auth.tokens_secret("Notes", url1)
    tok2_key = mcp_auth.tokens_secret("Notes", url2)

    mcp_auth.save_json(tok1_key, {"tokens": {"access_token": "tok1"}})
    mcp_auth.save_json(tok2_key, {"tokens": {"access_token": "tok2"}})

    # Adding and signing out of URL1
    await app.integrations.mcp.add("Notes", {"transport": "http", "url": url1, "enabled": False})
    out1 = await app.integrations.mcp.sign_out("Notes")
    assert out1["signed_in"] is False

    # URL1 tokens are removed, URL2 tokens remain completely untouched
    assert tok1_key not in keychain
    assert tok2_key in keychain
    assert mcp_auth.load_json(tok2_key)["tokens"]["access_token"] == "tok2"


async def test_sign_out_cannot_resurrect_when_marker_persistence_fails(app, keychain):
    """Even if saving to legacy entries fails or is not persistent, untagged tokens never resurrect after sign-out."""
    legacy_tok = {"tokens": {"access_token": "unverified_token", "token_type": "Bearer"}}
    mcp_auth.save_json("mcp:Notes:oauth", legacy_tok)

    # Server added with URL; url token key has an active token
    url_tok_key = mcp_auth.tokens_secret("Notes", URL)
    mcp_auth.save_json(url_tok_key, {"tokens": {"access_token": "active_token"}, "server_url": URL})
    await app.integrations.mcp.add("Notes", {"transport": "http", "url": URL, "enabled": False})

    # Sign out removes url_tok_key
    out = await app.integrations.mcp.sign_out("Notes")
    assert out["signed_in"] is False and out["status"] == "disabled"
    assert url_tok_key not in keychain

    # Even though legacy key has NO migrated_to marker and contains unverified_token:
    assert mcp_auth.load_json("mcp:Notes:oauth") == legacy_tok
    assert mcp_auth.load_json("mcp:Notes:oauth").get("migrated_to") is None

    # Restart / migration does NOT resurrect unverified_token
    mcp_auth.migrate_server_credentials("Notes", URL)
    assert url_tok_key not in keychain
    assert mcp_auth.KeychainTokenStorage("Notes", url=URL).has_tokens() is False
