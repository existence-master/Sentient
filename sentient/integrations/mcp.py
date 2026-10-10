"""MCP client: attach external MCP servers and expose their tools as ordinary Sentient tools.

Servers come from ``config.integrations.mcp_servers``::

    {name: {transport: "stdio" | "http", command, args, url, env, env_keys, enabled,
            auth: "none" | "headers" | "oauth", header_keys}}

Environment values added through the API are stored in the keychain (``mcp:<name>``);
config keeps only their names in ``env_keys``. Remote (http) servers can carry static
headers (values in the keychain, names in ``header_keys``) and can sign in with the MCP
authorization spec (OAuth 2.1 + PKCE, see ``mcp_auth``). The browser sign-in comes back to
the shared loopback listener of the integration manager. Each server runs in its own
background task that connects, lists tools, registers them as ``mcp_<server>_<tool>`` and
reconnects with backoff when the connection drops. Startup never waits for a server.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from sentient.integrations.base import JsonSchemaTool
from sentient.integrations.common import delete_secret, load_secret_json, store_secret_json
from sentient.integrations.mcp_auth import (
    KeychainTokenStorage,
    NeedsSignIn,
    SentientOAuthProvider,
    SignInFailed,
    client_metadata,
    client_secret,
    delete_json,
    forget_server,
    headers_secret,
    load_json,
    save_json,
    stale_sign_in,
    tokens_secret,
)
from sentient.integrations.mcp_shorten import shortener_for
from sentient.tools.base import Risk, ToolContext, ToolPlugin

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

log = logging.getLogger(__name__)

MAX_TOOL_NAME = 64
CONNECT_TIMEOUT_S = 30
MIN_BACKOFF_S = 2.0
MAX_BACKOFF_S = 300
SIGN_IN_TTL_S = 15 * 60
AUTH_MODES = ("none", "headers", "oauth")
HEADER_NAME = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]{1,128}$")
SIGN_IN_MESSAGES = {
    "none": "This server asks you to sign in.",
    "headers": "The server didn't accept the saved headers. Change their values with the key button on the server.",
    "oauth": "Sign in to use this server.",
}


def sanitize(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", value.lower())).strip("_") or "x"


def mcp_tool_name(server: str, tool: str) -> str:
    name = f"mcp_{sanitize(server)}_{sanitize(tool)}"
    if len(name) > MAX_TOOL_NAME:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        name = f"{name[: MAX_TOOL_NAME - 9]}_{digest}"
    return name


def plugin_id_for(server: str) -> str:
    return f"mcp_{sanitize(server)}"


class MCPServerPlugin(ToolPlugin):
    auth = "mcp"
    category = "utilities"
    icon = "plug"

    def __init__(self, server: str, tools: list, description: str = "") -> None:
        self.id = plugin_id_for(server)
        self.display_name = server
        self.description = description or f"Tools from the external MCP server '{server}'."
        self.selection_hint = f"tools provided by the {server} MCP server"
        self.tools = tools
        super().__init__()


@dataclass
class SignInFlow:
    server: str
    url: asyncio.Future
    code: asyncio.Future
    state: str = ""
    task: asyncio.Task | None = None
    created: float = field(default_factory=time.time)


@dataclass
class Probe:
    """What one connection attempt saw: did the MCP endpoint answer 401?"""

    unauthorized: bool = False


@dataclass
class ServerConn:
    name: str
    spec: dict
    status: str = "disconnected"  # disconnected | connecting | connected | needs_sign_in | error | disabled
    error: str | None = None
    signin: SignInFlow | None = None
    refresh_tried: bool = False
    tools: list[dict] = field(default_factory=list)  # [{name, mcp_name, description, risk}]
    client: Any = None
    task: asyncio.Task | None = None
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    broken: asyncio.Event = field(default_factory=asyncio.Event)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    plugin: MCPServerPlugin | None = None


def _risk_for(tool: Any) -> Risk:
    ann = getattr(tool, "annotations", None)
    if ann is not None and getattr(ann, "read_only_hint", None):
        return Risk.read
    if ann is not None and getattr(ann, "destructive_hint", None) is True:
        return Risk.send
    return Risk.write


def _result_to_json(result: Any) -> dict:
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            texts.append(block.text)
        elif kind == "resource":
            res = getattr(block, "resource", None)
            texts.append(getattr(res, "text", None) or f"[resource {getattr(res, 'uri', '')}]")
        elif kind == "resource_link":
            texts.append(f"[link {getattr(block, 'uri', '')}]")
        else:
            texts.append(f"[{kind or 'content'} omitted]")
    text = "\n".join(texts)
    if getattr(result, "is_error", False):
        return {"error": text or "The MCP tool reported an error."}
    out: dict[str, Any] = {"content": text}
    structured = getattr(result, "structured_content", None)
    if structured and not _repeats(structured, text):
        out["structured"] = structured
    return out


def _repeats(structured: Any, text: str) -> bool:
    """True when a result's structured content only repeats its text (a server returning a plain string gives
    ``{"result": text}``), so the model doesn't read the same result twice (#264)."""
    same = [structured]
    if isinstance(structured, dict) and list(structured) == ["result"]:
        same.append(structured["result"])
    if text in same:
        return True
    try:
        return json.loads(text) in same
    except ValueError:
        return False


class MCPManager:
    def __init__(self, mgr: IntegrationManager):
        self.mgr = mgr
        self.servers: dict[str, ServerConn] = {}
        self._flows: dict[str, SignInFlow] = {}  # OAuth state -> browser sign-in in progress
        self.http_transport: Any = None  # tests: an httpx2 transport for remote servers
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def app(self) -> Any:
        return self.mgr.app

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        for name, spec in dict(self.app.config.integrations.mcp_servers).items():
            self._launch(name, dict(spec or {}))

    async def stop(self) -> None:
        for conn in list(self.servers.values()):
            self._cancel_sign_in(conn)
        await asyncio.gather(*(self._shutdown(c) for c in list(self.servers.values())), return_exceptions=True)

    def connected_plugin_ids(self) -> set[str]:
        return {plugin_id_for(c.name) for c in self.servers.values() if c.status == "connected"}

    # ------------------------------------------------------------------ public operations
    def list(self) -> list[dict]:
        return [self.describe(c) for c in self.servers.values()]

    def describe(self, conn: ServerConn) -> dict:
        spec = conn.spec
        env_keys = sorted(set(spec.get("env_keys") or []) | set((spec.get("env") or {}).keys()))
        auth = _auth_of(spec)
        return {
            "name": conn.name,
            "transport": spec.get("transport", "stdio"),
            "command": spec.get("command"),
            "args": list(spec.get("args") or []),
            "url": spec.get("url"),
            "env_keys": env_keys,
            "auth": auth,
            "header_keys": sorted(spec.get("header_keys") or []),
            "missing_values": self._missing_values(conn.name, spec),
            "signed_in": auth == "oauth" and KeychainTokenStorage(conn.name).has_tokens(),
            "signing_in": conn.signin is not None,
            "enabled": bool(spec.get("enabled", True)),
            "status": conn.status,
            "tools": list(conn.tools),
            "error": conn.error,
        }

    async def add(self, name: str, spec: dict, *, wait_s: float = 15.0) -> dict:
        name = name.strip()
        if not name:
            raise ValueError("A server name is required.")
        transport = spec.get("transport", "stdio")
        if transport not in {"stdio", "http"}:
            raise ValueError("transport must be 'stdio' or 'http'.")
        if transport == "stdio" and not spec.get("command"):
            raise ValueError("A command is required for stdio servers.")
        if transport == "http" and not spec.get("url"):
            raise ValueError("A URL is required for http servers.")
        pid = plugin_id_for(name)
        for other in self.servers:
            if other != name and plugin_id_for(other) == pid:
                raise ValueError(f"'{name}' is too similar to the existing server '{other}'.")
        headers = {str(k).strip(): str(v).strip() for k, v in (spec.get("headers") or {}).items() if str(k).strip()}
        auth = spec.get("auth") or ("headers" if headers else "none")
        if auth not in AUTH_MODES:
            raise ValueError("auth must be 'none', 'headers' or 'oauth'.")
        if transport == "stdio" and (headers or auth != "none"):
            raise ValueError("Headers and sign-in are only for remote (http) servers.")
        if auth == "headers" and not headers:
            raise ValueError("Add at least one header, or pick another way to sign in.")
        for key, value in headers.items():
            if not HEADER_NAME.match(key):
                raise ValueError(f"'{key}' isn't a valid header name.")
            if not value or any(c in value for c in "\r\n"):
                raise ValueError(f"Enter a single-line value for the header '{key}'.")
        async with self._server_lock(name):
            env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
            if env and not store_secret_json(f"mcp:{name}", env):
                raise ValueError("The system keychain is unavailable, so environment values can't be stored safely.")
            if headers and not save_json(headers_secret(name), headers):
                raise ValueError("The system keychain is unavailable, so header values can't be stored safely.")
            if not headers:
                delete_json(headers_secret(name))
            stored = {
                "transport": transport,
                "command": spec.get("command"),
                "args": [str(a) for a in spec.get("args") or []],
                "url": spec.get("url"),
                "env_keys": sorted(env.keys()),
                "auth": auth,
                "header_keys": sorted(headers.keys()),
                "enabled": bool(spec.get("enabled", True)),
            }
            previous = self.app.config.integrations.mcp_servers.get(name)
            changed_here = previous is not None and (
                previous.get("url") != stored["url"] or previous.get("transport", "stdio") != transport)
            # A sign-in belongs to one server address. The keychain is shared with other setups on this computer,
            # so only clear one saved for another address, never one another setup made for this same server.
            for secret in stale_sign_in(name, stored["url"] if transport == "http" else None, changed_here=changed_here):
                delete_json(secret)
            if name in self.servers:
                old = self.servers.pop(name)
                self._cancel_sign_in(old)
                await self._shutdown(old)
            cfg = self.app.config
            cfg.integrations.mcp_servers = {**cfg.integrations.mcp_servers, name: stored}
            self.app.save_config()
            conn = self._launch(name, stored)
        if stored["enabled"] and wait_s > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(conn.ready.wait(), wait_s)
        return self.describe(conn)

    async def import_server(self, name: str, spec: dict) -> dict:
        """Add a server brought over from another assistant, turned off and without any secret values: only the
        names of its headers and environment settings are kept, and an OAuth server signs in again when turned on."""
        name = name.strip()
        transport = spec.get("transport", "stdio")
        if not name or transport not in {"stdio", "http"}:
            raise ValueError("This server can't be added.")
        if (transport == "stdio" and not spec.get("command")) or (transport == "http" and not spec.get("url")):
            raise ValueError("It has no address or command.")
        pid = plugin_id_for(name)
        if any(plugin_id_for(other) == pid for other in [*self.servers, *self.app.config.integrations.mcp_servers]):
            raise ValueError("Sentient already has a server with this name.")
        header_keys = sorted(str(k) for k in spec.get("header_keys") or []) if transport == "http" else []
        auth = spec.get("auth") if transport == "http" and spec.get("auth") in AUTH_MODES else "none"
        if auth == "headers" and not header_keys:
            auth = "none"
        stored = {
            "transport": transport,
            "command": spec.get("command") if transport == "stdio" else None,
            "args": [str(a) for a in spec.get("args") or []] if transport == "stdio" else [],
            "url": spec.get("url") if transport == "http" else None,
            "env_keys": sorted(str(k) for k in spec.get("env_keys") or []) if transport == "stdio" else [],
            "auth": auth,
            "header_keys": header_keys,
            "enabled": False,
        }
        cfg = self.app.config
        cfg.integrations.mcp_servers = {**cfg.integrations.mcp_servers, name: stored}
        self.app.save_config()
        return self.describe(self._launch(name, stored))

    @staticmethod
    def _missing_values(name: str, spec: dict) -> list[str]:
        """Header and environment names this server lists without a value (an imported server, for example)."""
        if spec.get("transport", "stdio") == "http":
            have = load_json(headers_secret(name)) or {}
            return sorted(k for k in spec.get("header_keys") or [] if not str(have.get(k) or "").strip())
        have = {**(spec.get("env") or {}), **(load_secret_json(f"mcp:{name}") or {})}
        return sorted(k for k in spec.get("env_keys") or [] if not str(have.get(k) or "").strip())

    def _server_lock(self, name: str) -> asyncio.Lock:
        """One change at a time per server, so overlapping requests can't leave a connection untracked."""
        return self._locks.setdefault(name, asyncio.Lock())

    async def _relaunch(self, name: str, stored: dict) -> ServerConn:
        old = self.servers.pop(name, None)
        if old is not None:
            self._cancel_sign_in(old)
            await self._shutdown(old)
        return self._launch(name, stored)

    async def set_values(self, name: str, values: dict, *, enable: bool = False, wait_s: float = 15.0) -> dict:
        """Fill in the header or environment values a server lists by name (the "Add values" form). Values go to the
        keychain, never to config; a blank value keeps the saved one. Then the server reconnects (and is turned on
        with ``enable``)."""
        async with self._server_lock(name):
            cfg = self.app.config
            spec = cfg.integrations.mcp_servers.get(name)
            if spec is None:
                raise KeyError(name)
            http = spec.get("transport", "stdio") == "http"
            allowed = set(spec.get("header_keys" if http else "env_keys") or [])
            given = {str(k).strip(): str(v).strip() for k, v in (values or {}).items() if str(v).strip()}
            for key, value in given.items():
                if key not in allowed:
                    raise ValueError(f"'{key}' isn't one of this server's {'headers' if http else 'settings'}.")
                if len(value.splitlines()) > 1:
                    raise ValueError(f"Enter a single-line value for '{key}'.")
            if http and given and not _protected_url(str(spec.get("url") or "")):
                raise ValueError("This server's address starts with http://, so these values would be sent unprotected. "
                                 "Change it to an https:// address first.")
            if given:
                secret = headers_secret(name) if http else f"mcp:{name}"
                current = (load_json(secret) if http else load_secret_json(secret)) or {}
                merged = {**{k: v for k, v in current.items() if k in allowed}, **given}
                if not (save_json(secret, merged) if http else store_secret_json(secret, merged)):
                    raise ValueError("The system keychain is unavailable, so these values can't be stored safely.")
            stored = dict(spec)
            if http and given and _auth_of(stored) == "none":
                stored["auth"] = "headers"
            if enable and not self._missing_values(name, stored):  # turned on only once nothing is missing
                stored["enabled"] = True
            if stored != spec:
                cfg.integrations.mcp_servers = {**cfg.integrations.mcp_servers, name: stored}
                self.app.save_config()
            conn = await self._relaunch(name, stored)
        if stored.get("enabled", True) and wait_s > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(conn.ready.wait(), wait_s)
        return self.describe(conn)

    async def set_enabled(self, name: str, enabled: bool) -> dict:
        """Turn a server on or off without changing anything else."""
        async with self._server_lock(name):
            cfg = self.app.config
            spec = cfg.integrations.mcp_servers.get(name)
            if spec is None:
                raise KeyError(name)
            stored = {**spec, "enabled": bool(enabled)}
            cfg.integrations.mcp_servers = {**cfg.integrations.mcp_servers, name: stored}
            self.app.save_config()
            return self.describe(await self._relaunch(name, stored))

    async def remove(self, name: str) -> bool:
        async with self._server_lock(name):
            conn = self.servers.pop(name, None)
            if conn is not None:
                self._cancel_sign_in(conn)
                await self._shutdown(conn)
            cfg = self.app.config
            existed = name in cfg.integrations.mcp_servers
            if existed:
                cfg.integrations.mcp_servers = {k: v for k, v in cfg.integrations.mcp_servers.items() if k != name}
                self.app.save_config()
            delete_secret(f"mcp:{name}")
            forget_server(name)
            return existed or conn is not None

    async def test(self, name: str) -> dict:
        conn = self.servers.get(name)
        if conn is None:
            raise KeyError(name)
        if conn.status == "connected" and conn.client is not None:
            try:
                tools = await self._list_tools(conn.client)
                return {"ok": True, "tools": [t.name for t in tools]}
            except Exception as exc:
                conn.broken.set()
                return {"ok": False, "tools": [], "error": _err(exc)}
        probe = Probe()
        try:
            async with asyncio.timeout(CONNECT_TIMEOUT_S):
                async with self._client(conn.name, conn.spec, probe=probe) as client:
                    tools = await self._list_tools(client)
            if conn.status in {"error", "disconnected", "needs_sign_in"} and conn.spec.get("enabled", True):
                conn.broken.set()  # wake the runner so it reconnects now
            return {"ok": True, "tools": [t.name for t in tools]}
        except Exception as exc:
            if probe.unauthorized or _has(exc, NeedsSignIn):
                return {"ok": False, "tools": [], "error": SIGN_IN_MESSAGES[_auth_of(conn.spec)]}
            return {"ok": False, "tools": [], "error": _err(exc)}

    # ------------------------------------------------------------------ sign-in (MCP authorization spec)
    async def sign_in(self, name: str) -> dict:
        """Start a browser sign-in. Returns ``{auth_url, state}``; the loopback callback finishes it."""
        conn = self.servers.get(name)
        if conn is None:
            raise KeyError(name)
        if conn.spec.get("transport", "stdio") != "http":
            raise ValueError("Only remote (http) servers can sign in.")
        self._cancel_sign_in(conn)
        listener = self.mgr.listener
        await listener.start(self.app.config.integrations.oauth_redirect_port)
        redirect_uri = listener.redirect_uri()
        store = KeychainTokenStorage(name, fresh=True, url=conn.spec.get("url"))
        registered = await store.get_client_info()
        if registered is not None and redirect_uri not in [str(u) for u in registered.redirect_uris or []]:
            delete_json(client_secret(name))  # registered for another port: register again
        loop = asyncio.get_running_loop()
        flow = SignInFlow(server=name, url=loop.create_future(), code=loop.create_future())
        conn.signin = flow
        flow.task = asyncio.create_task(self._sign_in_run(conn, flow, store, redirect_uri), name=f"mcp-sign-in:{name}")
        flow.task.add_done_callback(_log_sign_in_result)
        await asyncio.wait({flow.url, flow.task}, timeout=CONNECT_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED)
        if flow.url.done():
            return {"auth_url": flow.url.result(), "state": flow.state}
        self._cancel_sign_in(conn)
        if flow.task.done() and not flow.task.cancelled() and flow.task.exception() is not None:
            raise ValueError(_sign_in_error(flow.task.exception()))
        raise ValueError("The server didn't answer in time. Try again in a moment.")

    async def sign_out(self, name: str) -> dict:
        conn = self.servers.get(name)
        if conn is None:
            raise KeyError(name)
        self._cancel_sign_in(conn)
        delete_json(tokens_secret(name))
        if _auth_of(conn.spec) == "oauth" and conn.spec.get("enabled", True):
            await self._shutdown(self.servers.pop(name))
            conn = self._launch(name, conn.spec)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(conn.ready.wait(), 5)
        return self.describe(conn)

    def owns_state(self, state: str) -> bool:
        return bool(state) and state in self._flows

    async def oauth_callback(self, params: dict[str, str]) -> tuple[bool, str]:
        """The browser came back to the loopback listener with this server's state."""
        flow = self._flows.get(params.get("state", ""))
        if flow is None or time.time() - flow.created > SIGN_IN_TTL_S:
            return False, "This sign-in link has expired."
        if flow.code.done():
            return False, "This sign-in link was already used."
        if params.get("error") or not params.get("code"):
            reason = "You cancelled the sign-in." if params.get("error") == "access_denied" else (
                f"The server reported: {params.get('error_description') or params.get('error') or 'no sign-in code'}.")
            flow.code.set_exception(SignInFailed(reason))
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(asyncio.shield(flow.task), CONNECT_TIMEOUT_S)  # type: ignore[arg-type]
            return False, reason
        from mcp.shared.auth import AuthorizationCodeResult

        flow.code.set_result(AuthorizationCodeResult(code=params["code"], state=params.get("state"),
                                                     iss=params.get("iss")))
        try:
            await asyncio.wait_for(asyncio.shield(flow.task), CONNECT_TIMEOUT_S)  # type: ignore[arg-type]
        except TimeoutError:
            return True, f"You're signed in. Sentient is still connecting to {flow.server}."
        except Exception as exc:
            return False, _sign_in_error(exc)
        return True, f"{flow.server} is connected."

    async def _sign_in_run(self, conn: ServerConn, flow: SignInFlow, store: KeychainTokenStorage,
                           redirect_uri: str) -> None:
        async def open_browser(url: str) -> None:
            flow.state = (parse_qs(urlsplit(url).query).get("state") or [""])[0]
            self._flows[flow.state] = flow
            if not flow.url.done():
                flow.url.set_result(url)

        async def wait_for_code() -> Any:
            return await flow.code

        provider = SentientOAuthProvider(conn.spec["url"], client_metadata(redirect_uri), store, open_browser,
                                         wait_for_code)
        try:
            async with asyncio.timeout(SIGN_IN_TTL_S):
                async with self._client(conn.name, conn.spec, auth=provider, read_timeout=SIGN_IN_TTL_S) as client:
                    await self._list_tools(client)
        except Exception as exc:
            if flow.code.done() and conn.status != "connected":
                conn.error = _sign_in_error(exc)
            raise
        finally:
            self._flows.pop(flow.state, None)
            if conn.signin is flow:
                conn.signin = None
        if not store.saved:
            raise SignInFailed("This server didn't ask for a sign-in, so there is nothing to do.")
        if _auth_of(conn.spec) != "oauth":
            conn.spec = {**conn.spec, "auth": "oauth"}
            cfg = self.app.config
            if conn.name in cfg.integrations.mcp_servers:
                cfg.integrations.mcp_servers = {**cfg.integrations.mcp_servers, conn.name: conn.spec}
                self.app.save_config()
        conn.error = None
        conn.refresh_tried = False
        conn.broken.set()  # reconnect now with the new sign-in

    def _cancel_sign_in(self, conn: ServerConn) -> None:
        flow, conn.signin = conn.signin, None
        if flow is None:
            return
        self._flows.pop(flow.state, None)
        if flow.task is not None and not flow.task.done():
            flow.task.cancel()

    # ------------------------------------------------------------------ internals
    def _launch(self, name: str, spec: dict) -> ServerConn:
        conn = ServerConn(name=name, spec=spec)
        self.servers[name] = conn
        if not spec.get("enabled", True):
            conn.status = "disabled"
            conn.ready.set()
            return conn
        conn.task = asyncio.create_task(self._run(conn), name=f"mcp:{name}")
        return conn

    async def _shutdown(self, conn: ServerConn) -> None:
        conn.stop.set()
        conn.broken.set()
        if conn.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(conn.task), 10)
            except Exception:
                conn.task.cancel()
                with contextlib.suppress(BaseException):
                    await conn.task
        self._unregister(conn)
        conn.status = "disconnected"

    @contextlib.asynccontextmanager
    async def _client(self, name: str, spec: dict, *, auth: Any = None, probe: Probe | None = None,
                      read_timeout: float | None = None):
        from mcp import Client, StdioServerParameters

        timeout = float(read_timeout or self.app.config.integrations.mcp_timeout_s)
        if spec.get("transport", "stdio") == "http":
            import httpx2
            from mcp.client.streamable_http import streamable_http_client

            url = spec["url"]
            if auth is None and _auth_of(spec) == "oauth":
                auth = self._stored_sign_in(name, url)
            endpoint = httpx2.URL(url)

            async def on_response(response: Any) -> None:
                seen = response.request.url
                if (response.status_code == 401 and probe is not None and seen.host == endpoint.host
                        and seen.path.rstrip("/") == endpoint.path.rstrip("/")):
                    probe.unauthorized = True

            headers = {str(k): str(v) for k, v in (load_json(headers_secret(name)) or {}).items()}
            async with (
                httpx2.AsyncClient(
                    timeout=httpx2.Timeout(30.0, read=300.0), headers=headers, auth=auth,
                    transport=self.http_transport, event_hooks={"response": [on_response]},
                ) as http,
                Client(streamable_http_client(url, http_client=http), read_timeout_seconds=timeout) as client,
            ):
                yield client
            return
        env = dict(spec.get("env") or {})
        env.update(load_secret_json(f"mcp:{name}") or {})
        params = StdioServerParameters(command=spec["command"], args=list(spec.get("args") or []), env=env or None)
        async with Client(params, read_timeout_seconds=timeout) as client:
            yield client

    def _stored_sign_in(self, name: str, url: str) -> SentientOAuthProvider:
        """OAuth for background connections: uses and refreshes the stored sign-in, never opens a browser."""
        store = KeychainTokenStorage(name, url=url)
        if not store.has_tokens():
            raise NeedsSignIn(name)

        async def no_browser(_url: str) -> None:
            raise NeedsSignIn(name)

        async def no_code() -> Any:
            raise NeedsSignIn(name)

        registered = load_json(client_secret(name)) or {}
        redirect = (registered.get("redirect_uris") or ["http://127.0.0.1/oauth/callback"])[0]
        return SentientOAuthProvider(url, client_metadata(redirect), store, no_browser, no_code,
                                     discovery_transport=self.http_transport)

    @staticmethod
    async def _list_tools(client: Any) -> list:
        tools: list = []
        cursor = None
        for _ in range(50):
            res = await client.list_tools(cursor=cursor)
            tools.extend(res.tools)
            cursor = getattr(res, "next_cursor", None)
            if not cursor:
                break
        return tools

    async def _run(self, conn: ServerConn) -> None:
        backoff = MIN_BACKOFF_S
        while not conn.stop.is_set():
            conn.status, conn.error = "connecting", None
            conn.broken.clear()
            probe = Probe()
            retry_now = False
            try:
                async with self._client(conn.name, conn.spec, probe=probe) as client:
                    tools = await asyncio.wait_for(self._list_tools(client), CONNECT_TIMEOUT_S)
                    conn.client = client
                    self._register(conn, tools)
                    conn.status = "connected"
                    conn.refresh_tried = False
                    conn.ready.set()
                    backoff = MIN_BACKOFF_S
                    log.info("mcp server %s connected with %d tools", conn.name, len(tools))
                    await conn.broken.wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                auth = _auth_of(conn.spec)
                if probe.unauthorized or _has(exc, NeedsSignIn):
                    if auth == "oauth" and not conn.refresh_tried and KeychainTokenStorage(conn.name).mark_expired():
                        conn.refresh_tried = True  # the token may have died early: refresh once before asking
                        retry_now = True
                    conn.status, conn.error = "needs_sign_in", SIGN_IN_MESSAGES[auth]
                    log.info("mcp server %s needs a sign-in", conn.name)
                else:
                    conn.status, conn.error = "error", _err(exc)
                    log.warning("mcp server %s failed: %s", conn.name, conn.error)
            finally:
                conn.client = None
                self._unregister(conn)
                if not retry_now:
                    conn.ready.set()
            if conn.stop.is_set():
                break
            if conn.status == "connected":
                conn.status = "connecting"
            if retry_now:
                continue
            conn.broken.clear()  # set again by test(), a finished sign-in or shutdown to wake this loop early
            wait = MAX_BACKOFF_S if conn.status == "needs_sign_in" else backoff
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(conn.broken.wait(), wait)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
        conn.status = conn.status if conn.status in {"error", "needs_sign_in"} else "disconnected"

    def _register(self, conn: ServerConn, mcp_tools: list) -> None:
        self._unregister(conn)
        tools: list[JsonSchemaTool] = []
        described: list[dict] = []
        used: set[str] = set()
        pid = plugin_id_for(conn.name)
        for t in mcp_tools:
            name = mcp_tool_name(conn.name, t.name)
            if name in used:
                continue
            used.add(name)
            risk = _risk_for(t)
            jt = JsonSchemaTool(
                name=name,
                description=(t.description or getattr(t, "title", None) or t.name)[:1024],
                fn=self._caller(conn, t.name),
                input_schema=dict(getattr(t, "input_schema", None) or {}),
                risk=risk,
                plugin=pid,
            )
            # another program's tools: results are outside content, and any change may send data out (ADR 0018)
            jt.untrusted_output = True
            jt.exfiltrates = risk != Risk.read
            jt.shorten_fn = shortener_for(t.name)  # known long results keep their useful part (#264)
            tools.append(jt)
            described.append({"name": name, "mcp_name": t.name, "description": jt.description, "risk": risk.name})
        plugin = MCPServerPlugin(conn.name, tools)
        reg = self.app.registry
        plugin.tools = [t for t in tools if not reg.has_tool(t.name)]
        try:
            reg.register(plugin)
        except ValueError as exc:
            log.warning("mcp server %s tools not registered: %s", conn.name, exc)
            return
        conn.plugin = plugin
        conn.tools = [d for d in described if any(t.name == d["name"] for t in plugin.tools)]

    def _unregister(self, conn: ServerConn) -> None:
        plugin = conn.plugin
        if plugin is None:
            return
        reg = self.app.registry
        if reg.plugin(plugin.id) is plugin:
            reg.unregister(plugin.id)
        conn.plugin = None

    def _caller(self, conn: ServerConn, mcp_name: str):
        async def call(ctx: ToolContext, **arguments: Any) -> Any:
            client = conn.client
            if client is None or conn.status != "connected":
                return {"error": f"The MCP server '{conn.name}' isn't connected right now ({conn.error or conn.status})."}
            timeout = float(self.app.config.integrations.mcp_timeout_s)
            try:
                result = await client.call_tool(mcp_name, arguments, read_timeout_seconds=timeout)
            except Exception as exc:
                msg = _err(exc)
                if _looks_disconnected(exc):
                    conn.broken.set()
                return {"error": f"MCP server '{conn.name}' failed: {msg}"}
            return _result_to_json(result)

        return call


def _err(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        return _err(exc.exceptions[0])
    return str(exc) or type(exc).__name__


def _looks_disconnected(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(w in text for w in ("closed", "broken", "disconnect", "endofstream", "connection"))


def _protected_url(url: str) -> bool:
    """https, or plain http to this computer only (where nothing travels over a network)."""
    parts = urlsplit(url)
    if parts.scheme == "https":
        return True
    if parts.scheme != "http":
        return False
    host = (parts.hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback  # an IP literal only: a name like 127.example.com is not
    except ValueError:
        return False


def _auth_of(spec: dict) -> str:
    auth = spec.get("auth") or ("headers" if spec.get("header_keys") else "none")
    return auth if auth in AUTH_MODES else "none"


def _has(exc: BaseException, kind: type[BaseException]) -> bool:
    if isinstance(exc, kind):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_has(e, kind) for e in exc.exceptions)
    return False


def _first(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _sign_in_error(exc: BaseException) -> str:
    from mcp.client.auth import OAuthFlowError, OAuthRegistrationError, OAuthTokenError

    inner = _first(exc)
    if isinstance(inner, SignInFailed):
        return str(inner)
    if isinstance(inner, OAuthRegistrationError):
        return ("This server doesn't let apps sign in on their own. If it gives you an access token, add it as a "
                f"header instead. ({inner})")
    if isinstance(inner, OAuthTokenError):
        return f"The server turned down the sign-in. Try again. ({inner})"
    if isinstance(inner, OAuthFlowError):
        return f"The sign-in didn't work: {inner}"
    if isinstance(inner, TimeoutError):
        return "The sign-in took too long. Try again."
    return f"The sign-in didn't work: {_err(inner)}"


def _log_sign_in_result(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.info("mcp sign-in %s ended: %s", task.get_name(), _err(exc))
