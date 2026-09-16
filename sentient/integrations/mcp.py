"""MCP client: attach external MCP servers and expose their tools as ordinary Sentient tools.

Servers come from ``config.integrations.mcp_servers``::

    {name: {transport: "stdio" | "http", command, args, url, env, env_keys, enabled}}

Environment values added through the API are stored in the keychain (``mcp:<name>``);
config keeps only their names in ``env_keys``. Each server runs in its own background
task that connects, lists tools, registers them as ``mcp_<server>_<tool>`` and reconnects
with backoff when the connection drops. Startup never waits for a server.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sentient.integrations.base import JsonSchemaTool
from sentient.integrations.common import delete_secret, load_secret_json, store_secret_json
from sentient.tools.base import Risk, ToolContext, ToolPlugin

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

log = logging.getLogger(__name__)

MAX_TOOL_NAME = 64
CONNECT_TIMEOUT_S = 30
MAX_BACKOFF_S = 300


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
class ServerConn:
    name: str
    spec: dict
    status: str = "disconnected"  # disconnected | connecting | connected | error | disabled
    error: str | None = None
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
    if structured:
        out["structured"] = structured
    return out


class MCPManager:
    def __init__(self, mgr: IntegrationManager):
        self.mgr = mgr
        self.servers: dict[str, ServerConn] = {}

    @property
    def app(self) -> Any:
        return self.mgr.app

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        for name, spec in dict(self.app.config.integrations.mcp_servers).items():
            self._launch(name, dict(spec or {}))

    async def stop(self) -> None:
        await asyncio.gather(*(self._shutdown(c) for c in list(self.servers.values())), return_exceptions=True)

    def connected_plugin_ids(self) -> set[str]:
        return {plugin_id_for(c.name) for c in self.servers.values() if c.status == "connected"}

    # ------------------------------------------------------------------ public operations
    def list(self) -> list[dict]:
        return [self.describe(c) for c in self.servers.values()]

    def describe(self, conn: ServerConn) -> dict:
        spec = conn.spec
        env_keys = sorted(set(spec.get("env_keys") or []) | set((spec.get("env") or {}).keys()))
        return {
            "name": conn.name,
            "transport": spec.get("transport", "stdio"),
            "command": spec.get("command"),
            "args": list(spec.get("args") or []),
            "url": spec.get("url"),
            "env_keys": env_keys,
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
        env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
        if env and not store_secret_json(f"mcp:{name}", env):
            raise ValueError("The system keychain is unavailable, so environment values can't be stored safely.")
        stored = {
            "transport": transport,
            "command": spec.get("command"),
            "args": [str(a) for a in spec.get("args") or []],
            "url": spec.get("url"),
            "env_keys": sorted(env.keys()),
            "enabled": bool(spec.get("enabled", True)),
        }
        if name in self.servers:
            await self._shutdown(self.servers.pop(name))
        cfg = self.app.config
        cfg.integrations.mcp_servers = {**cfg.integrations.mcp_servers, name: stored}
        self.app.save_config()
        conn = self._launch(name, stored)
        if stored["enabled"] and wait_s > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(conn.ready.wait(), wait_s)
        return self.describe(conn)

    async def remove(self, name: str) -> bool:
        conn = self.servers.pop(name, None)
        if conn is not None:
            await self._shutdown(conn)
        cfg = self.app.config
        existed = name in cfg.integrations.mcp_servers
        if existed:
            cfg.integrations.mcp_servers = {k: v for k, v in cfg.integrations.mcp_servers.items() if k != name}
            self.app.save_config()
        delete_secret(f"mcp:{name}")
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
        try:
            async with asyncio.timeout(CONNECT_TIMEOUT_S):
                async with self._client(conn.name, conn.spec) as client:
                    tools = await self._list_tools(client)
            if conn.status in {"error", "disconnected"} and conn.spec.get("enabled", True):
                conn.broken.set()  # wake the runner so it reconnects now
            return {"ok": True, "tools": [t.name for t in tools]}
        except Exception as exc:
            return {"ok": False, "tools": [], "error": _err(exc)}

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

    def _client(self, name: str, spec: dict):
        from mcp import Client, StdioServerParameters

        if spec.get("transport", "stdio") == "http":
            return Client(spec["url"], read_timeout_seconds=float(self.app.config.integrations.mcp_timeout_s))
        env = dict(spec.get("env") or {})
        env.update(load_secret_json(f"mcp:{name}") or {})
        params = StdioServerParameters(command=spec["command"], args=list(spec.get("args") or []), env=env or None)
        return Client(params, read_timeout_seconds=float(self.app.config.integrations.mcp_timeout_s))

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
        backoff = 2.0
        while not conn.stop.is_set():
            conn.status, conn.error = "connecting", None
            conn.broken.clear()
            try:
                async with self._client(conn.name, conn.spec) as client:
                    tools = await asyncio.wait_for(self._list_tools(client), CONNECT_TIMEOUT_S)
                    conn.client = client
                    self._register(conn, tools)
                    conn.status = "connected"
                    conn.ready.set()
                    backoff = 2.0
                    log.info("mcp server %s connected with %d tools", conn.name, len(tools))
                    await conn.broken.wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                conn.status, conn.error = "error", _err(exc)
                log.warning("mcp server %s failed: %s", conn.name, conn.error)
            finally:
                conn.client = None
                self._unregister(conn)
                conn.ready.set()
            if conn.stop.is_set():
                break
            if conn.status == "connected":
                conn.status = "connecting"
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(conn.stop.wait(), backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
        conn.status = "disconnected" if conn.status != "error" else conn.status

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
