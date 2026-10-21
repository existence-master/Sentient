"""The MCP server itself: a small, safe tool set backed by the engine's REST API (ADR 0034).

Safety, in code rather than in prompts:

- The tool list is a whitelist. There is no tool that sends a message, deletes anything or runs a
  command, so nothing an MCP client asks for can do those things through Sentient.
- The server only starts when the user has turned it on in Settings (``integrations.mcp_server_enabled``),
  and every tool call checks the switch again against the running engine, so turning it off in
  Settings stops a server that is already running.
- The HTTP transport additionally requires the engine token as a bearer token: being another
  process on the same computer is not enough to call the tools. No client is ever given that
  token to paste into a configuration file: ``sentient mcp --connect <address>`` is the
  launcher. It reads the token the same way this server does, adds it when it connects, and
  serves the tools over stdio, so the configuration an app keeps holds only this command and
  the loopback address.
- Everything goes through the engine's REST API with the engine token, so a task is created the
  normal way (its plan waits for the user's approval in Sentient), memories an MCP app adds wait
  in Review (ADR 0021) unless the user allowed saving them straight away, and a chat turn's tools
  run under Sentient's own approvals, rules (ADR 0016) and outside-content checks (ADR 0018). The
  MCP client is outside content: nothing it says can approve anything.
"""

from __future__ import annotations

import json
import secrets
import sys
from typing import Any

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl

from sentient.mcp_server.client import EngineClient, EngineError, default_engine_url

OFF_MESSAGE = (
    "Sentient's MCP server is turned off. The user can turn it on in Sentient, under Integrations, "
    "in the section called Let other apps use Sentient."
)

_READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _error(message: str) -> str:
    return _json({"error": message})


def _memory_row(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "content": row.get("content"),
        "topics": row.get("topics") or [],
        "source": row.get("source"),
        "scope": row.get("scope"),
        "created_at": row.get("created_at"),
    }


def _task_row(row: dict) -> dict:
    plan = row.get("plan") or []
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "status": row.get("status"),
        "description": row.get("description"),
        "plan_steps": len(plan) if isinstance(plan, list) else 0,
        "updated_at": row.get("updated_at"),
    }


class Gate:
    """Reads the live switch from the engine before every call (see the module docstring)."""

    def __init__(self, client: EngineClient):
        self.client = client

    def check(self) -> dict | None:
        """The integrations config when the server may run, else None (turned off)."""
        section = self.client.integrations_config()
        return section if section.get("mcp_server_enabled") else None


class EngineTokenVerifier:
    """Bearer-token check for the HTTP transport: the token is the engine token.

    Loopback is not enough on its own: any process (or web page) on this computer could otherwise
    call the tools. stdio needs no verifier; the client starts the process itself, as with other
    local MCP servers.
    """

    def __init__(self, token: str):
        self._token = token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token and secrets.compare_digest(token, self._token):
            return AccessToken(token=token, client_id="sentient-mcp-http", scopes=[])
        return None


def build_server(
    client: EngineClient,
    gate: Gate | None = None,
    *,
    http_token: str | None = None,
    http_port: int = 8765,
) -> MCPServer:
    gate = gate or Gate(client)
    if http_token:
        base = f"http://127.0.0.1:{http_port}"
        server = MCPServer(
            "sentient",
            token_verifier=EngineTokenVerifier(http_token),
            auth=AuthSettings(
                issuer_url=AnyHttpUrl(base),
                resource_server_url=AnyHttpUrl(f"{base}/mcp"),
                validate_token_resource=False,
            ),
        )
    else:
        server = MCPServer("sentient")

    def live_config() -> tuple[dict | None, str]:
        """(section, "") when a call may run, else (None, the plain reason it may not).

        Only the turned-off switch reports OFF_MESSAGE; an engine problem (a rotated token, the
        engine stopped) reaches the caller with its own message, so the fix stays visible.
        """
        try:
            section = gate.check()
        except EngineError as exc:
            return None, str(exc)
        return (section, "") if section is not None else (None, OFF_MESSAGE)

    @server.tool(
        description=(
            "Search the user's memories in Sentient. Returns the matching memories with their ids. "
            "Memories still waiting for the user's review are never included."
        ),
        annotations=_READ,
    )
    def memory_search(query: str, limit: int = 10) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        try:
            rows = client.search_memories(query, limit)
        except EngineError as exc:
            return _error(str(exc))
        return _json({"memories": [_memory_row(r) for r in rows]})

    @server.tool(
        description=(
            "Save one fact to Sentient's memory. Unless the user allowed MCP apps to save directly, "
            "the memory waits in Sentient's Review list and is only used after the user approves it there."
        ),
        annotations=_WRITE,
    )
    def memory_add(content: str, scope: str | None = None) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        hold = not bool(section.get("mcp_server_memories_skip_review"))
        try:
            result = client.add_memory(content, hold_for_review=hold, scope=scope)
        except EngineError as exc:
            return _error(str(exc))
        out = {"action": result.get("action"), "id": result.get("id"), "content": result.get("content")}
        if result.get("status") == "pending" or hold:
            out["status"] = "pending"
            out["note"] = "Saved for review: the user approves it in Sentient before it is used."
        else:
            out["status"] = "active"
        return _json(out)

    @server.tool(description="Read one memory from Sentient by its id.", annotations=_READ)
    def memory_get(memory_id: int) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        try:
            row = client.get_memory(memory_id)
        except EngineError as exc:
            return _error(str(exc))
        return _json(_memory_row(row) | {"memory_type": row.get("memory_type"), "state": row.get("state")})

    @server.tool(
        description=(
            "Ask Sentient to do something as a task. The task is created the normal way: Sentient plans "
            "it and the plan waits for the user's approval in Sentient before anything runs."
        ),
        annotations=_WRITE,
    )
    def task_create(prompt: str) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        try:
            task = client.create_task(prompt)
        except EngineError as exc:
            return _error(str(exc))
        out = _task_row(task)
        out["note"] = "Created in Sentient. Its plan waits for the user's approval there before it runs."
        return _json(out)

    @server.tool(description="List the user's tasks in Sentient with their status.", annotations=_READ)
    def task_list() -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        try:
            rows = client.list_tasks()
        except EngineError as exc:
            return _error(str(exc))
        return _json({"tasks": [_task_row(r) for r in rows]})

    @server.tool(description="Read one task from Sentient by its id, with its status.", annotations=_READ)
    def task_status(task_id: str) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        try:
            task = client.get_task(task_id)
        except EngineError as exc:
            return _error(str(exc))
        return _json(_task_row(task))

    @server.tool(
        description="Show a notification to the user inside Sentient. Nothing is sent anywhere else.",
        annotations=_WRITE,
    )
    def notify_me(message: str, title: str | None = None) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        try:
            note = client.notify(message, title)
        except EngineError as exc:
            return _error(str(exc))
        return _json({"notified": True, "id": note.get("id")})

    @server.tool(
        description=(
            "Ask Sentient a question or give it an instruction, as one chat turn. The tools Sentient "
            "uses in that turn run under Sentient's own approvals and rules, and anything risky asks "
            "the user in the Sentient window first. Returns Sentient's reply."
        ),
        annotations=_WRITE,
    )
    def ask_sentient(message: str) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        try:
            turn = client.chat_turn(message)
        except EngineError as exc:
            return _error(str(exc))
        if turn.get("error"):
            return _json({"reply": turn.get("content") or "", "note": turn["error"]})
        return _json({"reply": turn.get("content") or ""})

    return server


def check_ready(client: EngineClient) -> str | None:
    """None when the server may start, else a plain sentence for the person starting it."""
    try:
        client.health()
        section = client.integrations_config()
    except EngineError as exc:
        return str(exc)
    if not section.get("mcp_server_enabled"):
        return OFF_MESSAGE
    return None


class BridgeError(Exception):
    """The launcher could not reach the HTTP MCP server, or it refused the sign-in."""


def validate_connect_url(url: str) -> str:
    """The launcher only ever connects to an MCP server on this computer."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in {"127.0.0.1", "localhost", "::1"}:
        raise BridgeError(
            f"{url} is not an address on this computer. The launcher only connects to a Sentient "
            "MCP server running on this computer, such as http://127.0.0.1:8765/mcp."
        )
    return url


async def _run_bridge(connect_url: str, token: str) -> None:
    """Serve the tools over stdio, forwarding to the HTTP server at ``connect_url``.

    This is the launcher: the MCP client starts this process, and the token is added here, by
    Sentient's own process, on the connection it opens. The token never appears in the
    client's configuration file, which names only this command and the loopback address.
    """
    import httpx2
    from mcp import ClientSession, types
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    http = httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx2.Timeout(10.0, read=1800.0),
    )
    try:
        async with (
            streamable_http_client(connect_url, http_client=http) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            bridge = Server("sentient-mcp-launcher")

            async def _list_tools(ctx, params):
                result = await session.list_tools()
                return types.ListToolsResult(tools=result.tools)

            async def _call_tool(ctx, params):
                return await session.call_tool(params.name, params.arguments or {})

            bridge.add_request_handler("tools/list", types.PaginatedRequestParams, _list_tools)
            bridge.add_request_handler("tools/call", types.CallToolRequestParams, _call_tool)
            async with stdio_server() as (stdin, stdout):
                await bridge.run(stdin, stdout, bridge.create_initialization_options())
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # connection failures can arrive grouped (anyio task groups)
        raise BridgeError(
            f"Could not connect to the Sentient MCP server at {connect_url}, or it refused "
            "Sentient's sign-in. Start the server first with `sentient mcp --transport http`, "
            "from this same Sentient install, then try again."
        ) from exc
    finally:
        await http.aclose()


def _launch(connect_url: str) -> int:
    """Launcher mode: forward a client's stdio to the HTTP server, signing in ourselves."""
    import anyio

    from sentient.gateway.auth import load_or_create_token

    try:
        url = validate_connect_url(connect_url)
    except BridgeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        anyio.run(_run_bridge, url, load_or_create_token())
    except BridgeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def main(
    transport: str = "stdio",
    *,
    port: int = 8765,
    engine_url: str | None = None,
    connect: str | None = None,
) -> int:
    """Entry point for ``sentient mcp``. Returns the process exit code."""
    from sentient.gateway.auth import load_or_create_token

    if connect is not None:
        return _launch(connect)

    url = engine_url or default_engine_url()
    token = load_or_create_token()
    try:
        client = EngineClient(url, token)
    except EngineError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    problem = check_ready(client)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    if transport == "http":
        # streamable HTTP, loopback only, and the client must send the engine token as a bearer
        # token: being on this computer is not enough on its own.
        server = build_server(client, http_token=token, http_port=port)
        print(
            f"Sentient MCP server on http://127.0.0.1:{port}/mcp "
            f"(this computer only; clients connect through the launcher: "
            f"sentient mcp --connect http://127.0.0.1:{port}/mcp)",
            file=sys.stderr,
        )
        server.run("streamable-http", host="127.0.0.1", port=port)
    else:
        build_server(client).run("stdio")
    return 0
