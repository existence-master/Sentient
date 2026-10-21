"""The MCP server itself: a small, safe tool set backed by the engine's REST API (ADR 0034).

Safety, in code rather than in prompts:

- The tool list is a whitelist. There is no tool that sends a message, deletes anything or runs a
  command, so nothing an MCP client asks for can do those things through Sentient.
- The server only starts when the user has turned it on in Settings (``integrations.mcp_server_enabled``),
  and every tool call checks the switch again against the running engine, so turning it off in
  Settings stops a server that is already running.
- Everything goes through the engine's REST API with the engine token, so a task is created the
  normal way (its plan waits for the user's approval in Sentient), memories an MCP app adds wait
  in Review (ADR 0021) unless the user allowed saving them straight away, and a chat turn's tools
  run under Sentient's own approvals, rules (ADR 0016) and outside-content checks (ADR 0018). The
  MCP client is outside content: nothing it says can approve anything.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

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


def build_server(client: EngineClient, gate: Gate | None = None) -> MCPServer:
    gate = gate or Gate(client)
    server = MCPServer("sentient")

    def live_config() -> dict | None:
        """None when a call must not run: turned off, or the engine went away."""
        try:
            return gate.check()
        except EngineError:
            return None

    @server.tool(
        description=(
            "Search the user's memories in Sentient. Returns the matching memories with their ids. "
            "Memories still waiting for the user's review are never included."
        ),
        annotations=_READ,
    )
    def memory_search(query: str, limit: int = 10) -> str:
        if live_config() is None:
            return _error(OFF_MESSAGE)
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
        section = live_config()
        if section is None:
            return _error(OFF_MESSAGE)
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
        if live_config() is None:
            return _error(OFF_MESSAGE)
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
        if live_config() is None:
            return _error(OFF_MESSAGE)
        try:
            task = client.create_task(prompt)
        except EngineError as exc:
            return _error(str(exc))
        out = _task_row(task)
        out["note"] = "Created in Sentient. Its plan waits for the user's approval there before it runs."
        return _json(out)

    @server.tool(description="List the user's tasks in Sentient with their status.", annotations=_READ)
    def task_list() -> str:
        if live_config() is None:
            return _error(OFF_MESSAGE)
        try:
            rows = client.list_tasks()
        except EngineError as exc:
            return _error(str(exc))
        return _json({"tasks": [_task_row(r) for r in rows]})

    @server.tool(description="Read one task from Sentient by its id, with its status.", annotations=_READ)
    def task_status(task_id: str) -> str:
        if live_config() is None:
            return _error(OFF_MESSAGE)
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
        if live_config() is None:
            return _error(OFF_MESSAGE)
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
        if live_config() is None:
            return _error(OFF_MESSAGE)
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


def main(
    transport: str = "stdio",
    *,
    port: int = 8765,
    engine_url: str | None = None,
) -> int:
    """Entry point for ``sentient mcp``. Returns the process exit code."""
    from sentient.gateway.auth import load_or_create_token

    url = engine_url or default_engine_url()
    try:
        client = EngineClient(url, load_or_create_token())
    except EngineError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    problem = check_ready(client)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    server = build_server(client)
    if transport == "http":
        # streamable HTTP, loopback only: the SDK's default host is 127.0.0.1 and we keep it that way.
        print(f"Sentient MCP server on http://127.0.0.1:{port}/mcp (this computer only)", file=sys.stderr)
        server.run("streamable-http", host="127.0.0.1", port=port)
    else:
        server.run("stdio")
    return 0
