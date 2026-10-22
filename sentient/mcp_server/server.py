"""The MCP server itself: a small, safe tool set backed by the engine's REST API (ADR 0034).

Safety, in code rather than in prompts:

- The tool list is a whitelist. There is no tool that sends a message, deletes anything or runs a
  command, so nothing an MCP client asks for can do those things through Sentient.
- The server only starts when the user has turned it on in Sentient (``integrations.mcp_server_enabled``),
  and every tool call checks the switch again against the running engine, so turning it off in
  Sentient stops a server that is already running.
- It signs in with the MCP-only token: a token the engine issues for this feature alone, kept in
  the keychain, which opens only the endpoints these tools use (never config, approvals or
  anything else). ``sentient mcp`` reads it from the keychain itself and finds the engine
  through the discovery file the engine writes at startup, so a client's configuration holds
  nothing but the ``sentient mcp`` command: no address to keep current, no secret to paste.
- Everything goes through the engine's REST API, so a task an app creates is marked as that
  app's outside content (its plan always waits for the user's approval, like a task by email),
  memories an app adds wait in Review (ADR 0021) unless the user allowed saving them straight
  away, and a chat turn an app steers is outside content from the start (ADR 0018): its words are
  never the user's, anything that sends or runs code asks the user first, and the question also
  lands as a notification in the app and on the user's phone. The MCP client is outside content:
  nothing it says can approve anything.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from sentient.mcp_server.client import EngineClient, EngineError, discovered_engine_url

OFF_MESSAGE = (
    "Sentient's MCP server is turned off. The user can turn it on in Sentient, under Integrations, "
    "in the section called Let other apps use Sentient (in the app: #/integrations?section=sentient-mcp)."
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


def _client_name(ctx: Any) -> str | None:
    """The name the connected app gave for itself ("Claude Code"), or None when it gave none.
    Used only to say where a task or a note came from; never trusted for anything else."""
    try:
        params = ctx.request_context.session.client_params
        info = getattr(params, "client_info", None) or getattr(params, "clientInfo", None)
        name = getattr(info, "name", None) if info else None
        if name is None:  # an explicit null in the handshake is no name, not the app "None"
            return None
        return str(name).strip() or None
    except Exception:
        return None


class Gate:
    """Reads the live switch from the engine before every call (see the module docstring)."""

    def __init__(self, client: EngineClient):
        self.client = client

    def check(self) -> dict | None:
        """The engine's MCP status when the server may run, else None (turned off).

        Before the status call carries the token anywhere, the client follows the discovery
        file if the engine moved ports, and the engine proves its identity again: the proof
        at startup alone is not enough, because the engine restarts on a fresh port and
        whatever takes over the old one would otherwise be sent the token on the next call."""
        self.client.refresh_address()
        if not self.client.prove_identity():
            raise EngineError(
                f"The engine answering at {self.client.base_url} could not prove it holds the MCP "
                "token, so the token was not sent to it. Sentient may have restarted, or its MCP "
                "token may have been replaced in the app; restart this MCP server to reconnect."
            )
        status = self.client.status()
        return status if status.get("enabled") else None


def build_server(client: EngineClient, gate: Gate | None = None) -> MCPServer:
    gate = gate or Gate(client)
    server = MCPServer("sentient")

    def live_config() -> tuple[dict | None, str]:
        """(status, "") when a call may run, else (None, the plain reason it may not).

        Only the turned-off switch reports OFF_MESSAGE; an engine problem (a replaced token, the
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
    def memory_add(content: str, ctx: Context, scope: str | None = None) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        hold = not bool(section.get("memories_skip_review"))
        try:
            result = client.add_memory(content, hold_for_review=hold, scope=scope, app=_client_name(ctx))
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
            "Ask Sentient to do something as a task. The task is created in your app's name and "
            "marked as outside content: Sentient plans it and the plan always waits for the user's "
            "approval in Sentient before anything runs."
        ),
        annotations=_WRITE,
    )
    def task_create(prompt: str, ctx: Context) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        try:
            task = client.create_task(prompt, app=_client_name(ctx))
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
        description=(
            "Show a notification to the user inside Sentient. The notification names your app, so "
            "it cannot be mistaken for a message from Sentient itself. Nothing is sent anywhere else."
        ),
        annotations=_WRITE,
    )
    def notify_me(message: str, ctx: Context, title: str | None = None) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        try:
            note = client.notify(message, title, app=_client_name(ctx))
        except EngineError as exc:
            return _error(str(exc))
        return _json({"notified": True, "id": note.get("id")})

    @server.tool(
        description=(
            "Ask Sentient a question or give it an instruction, as one chat turn. Sentient treats "
            "the turn as outside content: the tools it uses run under Sentient's own approvals and "
            "rules, and anything that sends, deletes or runs code asks the user first, in the "
            "Sentient window and as a notification. Returns Sentient's reply."
        ),
        annotations=_WRITE,
    )
    def ask_sentient(message: str, ctx: Context) -> str:
        section, why = live_config()
        if section is None:
            return _error(why)
        try:
            turn = client.chat_turn(message, app=_client_name(ctx))
        except EngineError as exc:
            return _error(str(exc))
        if turn.get("error"):
            return _json({"reply": turn.get("content") or "", "note": turn["error"]})
        out = {"reply": turn.get("content") or ""}
        notes = []
        if turn.get("approval_asked"):
            notes.append("Sentient asked the user to approve a step in this turn before it ran.")
        if turn.get("memory_held"):
            # the reply may still say "noted": the fact is not saved until the user approves it
            notes.append(
                "A memory from this exchange is waiting in the user's Review list in Sentient; "
                "it is only saved if they approve it there."
            )
        if notes:
            out["note"] = " ".join(notes)
        return _json(out)

    return server


def check_ready(client: EngineClient) -> str | None:
    """None when the server may start, else a plain sentence for the person starting it."""
    try:
        health = client.health()  # no token sent: this answer decides whether the token may travel
    except EngineError as exc:
        return str(exc)
    if not health.get("ok") or health.get("name") != "sentient":
        # the discovery file holds only a port, and ports get recycled: whatever answered there
        # is not Sentient's engine, and it must never be sent the MCP token
        return (
            f"Something else is answering at {client.base_url}, and it is not Sentient's engine, "
            "so this server will not send it the MCP token. Start Sentient first, then try again."
        )
    if not client.prove_identity():
        # the health marker is public and can be copied by anything listening on the port; only
        # a process holding the MCP token can answer the challenge, and the token is sent to
        # no one who has not answered it
        return (
            f"Whatever is answering at {client.base_url} says it is Sentient but cannot prove it "
            "holds the MCP token, so this server will not send the token there. Start Sentient "
            "first, then try again."
        )
    try:
        status = client.status()
    except EngineError as exc:
        return str(exc)
    if not status.get("enabled"):
        return OFF_MESSAGE
    return None


def main(*, engine_url: str | None = None) -> int:
    """Entry point for ``sentient mcp`` (stdio). Returns the process exit code."""
    from sentient.gateway import mcp_token

    follow_discovery = engine_url is None  # found through the file: follow it when the engine moves
    if engine_url is None:
        try:
            engine_url = discovered_engine_url()
        except EngineError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    in_use = mcp_token.current_token()
    if not in_use:
        print(
            "Sentient has not issued an MCP token yet. The user turns the feature on in Sentient "
            "(Integrations, in the section called Let other apps use Sentient), which creates it; "
            "this server reads it from the keychain itself, so there is nothing to paste anywhere.",
            file=sys.stderr,
        )
        return 2
    try:
        client = EngineClient(engine_url, in_use, follow_discovery=follow_discovery)
    except EngineError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    problem = check_ready(client)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    build_server(client).run("stdio")
    return 0
