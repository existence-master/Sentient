"""The tool list Sentient shows Claude Code: a tiny MCP server on stdio that can't do anything (issue #206, ADR 0022).

Claude Code starts it from the ``--mcp-config`` file Sentient writes for one reply::

    python -m sentient.llm.claude_code_tools <tools.json>     # development
    sentient-engine claude-code-tools <tools.json>             # installed app

It answers ``initialize`` and ``tools/list`` with the tools from the file, so Claude can ask for them by name. It
never runs one: ``tools/call`` answers with an error that says Sentient runs tools itself. The call Claude asked for
reaches Sentient as a tool call in Claude Code's output, and Sentient's own agent loop runs it with its approvals,
rules, outside-content checks and Stop everything. Claude Code is also started in a mode that refuses MCP calls, so
in practice this answer is never needed.

Plain JSON-RPC over newline-delimited stdin and stdout, standard library only, so it starts fast and does not
depend on the MCP package's version.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

PROTOCOL_VERSION = "2025-06-18"
NOT_HERE = (
    "Sentient runs its tools itself, after its own checks. This call was not run here; Sentient will run it and "
    "send the result in the next message."
)


def load_tools(path: str | Path) -> list[dict[str, Any]]:
    """The MCP tool list from the file Sentient wrote: ``[{name, description, inputSchema}]``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [t for t in data if isinstance(t, dict) and t.get("name")] if isinstance(data, list) else []


def handle(message: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The reply to one JSON-RPC message, or None for notifications."""
    method = message.get("method")
    if "id" not in message or not isinstance(method, str):
        return None  # a notification (initialized, cancelled) or a response: nothing to answer
    rid = message["id"]
    if method == "initialize":
        asked = (message.get("params") or {}).get("protocolVersion")
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": asked if isinstance(asked, str) and asked else PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "sentient", "version": "1"},
            "instructions": "Sentient's tools. Ask for one when you need it; Sentient runs it and sends the result.",
        }}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": tools}}
    if method == "tools/call":
        return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": NOT_HERE}], "isError": True}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def serve(tools: list[dict[str, Any]], stdin: TextIO, stdout: TextIO) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            reply: dict[str, Any] | None = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        else:
            reply = handle(message, tools) if isinstance(message, dict) else None
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: claude_code_tools <tools.json>", file=sys.stderr)
        return 2
    try:
        tools = load_tools(args[0])
    except (OSError, ValueError) as exc:
        print(f"could not read the tool list: {exc}", file=sys.stderr)
        return 1
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")  # type: ignore[union-attr]
    serve(tools, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
