"""A stand-in for the ``claude`` program (Claude Code) for tests/test_claude_code.py. Never calls a model.

Speaks the parts of print mode Sentient uses: ``--version``, one stream-json user message on stdin, and stream-json
events on stdout. Like Claude Code, it starts the MCP servers from ``--mcp-config`` over stdio, asks them for their
tools and lists them in the ``init`` event as ``mcp__<server>__<tool>``. ``FAKE_CLAUDE_SCENARIO`` picks the behaviour;
every run appends its arguments, stdin and what it saw to the JSON-lines file in ``FAKE_CLAUDE_LOG``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BUILTINS = ["Bash", "Read", "Edit", "Write", "WebFetch"]
# what decides which credentials Claude Code uses (code.claude.com/docs/en/authentication), recorded by name only
AUTH_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_", "CLAUDE_CODE_OAUTH_")
AUTH_NAMES = {"CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH", "CLAUDE_CONFIG_DIR"}


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


class Server:
    """An MCP stdio server from the --mcp-config file."""

    def __init__(self, spec: dict):
        self.proc = subprocess.Popen([spec["command"], *spec.get("args", [])], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, encoding="utf-8")
        self.next_id = 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def notify(self, method: str) -> None:
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def main() -> int:
    argv = sys.argv[1:]
    if argv == ["--version"]:
        print("9.9.9 (Claude Code)")
        return 0
    scenario = os.environ.get("FAKE_CLAUDE_SCENARIO", "chat")
    record: dict = {"argv": argv, "cwd": os.getcwd(), "scenario": scenario,
                    "has_gateway_token": "SENTIENT_GATEWAY_TOKEN" in os.environ,
                    "tool_search": os.environ.get("ENABLE_TOOL_SEARCH"),
                    "auth_env": sorted(k for k in os.environ if k.upper().startswith(AUTH_PREFIXES) or k.upper() in AUTH_NAMES)}
    stdin = sys.stdin.read()
    record["stdin"] = [json.loads(line) for line in stdin.splitlines() if line.strip()]
    system_file = flag(argv, "--system-prompt-file")
    record["system"] = Path(system_file).read_text(encoding="utf-8") if system_file else None

    def log() -> None:
        with open(os.environ["FAKE_CLAUDE_LOG"], "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    if scenario == "old":
        log()
        print("error: unknown option '--tools'", file=sys.stderr)
        return 1

    builtins = [] if flag(argv, "--tools") == "" and scenario != "builtin" else list(BUILTINS)
    tools = ["EndConversation", *builtins]
    servers: list[dict] = []
    config = flag(argv, "--mcp-config")
    if config:
        with open(config, encoding="utf-8") as f:
            for name, spec in json.load(f)["mcpServers"].items():
                server = Server(spec)
                init = server.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
                server.notify("notifications/initialized")
                listed = server.request("tools/list")["result"]["tools"]
                tools += [f"mcp__{name}__{t['name']}" for t in listed]
                if scenario == "tools" and listed:  # a careless host calling the tool: the bridge must not run it
                    record["bridge_call"] = server.request("tools/call", {"name": listed[0]["name"], "arguments": {}})["result"]
                record["bridge_init"] = init["result"]["serverInfo"]
                server.close()
                servers.append({"name": name, "status": "connected"})
    record["tools"] = tools
    log()
    emit({"type": "system", "subtype": "init", "session_id": "s1", "tools": tools, "mcp_servers": servers,
          "permissionMode": flag(argv, "--permission-mode") or "default", "model": flag(argv, "--model")})

    prompt = record["stdin"][0]["message"]["content"][0]["text"] if record["stdin"] else ""
    if scenario == "signed_out":
        emit({"type": "result", "subtype": "success", "is_error": True, "result": "Not logged in · Please run /login"})
        return 1
    if scenario == "hang":
        emit({"type": "stream_event", "event": {"type": "message_start", "message": {"id": "msg_h"}}})
        emit({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                "delta": {"type": "text_delta", "text": "Thinking about it"}}})
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        with open(os.environ["FAKE_CLAUDE_PIDS"], "w", encoding="utf-8") as f:
            json.dump({"claude": os.getpid(), "child": child.pid}, f)
        time.sleep(600)
        return 0
    usage = {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 7}
    if scenario == "tools" and "<tool_result" not in prompt:
        emit({"type": "stream_event", "event": {"type": "message_start", "message": {"id": "msg_t"}}})
        emit({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                "delta": {"type": "text_delta", "text": "Sending it now."}}})
        call = {"type": "tool_use", "id": "toolu_1", "name": "mcp__sentient__send_postcard", "input": {"text": "hi"}}
        emit({"type": "assistant", "message": {"id": "msg_t", "content": [{"type": "text", "text": "Sending it now."}, call],
                                               "usage": usage}, "parent_tool_use_id": None})
        emit({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True, "content": "Permission denied"}]},
            "parent_tool_use_id": None})
        emit({"type": "result", "subtype": "error_max_turns", "is_error": True, "errors": ["Reached max turns (1)"]})
        return 1
    reply = ["All done, ", "the postcard is on its way."] if scenario == "tools" else ["Hello ", "from Claude"]
    emit({"type": "stream_event", "event": {"type": "message_start", "message": {"id": "msg_1"}}})
    for piece in reply:
        emit({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                "delta": {"type": "text_delta", "text": piece}}})
    emit({"type": "assistant", "message": {"id": "msg_1", "content": [{"type": "text", "text": "".join(reply)}],
                                           "usage": usage}, "parent_tool_use_id": None})
    emit({"type": "result", "subtype": "success", "is_error": False, "result": "".join(reply), "usage": usage,
          "total_cost_usd": 0.01})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
