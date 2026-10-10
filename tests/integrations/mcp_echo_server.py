"""A tiny stdio MCP server used by test_mcp.py (run as a subprocess).

``--wait-for <file>`` holds the server back until that file exists, so a test can keep it from starting.
"""

import sys
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("echo")


@server.tool(description="Echo text back.", annotations=ToolAnnotations(read_only_hint=True))
def echo(text: str) -> str:
    return f"echo: {text}"


@server.tool(description="Add two integers.")
def add_numbers(a: int, b: int) -> int:
    return a + b


@server.tool(description="Always fails.")
def explode() -> str:
    raise ValueError("boom")


if __name__ == "__main__":
    if "--wait-for" in sys.argv:
        gate = Path(sys.argv[sys.argv.index("--wait-for") + 1])
        deadline = time.monotonic() + 120
        while not gate.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
    server.run("stdio")
