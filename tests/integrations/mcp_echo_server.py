"""A tiny stdio MCP server used by test_mcp.py (run as a subprocess)."""

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
    server.run("stdio")
