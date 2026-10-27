"""Sentient as an MCP server: ``sentient mcp`` (issue #443, ADR 0034).

A small, safe set of tools for other AI apps on this computer (Claude Code, Cursor and other MCP
clients), backed by the running engine's REST API. This package never touches the database or the
agent loop directly: every call goes through the gateway, so approvals, lasting rules (ADR 0016)
and the outside-content checks (ADR 0018) apply exactly as they do to any other client.
"""
