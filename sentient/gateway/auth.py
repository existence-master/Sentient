"""Gateway authentication: an invisible per-launch token.

The user never sees or types this. The desktop shell generates a random token,
passes it to the backend in ``SENTIENT_GATEWAY_TOKEN`` and to the renderer via
the preload bridge. It exists so another web page or local process cannot
drive the assistant through the loopback port (the class of bug behind
OpenClaw's CVE-2026-25253). When the backend is started by hand for
development, a token file under the Sentient home folder is used instead.
"""

from __future__ import annotations

import contextlib
import os
import secrets as pysecrets

from fastapi import HTTPException, Request, WebSocket, status

from sentient import paths

ENV_TOKEN = "SENTIENT_GATEWAY_TOKEN"


def load_or_create_token() -> str:
    env = os.environ.get(ENV_TOKEN, "").strip()
    if env:
        return env
    p = paths.token_file()
    if p.exists():
        tok = p.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    p.parent.mkdir(parents=True, exist_ok=True)
    tok = pysecrets.token_urlsafe(32)
    p.write_text(tok, encoding="utf-8")
    with contextlib.suppress(Exception):  # not all filesystems support modes
        p.chmod(0o600)
    return tok


def _extract(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.query_params.get("token")


def require_token(request: Request) -> None:
    expected = request.app.state.token
    given = _extract(request)
    if not given or not pysecrets.compare_digest(given, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing token")


# The only (method, route) pairs the MCP-only token opens: exactly what the eight MCP tools call
# (issue #443, ADR 0034). Anything else (config, approvals, sessions, deletes, the memory review
# inbox) refuses it, so a token an outside app holds can never drive the rest of the engine.
MCP_ALLOWED = frozenset({
    ("GET", "/api/memories"), ("GET", "/api/memories/{memory_id}"), ("POST", "/api/memories"),
    ("GET", "/api/tasks"), ("POST", "/api/tasks"), ("GET", "/api/tasks/{task_id}"),
    ("POST", "/api/notifications"),
    ("POST", "/api/chat"),
    ("GET", "/api/mcp/status"),
})


def require_gateway_or_mcp(request: Request) -> None:
    """For the routes the MCP tools use. The gateway token passes everywhere and marks the caller
    as one of the user's own surfaces; the MCP-only token passes only on the allowlist above and
    marks the caller as an outside app, which the handlers treat as outside content."""
    from sentient.gateway import mcp_token

    given = _extract(request)
    expected = request.app.state.token
    if given and expected and pysecrets.compare_digest(given, expected):
        request.state.caller = "gateway"
        return
    if mcp_token.token_matches(given):
        route = request.scope.get("route")
        key = (request.method, getattr(route, "path", ""))
        if key in MCP_ALLOWED:
            if key != ("GET", "/api/mcp/status"):
                # the feature switch rules the token too, not just the MCP server process: with the
                # feature off in Settings the token opens nothing (the status route stays reachable
                # so the server can report that the feature is off instead of a refusal)
                core = getattr(request.app.state, "sentient", None)
                if core is None or not core.config.integrations.mcp_server_enabled:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN, detail="the MCP server feature is turned off"
                    )
            request.state.caller = "mcp"
            return
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="the MCP token does not open this")
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing token")


def ws_token_ok(ws: WebSocket) -> bool:
    expected = ws.app.state.token
    given = ws.query_params.get("token") or ""
    header = ws.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        given = header[7:].strip()
    return bool(given) and pysecrets.compare_digest(given, expected)
