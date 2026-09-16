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


def ws_token_ok(ws: WebSocket) -> bool:
    expected = ws.app.state.token
    given = ws.query_params.get("token") or ""
    header = ws.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        given = header[7:].strip()
    return bool(given) and pysecrets.compare_digest(given, expected)
