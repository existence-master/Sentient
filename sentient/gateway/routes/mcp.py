"""The MCP server's own endpoints (issue #443, ADR 0034): a status the MCP process can check with
its narrow token, and the token replacement only the user's own surfaces may ask for."""

from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request
from fastapi.routing import APIRouter

from sentient.gateway import mcp_token
from sentient.gateway.deps import AUTH, MCP_OK, get_core

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


@router.get("/proof")  # no credentials on purpose: this runs before the caller sends any
async def proof(request: Request, nonce: str = ""):
    """How the MCP server knows it found the real engine before it sends the token (issue #443).

    The caller picks a fresh nonce; the answer is the HMAC of that nonce and the address this
    request arrived on (host and port), under the MCP token, which only a process holding the
    token (this engine, through the keychain) can produce. A health answer can be copied by
    anything that took over the discovery file's port, so the marker alone is never trusted.
    Binding the address in stops a relay: a process squatting on the engine's old address can
    forward the challenge to the real engine, but the proof it gets back is keyed to the
    address it was asked on, not to the squatter's own (the port alone would not notice a
    squatter on the same port at the other loopback address, ::1 against 127.0.0.1), so the
    caller's check fails and the token never travels. The token itself never travels either,
    and without an issued token there is nothing to prove with, so the route simply does not
    exist then."""
    current = mcp_token.current_token()
    if current is None or not 16 <= len(nonce) <= 128:
        raise HTTPException(404, "not found")
    server = request.scope.get("server") or (None, None)
    if server[0] is None or server[1] is None:
        raise HTTPException(404, "not found")
    signed = f"{nonce}:{server[0]}:{server[1]}".encode()
    digest = hmac.new(current.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    return {"proof": digest}


@router.get("/status", dependencies=MCP_OK)
async def status(request: Request):
    """What an MCP app may do right now: whether the feature is on, whether its memories skip
    Review, and whether a token has been issued at all."""
    cfg = get_core(request).config.integrations
    return {
        "enabled": bool(cfg.mcp_server_enabled),
        "memories_skip_review": bool(cfg.mcp_server_memories_skip_review),
        "token_issued": mcp_token.current_token() is not None,
    }


@router.post("/token/rotate", dependencies=AUTH)
async def rotate_token(request: Request):
    """Replace the MCP-only token. Every app still holding the old one stops working at once; the
    new one is never returned here, it only ever lives in the keychain and in the MCP process."""
    if mcp_token.rotate_token() is None:
        raise HTTPException(503, "the keychain is unavailable, so no token can be stored")
    return {"rotated": True}
