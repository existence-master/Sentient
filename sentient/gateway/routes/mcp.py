"""The MCP server's own endpoints (issue #443, ADR 0034): a status the MCP process can check with
its narrow token, and the token replacement only the user's own surfaces may ask for."""

from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.routing import APIRouter

from sentient.gateway import mcp_token
from sentient.gateway.deps import AUTH, MCP_OK, get_core

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


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
