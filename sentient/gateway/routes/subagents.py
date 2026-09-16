"""REST routes for chat subagents (owner: core). Contract: docs/API.md section 10."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from sentient.gateway.deps import AUTH, get_core

router = APIRouter(tags=["subagents"])


@router.get("/api/sessions/{session_id}/subagents", dependencies=AUTH)
async def session_subagents(request: Request, session_id: str, limit: int = 100):
    return await get_core(request).subagents.list_for_session(session_id, limit)


@router.get("/api/subagents/{subagent_id}", dependencies=AUTH)
async def get_subagent(request: Request, subagent_id: str):
    sub = await get_core(request).subagents.get(subagent_id)
    if sub is None:
        raise HTTPException(404, "no such subagent")
    return sub


@router.post("/api/subagents/{subagent_id}/cancel", dependencies=AUTH)
async def cancel_subagent(request: Request, subagent_id: str):
    sub = await get_core(request).subagents.cancel(subagent_id)
    if sub is None:
        raise HTTPException(404, "no such subagent")
    return sub
