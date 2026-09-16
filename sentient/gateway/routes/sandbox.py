"""REST routes for the sandbox package (owner: sandbox agent). Contract: docs/API.md section 11."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from sentient.gateway.deps import AUTH, get_core

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"], dependencies=AUTH)


class RunBody(BaseModel):
    code: str = Field(..., max_length=200_000)


@router.get("/status")
async def sandbox_status(request: Request) -> dict:
    return await get_core(request).sandbox.status()


@router.post("/run")
async def sandbox_run(body: RunBody, request: Request) -> dict:
    """Run a script whose tool calls are limited to risk ``read`` ("Run again", script-job tests)."""
    return await get_core(request).sandbox.run(body.code, channel="desktop", read_only=True)
