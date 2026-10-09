"""REST routes for the terminal package (owner: terminal). Contract: docs/API.md section 18."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from sentient.gateway.deps import AUTH, get_core

router = APIRouter(prefix="/api/terminal", tags=["terminal"], dependencies=AUTH)


class StopBody(BaseModel):
    id: str = Field(..., min_length=1, max_length=200)


@router.get("/status")
async def terminal_status(request: Request) -> dict:
    return get_core(request).terminal.status()


@router.post("/stop")
async def terminal_stop(body: StopBody, request: Request) -> dict:
    """Kill one running command (its card's Stop button). ``id`` is the tool call id."""
    return {"stopped": get_core(request).terminal.stop_command(body.id)}
