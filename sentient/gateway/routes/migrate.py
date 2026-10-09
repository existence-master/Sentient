"""REST routes for moving from another assistant (owner: core). Contract: docs/API.md section 18."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from sentient.gateway.deps import AUTH, get_core
from sentient.migrate import hermes

router = APIRouter(prefix="/api/import", tags=["import"], dependencies=AUTH)


class HermesPreviewBody(BaseModel):
    path: str | None = None


class HermesApplyBody(BaseModel):
    path: str | None = None
    parts: list[str] = Field(default_factory=list)
    skip: list[str] = Field(default_factory=list)


@router.get("/hermes")
async def hermes_default(request: Request):
    """Where Hermes usually lives and whether that folder is there (the import screen's starting point)."""
    home = hermes.default_home()
    return {"path": str(home), "exists": home.is_dir()}


@router.post("/hermes/preview")
async def hermes_preview(request: Request, body: HermesPreviewBody):
    try:
        return await hermes.preview(get_core(request), body.path)
    except hermes.HermesImportError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/hermes/apply")
async def hermes_apply(request: Request, body: HermesApplyBody):
    try:
        return await hermes.apply(get_core(request), body.path, body.parts, body.skip)
    except hermes.HermesImportError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/hermes/memories")
async def hermes_remove_memories(request: Request):
    return await hermes.remove_memories(get_core(request))
