"""REST routes for moving from another assistant (owner: core). Contract: docs/API.md section 19."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from sentient.gateway.deps import AUTH, get_core
from sentient.migrate import exports, hermes, memory_import

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


# ---------------------------------------------------------------------------- memory from ChatGPT, Claude, OpenClaw
class MemoryPreviewBody(BaseModel):
    path: str | None = None


class MemoryApplyBody(BaseModel):
    path: str | None = None
    parts: list[str] = Field(default_factory=list)
    skip: list[str] = Field(default_factory=list)


def _app_name(app: str) -> str:
    if app not in exports.APPS:
        raise HTTPException(404, f"Sentient can't import from {app}.")
    return app


@router.get("/memory/job")
async def memory_job(request: Request):
    """The import running now, or the last one since Sentient started (null when there was none)."""
    return get_core(request).memory_imports.current()


@router.post("/memory/job/cancel")
async def memory_job_cancel(request: Request):
    return get_core(request).memory_imports.cancel()


@router.get("/{app}")
async def memory_default(request: Request, app: str):
    """Where that app keeps its memory on this computer (OpenClaw), else ``path: null``."""
    path = exports.default_path(_app_name(app))
    return {"path": str(path) if path else None, "exists": bool(path and path.is_dir())}


@router.post("/{app}/preview")
async def memory_preview(request: Request, app: str, body: MemoryPreviewBody):
    try:
        return await memory_import.preview(get_core(request), _app_name(app), body.path)
    except exports.ExportError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{app}/apply")
async def memory_apply(request: Request, app: str, body: MemoryApplyBody):
    try:
        return await get_core(request).memory_imports.start(_app_name(app), body.path, body.parts, body.skip)
    except memory_import.ImportBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except exports.ExportError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/{app}/memories")
async def memory_remove(request: Request, app: str):
    try:
        return await memory_import.remove(get_core(request), _app_name(app))
    except memory_import.ImportBusy as exc:
        raise HTTPException(409, str(exc)) from exc
