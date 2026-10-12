"""REST routes for backups, restore and snapshots (owner: core). Contract: docs/API.md section 20."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from sentient.backup import BackupError
from sentient.gateway.deps import AUTH, get_core

router = APIRouter(prefix="/api/backups", tags=["backups"], dependencies=AUTH)


class BackupBody(BaseModel):
    folder: str | None = None


class PathBody(BaseModel):
    path: str


@router.get("")
async def overview(request: Request):
    return await get_core(request).backups.overview()


@router.post("")
async def backup_now(request: Request, body: BackupBody | None = None):
    try:
        return await get_core(request).backups.backup_now(body.folder if body else None)
    except BackupError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/inspect")
async def inspect(request: Request, body: PathBody):
    try:
        return await get_core(request).backups.inspect(body.path)
    except BackupError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/restore")
async def restore(request: Request, body: PathBody):
    try:
        return await get_core(request).backups.restore(body.path)
    except BackupError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/restore")
async def cancel_restore(request: Request):
    return {"cancelled": await get_core(request).backups.cancel_restore()}


@router.delete("/last-restore")
async def dismiss_last_restore(request: Request):
    get_core(request).backups.dismiss_last_restore()
    return {"ok": True}
