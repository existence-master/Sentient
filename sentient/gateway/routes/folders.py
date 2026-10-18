"""REST routes for folders and Obsidian vaults as searchable notes (owner: folders, #176). Contract: docs/API.md
section 22."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from sentient.folders import FolderError
from sentient.gateway.deps import AUTH, get_core

router = APIRouter(prefix="/api/folders", tags=["folders"], dependencies=AUTH)


class AddBody(BaseModel):
    path: str
    ignore: list[str] | None = None
    scope: str | None = None  # the chats its notes are used in (#392); always when left out


class PatchBody(BaseModel):
    ignore: list[str] | None = None
    scope: str | None = None


class SearchBody(BaseModel):
    query: str
    folder: str = ""
    limit: int = 0


@router.get("")
async def list_folders(request: Request):
    return await get_core(request).folders.overview()


@router.post("")
async def add_folder(request: Request, body: AddBody):
    try:
        return await get_core(request).folders.add(body.path, body.ignore, body.scope)
    except FolderError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.patch("/{folder_id}")
async def update_folder(request: Request, folder_id: str, body: PatchBody):
    folder = await get_core(request).folders.update(folder_id, body.ignore, body.scope)
    if folder is None:
        raise HTTPException(404, "no such folder")
    return folder


@router.delete("/{folder_id}")
async def remove_folder(request: Request, folder_id: str, forget: bool = False):
    """``?forget=true``: also forget the memories learned from its notes (#153)."""
    done = await get_core(request).folders.remove(folder_id, forget=forget)
    if not done:
        raise HTTPException(404, "no such folder")
    return {"removed": True, "forgotten": done} if isinstance(done, dict) else {"removed": True}


@router.post("/{folder_id}/rescan", status_code=202)
async def rescan_folder(request: Request, folder_id: str):
    if not await get_core(request).folders.rescan(folder_id):
        raise HTTPException(404, "no such folder")
    return {"queued": True}


@router.post("/search")
async def search_folders(request: Request, body: SearchBody):
    return await get_core(request).folders.search(body.query, limit=body.limit or None, folder=body.folder)
