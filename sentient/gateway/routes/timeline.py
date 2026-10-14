"""REST routes for the activity timeline (#147). OWNER: core. Contract: docs/API.md section 21."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from sentient.gateway.deps import AUTH, get_core

router = APIRouter(prefix="/api/timeline", tags=["timeline"], dependencies=AUTH)


def _kinds(raw: str | None) -> list[str] | None:
    return [k.strip() for k in raw.split(",") if k.strip()] if raw else None


@router.get("")
async def past(
    request: Request, before: str | None = None, limit: int = 50, kinds: str | None = None, app: str | None = None,
    q: str | None = None,
):
    return await get_core(request).timeline.past(before=before, limit=limit, kinds=_kinds(kinds), app=app, q=q)


@router.get("/upcoming")
async def upcoming(
    request: Request, days: int = 7, kinds: str | None = None, app: str | None = None, q: str | None = None,
):
    return await get_core(request).timeline.upcoming(days=days, kinds=_kinds(kinds), app=app, q=q)


@router.get("/apps")
async def apps(request: Request):
    return {"apps": await get_core(request).timeline.apps()}


class RestoreBody(BaseModel):
    id: str


@router.post("/restore")
async def restore(request: Request, body: RestoreBody):
    """Bring back what a chat tool call moved to the trash (#142): the item's ``undo`` says when this is offered."""
    out = await get_core(request).timeline.restore(body.id)
    if not out.get("ok"):
        raise HTTPException(409, out.get("error") or "That can't be restored.")
    return out


@router.post("/cancel")
async def cancel(request: Request, body: RestoreBody):
    """Cancel the timer or reminder of an upcoming item (#150): offered when its ``undo`` is ``{"action": "cancel"}``."""
    out = await get_core(request).timeline.cancel(body.id)
    if not out.get("ok"):
        raise HTTPException(409, out.get("error") or "That can't be cancelled.")
    return out
