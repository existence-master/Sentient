"""REST routes for notifications. OWNER: core. Contract: docs/API.md.

Proactive suggestion approve/dismiss lives in routes/proactivity.py.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from sentient.gateway.deps import AUTH, get_core

router = APIRouter(prefix="/api/notifications", tags=["notifications"], dependencies=AUTH)


@router.get("")
async def list_notifications(request: Request, limit: int = 100, unread_only: bool = False):
    svc = get_core(request).notifications
    return {"notifications": await svc.list(limit, unread_only), "unread": await svc.unread_count()}


@router.post("/{notification_id}/read")
async def mark_read(request: Request, notification_id: str):
    svc = get_core(request).notifications
    if not await svc.get(notification_id):
        raise HTTPException(404, "not found")
    await svc.mark_read(notification_id)
    return {"ok": True}


@router.post("/read-all")
async def mark_all_read(request: Request):
    await get_core(request).notifications.mark_read(None)
    return {"ok": True}


@router.delete("/{notification_id}")
async def delete_notification(request: Request, notification_id: str):
    await get_core(request).notifications.delete(notification_id)
    return {"ok": True}


@router.delete("")
async def delete_all(request: Request):
    await get_core(request).notifications.delete(None)
    return {"ok": True}
