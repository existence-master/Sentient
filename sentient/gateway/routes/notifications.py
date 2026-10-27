"""REST routes for notifications. OWNER: core. Contract: docs/API.md.

Proactive suggestion approve/dismiss lives in routes/proactivity.py.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from sentient.app_names import clean_app_name
from sentient.gateway.deps import MCP_OK, get_core

router = APIRouter(prefix="/api/notifications", tags=["notifications"], dependencies=MCP_OK)


@router.get("")
async def list_notifications(request: Request, limit: int = 100, unread_only: bool = False):
    svc = get_core(request).notifications
    return {"notifications": await svc.list(limit, unread_only), "unread": await svc.unread_count()}


@router.post("")
async def create_notification(request: Request, body: dict):
    """A plain info notification, for the MCP server's notify_me tool (ADR 0034).

    Only ever an ``info`` note in Sentient itself: nothing is sent anywhere, and a caller cannot
    dress a note up as an approval, a task result or an error by picking the kind.
    """
    message = str(body.get("message") or "").strip()
    if not message:
        raise HTTPException(400, "message is required")
    title = str(body.get("title") or "").strip() or None
    source = str(body.get("source") or "").strip()
    payload = {"source": source} if source else None
    if getattr(request.state, "caller", "gateway") == "mcp":
        # a note from an outside app is titled as one, under a fixed attribution the app cannot
        # replace: naming itself "Sentient" in `app` still lands as "From MCP app (Sentient)"
        # (issue #443)
        app_name = clean_app_name(body.get("app")) or "An MCP app"  # the name itself is cleaned too
        title = f"From MCP app ({app_name}): {title}" if title else f"From MCP app ({app_name})"
        payload = {**(payload or {}), "app": app_name}
    return await get_core(request).notify("info", message, title=title, payload=payload)


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
