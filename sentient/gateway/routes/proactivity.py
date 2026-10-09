"""REST routes for proactivity. OWNER: MEMORY/PROACTIVITY AGENT. Contract: docs/API.md section 6."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from sentient.gateway.deps import AUTH, get_core
from sentient.proactivity.brief import BriefError
from sentient.proactivity.service import SuggestionError

router = APIRouter(prefix="/api/proactivity", tags=["proactivity"], dependencies=AUTH)


@router.post("/suggestions/{notification_id}")
async def act_on_suggestion(request: Request, notification_id: str, body: dict):
    s = get_core(request)
    try:
        return await s.proactivity.act_on_suggestion(notification_id, str(body.get("action", "")))
    except SuggestionError as exc:
        raise HTTPException(exc.status, exc.detail) from exc


@router.get("/status")
async def status(request: Request):
    return await get_core(request).proactivity.status()


@router.post("/poll-now")
async def poll_now(request: Request):
    return await get_core(request).proactivity.poll_now()


@router.get("/preferences")
async def preferences(request: Request):
    return await get_core(request).proactivity.preferences()


@router.delete("/preferences/{suggestion_type}")
async def reset_preference(request: Request, suggestion_type: str):
    return {"ok": await get_core(request).proactivity.reset_preference(suggestion_type)}


# ----------------------------------------------------------------------------- Daily Brief
@router.get("/brief")
async def brief_state(request: Request):
    return await get_core(request).proactivity.brief.state()


@router.post("/brief")
async def brief_setup(request: Request, body: dict | None = None):
    try:
        return await get_core(request).proactivity.brief.setup(body or {})
    except BriefError as exc:
        raise HTTPException(exc.status, exc.detail) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/brief/run")
async def brief_run(request: Request):
    try:
        return await get_core(request).proactivity.brief.run_now()
    except BriefError as exc:
        raise HTTPException(exc.status, exc.detail) from exc


@router.post("/brief/feedback")
async def brief_feedback(request: Request, body: dict):
    try:
        return await get_core(request).proactivity.brief.feedback(
            str(body.get("brief_id") or ""), str(body.get("value") or ""),
            item=body.get("item_id") or None, section=body.get("section") or None,
        )
    except BriefError as exc:
        raise HTTPException(exc.status, exc.detail) from exc
