"""REST routes for the Skills hub (#210): sources, browsing, preview, install and updates. Contract: docs/API.md section 8.

Installs and updates land in pending review (``/api/skills`` ``pending``) and are approved or declined there.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from sentient.gateway.deps import AUTH, get_core
from sentient.skills.hub import HubConfirm, HubError

router = APIRouter(prefix="/api/skills/hub", tags=["skills"], dependencies=AUTH)


def _hub(request: Request):
    return get_core(request).skills_hub


def _confirm(exc: HubConfirm) -> JSONResponse:
    """409 with a ``code`` the window answers with an extra yes: ``confirm_source`` or ``high_risk``."""
    return JSONResponse(status_code=409, content={"detail": str(exc), "code": exc.code, "vetting": exc.vetting})


def _error(exc: HubError) -> HTTPException:
    return HTTPException(exc.status, str(exc))


@router.get("/sources")
async def list_sources(request: Request):
    return [s.to_dict() for s in _hub(request).sources()]


@router.post("/sources")
async def add_source(request: Request, body: dict):
    try:
        return _hub(request).add_source(str(body.get("url") or ""), body.get("name"))
    except HubError as exc:
        raise _error(exc) from exc


@router.delete("/sources/{sid}")
async def remove_source(request: Request, sid: str):
    try:
        _hub(request).remove_source(sid)
    except HubError as exc:
        raise _error(exc) from exc
    return {"ok": True}


@router.get("/browse")
async def browse(request: Request, source: str, refresh: bool = False):
    try:
        return await _hub(request).browse(source, refresh=refresh)
    except HubError as exc:
        raise _error(exc) from exc


@router.get("/preview")
async def preview(request: Request, source: str, path: str):
    try:
        return await _hub(request).preview(source, path)
    except HubError as exc:
        raise _error(exc) from exc


@router.post("/install")
async def install(request: Request, body: dict):
    """Fetch a skill, run the safety check and stage it in pending review, pinned in the lockfile."""
    from sentient.gateway.routes.skills import _changed, _skill

    s = get_core(request)
    try:
        name = await s.skills_hub.install(
            s, str(body.get("source") or ""), str(body.get("path") or ""),
            confirm_source=body.get("confirm_source") is True, accept_risk=body.get("accept_risk") is True,
        )
    except HubConfirm as exc:
        return _confirm(exc)
    except HubError as exc:
        raise _error(exc) from exc
    await _changed(s, name, "pending_review")
    return await _skill(s, name, pending=True)


@router.get("/installed")
async def installed(request: Request, check: bool = False):
    """Skills installed from the hub, from the lockfile. ``check=true`` asks each source for a newer version."""
    hub = _hub(request)
    if check:
        return {"skills": await hub.check_updates()}
    return {"skills": [hub.summary(n, e) for n, e in sorted(hub.lock.all().items())]}


@router.post("/update/{name}")
async def update(request: Request, name: str, body: dict | None = None):
    """Fetch the newest version of an installed skill into pending review, with what changed."""
    from sentient.gateway.routes.skills import _changed, _skill

    s = get_core(request)
    try:
        res = await s.skills_hub.update(s, name, accept_risk=(body or {}).get("accept_risk") is True)
    except HubConfirm as exc:
        return _confirm(exc)
    except HubError as exc:
        raise _error(exc) from exc
    if res["changed"]:
        await _changed(s, res["name"], "pending_review")
        res["skill"] = await _skill(s, res["name"], pending=True)
    return res
