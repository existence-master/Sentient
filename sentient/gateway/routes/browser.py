"""REST routes for the browser package (owner: browser agent). Contract: docs/API.md section 12."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from sentient.browser.service import BrowserError
from sentient.gateway.deps import AUTH, get_core

router = APIRouter(tags=["browser"], dependencies=AUTH)


class OpenBody(BaseModel):
    url: str | None = None
    profile: str | None = None


class ProfileBody(BaseModel):
    name: str
    kind: str = "launch"
    engine: str = ""
    endpoint: str = ""
    notes: str = ""


class ProfilePatch(BaseModel):
    name: str | None = None
    engine: str | None = None
    endpoint: str | None = None
    notes: str | None = None


@router.get("/api/browser/status")
async def browser_status(request: Request):
    return await get_core(request).browser.status()


@router.post("/api/browser/open")
async def browser_open(request: Request, body: OpenBody | None = None):
    """Show the browser in a visible window on the same profile so the user can sign in themselves."""
    try:
        return await get_core(request).browser.open_for_user(
            (body.url if body else None) or None, (body.profile if body else None) or None
        )
    except BrowserError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/api/browser/profiles")
async def browser_profiles(request: Request):
    return get_core(request).browser.profiles()


@router.post("/api/browser/profiles")
async def browser_profile_create(request: Request, body: ProfileBody):
    try:
        return await get_core(request).browser.create_profile(
            body.name, body.kind, body.engine, body.endpoint, body.notes
        )
    except BrowserError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.patch("/api/browser/profiles/{name}")
async def browser_profile_update(request: Request, name: str, body: ProfilePatch):
    try:
        return await get_core(request).browser.update_profile(
            name, new_name=body.name, engine=body.engine, endpoint=body.endpoint, notes=body.notes
        )
    except BrowserError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/api/browser/profiles/{name}")
async def browser_profile_delete(request: Request, name: str):
    try:
        return await get_core(request).browser.delete_profile(name)
    except BrowserError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/browser/close")
async def browser_close(request: Request):
    return await get_core(request).browser.close()


@router.get("/api/browser/screenshot")
async def browser_screenshot(request: Request):
    image = await get_core(request).browser.capture_jpeg()
    if image is None:
        raise HTTPException(status_code=409, detail="The browser isn't open right now.")
    return Response(content=image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
