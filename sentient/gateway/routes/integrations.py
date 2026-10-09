"""REST routes for integrations. OWNER: INTEGRATIONS AGENT. Contract: docs/API.md section 5."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from sentient.gateway.deps import AUTH, get_core
from sentient.integrations.base import IntegrationError

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

router = APIRouter(prefix="/api/integrations", tags=["integrations"], dependencies=AUTH)


def _mgr(request: Request) -> IntegrationManager:
    return get_core(request).integrations


class ConnectBody(BaseModel):
    fields: dict[str, Any] = Field(default_factory=dict)


class PrivacyFilters(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class MCPServerBody(BaseModel):
    name: str
    transport: Literal["stdio", "http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    auth: Literal["none", "headers", "oauth"] | None = None
    enabled: bool = True


# ----------------------------------------------------------------------------- list
@router.get("")
async def list_integrations(request: Request):
    return await _mgr(request).list_integrations()


# ----------------------------------------------------------------------------- MCP servers (before /{id})
@router.get("/mcp")
async def list_mcp(request: Request):
    return _mgr(request).mcp.list()


@router.post("/mcp")
async def add_mcp(request: Request, body: MCPServerBody):
    try:
        return await _mgr(request).mcp.add(body.name, body.model_dump(exclude={"name"}))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/mcp/{name}")
async def delete_mcp(request: Request, name: str):
    if not await _mgr(request).mcp.remove(name):
        raise HTTPException(404, f"no MCP server named {name}")
    return {"ok": True}


@router.post("/mcp/{name}/test")
async def test_mcp(request: Request, name: str):
    try:
        return await _mgr(request).mcp.test(name)
    except KeyError as exc:
        raise HTTPException(404, f"no MCP server named {name}") from exc


@router.post("/mcp/{name}/sign-in")
async def sign_in_mcp(request: Request, name: str):
    try:
        return await _mgr(request).mcp.sign_in(name)
    except KeyError as exc:
        raise HTTPException(404, f"no MCP server named {name}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/mcp/{name}/sign-out")
async def sign_out_mcp(request: Request, name: str):
    try:
        return await _mgr(request).mcp.sign_out(name)
    except KeyError as exc:
        raise HTTPException(404, f"no MCP server named {name}") from exc


# ----------------------------------------------------------------------------- change feeds (before /{id})
@router.get("/feeds")
async def feed_status(request: Request):
    return await _mgr(request).feed_status()


@router.post("/feeds/{source}/sync")
async def feed_sync(request: Request, source: str):
    mgr = _mgr(request)
    p = mgr.plugin(source)
    if p is None or getattr(p, "change_feed", None) is None:
        raise HTTPException(404, f"{source} has no change feed")
    return await mgr.feeds.sync(source)


# ----------------------------------------------------------------------------- one integration
@router.get("/{integration_id}")
async def get_integration(request: Request, integration_id: str):
    try:
        return await _mgr(request).integration(integration_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown integration {integration_id}") from exc


@router.post("/{integration_id}/connect")
async def connect(request: Request, integration_id: str, body: ConnectBody | None = None):
    try:
        return await _mgr(request).connect(integration_id, (body or ConnectBody()).fields)
    except KeyError as exc:
        raise HTTPException(404, f"unknown integration {integration_id}") from exc
    except IntegrationError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{integration_id}/cancel")
async def cancel_connect(request: Request, integration_id: str):
    """Abandon a pending OAuth/device sign-in so the integration stops showing 'connecting'."""
    try:
        return await _mgr(request).cancel_connect(integration_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown integration {integration_id}") from exc


@router.post("/{integration_id}/disconnect")
async def disconnect(request: Request, integration_id: str):
    try:
        return await _mgr(request).disconnect(integration_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown integration {integration_id}") from exc


@router.post("/{integration_id}/test")
async def test(request: Request, integration_id: str):
    try:
        return await _mgr(request).test(integration_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown integration {integration_id}") from exc


@router.get("/{integration_id}/privacy-filters")
async def get_privacy_filters(request: Request, integration_id: str):
    try:
        return await _mgr(request).get_privacy_filters(integration_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown integration {integration_id}") from exc


@router.put("/{integration_id}/privacy-filters")
async def put_privacy_filters(request: Request, integration_id: str, body: PrivacyFilters):
    mgr = _mgr(request)
    plugin = mgr.plugin(integration_id)
    if plugin is None:
        raise HTTPException(404, f"unknown integration {integration_id}")
    if not plugin.privacy_fields:
        raise HTTPException(400, f"{plugin.display_name} doesn't support privacy filters")
    await mgr.set_privacy_filters(integration_id, body.model_dump())
    return {"ok": True}
