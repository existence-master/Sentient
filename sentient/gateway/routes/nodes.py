"""Device (node) routes. OWNER: NODES AGENT. Contract: docs/API.md section 13, docs/NODES.md.

``router`` is mounted on the loopback gateway: REST under ``/api/nodes`` (gateway token) plus
everything in ``device_router``. ``device_router`` is also the whole HTTP surface of the LAN
listener (``sentient.nodes.lan``): the node socket, the web device app and uploads, all
authenticated with node tokens, never the gateway token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from sentient.gateway.deps import AUTH, get_core
from sentient.nodes.service import MAX_UPLOAD_BYTES

router = APIRouter(tags=["nodes"])
device_router = APIRouter(tags=["nodes"])

WEB_DIR = (Path(__file__).resolve().parents[2] / "nodes" / "web").resolve()
MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml", ".png": "image/png", ".json": "application/json", ".webmanifest": "application/manifest+json",
    ".ico": "image/x-icon",
}  # fmt: skip


# ---------------------------------------------------------------------- device-facing (gateway + LAN)
@device_router.websocket("/ws/node")
async def node_socket(ws: WebSocket):
    core = ws.app.state.sentient
    await core.nodes.serve_socket(ws, lan=bool(getattr(ws.app.state, "lan", False)))


@device_router.get("/node", include_in_schema=False)
async def web_app_redirect():
    return RedirectResponse("/node/")


@device_router.get("/node/{path:path}", include_in_schema=False)
async def web_app(path: str):
    target = (WEB_DIR / (path or "index.html")).resolve()
    if target.is_dir():
        target = target / "index.html"
    if not target.is_relative_to(WEB_DIR) or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(
        target,
        media_type=MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream"),
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"},
    )


@device_router.post("/api/nodes/upload")
async def upload(request: Request):
    """Large payloads (photos) from a device. ``Authorization: Bearer <node token>``; raw body or multipart ``file``."""
    core = request.app.state.sentient
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else request.query_params.get("node_token", "")
    node = await core.nodes.verify_token(token)
    if node is None:
        raise HTTPException(401, "invalid or missing device token")
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "upload too large (max 20 MB)")
    ctype = request.headers.get("content-type", "application/octet-stream")
    if ctype.lower().startswith("multipart/"):
        form = await request.form()
        file: Any = form.get("file")
        if file is None or not hasattr(file, "read"):
            raise HTTPException(400, "multipart upload needs a 'file' field")
        data = await file.read()
        ctype = file.content_type or "application/octet-stream"
    else:
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "upload too large (max 20 MB)")
            chunks.append(chunk)
        data = b"".join(chunks)
    if not data:
        raise HTTPException(400, "empty upload")
    return await core.nodes.save_upload(node, data, ctype)


# ---------------------------------------------------------------------- desktop-facing REST (gateway token)
class RenameBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)


class InvokeBody(BaseModel):
    capability: str
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int | None = None


@router.get("/api/nodes", dependencies=AUTH)
async def list_nodes(request: Request):
    return await get_core(request).nodes.list_nodes()


@router.post("/api/nodes/pairing", dependencies=AUTH)
async def create_pairing(request: Request):
    return get_core(request).nodes.create_pairing(str(request.base_url))


@router.get("/api/nodes/lan", dependencies=AUTH)
async def lan_status(request: Request):
    return get_core(request).nodes.lan_status()


@router.get("/api/nodes/{node_id}", dependencies=AUTH)
async def get_node(request: Request, node_id: str):
    node = await get_core(request).nodes.get_node(node_id)
    if node is None:
        raise HTTPException(404, "device not found")
    return node


@router.patch("/api/nodes/{node_id}", dependencies=AUTH)
async def rename_node(request: Request, node_id: str, body: RenameBody):
    node = await get_core(request).nodes.rename(node_id, body.name)
    if node is None:
        raise HTTPException(404, "device not found")
    return node


@router.delete("/api/nodes/{node_id}", dependencies=AUTH)
async def delete_node(request: Request, node_id: str):
    try:
        ok = await get_core(request).nodes.delete(node_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not ok:
        raise HTTPException(404, "device not found")
    return {"ok": True}


@router.post("/api/nodes/{node_id}/invoke", dependencies=AUTH)
async def invoke_node(request: Request, node_id: str, body: InvokeBody):
    svc = get_core(request).nodes
    if await svc.get_node(node_id) is None:
        raise HTTPException(404, "device not found")
    return await svc.invoke(node_id, body.capability, body.params, body.timeout_ms)


router.include_router(device_router)
