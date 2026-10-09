"""Inbound webhooks. OWNER: INTEGRATIONS AGENT. Contract: docs/API.md section 16.

- ``/api/hooks``          manage hooks (bearer token, like every /api route)
- ``POST /hooks/{id}``    public: no bearer token; the caller proves itself with the hook secret
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from sentient.gateway.deps import AUTH, get_core
from sentient.integrations.hooks import SECRET_HEADER

manage = APIRouter(prefix="/api/hooks", tags=["hooks"], dependencies=AUTH)
public = APIRouter(tags=["hooks"])  # deliberately no AUTH: the per-hook secret authenticates


class HookBody(BaseModel):
    name: str


def _base_url(request: Request) -> str:
    # The same /hooks/{id} path is also served by the devices LAN listener when it is on.
    return str(request.base_url).rstrip("/")


@manage.get("")
async def list_hooks(request: Request):
    return await get_core(request).integrations.hooks.list(_base_url(request))


@manage.post("")
async def create_hook(request: Request, body: HookBody):
    try:
        return await get_core(request).integrations.hooks.create(body.name, _base_url(request))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@manage.delete("/{hook_id}")
async def delete_hook(request: Request, hook_id: str):
    if not await get_core(request).integrations.hooks.delete(hook_id):
        raise HTTPException(404, "There is no webhook with that id.")
    return {"ok": True}


async def _read_limited(request: Request, limit: int) -> bytes:
    too_big = HTTPException(413, f"The request body is larger than {limit // 1024} KB.")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise too_big
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > limit:
            raise too_big
    return bytes(buf)


async def _parse_body(request: Request, raw: bytes) -> tuple[Any, str]:
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if not raw:
        return None, ctype
    if ctype.endswith("json") or (not ctype and raw.lstrip()[:1] in (b"{", b"[")):
        try:
            return json.loads(raw), ctype
        except ValueError as exc:
            if ctype.endswith("json"):
                raise HTTPException(400, "The request body isn't valid JSON.") from exc
    if ctype == "application/x-www-form-urlencoded":
        parsed = parse_qs(raw.decode("utf-8", errors="replace"), keep_blank_values=True)
        return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}, ctype
    if ctype == "multipart/form-data":
        request._body = raw  # already read with the size limit; let Starlette parse the cached body
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(400, "The form data couldn't be read.") from exc
        out: dict[str, Any] = {}
        for key, value in form.multi_items():
            v: Any = ({"filename": value.filename, "content_type": value.content_type, "size": value.size}
                      if isinstance(value, UploadFile) else value)
            if key in out:
                out[key] = [*out[key], v] if isinstance(out[key], list) else [out[key], v]
            else:
                out[key] = v
        return out, ctype
    return raw.decode("utf-8", errors="replace"), ctype


@public.post("/hooks/{hook_id}")
async def call_hook(request: Request, hook_id: str):
    core = get_core(request)
    hooks = core.integrations.hooks
    hook = await hooks.get(hook_id)
    if hook is None:
        raise HTTPException(404, "Unknown webhook.")
    given = request.headers.get(SECRET_HEADER) or request.query_params.get("secret")
    if not hooks.secret_ok(hook, given):
        raise HTTPException(401, "Wrong or missing webhook secret.")
    if core.stopped:  # Stop everything: callers retry later instead of losing the call
        raise HTTPException(503, "Sentient is stopped right now. Try again after it is resumed.", headers={"Retry-After": "60"})
    # After the secret check, so callers without the secret can't use up a real caller's budget.
    wait = hooks.rate_limited(hook["id"], core.config.integrations.webhook_rate_limit_per_minute)
    if wait is not None:
        raise HTTPException(
            429, f"This webhook is being called too often. Try again in {wait} s.", headers={"Retry-After": str(wait)}
        )
    raw = await _read_limited(request, core.config.integrations.webhook_max_body_kb * 1024)
    body, ctype = await _parse_body(request, raw)
    query = {k: v for k, v in request.query_params.items() if k != "secret"}
    item = await hooks.receive(hook, body, content_type=ctype, query=query)
    return {"ok": True, "item_id": item["id"]}


router = APIRouter()
router.include_router(manage)
router.include_router(public)
