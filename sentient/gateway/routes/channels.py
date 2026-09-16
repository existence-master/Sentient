"""REST routes for messaging channels (owner: channels agent). Contract: docs/API.md section 14."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from sentient.channels.base import ChannelError
from sentient.gateway.deps import AUTH, get_core

router = APIRouter(prefix="/api/channels", tags=["channels"], dependencies=AUTH)


async def _run(coro: Any) -> Any:
    try:
        return await coro
    except ChannelError as exc:
        raise HTTPException(exc.status, exc.message) from exc


@router.get("")
async def list_channels(request: Request):
    return await get_core(request).channels.list()


@router.post("/{channel_id}/connect")
async def connect(request: Request, channel_id: str, body: dict | None = Body(None)):
    fields = (body or {}).get("fields") or {}
    if not isinstance(fields, dict):
        raise HTTPException(400, "fields must be an object")
    return await _run(get_core(request).channels.connect(channel_id, fields))


@router.post("/{channel_id}/disconnect")
async def disconnect(request: Request, channel_id: str):
    return await _run(get_core(request).channels.disconnect(channel_id))


@router.post("/{channel_id}/pairing")
async def create_pairing(request: Request, channel_id: str):
    return await _run(get_core(request).channels.create_pairing(channel_id))


@router.patch("/{channel_id}/paired/{chat_id}")
async def update_paired(request: Request, channel_id: str, chat_id: str, body: dict | None = Body(None)):
    if not isinstance((body or {}).get("deliver"), bool):
        raise HTTPException(400, "deliver must be true or false")
    return await _run(get_core(request).channels.set_deliver(channel_id, chat_id, body["deliver"]))  # type: ignore[index]


@router.delete("/{channel_id}/paired/{chat_id}")
async def remove_paired(request: Request, channel_id: str, chat_id: str):
    return await _run(get_core(request).channels.remove_chat(channel_id, chat_id))


@router.post("/{channel_id}/test")
async def test_channel(request: Request, channel_id: str, body: dict | None = Body(None)):
    chat_id = (body or {}).get("chat_id")
    return await _run(get_core(request).channels.test(channel_id, str(chat_id) if chat_id else None))
