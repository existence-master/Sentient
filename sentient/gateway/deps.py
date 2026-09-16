"""Shared FastAPI dependencies for route modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Request, WebSocket

from sentient.gateway.auth import require_token

if TYPE_CHECKING:  # pragma: no cover
    from sentient.app import SentientApp

AUTH = [Depends(require_token)]


def get_core(request: Request) -> SentientApp:
    return request.app.state.sentient


def ws_core(ws: WebSocket) -> SentientApp:
    return ws.app.state.sentient
