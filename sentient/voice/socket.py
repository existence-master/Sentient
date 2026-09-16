"""The ``/ws/voice`` websocket transport, reusable by any Starlette/FastAPI app.

The gateway mounts it at ``/ws/voice``; the nodes LAN listener can mount the same
endpoint so devices talk to Sentient over the local network::

    from sentient.voice.socket import voice_socket_endpoint
    lan_app.state.sentient = core               # required
    lan_app.add_api_websocket_route("/ws/voice", voice_socket_endpoint)

or, when it authenticates the device itself::

    from sentient.voice.socket import serve_voice_socket
    await serve_voice_socket(ws, core, node=node)   # ws not yet accepted

Authentication (``authorize_voice_socket``): the gateway token (``?token=`` or a
bearer header) when the app has ``state.token``, or ``?node_token=`` checked with
``await core.nodes.verify_token(token)``, which returns the Node dict or None.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import secrets as pysecrets
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)

UNAUTHORIZED = 4401


def _gateway_token_ok(ws: WebSocket) -> bool:
    expected = getattr(ws.app.state, "token", None)
    if not expected:
        return False
    given = ws.query_params.get("token") or ""
    header = ws.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        given = header[7:].strip()
    return bool(given) and pysecrets.compare_digest(given, expected)


def _node_dict(node: Any) -> dict[str, Any]:
    if isinstance(node, dict):
        return dict(node)
    if hasattr(node, "model_dump"):
        return node.model_dump()
    return {k: getattr(node, k, None) for k in ("node_id", "name", "kind", "platform")}


async def authorize_voice_socket(ws: WebSocket, core: Any) -> tuple[bool, dict[str, Any] | None]:
    """-> (allowed, node). ``node`` is None for the desktop app (gateway token).

    On the nodes LAN listener a middleware has already validated ``?node_token=``, swapped in the listener's own
    token and put the Node in ``scope["state"]["node"]``; that node is used as is (no second lookup).
    """
    if _gateway_token_ok(ws):
        scope_state = ws.scope.get("state") or {}
        scope_node = scope_state.get("node") if isinstance(scope_state, dict) else getattr(scope_state, "node", None)
        return True, (_node_dict(scope_node) if scope_node else None)
    node_token = ws.query_params.get("node_token") or ""
    if not node_token:
        return False, None
    verify = getattr(getattr(core, "nodes", None), "verify_token", None)
    if not callable(verify):
        log.warning("voice socket: node_token given but the nodes service cannot verify tokens")
        return False, None
    try:
        node = verify(node_token)
        if inspect.isawaitable(node):
            node = await node
    except Exception as exc:
        log.warning("voice socket: node token check failed: %s", exc)
        return False, None
    if not node:
        return False, None
    return True, _node_dict(node)


async def serve_voice_socket(ws: WebSocket, core: Any, *, node: dict[str, Any] | None = None) -> None:
    """Accept an already-authorized websocket and run one voice session on it until it closes."""
    await ws.accept()

    async def send_json(obj: dict) -> None:
        await ws.send_text(json.dumps(obj, ensure_ascii=False, default=str))

    session = core.voice.open_session(send_json, ws.send_bytes, node=node)
    stopped = False
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session.handle_audio(message["bytes"])
                continue
            raw = message.get("text")
            if raw is None:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await session.send({"type": "error", "message": "invalid json", "recoverable": True})
                continue
            if not isinstance(msg, dict):
                continue
            if await session.handle_json(msg) == "stop":
                stopped = True
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        # an explicit stop cancels the turn; a dropped socket lets the reply finish and persist
        await core.voice.close_session(session, cancel_turn=stopped)
        if stopped:
            with contextlib.suppress(Exception):
                await ws.close()


async def voice_socket_endpoint(ws: WebSocket) -> None:
    """Complete endpoint: authorize (gateway token or node_token), then serve. Needs ``app.state.sentient``."""
    core = getattr(ws.app.state, "sentient", None)
    if core is None:
        await ws.close(code=1011)
        return
    allowed, node = await authorize_voice_socket(ws, core)
    if not allowed:
        await ws.close(code=UNAUTHORIZED)
        return
    await serve_voice_socket(ws, core, node=node)
