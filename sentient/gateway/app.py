"""The gateway: the single local process the desktop app talks to.

REST routers live in ``sentient/gateway/routes`` (one module per feature
package, each owned by that package). This module wires them together and
implements the live WebSocket at ``/ws`` which carries:

- chat turns (client ``chat.send`` / ``chat.steer`` / ``chat.cancel``; server agent events), and
- every domain event published on the in-process EventBus
  (``task.updated``, ``notification.new``, ``integration.updated`` ...).

See docs/API.md for the full contract.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sentient import __version__
from sentient.app import SentientApp
from sentient.gateway.auth import load_or_create_token, ws_token_ok
from sentient.gateway.routes import ROUTERS

log = logging.getLogger(__name__)

DROPPED_NOTE = "Stopped. Your queued message wasn't sent."

# When present, the built desktop renderer is also served over HTTP so the UI
# can be opened in a normal browser during development.
UI_DIST = Path(__file__).resolve().parents[2] / "desktop" / "out" / "renderer"


def create_app(sentient: SentientApp | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.token = load_or_create_token()
        # SENTIENT_DISABLE_BACKGROUND=1 starts the engine without schedulers, pollers, resume or reviews
        # (demo/screenshot profiles, debugging). The desktop app never sets it.
        background = os.environ.get("SENTIENT_DISABLE_BACKGROUND", "").strip().lower() not in {"1", "true", "yes"}
        app.state.sentient = sentient or SentientApp(enable_background=background)
        await app.state.sentient.start()
        yield
        await app.state.sentient.stop()

    app = FastAPI(title="Sentient", version=__version__, lifespan=lifespan)
    # The renderer runs on file:// or the Vite dev server; every API call carries the
    # bearer token, so a permissive CORS policy without credentials is safe.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
    )
    for router in ROUTERS:
        app.include_router(router)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        if not ws_token_ok(ws):
            await ws.close(code=4401)
            return
        await ws.accept()
        s: SentientApp = ws.app.state.sentient
        assert s.agent is not None
        send_lock = asyncio.Lock()
        # per session: the running turn plus turns queued behind it (a message sent while a turn was finishing)
        turns: dict[str, list[asyncio.Task]] = {}
        closed = False

        async def send(obj: dict) -> None:
            if closed:
                return
            async with send_lock:
                with contextlib.suppress(Exception):
                    await ws.send_text(json.dumps(obj, ensure_ascii=False, default=str))

        async def run_chat(session_id: str, msg: dict, previous: asyncio.Task | None, generation: int) -> None:
            me = asyncio.current_task()
            turn = None
            stopped: dict = {}
            try:
                if previous is not None and not previous.done():
                    await asyncio.wait({previous})
                if s.stop_generation != generation:
                    # queued behind a reply when Stop everything was pressed: never sent (docs/API.md section 17)
                    # client_id lets the window match these to the message it queued, not the latest turn
                    client_id = msg.get("client_id")
                    await send({"type": "error", "message": DROPPED_NOTE, "session_id": session_id, "recoverable": True,
                                "dropped": True, "client_id": client_id})
                    await send({"type": "done", "content": "", "session_id": session_id, "cancelled": True,
                                "dropped": [str(msg.get("text", ""))], "client_id": client_id})
                    return
                turn = s.agent.run_turn(
                    session_id,
                    str(msg.get("text", "")),
                    channel=str(msg.get("channel", "desktop")),
                    attachments=list(msg.get("attachments") or []),
                    model=msg.get("model") or None,
                    on_stopped=stopped.update,
                )
                async for event in turn:
                    # keep consuming even if the window went away so the turn is fully persisted
                    await send(event.model_dump())
            except asyncio.CancelledError:
                if turn is not None:  # stopped between events: close it now so the kept reply is saved first
                    with contextlib.suppress(Exception):
                        await turn.aclose()
                await send({"type": "error", "message": "Stopped.", "session_id": session_id, "recoverable": True})
                # the kept reply's id and memory sources, so the window shows them without a reload
                kept = {"message_id": stopped["message_id"]} if stopped.get("message_id") else {}
                await send({"type": "done", "content": "", "session_id": session_id, "cancelled": True, **kept,
                            "memory_sources": stopped.get("memory_sources") or []})
                raise
            except Exception as exc:
                log.exception("chat turn failed")
                await send({"type": "error", "message": str(exc), "session_id": session_id, "recoverable": False})
            finally:
                chain = turns.get(session_id) or []
                if me in chain:
                    chain.remove(me)
                if not chain:
                    turns.pop(session_id, None)

        async def forward_bus() -> None:
            async with s.bus.subscribe() as q:
                while True:
                    await send(await q.get())

        bus_task = asyncio.create_task(forward_bus())
        try:
            await send({"type": "hello", "version": __version__, "assistant": s.config.assistant.name})
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await send({"type": "error", "message": "invalid json"})
                    continue
                kind = msg.get("type")
                if kind == "ping":
                    await send({"type": "pong"})
                elif kind in {"chat.send", "chat.steer"}:
                    text = str(msg.get("text", ""))
                    if not text.strip() and not msg.get("attachments"):
                        continue
                    session_id = msg.get("session_id")
                    if session_id and not msg.get("attachments") and s.agent.steer(str(session_id), text):
                        # a reply is running: the text is applied at its next model round
                        await send({"type": "steer_ack", "session_id": session_id, "queued": True,
                                    "client_id": msg.get("client_id")})
                        continue
                    if kind == "chat.steer" and session_id:
                        await send({"type": "steer_ack", "session_id": session_id, "queued": False,
                                    "client_id": msg.get("client_id")})
                    session_id = session_id or await s.store.create_session(
                        channel=str(msg.get("channel", "desktop"))
                    )
                    await send({"type": "session", "session_id": session_id, "client_id": msg.get("client_id")})
                    chain = turns.setdefault(session_id, [])
                    previous = chain[-1] if chain else None
                    chain.append(asyncio.create_task(run_chat(session_id, msg, previous, s.stop_generation)))
                elif kind == "chat.cancel":
                    for t in list(turns.get(str(msg.get("session_id"))) or []):
                        t.cancel()
                elif kind == "approval.respond":
                    ok = s.approvals.resolve(str(msg.get("approval_id")), str(msg.get("decision", "deny")))
                    await send({"type": "approval.ack", "approval_id": msg.get("approval_id"), "resolved": ok})
                else:
                    await send({"type": "error", "message": f"unknown message type {kind}"})
        except WebSocketDisconnect:
            pass
        finally:
            closed = True
            bus_task.cancel()
            # turns keep running to completion so nothing is lost; they just stop sending

    if UI_DIST.exists():
        app.mount("/", StaticFiles(directory=str(UI_DIST), html=True), name="ui")

    return app
