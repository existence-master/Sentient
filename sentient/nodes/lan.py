"""The LAN listener: a second in-process HTTPS/WSS server for devices on the local network.

It serves ONLY what devices need (never the main ``/api``):

- ``WS /ws/node``            node protocol (node tokens and pairing codes; the gateway token is refused)
- ``WS /ws/voice``           voice socket for devices, authenticated with ``?node_token=``
- ``GET /node/``             the web device app
- ``POST /api/nodes/upload`` large payloads (photos) from devices, node token required

It uses a self-signed certificate from ``sentient.nodes.tls`` and announces itself as
``_sentient._tcp.local.`` over mDNS so glasses can find the engine without an address.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import socket
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from sentient import __version__
from sentient.nodes.tls import ensure_certificate, lan_ipv4s

if TYPE_CHECKING:  # pragma: no cover
    from sentient.nodes.service import NodeService

log = logging.getLogger(__name__)

SERVICE_TYPE = "_sentient._tcp.local."


class _EmbeddedServer(uvicorn.Server):
    """uvicorn server that leaves signal handling to the main gateway server."""

    @contextlib.contextmanager
    def capture_signals(self):  # type: ignore[override]
        yield

    def install_signal_handlers(self) -> None:  # older uvicorn
        return


class _NodeTokenGate:
    """Authenticates ``/ws/voice`` on the LAN with ``?node_token=`` before the voice route sees it.

    The voice route checks ``?token=`` against ``app.state.token``; on the LAN app that is a secret
    that never leaves the process, so it is added only after the node token was verified.
    """

    def __init__(self, app: Any, core: Any, secret: str):
        self.app = app
        self.core = core
        self.secret = secret

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket" and scope.get("path") == "/ws/voice":
            params = [(k, v) for k, v in parse_qsl(scope.get("query_string", b"").decode()) if k != "token"]
            token = next((v for k, v in params if k == "node_token"), "")
            node = await self.core.nodes.verify_token(token) if token else None
            if node is None:
                await send({"type": "websocket.close", "code": 4401})
                return
            scope = dict(scope, query_string=urlencode([*params, ("token", self.secret)]).encode())
            scope.setdefault("state", {})
            scope["state"]["node"] = node  # TODO(voice): read the device from ws.state.node for channel attribution
        await self.app(scope, receive, send)


def create_lan_app(core: Any, secret: str) -> Any:
    from sentient.gateway.routes.nodes import device_router

    app = FastAPI(title="Sentient devices", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.sentient = core
    app.state.token = secret
    app.state.lan = True
    app.include_router(device_router)
    try:  # webhooks from smart-home gadgets and phones on the same network; each hook has its own secret
        from sentient.gateway.routes.hooks import public as hooks_public

        app.include_router(hooks_public)
    except Exception:  # pragma: no cover - integrations package unavailable
        log.warning("webhooks unavailable on the LAN listener")
    try:
        from sentient.gateway.routes import voice as voice_routes

        app.add_api_websocket_route("/ws/voice", voice_routes.voice_socket)
    except Exception:  # pragma: no cover - voice package unavailable
        log.warning("voice socket unavailable on the LAN listener")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/node/")

    return _NodeTokenGate(app, core, secret)


class LanListener:
    def __init__(self, service: NodeService):
        self.service = service
        self.port: int | None = None
        self.running = False
        self.error: str | None = None
        self.fingerprint: str | None = None
        self.mdns = False
        self.mdns_wanted = False
        self._server: _EmbeddedServer | None = None
        self._task: asyncio.Task | None = None
        self._zc: Any = None

    def ws_urls(self) -> list[str]:
        return [f"wss://{ip}:{self.port}/ws/node" for ip in lan_ipv4s()]

    def web_urls(self) -> list[str]:
        return [f"https://{ip}:{self.port}/node/" for ip in lan_ipv4s()]

    async def start(self, port: int, *, mdns: bool = True, host: str = "0.0.0.0") -> None:
        self.port, self.mdns_wanted, self.error = port, mdns, None
        try:
            cert = await asyncio.to_thread(ensure_certificate)
        except Exception as exc:
            log.exception("LAN certificate failed")
            self.error = f"Could not create the encryption certificate: {exc}"
            return
        self.fingerprint = cert.fingerprint
        config = uvicorn.Config(
            create_lan_app(self.service.app, self.service.lan_secret),
            host=host,
            port=port,
            ssl_certfile=str(cert.cert_file),
            ssl_keyfile=str(cert.key_file),
            log_level="warning",
            lifespan="off",
            access_log=False,
            server_header=False,
            proxy_headers=False,
            ws_ping_interval=None,  # battery: devices keep the link alive with their own pings
            ws_ping_timeout=None,
            ws_max_size=32 * 1024 * 1024,
        )
        server = _EmbeddedServer(config)
        self._server = server

        async def run() -> None:
            try:
                await server.serve()
            except SystemExit:  # uvicorn exits when it cannot bind
                pass
            except Exception:
                log.exception("LAN listener crashed")

        self._task = asyncio.create_task(run(), name="nodes:lan")
        for _ in range(200):
            if server.started or self._task.done():
                break
            await asyncio.sleep(0.025)
        if not server.started:
            self.error = f"Could not listen on port {port}. Another program may be using it."
            log.warning("LAN listener: %s", self.error)
            await self.stop()
            self.error = f"Could not listen on port {port}. Another program may be using it."
            return
        self.running = True
        log.info("LAN device listener on https://0.0.0.0:%s (sha256 %s)", port, cert.fingerprint)
        if mdns:
            await self._advertise()

    async def _advertise(self) -> None:
        try:
            from zeroconf import IPVersion
            from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

            ips = lan_ipv4s()
            if not ips:
                return
            host = re.sub(r"[^A-Za-z0-9-]", "-", socket.gethostname() or "sentient")[:40].strip("-") or "sentient"
            info = AsyncServiceInfo(
                SERVICE_TYPE,
                f"Sentient on {host}.{SERVICE_TYPE}",
                addresses=[socket.inet_aton(ip) for ip in ips],
                port=self.port,
                properties={
                    "version": __version__,
                    "protocol": "1",
                    "fingerprint": self.fingerprint or "",
                    "port": str(self.port),
                    "path": "/ws/node",
                    "tls": "1",
                },
                server=f"{host}.local.",
            )
            self._zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
            await self._zc.async_register_service(info, allow_name_change=True)
            self.mdns = True
        except Exception as exc:
            log.warning("mDNS announcement failed: %s", exc)
            await self._close_zeroconf()

    async def _close_zeroconf(self) -> None:
        zc, self._zc = self._zc, None
        self.mdns = False
        if zc is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(zc.async_unregister_all_services(), 3)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(zc.async_close(), 3)

    async def stop(self) -> None:
        await self._close_zeroconf()
        server, task = self._server, self._task
        self._server = self._task = None
        self.running = False
        if server is not None:
            server.should_exit = True
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), 5)
            except (TimeoutError, Exception):
                if server is not None:
                    server.force_exit = True
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
