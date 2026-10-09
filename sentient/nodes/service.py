"""Devices (phones, glasses, the desktop itself) on the node protocol (docs/API.md section 13, docs/NODES.md).

A node connects to ``WS /ws/node`` (loopback gateway or the TLS LAN listener), sends ``hello``
and authenticates with a stored token, a one-time pairing code, or (desktop app only, loopback
only) the gateway token. While connected the engine can ``invoke`` its capabilities and the node
sends ``event`` and ``state`` messages. Other packages use:

- ``await app.nodes.verify_token(token) -> Node | None`` (voice authenticates ``/ws/voice?node_token=``)
- ``await app.nodes.invoke(node_id, capability, params, timeout_ms) -> {ok, data?, error?, code?}``
- ``app.nodes.online()`` and the bus events ``node.updated`` / ``node.deleted`` / ``node.event``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import re
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from sentient import __version__, paths
from sentient.services import Service
from sentient.store.db import now_iso

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import WebSocket

    from sentient.config.schema import NodesConfig
    from sentient.nodes.lan import LanListener

log = logging.getLogger(__name__)

PROTOCOL = 1
DESKTOP_ID = "desktop"
KINDS = {"phone", "glasses", "desktop", "watch", "custom"}
CAPABILITIES = [
    "camera.photo", "screen.capture", "location.get", "notify.show", "display.text", "display.card",
    "audio.play", "audio.pcm", "speak", "mic.stream", "clipboard.read", "clipboard.write", "button.events", "battery",
]  # fmt: skip
EVENTS = {"button", "wake", "gesture", "battery", "presence", "notification_action"}
# capabilities the agent tools use; a device offering none of them does not make the tools visible
TOOL_CAPABILITIES = {"camera.photo", "screen.capture", "location.get", "notify.show", "display.text",
                     "display.card", "speak", "audio.play", "audio.pcm"}  # fmt: skip

PAIR_CODE_TTL_S = 600
PAIR_MAX_ACTIVE_CODES = 5
PAIR_FAILURES_PER_MINUTE = 5
PAIR_GLOBAL_FAILURES = 30  # within 10 minutes: every outstanding code is cancelled
HELLO_TIMEOUT_S = 15
UPLOAD_TTL_S = 3600
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
LAST_SEEN_WRITE_S = 60

CLOSE_CODES = {
    "pairing_required": 4401, "bad_token": 4401, "bad_code": 4401, "revoked": 4401,
    "rate_limited": 4429, "disabled": 4403, "protocol": 4400, "replaced": 4409,
}  # fmt: skip
_NAME_RE = re.compile(r"^[a-z0-9_.-]{1,40}$")
MIME_EXT = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "audio/wav": ".wav",
    "audio/x-wav": ".wav", "application/octet-stream": ".bin",
}  # fmt: skip


class DeviceError(RuntimeError):
    """A friendly, user-facing reason a device request could not be done."""


class NodeAuthError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass
class NodeConnection:
    node_id: str
    name: str
    kind: str
    platform: str
    capabilities: list[str]
    lan: bool
    ws: Any
    connected_at: float
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending: dict[str, asyncio.Future] = field(default_factory=dict)
    awaiting_binary: tuple[str, dict] | None = None
    battery: int | None = None
    charging: bool | None = None
    worn: bool | None = None
    last_seen_write: float = 0.0
    closed: bool = False

    async def send(self, obj: dict, payload: bytes | None = None) -> None:
        """Send one JSON message, optionally followed by its binary payload frame (same lock, so
        the two frames are never interleaved with another message)."""
        if self.closed:
            raise ConnectionError("device disconnected")
        async with self.send_lock:
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False, default=str))
            if payload is not None:
                await self.ws.send_bytes(payload)

    def fail_pending(self, message: str) -> None:
        for fut in self.pending.values():
            if not fut.done():
                fut.set_result({"ok": False, "code": "offline", "error": message})
        self.pending.clear()


def _clean_caps(raw: Any) -> list[str]:
    out: list[str] = []
    for c in raw if isinstance(raw, list) else []:
        if isinstance(c, str) and _NAME_RE.match(c) and c not in out:
            out.append(c)
    return out[:50]


def _bool_or_none(v: Any) -> bool | None:
    return bool(v) if isinstance(v, bool | int) else None


def _battery(v: Any) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    if 0 < v <= 1 and isinstance(v, float):
        v = v * 100
    return max(0, min(100, round(v)))


def _normalize_result(msg: dict) -> dict:
    if msg.get("ok"):
        data = msg.get("data")
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            data = {"value": data}
        return {"ok": True, "data": data}
    err = msg.get("error")
    if isinstance(err, dict):
        out = {"ok": False, "error": str(err.get("message") or err.get("code") or "The device reported an error.")}
        if err.get("code"):
            out["code"] = str(err["code"])
        return out
    return {"ok": False, "error": str(err or "The device reported an error.")}


STOP_STATE_SEND_S = 2.0  # deadline per device for a stop_state message


class NodeService(Service):
    name = "nodes"

    def __init__(self, app):
        super().__init__(app)
        self.clock = time.monotonic  # tests move time forward
        self._conns: dict[str, NodeConnection] = {}
        self._codes: dict[str, tuple[float, str]] = {}  # code -> (monotonic expiry, iso expiry)
        self._failures: dict[str, deque[float]] = {}
        self._global_failures: deque[float] = deque()
        self._uploads: dict[str, tuple[Path, str, str, float]] = {}  # id -> (path, node_id, mime, created)
        self._lan_lock = asyncio.Lock()
        self.lan: LanListener | None = None
        self.lan_host = "0.0.0.0"  # tests bind loopback
        # authenticates the LAN app's internal hop into the voice socket; never leaves the process
        self.lan_secret = secrets.token_urlsafe(32)
        self._schema_ready = False

    @property
    def cfg(self) -> NodesConfig:
        return self.app.config.nodes

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        await self._ensure_schema()
        from sentient.nodes.tools import DevicesPlugin

        if self.app.registry.plugin(DevicesPlugin.id) is None:
            self.app.registry.register(DevicesPlugin())
        self._refresh_tools()
        self._loops.append(asyncio.create_task(self._watch_config(), name="nodes:config"))
        await self.apply_config()

    async def stop(self) -> None:
        await super().stop()
        for conn in list(self._conns.values()):
            await self._close_conn(conn, 1001, None)
        self._conns.clear()
        async with self._lan_lock:
            if self.lan is not None:
                await self.lan.stop()
                self.lan = None

    async def _ensure_schema(self) -> None:
        # TODO(core): add "nodes" to store.db.PACKAGE_SCHEMAS; until then the service applies its own schema.
        if self._schema_ready:
            return
        db = self.app.store.db
        await db.executescript(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))
        await db.commit()
        self._schema_ready = True

    async def _watch_config(self) -> None:
        async with self.app.bus.subscribe() as q:
            while True:
                event = await q.get()
                if event.get("type") == "config.updated":
                    try:
                        await self.apply_config()
                    except Exception:
                        log.exception("nodes: applying config failed")

    async def apply_config(self) -> None:
        cfg = self.cfg
        if not cfg.enabled:
            for conn in list(self._conns.values()):
                await self._close_conn(conn, CLOSE_CODES["disabled"], ("disabled", "Devices are turned off in Sentient."))
        want_lan = cfg.enabled and cfg.lan_enabled
        async with self._lan_lock:
            lan = self.lan
            if want_lan:
                if lan is None or not lan.running or lan.port != cfg.lan_port or lan.mdns_wanted != cfg.mdns_enabled:
                    if lan is not None:
                        await lan.stop()
                    from sentient.nodes.lan import LanListener

                    self.lan = LanListener(self)
                    await self.lan.start(cfg.lan_port, mdns=cfg.mdns_enabled, host=self.lan_host)
            elif lan is not None:
                await lan.stop()
                self.lan = None
        self._refresh_tools()

    def lan_status(self) -> dict:
        cfg = self.cfg
        lan = self.lan
        running = bool(lan and lan.running)
        return {
            "enabled": bool(cfg.enabled and cfg.lan_enabled),
            "running": running,
            "port": cfg.lan_port,
            "urls": lan.ws_urls() if running and lan else [],
            "web_urls": lan.web_urls() if running and lan else [],
            "fingerprint": lan.fingerprint if lan else None,
            "mdns": bool(lan and lan.mdns),
            "error": lan.error if lan else None,
        }

    # ------------------------------------------------------------------ nodes table
    async def _row(self, node_id: str) -> dict | None:
        row = await self.app.store.fetchone("SELECT * FROM nodes WHERE id = ? AND revoked = 0", (node_id,))
        return dict(row) if row else None

    def _node_dict(self, row: dict) -> dict:
        conn = self._conns.get(row["id"])
        try:
            stored_caps = json.loads(row.get("capabilities") or "[]")
        except json.JSONDecodeError:
            stored_caps = []
        return {
            "node_id": row["id"],
            "name": row["name"],
            "kind": row["kind"],
            "platform": row["platform"],
            "app_version": row.get("app_version") or "",
            "capabilities": list(conn.capabilities) if conn else stored_caps,
            "online": conn is not None,
            "connection": ("lan" if conn.lan else "local") if conn else None,
            "last_seen_at": now_iso() if conn else row.get("last_seen_at"),
            "battery": conn.battery if conn and conn.battery is not None else row.get("battery"),
            "charging": conn.charging if conn else None,
            "worn": conn.worn if conn else None,
            "created_at": row["created_at"],
        }

    async def get_node(self, node_id: str) -> dict | None:
        row = await self._row(node_id)
        return self._node_dict(row) if row else None

    async def list_nodes(self) -> list[dict]:
        rows = await self.app.store.fetchall("SELECT * FROM nodes WHERE revoked = 0")
        nodes = [self._node_dict(dict(r)) for r in rows]
        nodes.sort(key=lambda n: n["last_seen_at"] or "", reverse=True)
        nodes.sort(key=lambda n: not n["online"])
        return nodes

    def online(self) -> list[dict]:
        """Connected devices (lightweight, no database access)."""
        return [
            {"node_id": c.node_id, "name": c.name, "kind": c.kind, "capabilities": list(c.capabilities)}
            for c in self._conns.values()
        ]

    async def verify_token(self, token: str | None) -> dict | None:
        if not token:
            return None
        row = await self.app.store.fetchone(
            "SELECT * FROM nodes WHERE token_hash = ? AND revoked = 0", (hash_token(str(token)),)
        )
        return self._node_dict(dict(row)) if row else None

    async def rename(self, node_id: str, name: str) -> dict | None:
        name = name.strip()[:60]
        if not name or await self._row(node_id) is None:
            return None
        await self.app.store.execute("UPDATE nodes SET name = ? WHERE id = ?", (name, node_id))
        if node_id in self._conns:
            self._conns[node_id].name = name
        node = await self.get_node(node_id)
        self.app.bus.publish("node.updated", node)
        return node

    async def delete(self, node_id: str) -> bool:
        """Forget a device and revoke its token. The built-in desktop node cannot be deleted."""
        if node_id == DESKTOP_ID:
            raise ValueError("The desktop app is built in and cannot be removed.")
        if await self._row(node_id) is None:
            return False
        await self.app.store.execute("UPDATE nodes SET revoked = 1, token_hash = NULL WHERE id = ?", (node_id,))
        conn = self._conns.get(node_id)
        if conn is not None:
            await self._close_conn(conn, CLOSE_CODES["revoked"], ("revoked", "This device was removed in Sentient."))
        self.app.bus.publish("node.deleted", {"node_id": node_id})
        self._refresh_tools()
        return True

    # ------------------------------------------------------------------ pairing
    def _prune_codes(self, now: float) -> None:
        for code in [c for c, (exp, _) in self._codes.items() if exp <= now]:
            del self._codes[code]

    def create_pairing(self, base_url: str | None = None) -> dict:
        now = self.clock()
        self._prune_codes(now)
        while len(self._codes) >= PAIR_MAX_ACTIVE_CODES:
            del self._codes[min(self._codes, key=lambda c: self._codes[c][0])]
        code = f"{secrets.randbelow(1_000_000):06d}"
        while code in self._codes:
            code = f"{secrets.randbelow(1_000_000):06d}"
        expires_at = (datetime.now(UTC) + timedelta(seconds=PAIR_CODE_TTL_S)).isoformat()
        self._codes[code] = (now + PAIR_CODE_TTL_S, expires_at)

        lan = self.lan_status()
        local_ws = local_web = None
        if base_url:
            base = base_url.rstrip("/")
            local_web = f"{base}/node/#code={code}"
            local_ws = ("wss" + base[5:] if base.startswith("https") else "ws" + base[4:]) + "/ws/node"
        urls = lan["urls"] or ([local_ws] if local_ws else [])
        web_urls = [f"{u}#code={code}" for u in lan["web_urls"]]
        web_url = web_urls[0] if web_urls else local_web
        fp = lan["fingerprint"] if lan["running"] else None
        qr = ""
        if urls:
            qr = f"sentient://pair?url={quote(urls[0], safe='')}&code={code}" + (f"&fp={fp}" if fp else "")
        qr_svg = ""
        if web_url:
            with contextlib.suppress(Exception):
                import segno

                qr_svg = segno.make(web_url, error="m").svg_inline(scale=5, border=2)
        return {
            "code": code,
            "expires_at": expires_at,
            "lan_enabled": lan["enabled"],
            "urls": urls,
            "web_url": web_url,
            "fingerprint": fp,
            "qr": qr,
            "qr_svg": qr_svg,
        }

    def _consume_code(self, code: str, remote: str) -> None:
        now = self.clock()
        fails = self._failures.setdefault(remote, deque())
        while fails and now - fails[0] > 60:
            fails.popleft()
        if len(fails) >= PAIR_FAILURES_PER_MINUTE:
            raise NodeAuthError("rate_limited", "Too many wrong codes. Wait a minute, then try again.")
        self._prune_codes(now)
        if not re.fullmatch(r"\d{6}", code) or code not in self._codes:
            fails.append(now)
            self._global_failures.append(now)
            while self._global_failures and now - self._global_failures[0] > 600:
                self._global_failures.popleft()
            if len(self._global_failures) >= PAIR_GLOBAL_FAILURES and self._codes:
                log.warning("nodes: too many wrong pairing codes; cancelling outstanding codes")
                self._codes.clear()
            raise NodeAuthError("bad_code", "That pairing code is wrong or has expired. Ask Sentient for a new one.")
        del self._codes[code]

    # ------------------------------------------------------------------ authentication
    async def _authenticate(self, hello: dict, *, gateway_ok: bool, remote: str) -> tuple[dict, str | None]:
        now = now_iso()
        caps = _clean_caps(hello.get("capabilities"))
        platform = str(hello.get("platform") or "")[:60]
        app_version = str(hello.get("app_version") or "")[:40]
        if gateway_ok:
            name = str(hello.get("name") or "").strip()[:60] or "This computer"
            await self.app.store.execute(
                "INSERT INTO nodes(id, name, kind, platform, app_version, capabilities, created_at, last_seen_at) "
                "VALUES(?, ?, 'desktop', ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET platform = excluded.platform, "
                "app_version = excluded.app_version, capabilities = excluded.capabilities, "
                "last_seen_at = excluded.last_seen_at, revoked = 0",
                (DESKTOP_ID, name, platform, app_version, json.dumps(caps), now, now),
            )
            row = await self._row(DESKTOP_ID)
            assert row is not None
            return row, None

        kind = str(hello.get("kind") or "custom")
        if kind not in KINDS or kind == "desktop":
            kind = "custom"
        token = str(hello.get("token") or "")
        code = str(hello.get("pair_code") or "").strip()
        if token:
            node = await self.verify_token(token)
            if node is not None:
                await self.app.store.execute(
                    "UPDATE nodes SET kind = ?, platform = ?, app_version = ?, capabilities = ?, last_seen_at = ? "
                    "WHERE id = ?",
                    (kind, platform, app_version, json.dumps(caps), now, node["node_id"]),
                )
                row = await self._row(node["node_id"])
                assert row is not None
                return row, None
            if not code:
                raise NodeAuthError("bad_token", "This device is no longer paired. Pair it again with a new code.")
        if code:
            self._consume_code(code, remote)
            default_name = {"phone": "Phone", "glasses": "Glasses", "watch": "Watch"}.get(kind, "Device")
            name = str(hello.get("name") or "").strip()[:60] or default_name
            node_id = secrets.token_hex(8)
            token = secrets.token_urlsafe(32)
            await self.app.store.execute(
                "INSERT INTO nodes(id, name, kind, platform, app_version, capabilities, token_hash, created_at, "
                "last_seen_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (node_id, name, kind, platform, app_version, json.dumps(caps), hash_token(token), now, now),
            )
            row = await self._row(node_id)
            assert row is not None
            return row, token
        raise NodeAuthError(
            "pairing_required",
            "Pair this device first: in Sentient open Devices, choose Pair a device, and enter the 6-digit code here.",
        )

    # ------------------------------------------------------------------ the socket
    async def serve_socket(self, ws: WebSocket, *, lan: bool) -> None:
        from fastapi import WebSocketDisconnect

        from sentient.gateway.auth import ws_token_ok

        gateway_ok = (not lan) and ws_token_ok(ws)
        remote = ws.client.host if ws.client else "unknown"
        await ws.accept()

        async def reject(code: str, message: str) -> None:
            with contextlib.suppress(Exception):
                await ws.send_text(json.dumps({"type": "error", "code": code, "message": message}))
                await ws.close(code=CLOSE_CODES.get(code, 4400))

        if not self.cfg.enabled:
            await reject("disabled", "Devices are turned off in Sentient.")
            return
        try:
            first = await asyncio.wait_for(ws.receive(), HELLO_TIMEOUT_S)
            hello = json.loads(first.get("text") or "null") if first.get("type") == "websocket.receive" else None
        except (TimeoutError, json.JSONDecodeError):
            hello = None
        except (WebSocketDisconnect, RuntimeError):
            return
        if not isinstance(hello, dict) or hello.get("type") != "hello":
            await reject("protocol", 'The first message must be {"type": "hello", ...}.')
            return
        try:
            row, new_token = await self._authenticate(hello, gateway_ok=gateway_ok, remote=remote)
        except NodeAuthError as exc:
            await reject(exc.code, exc.message)
            return

        conn = NodeConnection(
            node_id=row["id"], name=row["name"], kind=row["kind"], platform=row["platform"],
            capabilities=_clean_caps(hello.get("capabilities")), lan=lan, ws=ws, connected_at=self.clock(),
            last_seen_write=self.clock(),
        )  # fmt: skip
        self._apply_state(conn, hello.get("state") if isinstance(hello.get("state"), dict) else {})
        old = self._conns.get(conn.node_id)
        if old is not None:
            await self._close_conn(old, CLOSE_CODES["replaced"], ("replaced", "This device connected again elsewhere."))
        idle = self.cfg.idle_timeout_s
        welcome = {
            "type": "welcome",
            "protocol": PROTOCOL,
            "node_id": conn.node_id,
            "name": conn.name,
            "assistant": self.app.config.assistant.name,
            "server_version": __version__,
            "keepalive_s": max(15, idle // 3),
            "idle_timeout_s": idle,
            "stopped": self.app.stopped,
        }
        if new_token:
            welcome["token"] = new_token
        try:
            await conn.send(welcome)
        except Exception:
            return
        self._conns[conn.node_id] = conn
        self._refresh_tools()
        self.app.bus.publish("node.updated", self._node_dict(row))
        log.info("device connected: %s (%s, %s)", conn.name, conn.kind, "lan" if lan else "local")

        try:
            while True:
                try:
                    message = await asyncio.wait_for(ws.receive(), idle)
                except TimeoutError:
                    log.info("device %s idle for %ss; disconnecting", conn.name, idle)
                    with contextlib.suppress(Exception):
                        await ws.close(code=1001)
                    break
                if message.get("type") == "websocket.disconnect":
                    break
                await self._touch(conn)
                if message.get("bytes") is not None:
                    self._on_binary(conn, message["bytes"])
                    continue
                raw = message.get("text")
                if raw is None:
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._safe_send(conn, {"type": "error", "code": "protocol", "message": "invalid json"})
                    continue
                if isinstance(msg, dict):
                    await self._dispatch(conn, msg)
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            pass
        finally:
            conn.closed = True
            conn.fail_pending("The device disconnected before answering.")
            if self._conns.get(conn.node_id) is conn:
                del self._conns[conn.node_id]
                self._refresh_tools()
                with contextlib.suppress(Exception):
                    await self.app.store.execute(
                        "UPDATE nodes SET last_seen_at = ?, battery = COALESCE(?, battery) WHERE id = ?",
                        (now_iso(), conn.battery, conn.node_id),
                    )
                    node = await self.get_node(conn.node_id)
                    if node is not None:
                        self.app.bus.publish("node.updated", node)
                self._refresh_tools()
                log.info("device disconnected: %s", conn.name)

    async def send_stop_state(self, timeout: float | None = None) -> None:
        """Tell every connected device that Stop everything was turned on or off (``stop_state``).

        Sends go out side by side, each with a deadline, so a device that is slow to read (or busy receiving
        audio) never holds up the others or the caller."""
        message = {"type": "stop_state", **self.app.stop_state}

        async def one(conn: NodeConnection) -> None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(conn.send(message), timeout or STOP_STATE_SEND_S)

        await asyncio.gather(*(one(c) for c in list(self._conns.values())))

    async def _safe_send(self, conn: NodeConnection, obj: dict) -> None:
        with contextlib.suppress(Exception):
            await conn.send(obj)

    async def _close_conn(self, conn: NodeConnection, code: int, error: tuple[str, str] | None) -> None:
        if error is not None:
            await self._safe_send(conn, {"type": "error", "code": error[0], "message": error[1]})
        conn.closed = True
        conn.fail_pending("The device disconnected before answering.")
        if self._conns.get(conn.node_id) is conn:
            del self._conns[conn.node_id]
            self._refresh_tools()
        with contextlib.suppress(Exception):
            await conn.ws.close(code=code)

    async def _touch(self, conn: NodeConnection) -> None:
        now = self.clock()
        if now - conn.last_seen_write >= LAST_SEEN_WRITE_S:
            conn.last_seen_write = now
            with contextlib.suppress(Exception):
                await self.app.store.execute("UPDATE nodes SET last_seen_at = ? WHERE id = ?", (now_iso(), conn.node_id))

    def _apply_state(self, conn: NodeConnection, state: dict) -> bool:
        changed = False
        if "battery" in state and (b := _battery(state.get("battery"))) is not None:
            changed |= b != conn.battery
            conn.battery = b
        for key in ("charging", "worn"):
            if key in state and (v := _bool_or_none(state.get(key))) is not None:
                changed |= v != getattr(conn, key)
                setattr(conn, key, v)
        return changed

    async def _publish_conn(self, conn: NodeConnection) -> None:
        if conn.battery is not None:
            with contextlib.suppress(Exception):
                await self.app.store.execute("UPDATE nodes SET battery = ? WHERE id = ?", (conn.battery, conn.node_id))
        node = await self.get_node(conn.node_id)
        if node is not None:
            self.app.bus.publish("node.updated", node)

    async def _dispatch(self, conn: NodeConnection, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "ping":
            await self._safe_send(conn, {"type": "pong", "ts": msg.get("ts")})
        elif kind == "result":
            await self._on_result(conn, msg)
        elif kind == "event":
            event = str(msg.get("event") or "")
            data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
            if not _NAME_RE.match(event):
                await self._safe_send(conn, {"type": "error", "code": "protocol", "message": "event needs a name"})
                return
            state = {}
            if event == "battery":
                state = {"battery": data.get("level", data.get("battery")), "charging": data.get("charging")}
            elif event == "presence" and "worn" in data:
                state = {"worn": data.get("worn")}
            if state and self._apply_state(conn, state):
                await self._publish_conn(conn)
            self.app.bus.publish("node.event", {"node_id": conn.node_id, "event": event, "data": data})
        elif kind == "state":
            state = msg.get("data") if isinstance(msg.get("data"), dict) else msg
            if self._apply_state(conn, state):
                await self._publish_conn(conn)
        elif kind in {"stop_all", "resume"}:
            # Stop everything from a paired device (docs/API.md section 17): deterministic, never the model
            was = self.app.stopped
            if kind == "stop_all":
                await self.app.stop_all(source="device")
            else:
                await self.app.resume(source="device")
            if self.app.stopped == was:  # nothing changed, so no broadcast went out: answer this device
                await self._safe_send(conn, {"type": "stop_state", **self.app.stop_state})
        elif kind == "hello":
            caps = _clean_caps(msg.get("capabilities"))
            if caps != conn.capabilities:
                conn.capabilities = caps
                await self.app.store.execute(
                    "UPDATE nodes SET capabilities = ? WHERE id = ?", (json.dumps(caps), conn.node_id)
                )
                self._refresh_tools()
                await self._publish_conn(conn)
        else:
            await self._safe_send(conn, {"type": "error", "code": "unknown_type", "message": f"unknown message type {kind}"})

    async def _on_result(self, conn: NodeConnection, msg: dict) -> None:
        call_id = str(msg.get("id") or "")
        fut = conn.pending.get(call_id)
        if fut is None or fut.done():
            return  # late answer after a timeout
        result = _normalize_result(msg)
        data = result.get("data") or {}
        if result["ok"] and data.get("binary"):
            conn.awaiting_binary = (call_id, data)  # the payload follows as one binary frame
            return
        if result["ok"] and data.get("upload_id"):
            upload = self._uploads.pop(str(data["upload_id"]), None)
            if upload is None or upload[1] != conn.node_id:
                result = {"ok": False, "code": "bad_upload", "error": "The device's upload was not found."}
            else:
                raw = await asyncio.to_thread(upload[0].read_bytes)
                with contextlib.suppress(OSError):
                    upload[0].unlink()
                data.pop("upload_id", None)
                data.setdefault("mime", upload[2])
                data["base64"] = base64.b64encode(raw).decode()
        fut.set_result(result)

    def _on_binary(self, conn: NodeConnection, payload: bytes) -> None:
        if conn.awaiting_binary is None:
            return
        call_id, data = conn.awaiting_binary
        conn.awaiting_binary = None
        fut = conn.pending.get(call_id)
        if fut is None or fut.done():
            return
        data.pop("binary", None)
        data["base64"] = base64.b64encode(payload).decode()
        fut.set_result({"ok": True, "data": data})

    # ------------------------------------------------------------------ invoking capabilities
    async def invoke(
        self,
        node_id: str,
        capability: str,
        params: dict | None = None,
        timeout_ms: int | None = None,
        payload: bytes | None = None,
    ) -> dict:
        """Ask a connected device to do something. Returns ``{ok, data?, error?, code?}``; never raises.

        ``payload`` sends raw bytes as one binary frame right after the invoke (``params.binary``,
        ``params.bytes``), so small devices never have to parse a large base64 JSON string."""
        conn = self._conns.get(node_id)
        if conn is None:
            return {"ok": False, "code": "offline", "error": "That device is not connected right now."}
        if capability not in conn.capabilities:
            return {"ok": False, "code": "unsupported", "error": f"{conn.name} does not support {capability}."}
        if node_id == DESKTOP_ID and capability == "screen.capture" and not self.cfg.allow_desktop_screen:
            return {"ok": False, "code": "not_allowed", "error": "Screen capture of this computer is turned off in Settings."}
        timeout = timeout_ms if timeout_ms and timeout_ms > 0 else self.cfg.invoke_timeout_s * 1000
        timeout = max(50, min(int(timeout), 120_000))
        call_id = secrets.token_hex(6)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        conn.pending[call_id] = fut
        sent_params = dict(params or {})
        if payload is not None:
            sent_params.update({"binary": True, "bytes": len(payload)})
        try:
            await conn.send(
                {"type": "invoke", "id": call_id, "capability": capability, "params": sent_params, "timeout_ms": timeout},
                payload,
            )
            return await asyncio.wait_for(fut, timeout / 1000)
        except TimeoutError:
            return {"ok": False, "code": "timeout", "error": f"{conn.name} did not answer within {timeout / 1000:g} s."}
        except Exception:
            return {"ok": False, "code": "offline", "error": "That device is not connected right now."}
        finally:
            conn.pending.pop(call_id, None)
            if conn.awaiting_binary and conn.awaiting_binary[0] == call_id:
                conn.awaiting_binary = None

    def resolve(self, device: str, capabilities: list[str]) -> tuple[NodeConnection, str]:
        """Pick a connected device and the first of ``capabilities`` it supports. Raises DeviceError."""
        conns = list(self._conns.values())
        label = capabilities[0].replace(".", " ")
        if not conns:
            raise DeviceError(
                "No device is connected. Ask the user to open Sentient on their phone or turn on their glasses."
            )
        names = ", ".join(c.name for c in conns)
        wanted = device.strip().lower()
        if wanted:
            pool = [c for c in conns if wanted in (c.node_id.lower(), c.name.lower(), c.kind)]
            pool = pool or [c for c in conns if wanted in c.name.lower()]
            if not pool:
                raise DeviceError(f"No connected device matches '{device}'. Connected: {names}.")
        else:
            pool = [c for c in conns if c.node_id != DESKTOP_ID or capabilities[0] == "screen.capture"]
        for cap in capabilities:
            capable = [c for c in pool if cap in c.capabilities]
            if cap == "screen.capture" and not self.cfg.allow_desktop_screen:
                capable = [c for c in capable if c.node_id != DESKTOP_ID]
            if capable:
                capable.sort(
                    key=lambda c: (cap == "camera.photo" and c.kind == "glasses", c.node_id != DESKTOP_ID, c.connected_at),
                    reverse=True,
                )
                return capable[0], cap
        if wanted:
            raise DeviceError(f"{pool[0].name} cannot do that ({label}).")
        raise DeviceError(f"None of the connected devices ({names}) can do that ({label}).")

    async def call(
        self,
        conn: NodeConnection,
        capability: str,
        params: dict | None = None,
        timeout_ms: int | None = None,
        payload: bytes | None = None,
    ) -> dict:
        """``invoke`` for tools: returns ``data`` or raises DeviceError with the device's reason."""
        result = await self.invoke(conn.node_id, capability, params, timeout_ms, payload)
        if not result.get("ok"):
            raise DeviceError(f"{conn.name}: {result.get('error') or 'the request failed'}")
        return result.get("data") or {}

    def _refresh_tools(self) -> None:
        from sentient.nodes.tools import DevicesPlugin

        usable = False
        for c in self._conns.values():
            caps = set(c.capabilities) & TOOL_CAPABILITIES
            if c.node_id == DESKTOP_ID:
                caps &= {"screen.capture"} if self.cfg.allow_desktop_screen else set()
            usable = usable or bool(caps)
        with contextlib.suppress(Exception):
            self.app.registry.set_hidden(DevicesPlugin.id, not usable)

    # ------------------------------------------------------------------ uploads (large payloads over HTTP)
    def uploads_dir(self) -> Path:
        return paths.files_dir() / "outputs" / "devices" / "uploads"

    async def save_upload(self, node: dict, data: bytes, mime: str) -> dict:
        now = self.clock()
        for uid, (p, _, _, created) in list(self._uploads.items()):
            if now - created > UPLOAD_TTL_S:
                self._uploads.pop(uid, None)
                with contextlib.suppress(OSError):
                    p.unlink()
        mime = (mime or "application/octet-stream").split(";")[0].strip().lower()
        upload_id = secrets.token_hex(10)
        path = self.uploads_dir() / f"{upload_id}{MIME_EXT.get(mime, '.bin')}"

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        await asyncio.to_thread(write)
        self._uploads[upload_id] = (path, node["node_id"], mime, now)
        return {"upload_id": upload_id, "mime": mime, "size": len(data)}
