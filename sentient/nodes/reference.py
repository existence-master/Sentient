"""Reference device node: simulates a phone or smart glasses from a PC (docs/NODES.md).

    sentient node --url wss://192.168.1.20:7778/ws/node --code 123456 --fingerprint 3f9a...
    sentient node --url "sentient://pair?url=...&code=...&fp=..."      # the pairing link works too

Capabilities: ``display.text``, ``display.card``, ``notify.show`` (printed), ``speak`` (pyttsx3 when
available, else printed), ``camera.photo`` (OpenCV webcam, with ``--camera``), ``button.events``
(press Enter) and ``battery`` (simulated). The token is stored in ``~/.sentient-node/<name>.json``.
Self-signed LAN certificates are accepted only when their SHA-256 fingerprint matches the one
Sentient shows at pairing (remembered afterwards), or with ``--insecure`` for development.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import platform
import random
import re
import ssl
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from sentient import __version__
from sentient.nodes.tls import fingerprint_der, normalize_fingerprint

PROTOCOL = 1


class NodeFatal(Exception):
    """Stop the node: reconnecting will not help (bad pairing, pin mismatch...)."""


def state_dir() -> Path:
    raw = os.environ.get("SENTIENT_NODE_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".sentient-node"


def state_file(name: str, directory: Path | None = None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-").lower() or "node"
    return (directory or state_dir()) / f"{safe}.json"


def parse_pair_link(link: str) -> tuple[str, str | None, str | None]:
    """``sentient://pair?url=...&code=...&fp=...`` -> (url, code, fingerprint)."""
    q = parse_qs(urlparse(link).query)
    url = (q.get("url") or [""])[0]
    if not url:
        raise NodeFatal("The pairing link has no url.")
    return url, (q.get("code") or [None])[0], (q.get("fp") or [None])[0]


def normalize_url(url: str) -> str:
    url = url.strip()
    if url.startswith("https://"):
        url = "wss://" + url[8:]
    elif url.startswith("http://"):
        url = "ws://" + url[7:]
    elif not url.startswith(("ws://", "wss://")):
        url = "wss://" + url
    parsed = urlparse(url)
    if parsed.path in ("", "/"):
        url = url.rstrip("/") + "/ws/node"
    return url


class ReferenceNode:
    def __init__(
        self,
        url: str,
        *,
        code: str | None = None,
        name: str = "Reference node",
        kind: str = "glasses",
        camera: int | None = None,
        fingerprint: str | None = None,
        insecure: bool = False,
        speech: bool = True,
        directory: Path | None = None,
        out: Callable[[str], None] | None = None,
        interactive: bool | None = None,
    ):
        self.url = normalize_url(url)
        self.code = code
        self.name = name
        self.kind = kind
        self.camera = camera
        self.fingerprint = normalize_fingerprint(fingerprint) if fingerprint else None
        self.insecure = insecure
        self.speech = speech
        self.file = state_file(name, directory)
        self.out = out or (lambda text: print(text, flush=True))
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.battery = 87.0
        self.charging = False
        self.node_id: str | None = None
        self.keepalive_s = 60
        self.welcomed = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._lines: asyncio.Queue[str] | None = None
        self._answers: set[asyncio.Task] = set()
        self.saved: dict[str, Any] = self._load()

    # ------------------------------------------------------------------ state file
    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, **patch: Any) -> None:
        self.saved.update(patch)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.saved, indent=2), encoding="utf-8")
        with contextlib.suppress(Exception):
            self.file.chmod(0o600)

    # ------------------------------------------------------------------ protocol
    @property
    def capabilities(self) -> list[str]:
        caps = ["display.text", "display.card", "notify.show", "speak", "button.events", "battery"]
        if self.camera is not None:
            caps.append("camera.photo")
        return caps

    def hello(self) -> dict:
        msg: dict[str, Any] = {
            "type": "hello",
            "protocol": PROTOCOL,
            "name": self.name,
            "kind": self.kind,
            "platform": f"{platform.system()} (reference node)",
            "app_version": __version__,
            "capabilities": self.capabilities,
            "state": {"battery": round(self.battery), "charging": self.charging},
        }
        token = self.saved.get("token") if self.saved.get("url") in (None, self.url) else None
        if token:
            msg["token"] = token
        if self.code:
            msg["pair_code"] = self.code
        return msg

    async def handle_invoke(self, msg: dict) -> dict:
        """Run one ``invoke`` and build its ``result`` message."""
        call_id, cap = msg.get("id"), str(msg.get("capability") or "")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        try:
            if cap not in self.capabilities:
                return {"type": "result", "id": call_id, "ok": False,
                        "error": {"code": "unsupported", "message": f"{self.name} cannot do {cap}."}}  # fmt: skip
            data = await self._run(cap, params)
            return {"type": "result", "id": call_id, "ok": True, "data": data}
        except Exception as exc:
            return {"type": "result", "id": call_id, "ok": False, "error": {"code": "failed", "message": str(exc)}}

    async def _run(self, cap: str, params: dict) -> dict:
        if cap == "display.text":
            self._box(str(params.get("text", "")))
            return {"shown": True}
        if cap == "display.card":
            self._box(str(params.get("text", "")), title=str(params.get("title") or ""))
            return {"shown": True}
        if cap == "notify.show":
            title = params.get("title") or "Sentient"
            self.out(f"[notification] {title}: {params.get('text', '')}")
            return {"shown": True}
        if cap == "speak":
            text = str(params.get("text", ""))
            spoken = await asyncio.to_thread(self._speak, text) if self.speech else False
            if not spoken:
                self.out(f'[says] "{text}"')
            return {"spoken": spoken}
        if cap == "battery":
            return {"battery": round(self.battery), "charging": self.charging}
        if cap == "camera.photo":
            self.out("[camera] taking a photo...")
            return await asyncio.to_thread(self._photo, params)
        return {}

    def _box(self, text: str, title: str = "") -> None:
        lines = [title.upper()] if title else []
        lines += text.splitlines() or [""]
        width = min(max(len(line) for line in lines), 76)
        self.out("+" + "-" * (width + 2) + "+")
        for line in lines:
            self.out(f"| {line[:width].ljust(width)} |")
        self.out("+" + "-" * (width + 2) + "+")

    @staticmethod
    def _speak(text: str) -> bool:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception:
            return False

    def _photo(self, params: dict) -> dict:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed (pip install opencv-python-headless).") from exc
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(self.camera or 0), backend)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Webcam {self.camera} could not be opened.")
            frame = None
            for _ in range(8):  # the first frames are often dark while exposure settles
                ok, frame = cap.read()
            if frame is None or not ok:
                raise RuntimeError("The webcam returned no image.")
            max_w = int(params.get("max_width") or 1280)
            h, w = frame.shape[:2]
            if w > max_w:
                frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise RuntimeError("Could not encode the photo.")
            h, w = frame.shape[:2]
            return {"mime": "image/jpeg", "base64": base64.b64encode(buf.tobytes()).decode(), "width": w, "height": h}
        finally:
            cap.release()

    # ------------------------------------------------------------------ TLS pinning
    def ssl_context(self) -> ssl.SSLContext | None:
        if not self.url.startswith("wss://"):
            return None
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # the certificate is self-signed; the fingerprint is checked instead
        return ctx

    def expected_fingerprint(self) -> str | None:
        return self.fingerprint or normalize_fingerprint(self.saved.get("fingerprint") or "") or None

    def check_pin(self, der: bytes | None) -> str:
        """Compare the server certificate with the pinned fingerprint. Returns the observed fingerprint."""
        if not der:
            raise NodeFatal("The server did not present a certificate.")
        seen = fingerprint_der(der)
        expected = self.expected_fingerprint()
        if expected:
            if seen != expected:
                raise NodeFatal(
                    "The server certificate does not match the pinned fingerprint.\n"
                    f"  expected {expected}\n  got      {seen}\n"
                    "Someone may be intercepting the connection. If you reset Sentient, pair again."
                )
        elif not self.insecure:
            raise NodeFatal(
                "Pass --fingerprint with the value shown in Sentient next to the pairing code "
                f"(this server's is {seen}), or --insecure for development."
            )
        return seen

    # ------------------------------------------------------------------ connection
    async def _send(self, ws: Any, obj: dict) -> None:
        async with self._send_lock:
            await ws.send(json.dumps(obj))

    def _start_stdin(self) -> None:
        if not self.interactive or self._lines is not None:
            return
        loop = asyncio.get_running_loop()
        self._lines = asyncio.Queue()
        queue = self._lines

        def reader() -> None:
            for line in sys.stdin:
                loop.call_soon_threadsafe(queue.put_nowait, line.strip())
            loop.call_soon_threadsafe(queue.put_nowait, "q")

        threading.Thread(target=reader, daemon=True, name="node-stdin").start()

    async def _input_loop(self, ws: Any) -> None:
        if self._lines is None:
            return
        while True:
            line = await self._lines.get()
            if line.lower() in {"q", "quit", "exit"}:
                await ws.close()
                raise NodeFatal("Bye.")
            if line.lower().startswith("b "):
                with contextlib.suppress(ValueError):
                    self.battery = float(line.split()[1])
                    await self._send(ws, {"type": "state", "battery": round(self.battery), "charging": self.charging})
                    self.out(f"[battery] {round(self.battery)}%")
                continue
            await self._send(ws, {"type": "event", "event": "button", "data": {"action": "press"}})
            self.out("[button] pressed")

    async def _keepalive(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(self.keepalive_s)
            self.battery = max(5.0, self.battery - 0.2)
            await self._send(ws, {"type": "ping"})

    async def session(self) -> None:
        """One connection: hello, welcome, then answer invokes until the socket closes."""
        from websockets.asyncio.client import connect

        self.welcomed.clear()
        async with connect(self.url, ssl=self.ssl_context(), max_size=32 * 1024 * 1024, ping_interval=None,
                           open_timeout=10, close_timeout=3) as ws:  # fmt: skip
            seen = None
            if self.url.startswith("wss://"):
                ssl_obj = ws.transport.get_extra_info("ssl_object")
                seen = self.check_pin(ssl_obj.getpeercert(binary_form=True) if ssl_obj else None)
            await self._send(ws, self.hello())
            first = json.loads(await asyncio.wait_for(ws.recv(), 15))
            if first.get("type") == "error":
                self._on_auth_error(first)
            if first.get("type") != "welcome":
                raise NodeFatal(f"Unexpected first message: {first}")
            self.node_id = first.get("node_id")
            self.keepalive_s = int(first.get("keepalive_s") or 60)
            patch: dict[str, Any] = {"url": self.url, "node_id": self.node_id, "name": self.name}
            if first.get("token"):
                patch["token"] = first["token"]
            if seen and self.expected_fingerprint():
                patch["fingerprint"] = seen
            self._save(**patch)
            if self.code:
                self.out(f"Paired with {first.get('assistant', 'Sentient')} as '{first.get('name')}'.")
                self.code = None
            self.out(f"Connected to {first.get('assistant', 'Sentient')}. "
                     + ("Press Enter to send a button press, 'q' to quit." if self.interactive else ""))  # fmt: skip
            self.welcomed.set()
            helpers = [asyncio.create_task(self._keepalive(ws))]
            if self.interactive:
                helpers.append(asyncio.create_task(self._input_loop(ws)))
            try:
                async for raw in ws:
                    for t in helpers:
                        if t.done() and t.exception():
                            raise t.exception()  # type: ignore[misc]
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    kind = msg.get("type")
                    if kind == "invoke":
                        answer = asyncio.create_task(self._answer(ws, msg))
                        self._answers.add(answer)
                        answer.add_done_callback(self._answers.discard)
                    elif kind == "error":
                        self._on_auth_error(msg)
                        self.out(f"[error] {msg.get('message')}")
            finally:
                for t in helpers:
                    t.cancel()
                for t in helpers:
                    with contextlib.suppress(BaseException):
                        await t
                    if t.done() and not t.cancelled() and isinstance(t.exception(), NodeFatal):
                        raise t.exception()  # type: ignore[misc]

    async def _answer(self, ws: Any, msg: dict) -> None:
        result = await self.handle_invoke(msg)
        with contextlib.suppress(Exception):
            await self._send(ws, result)

    def _on_auth_error(self, msg: dict) -> None:
        code = msg.get("code")
        text = str(msg.get("message") or code)
        if code in {"bad_token", "revoked"}:
            self._save(token=None)
            raise NodeFatal(f"{text}\nRun again with --code <new pairing code>.")
        if code in {"pairing_required", "bad_code", "rate_limited", "replaced"}:
            raise NodeFatal(text)
        if code == "disabled":
            raise ConnectionError(text)

    async def run(self) -> int:
        from websockets.exceptions import WebSocketException

        self._start_stdin()
        delay = 1.0
        while True:
            try:
                await self.session()
            except NodeFatal as exc:
                self.out(str(exc))
                return 0 if str(exc) == "Bye." else 1
            except (OSError, TimeoutError, WebSocketException, ConnectionError, json.JSONDecodeError) as exc:
                self.out(f"Connection problem: {exc or type(exc).__name__}")
            if self.welcomed.is_set():
                delay = 1.0
            wait = delay + random.uniform(0, delay / 2)
            self.out(f"Reconnecting in {wait:.0f} s...")
            await asyncio.sleep(wait)
            delay = min(delay * 2, 30.0)


def main(
    url: str,
    code: str | None = None,
    name: str = "Reference node",
    kind: str = "glasses",
    camera: int | None = None,
    fingerprint: str | None = None,
    insecure: bool = False,
    speech: bool = True,
) -> None:
    if url.startswith("sentient://"):
        url, link_code, link_fp = parse_pair_link(url)
        code = code or link_code
        fingerprint = fingerprint or link_fp
    node = ReferenceNode(
        url, code=code, name=name, kind=kind, camera=camera, fingerprint=fingerprint, insecure=insecure, speech=speech
    )
    try:
        rc = asyncio.run(node.run())
    except KeyboardInterrupt:
        rc = 0
    raise SystemExit(rc)
