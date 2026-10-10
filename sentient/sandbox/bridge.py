"""The engine half of the tool bridge (Hermes-style programmatic tool calling).

One bridge per run. ``tcp`` transport: a tiny HTTP endpoint on a random loopback port that only
accepts ``POST`` requests carrying the run's one-time token. ``mailbox`` transport: request and
response JSON files in ``<run_dir>/.rpc`` (used for Docker containers without a network).
Both call ``ToolBridge.handle`` which applies the ``BridgePolicy`` and runs the tool.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

from sentient.sandbox.policy import BridgePolicy, Refused
from sentient.tools.base import ToolContext

log = logging.getLogger(__name__)

MAX_BODY = 8 * 1024 * 1024
MAX_RESULT_CHARS = 2_000_000
# A refused request's body is still read, so the refusal arrives instead of a reset connection: small ones, briefly.
REFUSED_BODY_MAX = 64 * 1024
REFUSED_BODY_WAIT_S = 10


class ToolBridge:
    def __init__(
        self,
        registry: Any,
        policy: BridgePolicy,
        ctx: ToolContext,
        *,
        run_dir: Path,
        transport: str = "tcp",
        host: str = "127.0.0.1",
    ):
        self.registry = registry
        self.policy = policy
        self.ctx = ctx
        self.run_dir = run_dir
        self.transport = transport
        self.host = host
        self.token = secrets.token_urlsafe(32)
        self.tool_calls = 0   # calls that ran
        self.refused = 0      # calls the policy refused
        self.port: int | None = None
        self._server: asyncio.Server | None = None
        self._poller: asyncio.Task | None = None
        self._handlers: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> dict:
        """Start listening; returns ``{url, mailbox}`` for the generated client."""
        if self.transport == "mailbox":
            (self.run_dir / ".rpc").mkdir(parents=True, exist_ok=True)
            self._poller = asyncio.create_task(self._poll_mailbox(), name="sandbox:mailbox")
            return {"url": None, "mailbox": ".rpc"}
        self._server = await asyncio.start_server(self._on_connection, host=self.host, port=0)
        self.port = self._server.sockets[0].getsockname()[1]
        return {"url": f"http://127.0.0.1:{self.port}/call", "mailbox": None}

    async def stop(self) -> None:
        self.token = secrets.token_urlsafe(32)  # the one-time token dies with the run
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 2)
            self._server = None
        if self._poller is not None:
            self._poller.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._poller
            self._poller = None
        for t in list(self._handlers):
            t.cancel()
        if self._handlers:
            await asyncio.gather(*self._handlers, return_exceptions=True)
        self._handlers.clear()

    # ------------------------------------------------------------------ dispatch
    async def handle(self, name: Any, arguments: Any) -> dict:
        if not isinstance(name, str) or not name:
            return {"ok": False, "error": "A tool name is required."}
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "Tool arguments must be keyword arguments."}
        tool = self.registry.get(name)
        try:
            await self.policy.check(tool, name, arguments, self.ctx, self.tool_calls)
        except Refused as exc:
            self.refused += 1
            return {"ok": False, "refused": True, "error": str(exc)}
        assert tool is not None
        self.tool_calls += 1
        try:
            value = await tool.call(self.ctx, arguments)
        except Exception as exc:
            log.info("sandbox tool %s failed: %s", name, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "result": value}

    @staticmethod
    def encode(payload: dict) -> bytes:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) > MAX_RESULT_CHARS:
            text = json.dumps({"ok": False, "error": "The tool result was too large to pass to the script."})
        return text.encode("utf-8")

    # ------------------------------------------------------------------ tcp transport
    async def _on_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        try:
            status, body = await self._http_exchange(reader)
            head = (
                f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            )
            writer.write(head.encode("latin-1") + body)
            await writer.drain()
        except (asyncio.CancelledError, ConnectionError, OSError):
            pass
        finally:
            if task is not None:
                self._handlers.discard(task)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _http_exchange(self, reader: asyncio.StreamReader) -> tuple[str, bytes]:
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError):
            return "400 Bad Request", self.encode({"ok": False, "error": "bad request"})
        lines = raw.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            return "413 Payload Too Large", self.encode({"ok": False, "error": "request too large"})
        if len(parts) < 2 or parts[0] != "POST" or parts[1] != "/call":
            await self._discard_body(reader, length)
            return "404 Not Found", self.encode({"ok": False, "error": "not found"})
        if not hmac.compare_digest(headers.get("x-sentient-token", ""), self.token):
            await self._discard_body(reader, length)
            return "403 Forbidden", self.encode({"ok": False, "error": "forbidden"})
        try:
            body = await asyncio.wait_for(reader.readexactly(length), 30)
            payload = json.loads(body.decode("utf-8") or "{}")
        except (asyncio.IncompleteReadError, TimeoutError, ValueError):
            return "400 Bad Request", self.encode({"ok": False, "error": "invalid JSON"})
        if not isinstance(payload, dict):
            return "400 Bad Request", self.encode({"ok": False, "error": "invalid request"})
        response = await self.handle(payload.get("tool"), payload.get("arguments", {}))
        return "200 OK", self.encode(response)

    @staticmethod
    async def _discard_body(reader: asyncio.StreamReader, length: int) -> None:
        """Read a refused request's body before answering. Closing with it unread (or still on its way, as clients
        send it after the headers) resets the connection instead of delivering the refusal."""
        if 0 < length <= REFUSED_BODY_MAX:
            with contextlib.suppress(asyncio.IncompleteReadError, TimeoutError):
                await asyncio.wait_for(reader.readexactly(length), REFUSED_BODY_WAIT_S)

    # ------------------------------------------------------------------ mailbox transport
    async def _poll_mailbox(self) -> None:
        box = self.run_dir / ".rpc"
        while True:
            try:
                requests = sorted(box.glob("req-*.json"))
            except OSError:
                requests = []
            for path in requests:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    path.unlink()
                except (OSError, ValueError):
                    continue  # still being renamed into place; try again next tick
                request_id = path.stem[len("req-"):]
                task = asyncio.create_task(self._answer_mailbox(box, request_id, payload))
                self._handlers.add(task)
                task.add_done_callback(self._handlers.discard)
            await asyncio.sleep(0.02)

    async def _answer_mailbox(self, box: Path, request_id: str, payload: Any) -> None:
        if not isinstance(payload, dict) or not hmac.compare_digest(str(payload.get("token", "")), self.token):
            response = {"ok": False, "error": "forbidden"}
        else:
            response = await self.handle(payload.get("tool"), payload.get("arguments", {}))
        tmp = box / f"resp-{request_id}.tmp"
        with contextlib.suppress(OSError):
            tmp.write_bytes(self.encode(response))
            os.replace(tmp, box / f"resp-{request_id}.json")
