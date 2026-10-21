"""A thin client for the running engine's REST API, used by the MCP server (ADR 0034).

The MCP server is a separate process started by the MCP client (for example Claude Code running
``sentient mcp``). It holds no state of its own: it reads the engine's address from the config and
its token the same way ``sentient serve`` does (the ``SENTIENT_GATEWAY_TOKEN`` environment variable
or the token file under the Sentient home folder), and it only ever talks to a loopback address.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import httpx

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class EngineError(RuntimeError):
    """The engine could not be reached or refused the call. The message is safe to show the caller."""


def default_engine_url() -> str:
    """The running engine's address from the saved config, always on loopback."""
    from sentient.config import load_config

    cfg = load_config()
    host = cfg.gateway.host if cfg.gateway.host in LOOPBACK_HOSTS else "127.0.0.1"
    return f"http://{host}:{cfg.gateway.port}"


class EngineClient:
    """Synchronous REST calls to the engine. One instance per MCP server process."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0):
        parsed = urlparse(base_url)
        if parsed.hostname not in LOOPBACK_HOSTS:
            raise EngineError(
                f"Refusing to talk to a non-local engine address ({base_url!r}). "
                "The MCP server only works on this computer."
            )
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout, read=timeout),
            # never through a proxy from the environment: the engine is on this computer, and its
            # token must not travel through anything (a proxy setting also breaks loopback URLs
            # whose no_proxy list carries bracketed IPv6 hosts)
            trust_env=False,
        )

    # ------------------------------------------------------------------ plumbing
    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise EngineError(
                f"Sentient's engine is not answering at {self.base_url}. Start Sentient first, then try again."
            ) from exc
        if resp.status_code == 401:
            raise EngineError(
                "The engine refused this server's token. Restart the MCP server so it picks up the current token."
            )
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = str(resp.json().get("detail") or "")
            except ValueError:
                detail = resp.text[:200]
            raise EngineError(f"The engine said no ({resp.status_code}){': ' + detail if detail else ''}.")
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {"text": resp.text}

    def health(self) -> dict:
        """``GET /api/health`` (no token needed): is an engine up at this address?"""
        try:
            resp = self._http.get("/api/health")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise EngineError(
                f"No Sentient engine is running at {self.base_url}. Start Sentient first, then try again."
            ) from exc

    def get_config(self) -> dict:
        return self._call("GET", "/api/config")

    def integrations_config(self) -> dict:
        """The live ``integrations`` section: the MCP server's on/off switch lives here."""
        cfg = self.get_config()
        section = cfg.get("integrations") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}

    # ------------------------------------------------------------------ memories
    def search_memories(self, query: str, limit: int = 10) -> list[dict]:
        rows = self._call("GET", "/api/memories", params={"q": query, "limit": max(1, min(limit, 100))})
        return rows if isinstance(rows, list) else []

    def get_memory(self, memory_id: int) -> dict:
        return self._call("GET", f"/api/memories/{memory_id}")

    def add_memory(self, content: str, *, hold_for_review: bool, scope: str | None = None) -> dict:
        body: dict[str, Any] = {"content": content, "source": "mcp"}
        if scope:
            body["scope"] = scope
        if hold_for_review:
            body["hold_for_review"] = True
            body["review_from"] = "an MCP app"
        return self._call("POST", "/api/memories", json=body)

    # ------------------------------------------------------------------ tasks
    @staticmethod
    def _task_view(task: dict) -> dict:
        """The engine calls it ``task_id``; the MCP surface presents one ``id`` field."""
        if isinstance(task, dict) and task.get("id") is None and task.get("task_id") is not None:
            return {"id": task["task_id"], **task}
        return task

    def create_task(self, prompt: str) -> dict:
        return self._task_view(self._call("POST", "/api/tasks", json={"prompt": prompt}))

    def list_tasks(self) -> list[dict]:
        rows = self._call("GET", "/api/tasks")
        return [self._task_view(r) for r in rows] if isinstance(rows, list) else []

    def get_task(self, task_id: str) -> dict:
        return self._task_view(self._call("GET", f"/api/tasks/{task_id}"))

    # ------------------------------------------------------------------ notifications
    def notify(self, message: str, title: str | None = None) -> dict:
        body: dict[str, Any] = {"message": message, "source": "mcp"}
        if title:
            body["title"] = title
        return self._call("POST", "/api/notifications", json=body)

    # ------------------------------------------------------------------ chat
    def chat_turn(self, text: str, *, timeout: float = 180.0) -> dict:
        """One chat turn through ``POST /api/chat`` (NDJSON). The turn's own tools run inside the
        engine, under Sentient's approvals and rules; when a call has to ask, the question appears
        in the Sentient window like any other, and this call waits for the answer until ``timeout``.
        Returns ``{content, session_id}`` or ``{content, error, session_id}``."""
        session_id: str | None = None
        content = ""
        try:
            with self._http.stream(
                "POST", "/api/chat", json={"text": text, "channel": "mcp"},
                timeout=httpx.Timeout(30.0, read=timeout),
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise EngineError(f"The engine said no ({resp.status_code}) while starting the chat turn.")
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("session_id"):
                        session_id = event["session_id"]
                    if event.get("type") == "done":
                        content = str(event.get("content") or "")
                        if event.get("error"):
                            return {"content": content, "error": str(event["error"]), "session_id": session_id}
                    elif event.get("type") == "error":
                        return {
                            "content": content,
                            "error": str(event.get("message") or "the chat turn failed"),
                            "session_id": session_id,
                        }
        except httpx.TimeoutException:
            return {
                "content": content,
                "error": (
                    "No answer yet: the turn is still running in Sentient, most likely waiting for the user "
                    "to answer an approval in the Sentient window."
                ),
                "session_id": session_id,
            }
        except httpx.HTTPError as exc:
            raise EngineError(f"The chat turn broke off: {exc}") from exc
        return {"content": content, "session_id": session_id}
