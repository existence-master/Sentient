"""A thin client for the running engine's REST API, used by the MCP server (ADR 0034).

The MCP server is a separate process started by the MCP client (for example Claude Code running
``sentient mcp``). It holds no state of its own. It finds the engine through the discovery file
the engine writes at startup (its loopback port; the desktop picks a fresh port every launch, so
the configured port is not where it listens), and it signs in with the MCP-only token from the
keychain: a token that opens only the endpoints these tools use, never the gateway token. It
never reads or writes the gateway token.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import httpx

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class EngineError(RuntimeError):
    """The engine could not be reached or refused the call. The message is safe to show the caller."""


def discovered_engine_url() -> str:
    """The running engine's address from its discovery file (loopback, port only)."""
    from sentient import paths

    try:
        data = json.loads(paths.discovery_file().read_text(encoding="utf-8"))
        port = int(data["port"])
    except Exception:
        port = 0
    if not port:
        raise EngineError(
            "Sentient's engine is not running (no discovery file). Start Sentient first, then try again."
        )
    return f"http://127.0.0.1:{port}"


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
    def _call(self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any) -> Any:
        if timeout is not None:
            kwargs["timeout"] = httpx.Timeout(timeout, read=timeout)
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            # not "not answering": the engine is up but slow (a memory save runs a model call), and
            # the change may well have landed. Say so, so the caller checks before retrying and
            # does not save the same thing twice.
            raise EngineError(
                "The engine did not answer in time. The change may have gone through anyway; "
                "check in Sentient before trying again."
            ) from exc
        except httpx.HTTPError as exc:
            raise EngineError(
                f"Sentient's engine is not answering at {self.base_url}. Start Sentient first, then try again."
            ) from exc
        if resp.status_code == 401:
            raise EngineError(
                "The engine refused this server's token. It may have been replaced in Sentient: "
                "restart this MCP server so it reads the current one."
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

    def status(self) -> dict:
        """``GET /api/mcp/status``: whether the feature is on and what its memories do, readable
        with the MCP-only token (the full config is not, on purpose)."""
        data = self._call("GET", "/api/mcp/status")
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ memories
    def search_memories(self, query: str, limit: int = 10) -> list[dict]:
        rows = self._call("GET", "/api/memories", params={"q": query, "limit": max(1, min(limit, 100))})
        return rows if isinstance(rows, list) else []

    def get_memory(self, memory_id: int) -> dict:
        return self._call("GET", f"/api/memories/{memory_id}")

    def add_memory(
        self, content: str, *, hold_for_review: bool, scope: str | None = None, app: str | None = None
    ) -> dict:
        body: dict[str, Any] = {"content": content, "source": "mcp"}
        if scope:
            body["scope"] = scope
        if app:
            body["app"] = app  # the engine names the review note (and forces the wait) from this
        if hold_for_review:
            body["hold_for_review"] = True
            body["review_from"] = app or "an MCP app"
        # saving runs the memory pipeline (a model call) inside the engine: give it room, so a
        # slow save is not reported as a failure after it already landed
        return self._call("POST", "/api/memories", json=body, timeout=120.0)

    # ------------------------------------------------------------------ tasks
    @staticmethod
    def _task_view(task: dict) -> dict:
        """The engine calls it ``task_id``; the MCP surface presents one ``id`` field."""
        if isinstance(task, dict) and task.get("id") is None and task.get("task_id") is not None:
            return {"id": task["task_id"], **task}
        return task

    def create_task(self, prompt: str, *, app: str | None = None) -> dict:
        body: dict[str, Any] = {"prompt": prompt}
        if app:
            body["app"] = app
        return self._task_view(self._call("POST", "/api/tasks", json=body))

    def list_tasks(self) -> list[dict]:
        rows = self._call("GET", "/api/tasks")
        return [self._task_view(r) for r in rows] if isinstance(rows, list) else []

    def get_task(self, task_id: str) -> dict:
        return self._task_view(self._call("GET", f"/api/tasks/{task_id}"))

    # ------------------------------------------------------------------ notifications
    def notify(self, message: str, title: str | None = None, *, app: str | None = None) -> dict:
        body: dict[str, Any] = {"message": message, "source": "mcp"}
        if title:
            body["title"] = title
        if app:
            body["app"] = app
        return self._call("POST", "/api/notifications", json=body)

    # ------------------------------------------------------------------ chat
    def chat_turn(self, text: str, *, timeout: float = 180.0, app: str | None = None) -> dict:
        """One chat turn through ``POST /api/chat`` (NDJSON). The turn is outside content inside
        the engine: its tools run under Sentient's approvals and rules, and a call that has to ask
        appears in the Sentient window and as an approval notification the user can answer there
        or on their phone. Returns ``{content, session_id}`` (plus ``approval_asked`` when a
        question went to the user, and ``error`` when the turn failed or is still waiting)."""
        session_id: str | None = None
        content = ""
        approval_asked = False
        try:
            body: dict[str, Any] = {"text": text, "channel": "mcp"}
            if app:
                body["app"] = app  # display only: the engine names the turn's outside-content mark with it
            with self._http.stream(
                "POST", "/api/chat", json=body,
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
                    if event.get("type") == "approval_request":
                        approval_asked = True
                    elif event.get("type") == "done":
                        content = str(event.get("content") or "")
                        if event.get("error"):
                            return {"content": content, "error": str(event["error"]),
                                    "session_id": session_id, "approval_asked": approval_asked}
                    elif event.get("type") == "error":
                        return {
                            "content": content,
                            "error": str(event.get("message") or "the chat turn failed"),
                            "session_id": session_id,
                            "approval_asked": approval_asked,
                        }
        except httpx.TimeoutException:
            return {
                "content": content,
                "error": (
                    "No answer yet: the turn is still running in Sentient, most likely waiting for the user "
                    "to answer an approval in Sentient (it is in the window and in Sentient's notifications)."
                ),
                "session_id": session_id,
                "approval_asked": approval_asked,
            }
        except httpx.HTTPError as exc:
            raise EngineError(f"The chat turn broke off: {exc}") from exc
        return {"content": content, "session_id": session_id, "approval_asked": approval_asked}
