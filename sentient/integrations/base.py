"""Integration plugin framework.

An integration is a ``ToolPlugin`` that also describes how it authenticates, what
the settings UI must ask for, which privacy filters it honours and which events
it can trigger tasks on. Tools are written with :func:`itool`, which wraps the
function so that a missing connection, an HTTP failure or a network error comes
back to the model as a friendly ``{"error": ...}`` instead of an exception.

Tool bodies get credentials through ``await creds(ctx, "<id>")`` (raises
:class:`NotConnected`) and the manager through :func:`manager_from`.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel

from sentient.tools.base import Risk, Tool, ToolContext, ToolPlugin, tool

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

log = logging.getLogger(__name__)

USER_AGENT = "Sentient/3 (local personal assistant; https://github.com/existence-master/Sentient)"

# Set by IntegrationManager.start() so tools called without an app in ctx still work.
_CURRENT_MANAGER: IntegrationManager | None = None


class IntegrationError(Exception):
    """A failure whose message is safe and useful to show the user/model."""


class NotConnected(IntegrationError):
    pass


@dataclass
class SetupField:
    key: str
    label: str
    secret: bool = False
    required: bool = True
    help: str = ""
    placeholder: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeedBatch:
    """Result of one change-feed sync (``IntegrationPlugin.change_feed``).

    ``items`` are normalized items; each may carry ``_key`` (dedupe key, default ``id``) and
    ``_event`` (event name, default the source's first trigger). ``rebaselined`` means the
    cursor was (re)established from "now" and nothing older was emitted.
    """

    cursor: str | None
    items: list[dict] = field(default_factory=list)
    rebaselined: bool = False
    note: str | None = None


class IntegrationPlugin(ToolPlugin):
    """Base class for every integration (builtin, OAuth, API key, manual).

    Optional hooks the manager looks for:
    - ``async poll(mgr, since) -> list[dict]``           timer polling (poll_source)
    - ``async change_feed(mgr, cursor) -> FeedBatch``     incremental change feed (every fast_sync_seconds)
    - ``async watch(mgr) -> None``                        long-running push watcher (IMAP IDLE)
    - ``async dynamic_triggers(mgr) -> list[dict]``       triggers computed at listing time (webhooks)
    """

    auth_type: str = "builtin"  # builtin | oauth | api_key | manual | mcp
    setup_fields: list[SetupField] = []
    instructions_md: str = ""
    docs_url: str = ""
    privacy_fields: list[str] = []  # subset of keywords | emails | labels
    privacy_kind: str | None = None  # "email" | "event": which privacy rule applies to emitted items
    feed_kind: str | None = None  # label for feed_status: gmail_history | calendar_sync_token | imap_idle
    triggers: list[dict] = []  # [{"event": "new_email", "label": "New email"}]
    optional_alternative_for: str | None = None  # e.g. "weather" for accuweather

    def __init__(self) -> None:
        super().__init__()
        self.auth = self.auth_type

    @property
    def is_builtin(self) -> bool:
        return self.auth_type == "builtin"

    # ------------------------------------------------------------------ auth hooks
    async def begin_oauth(self, fields: dict[str, str], mgr: IntegrationManager) -> dict | None:
        """Start a browser-based flow and return ``{auth_url, state, ...}``, or None to use ``validate``."""
        return None

    async def validate(self, fields: dict[str, str], mgr: IntegrationManager) -> tuple[dict, str | None]:
        """api_key/manual: check the fields against the service.

        Returns ``(credentials_to_store, account_label)``; raise IntegrationError on failure.
        """
        missing = [f.label for f in self.setup_fields if f.required and not str(fields.get(f.key, "")).strip()]
        if missing:
            raise IntegrationError("Please fill in: " + ", ".join(missing))
        return {f.key: str(fields.get(f.key, "")).strip() for f in self.setup_fields}, None

    async def test(self, credentials: dict | None, mgr: IntegrationManager) -> str:
        """Return a short human detail when the connection works; raise on failure."""
        if self.auth_type == "builtin":
            return "Built in, nothing to set up."
        _, label = await self.validate(credentials or {}, mgr)
        return f"Connected{f' as {label}' if label else ''}."


# ---------------------------------------------------------------------------- tool helpers
def manager_from(ctx: ToolContext | None) -> IntegrationManager:
    app = (ctx.extra or {}).get("app") if ctx is not None else None
    mgr = getattr(app, "integrations", None) if app is not None else None
    if mgr is None or not hasattr(mgr, "get_credentials"):
        mgr = _CURRENT_MANAGER
    if mgr is None:
        raise IntegrationError("Integrations are not running.")
    return mgr


async def creds(ctx: ToolContext, plugin_id: str) -> dict:
    mgr = manager_from(ctx)
    if not await mgr.is_connected(plugin_id):
        raise NotConnected(plugin_id)
    c = await mgr.get_credentials(plugin_id)
    if not c:
        raise NotConnected(plugin_id)
    return c


def _display(plugin_id: str) -> str:
    mgr = _CURRENT_MANAGER
    if mgr is not None:
        p = mgr.plugin(plugin_id)
        if p is not None:
            return p.display_name
    return plugin_id


def _http_error_message(exc: httpx.HTTPStatusError) -> str:
    resp = exc.response
    detail = ""
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = err.get("message") or err.get("status") or ""
            elif isinstance(err, str):
                detail = body.get("error_description") or err
            detail = detail or body.get("message") or ""
    except Exception:
        detail = resp.text[:300]
    return f"HTTP {resp.status_code}{': ' + str(detail) if detail else ''}"


def itool(plugin_id: str, name: str, *, risk: Risk = Risk.read, description: str | None = None, internal: bool = False):
    """Like ``@tool`` but errors become friendly ``{"error": ...}`` results."""

    def wrap(fn: Callable[..., Awaitable[Any]]) -> Tool:
        t = tool(name, risk=risk, description=description, internal=internal)(fn)

        @functools.wraps(fn)
        async def safe(ctx: ToolContext, **kwargs: Any) -> Any:
            try:
                return await fn(ctx, **kwargs)
            except NotConnected:
                d = _display(plugin_id)
                return {"error": f"{d} isn't connected yet. Connect it from Integrations."}
            except IntegrationError as exc:
                return {"error": str(exc)}
            except httpx.HTTPStatusError as exc:
                return {"error": f"{_display(plugin_id)} request failed ({_http_error_message(exc)})."}
            except httpx.RequestError as exc:
                return {"error": f"Couldn't reach {_display(plugin_id)}: {type(exc).__name__}. Check the internet connection."}

        t.fn = safe
        return t

    return wrap


class _NoParams(BaseModel):
    pass


class JsonSchemaTool(Tool):
    """A tool whose parameters are an externally supplied JSON schema (MCP tools)."""

    def __init__(self, *, name: str, description: str, fn: Callable[..., Awaitable[Any]], input_schema: dict,
                 risk: Risk, plugin: str):
        super().__init__(name=name, description=description, fn=fn, params_model=_NoParams, risk=risk, plugin=plugin)
        schema = dict(input_schema or {})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        self.input_schema = schema

    def openai_schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                  "parameters": self.input_schema}}

    async def call(self, ctx: ToolContext, arguments: dict[str, Any]) -> Any:
        return await self.fn(ctx, **(arguments or {}))
