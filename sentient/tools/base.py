"""The tool plugin interface.

A tool is a plain async Python function with type hints and a docstring; the
JSON schema the model sees is derived from the signature. A plugin groups tools
that share credentials (an integration such as Gmail) or a purpose (memory).

Every tool declares a ``risk`` so the approvals layer can decide whether to
ask the user first:

- ``read``   : looks things up, no side effects (search mail, get weather)
- ``write``  : creates or changes something reversible (draft, create task, save memory)
- ``send``   : irreversible or outward-facing (send email, post message, delete)
- ``exec``   : runs code or shell commands

A tool may also declare ``risk_fn(arguments, ctx) -> Risk | None`` to raise or
lower the risk per call (a browser click on "Place order" is ``send``). The
approvals layer and ``approval_request.risk`` use the *effective* risk.

Two more tags guard against outside content steering Sentient (ADR 0018):
``untrusted_output`` marks a tool whose result brings in content someone else wrote
(an email, a web page, a message); ``None`` picks the default in
``sentient.tools.rules.brings_untrusted``. ``exfiltrates`` (a bool, or ``fn(arguments, ctx) -> bool``)
marks a call that can move data out of Sentient even below ``send`` (typing into a web page,
inviting people). ``url_fn(arguments, ctx) -> str | None`` names the web address a call loads,
so an address that could carry data to a new site asks first.

Long-running tools stream output with ``ctx.progress({"kind": ..., "text": ...})``
(docs/API.md section 10). Inside an agent loop that becomes a ``tool_progress``
chat event; anywhere else it is a no-op.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, get_args, get_origin

from pydantic import BaseModel, create_model

log = logging.getLogger(__name__)


class Risk(IntEnum):
    read = 0
    write = 1
    send = 2
    exec = 3


# Who a run works for (ADR 0017). "user": someone asked for it (a chat, a task the user created or approved, a
# suggestion the user accepted). These origins are work nobody asked for, which may only read.
UNPROMPTED_ORIGINS = frozenset({"proactive", "heartbeat", "followups", "dreaming", "background"})


def is_unprompted(origin: str | None) -> bool:
    """True for work nobody asked for (an origin or a ``run_loop`` source in ``UNPROMPTED_ORIGINS``)."""
    return str(origin or "").strip().lower() in UNPROMPTED_ORIGINS


# (sink, call_id, tool_name) of the tool call running in the current asyncio task.
# The agent loop sets it before running a tool in its own task; nested work the tool
# starts (threads via asyncio.to_thread, child tasks) inherits it.
ProgressSink = Callable[[str, str, dict], None]
_CURRENT_CALL: contextvars.ContextVar[tuple[ProgressSink | None, str, str] | None] = contextvars.ContextVar(
    "sentient_tool_call", default=None
)


class _Awaitable:
    """Returned by ``ToolContext.progress`` so both ``ctx.progress(x)`` and ``await ctx.progress(x)`` work."""

    __slots__ = ()

    def __await__(self):
        return iter(())


_DONE = _Awaitable()


def current_call() -> tuple[str, str] | None:
    """``(call_id, tool_name)`` of the tool call being executed, if any."""
    cur = _CURRENT_CALL.get()
    return (cur[1], cur[2]) if cur else None


def bind_call(sink: ProgressSink | None, call_id: str, name: str) -> contextvars.Token:
    """Used by the agent loop: route ``ctx.progress`` in this task to ``sink``."""
    return _CURRENT_CALL.set((sink, call_id, name))


@dataclass
class ToolContext:
    """Passed to every tool call. Gives tools access to shared services without globals."""

    store: Any
    config: Any
    llm: Any
    memory: Any = None
    session_id: str | None = None
    channel: str = "web"
    extra: dict[str, Any] = field(default_factory=dict)
    # "user", or an ``UNPROMPTED_ORIGINS`` name when nobody asked for this work (then only reads may run)
    origin: str = "user"
    # where outside content came into this run ("Gmail"), or "" while it has none (ADR 0018). Set in code, never
    # cleared during the run; once set, anything that can send data out asks the user first.
    untrusted: str = ""
    # web hosts this run (or chat) already loaded; an address on one of them never asks for carrying data (ADR 0018)
    visited: set[str] = field(default_factory=set)

    @property
    def call_id(self) -> str | None:
        """Id of the tool call currently running (None outside the agent loop)."""
        cur = current_call()
        return cur[0] if cur else None

    def progress(self, payload: dict) -> _Awaitable:
        """Stream progress of a long-running tool: ``{kind: "stdout"|"stderr"|"status"|"frame"|"subagent",
        text?, image?, data?}``. Outside a chat turn (or with no listener) it does nothing.
        Safe to call from worker threads started with ``asyncio.to_thread``; may be awaited or not."""
        cur = _CURRENT_CALL.get()
        if cur is None or cur[0] is None:
            return _DONE
        sink, call_id, name = cur
        try:
            sink(call_id, name, dict(payload or {}))
        except Exception as exc:  # progress must never break the tool
            log.debug("tool progress dropped: %s", exc)
        return _DONE


ToolFn = Callable[..., Awaitable[Any]]
RiskFn = Callable[[dict, ToolContext], "Risk | Awaitable[Risk | None] | None"]
DescribeFn = Callable[[dict, ToolContext], "dict | Awaitable[dict | None] | None"]
ExfilFn = Callable[[dict, ToolContext], bool]
UrlFn = Callable[[dict, ToolContext], "str | None"]

# Default wording for approval prompts, by effective risk.
RISK_LABELS = {
    Risk.read: "Looks something up",
    Risk.write: "Changes something",
    Risk.send: "Sends",
    Risk.exec: "Runs code",
}


@dataclass
class Tool:
    name: str
    description: str
    fn: ToolFn
    params_model: type[BaseModel]
    risk: Risk = Risk.read
    plugin: str = "core"
    # True when the effect stays inside Sentient (its own memory, skills, files folder, task list).
    # Approvals mode "ask" does not prompt for internal tools; mode "always" still does.
    internal: bool = False
    # Optional per-call risk: ``risk_fn(arguments, ctx) -> Risk | None`` (sync or async). None keeps ``risk``.
    risk_fn: RiskFn | None = None
    # Optional approval wording: ``describe_fn(arguments, ctx) -> {risk_label?, target?}`` (sync or async).
    describe_fn: DescribeFn | None = None
    # True when the result brings in content someone else wrote; None uses the default (``rules.brings_untrusted``).
    untrusted_output: bool | None = None
    # True (or ``fn(arguments, ctx) -> True`` for a call) when it can move data out below ``send`` (typing into a
    # page, inviting people). Checked with ``rules.sends_out``.
    exfiltrates: bool | ExfilFn = False
    # The web address a call loads (``url_fn(arguments, ctx) -> str | None``), for ``rules.address_carries_data``.
    url_fn: UrlFn | None = None
    # False when every call must ask again: "Allow for this chat" never covers this tool (commands on the host)
    allow_for_chat: bool = True
    # Optional shorter version of a result too long for the model (``shorten_fn(result) -> result | None``), read
    # instead of a plain cut; the full result is still saved and shown (#264).
    shorten_fn: Callable[[Any], Any] | None = None

    def openai_schema(self) -> dict:
        schema = self.params_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    async def call(self, ctx: ToolContext, arguments: dict[str, Any]) -> Any:
        parsed = self.params_model.model_validate(arguments or {})
        kwargs = parsed.model_dump()
        return await self.fn(ctx, **kwargs)


async def effective_risk(tool: Tool, arguments: dict, ctx: ToolContext) -> Risk:
    """``tool.risk_fn(arguments, ctx)`` when the tool has one and it returns a risk, else ``tool.risk``.
    A risk function that raises is treated as ``exec``: it must never make a call look safer."""
    fn = getattr(tool, "risk_fn", None)
    if fn is None:
        return tool.risk
    try:
        value: Any = fn(arguments or {}, ctx)
        if inspect.isawaitable(value):
            value = await value
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("risk_fn of %s failed (%s); treating the call as exec", tool.name, exc)
        return Risk.exec
    if value is None:
        return tool.risk
    if isinstance(value, Risk):
        return value
    if isinstance(value, str):
        try:
            return Risk[value]
        except KeyError:
            return Risk.exec
    try:
        return Risk(int(value))
    except (TypeError, ValueError):
        return Risk.exec


async def describe_call(tool: Tool, arguments: dict, ctx: ToolContext, risk: Risk) -> dict:
    """``{risk_label, target}`` for an approval prompt: ``tool.describe_fn`` when it gives them,
    else the label for the effective risk and no target. Never raises."""
    out: dict[str, str | None] = {"risk_label": RISK_LABELS.get(Risk(risk), "Changes something"), "target": None}
    fn = getattr(tool, "describe_fn", None)
    if fn is None:
        return out
    try:
        value: Any = fn(arguments or {}, ctx)
        if inspect.isawaitable(value):
            value = await value
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.debug("describe_fn of %s failed: %s", tool.name, exc)
        return out
    if isinstance(value, dict):
        for key in ("risk_label", "target"):
            if value.get(key):
                out[key] = str(value[key])[:120]
    return out


def _params_model_from_signature(fn: Callable, name: str) -> type[BaseModel]:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "ctx":
            continue
        annotation = hints.get(pname, str)
        default = ... if param.default is inspect.Parameter.empty else param.default
        # unwrap Optional[X] defaults to None
        if default is ... and get_origin(annotation) is typing.Union and type(None) in get_args(annotation):
            default = None
        fields[pname] = (annotation, default)
    return create_model(f"{name}_params", **fields)  # type: ignore[call-overload]


def tool(
    name: str | None = None,
    *,
    risk: Risk = Risk.read,
    description: str | None = None,
    internal: bool = False,
    risk_fn: RiskFn | None = None,
    describe_fn: DescribeFn | None = None,
    untrusted_output: bool | None = None,
    exfiltrates: bool | ExfilFn = False,
    url_fn: UrlFn | None = None,
    allow_for_chat: bool = True,
):
    """Decorator turning ``async def fn(ctx, arg: type = default)`` into a Tool."""

    def wrap(fn: ToolFn) -> Tool:
        tool_name = name or fn.__name__
        desc = (description or inspect.getdoc(fn) or "").strip()
        model = _params_model_from_signature(fn, tool_name)
        return Tool(
            name=tool_name, description=desc, fn=fn, params_model=model, risk=risk, internal=internal,
            risk_fn=risk_fn, describe_fn=describe_fn, untrusted_output=untrusted_output, exfiltrates=exfiltrates,
            url_fn=url_fn, allow_for_chat=allow_for_chat,
        )

    return wrap


class ToolPlugin:
    """Subclass and set the class attributes; list Tool objects in ``tools``.

    Integrations additionally describe how they authenticate so the settings
    UI can render a Connect button without knowing anything about the service.
    """

    id: str = "plugin"
    display_name: str = "Plugin"
    description: str = ""
    category: str = "core"          # core | productivity | communication | knowledge | utilities
    icon: str = "IconPuzzle"        # tabler icon name used by the UI
    auth: str = "none"              # none | api_key | oauth | manual
    selection_hint: str = ""        # one line telling the model when these tools are relevant
    scoped: bool = False            # offered only to callers that name the tools (never in chat or the catalog)
    tools: list[Tool] = []

    def __init__(self) -> None:
        for t in self.tools:
            t.plugin = self.id

    async def is_connected(self, ctx: ToolContext) -> bool:
        return True

    async def setup(self, ctx: ToolContext) -> None:
        """Called once when the plugin is loaded (open clients, warm caches)."""
        return
