"""Subagent tools: hand self-contained work to workers that run alongside the chat (docs/API.md section 10)."""

from __future__ import annotations

import json
from typing import Any

from sentient.agent import reader
from sentient.tools.base import Risk, ToolContext, ToolPlugin, effective_risk, is_unprompted, tool
from sentient.tools.rules import (
    OUTSIDE_KEY,
    address_carries_data,
    address_host,
    brings_untrusted,
    call_address,
    result_outside_source,
    sends_out,
    untrusted_source,
)

MAX_PARALLEL_TASKS = 10
PLUGIN_ID = "subagents"


def _manager(ctx: ToolContext) -> Any:
    app = ctx.extra.get("app")
    return getattr(app, "subagents", None) if app is not None else None


def _progress(ctx: ToolContext):
    def on_update(subagent_id: str, message: str) -> None:
        ctx.progress({"kind": "subagent", "text": message, "data": {"subagent_id": subagent_id, "message": message}})

    return on_update


# a subagent's summary can carry what it read on the web or in mail, so it counts as outside content (ADR 0018)
@tool("delegate_task", risk=Risk.write, internal=True, untrusted_output=True)
async def delegate_task(
    ctx: ToolContext,
    goal: str,
    context: str = "",
    tools: list[str] | None = None,
    background: bool = False,
) -> dict:
    """Hand one self-contained piece of work to a subagent that works on its own with your tools and returns a
    summary. Good for multi-step research or long jobs. The subagent cannot see this chat: put everything it needs in
    goal and context. tools optionally limits it to these tool names or plugin ids. background=true returns at once
    and posts the result to this chat when it is done. Subagents cannot send, buy, delete or run code."""
    mgr = _manager(ctx)
    if ctx.extra.get("subagent_id"):
        return {"error": "Subagents cannot start other subagents. Do this part yourself or report it back."}
    if mgr is None:
        return {"error": "Subagents are not available."}
    return await mgr.delegate(
        goal,
        context=context,
        tools=tools,
        session_id=ctx.session_id,
        parent_call_id=ctx.call_id,
        background=background,
        on_update=None if background else _progress(ctx),
        origin=ctx.origin,
        untrusted=ctx.untrusted,
        memory_allowed=ctx.memory_allowed,  # never wider than this turn (#525)
    )


@tool("delegate_tasks", risk=Risk.write, internal=True, untrusted_output=True)
async def delegate_tasks(ctx: ToolContext, tasks: list[dict]) -> dict:
    """Run several independent pieces of work in parallel, one subagent each, and get every summary back.
    tasks is a list of objects like {"goal": "...", "context": "..."}. Each subagent cannot see this chat,
    so give each one everything it needs."""
    mgr = _manager(ctx)
    if ctx.extra.get("subagent_id"):
        return {"error": "Subagents cannot start other subagents. Do this part yourself or report it back."}
    if mgr is None:
        return {"error": "Subagents are not available."}
    valid = [t for t in tasks if isinstance(t, dict) and str(t.get("goal") or "").strip()]
    if not valid:
        return {"error": 'Give at least one task like {"goal": "...", "context": "..."}.'}
    if len(valid) > MAX_PARALLEL_TASKS:
        return {"error": f"At most {MAX_PARALLEL_TASKS} tasks at once. Split the work or combine related goals."}
    results = await mgr.delegate_many(
        valid, session_id=ctx.session_id, parent_call_id=ctx.call_id, on_update=_progress(ctx), origin=ctx.origin,
        untrusted=ctx.untrusted, memory_allowed=ctx.memory_allowed,
    )
    return {"results": results}


def _refuse_source(app: Any, name: str, ctx: ToolContext) -> str:
    """Why the reader can't use ``name`` to fetch its document, or "". Never a tool the turn itself can't use: in a
    turn without memory (#525) the memory and history look-ups are refused here before anything runs."""
    source = app.registry.get(name) if app is not None else None
    if source is None:
        return f"There is no tool called {name}. Give the look-up tool that fetches the email, page or file."
    if ctx.withholds(name):
        return f"{name} isn't available in this chat: it doesn't use what Sentient knows about the user."
    if source.plugin == PLUGIN_ID or app.registry.is_hidden(source.plugin) or app.approvals.is_hidden(source):
        return f"{name} can't be used here."
    if Risk(source.risk) != Risk.read:
        return f"{name} changes or sends something. The reader only takes look-up tools."
    if app.approvals.rule(source) in {"ask", "if_asked", "hand_off"}:
        return f"{name} is set to ask first, so call it yourself."
    return ""


# Reads outside content for the model through a tool-less reader (#144, ADR 0031). Its result marks the run only when
# it carries text, an address or a link (``outside_content`` first in the result, ADR 0025's way of saying so).
@tool("read_with_reader", risk=Risk.read, internal=True, untrusted_output=False)
async def read_with_reader(ctx: ToolContext, tool: str, fields: dict[str, str], arguments: dict | None = None) -> dict:
    """Read an email, web page or file through a reader that has no tools and returns only the fields you ask for, so
    instructions hidden in it can't reach you. tool is the look-up that fetches it (gmail_read_message,
    email_imap_read, web_fetch...) with its arguments. fields maps each name to its kind: "text" (one short sentence),
    "email", "url", "date", "number", "yes_no", "id" (a message id to reply to or forward) or choices like
    "urgent|normal|low". Use it when you need a few facts from mail or a page rather than its full text."""
    app = ctx.extra.get("app")
    mgr = getattr(app, "subagents", None) if app is not None else None
    if mgr is None or getattr(app, "registry", None) is None:
        return {"error": "The reader is not available."}
    try:
        wanted = reader.parse_fields(fields)
    except ValueError as exc:
        return {"error": str(exc)}
    refusal = _refuse_source(app, tool, ctx)
    if refusal:
        return {"error": refusal}
    source = app.registry.get(tool)
    args = dict(arguments or {})
    risk = await effective_risk(source, args, ctx)
    if risk != Risk.read or sends_out(source, risk, args, ctx):
        return {"error": f"{tool} would change or send something with these arguments. The reader only reads."}
    address = call_address(source, args, ctx)
    host = address_host(address)
    if ctx.untrusted and host and host not in ctx.visited and address_carries_data(address):
        # the address itself could carry data out (ADR 0018): only the agent loop may ask the user about that
        return {"error": f"This address could carry data to {host}. Call {tool} yourself so Sentient can check first."}
    try:
        result = await source.call(ctx, args)
    except Exception as exc:  # bad arguments for that tool
        return {"error": f"{tool} failed: {type(exc).__name__}: {str(exc)[:300]}"}
    if isinstance(result, dict) and result.get("error"):
        return {"error": f"{tool} failed: {result['error']}"}
    if host:
        ctx.visited.add(host)
    document = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    where = untrusted_source(source, app.registry)
    attended = bool(ctx.session_id) and not (ctx.extra.get("task_id") or ctx.extra.get("subagent_id")) and not (
        is_unprompted(ctx.origin)
    )
    out = await mgr.read(document, wanted, source=where, session_id=ctx.session_id, parent_call_id=ctx.call_id,
                         attended=attended)
    if out.get("status") != "completed":
        return {"error": out.get("error") or "The reader failed.", "subagent_id": out.get("subagent_id")}
    values = out["fields"]
    answer = {"source": where, "fields": values, "subagent_id": out["subagent_id"]}
    outside = brings_untrusted(source) or bool(result_outside_source(result))
    return {OUTSIDE_KEY: where, **answer} if outside and reader.brings_outside(wanted, values) else answer


class SubagentsPlugin(ToolPlugin):
    id = "subagents"
    display_name = "Subagents"
    description = "Hand self-contained work to subagents that run alongside the chat, in parallel or in the background."
    category = "core"
    icon = "IconAffiliate"
    selection_hint = "large or multi-part work: research several things in parallel, long jobs in the background, deep dives"
    tools = [delegate_task, delegate_tasks, read_with_reader]


PLUGIN = SubagentsPlugin()
