"""Subagent tools: hand self-contained work to workers that run alongside the chat (docs/API.md section 10)."""

from __future__ import annotations

from typing import Any

from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

MAX_PARALLEL_TASKS = 10


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
        untrusted=ctx.untrusted,
    )
    return {"results": results}


class SubagentsPlugin(ToolPlugin):
    id = "subagents"
    display_name = "Subagents"
    description = "Hand self-contained work to subagents that run alongside the chat, in parallel or in the background."
    category = "core"
    icon = "IconAffiliate"
    selection_hint = "large or multi-part work: research several things in parallel, long jobs in the background, deep dives"
    tools = [delegate_task, delegate_tasks]


PLUGIN = SubagentsPlugin()
