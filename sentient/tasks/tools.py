"""Chat-facing task tools (v2 mcp_hub/tasks): create, search and check tasks.

Registered by ``TaskService.start()``. Never offered inside a task run or swarm worker.
"""

from __future__ import annotations

import json
from typing import Any

from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool


def _service(ctx: ToolContext) -> Any:
    app = ctx.extra.get("app")
    return getattr(app, "tasks", None)


def _compact(task: dict) -> dict:
    runs = task.get("runs") or []
    last = runs[-1] if runs else None
    latest = None
    if last:
        latest = {
            "run_id": last["run_id"],
            "status": last["status"],
            "started_at": last.get("execution_start_time"),
            "finished_at": last.get("finished_at"),
            "summary": (last.get("result") or {}).get("summary"),
            "error": last.get("error"),
        }
    return {
        "task_id": task["task_id"],
        "name": task["name"],
        "description": task["description"][:500],
        "status": task["status"],
        "priority": task["priority"],
        "task_type": task["task_type"],
        "schedule": task["schedule"],
        "enabled": task["enabled"],
        "next_execution_at": task["next_execution_at"],
        "last_execution_at": task["last_execution_at"],
        "plan": task["plan"],
        "runs": len(runs),
        "latest_run": latest,
        "open_questions": [q["text"] for q in task.get("clarifying_questions") or [] if not q.get("answer")],
        "error": task.get("error"),
        "script": _script_summary(task.get("script")),
    }


def _script_summary(script: Any) -> dict | None:
    if not isinstance(script, dict):
        return None
    last = script.get("last_result")
    shown = last if isinstance(last, str) else json.dumps(last, ensure_ascii=False, default=str)
    return {
        "code": str(script.get("code") or "")[:3000],
        "condition": script.get("condition"),
        "then": script.get("then"),
        "last_result": shown[:500] if last is not None else None,
        "last_run_at": script.get("last_run_at"),
        "last_error": script.get("last_error"),
    }


@tool("create_task_from_prompt", risk=Risk.write, internal=True)
async def create_task_from_prompt(ctx: ToolContext, prompt: str, is_swarm: bool = False) -> dict:
    """Hand work to the background task system. Use it for work that should happen later or
    repeatedly (a specific future time, recurring like "every weekday at 9am", or triggered by new
    emails or calendar events) and for long multi-step work that should run in the background.
    Set is_swarm=true to process a list of many items in parallel. Do NOT use it for things you can
    do right now in this conversation. `prompt` is a complete natural-language description of the
    work including any schedule; the task is planned and then waits for the user's approval."""
    svc = _service(ctx)
    if svc is None:
        return {"status": "failure", "error": "The task system is not available."}
    context = {"source": "chat"}
    if ctx.session_id:
        context["session_id"] = ctx.session_id
    task = await svc.create_task(prompt, is_swarm=is_swarm, source="chat", original_context=context)
    short = prompt[:50] + "..." if len(prompt) > 50 else prompt
    return {
        "status": "success",
        "task_id": task["task_id"],
        "result": f"Task '{short}' has been created and is being planned. It will appear in Tasks for approval.",
    }


@tool("search_tasks", risk=Risk.read)
async def search_tasks(ctx: ToolContext, query: str = "", status: str | None = None) -> dict:
    """Find the user's existing tasks by keyword and/or status (planning, approval_pending, pending,
    active, processing, completed, error, declined, archived). Returns up to 20, highest priority first."""
    svc = _service(ctx)
    if svc is None:
        return {"status": "failure", "error": "The task system is not available."}
    tasks = await svc.search(query, status or None)
    return {"status": "success", "tasks": [_compact(t) for t in tasks]}


@tool("get_task_status", risk=Risk.read)
async def get_task_status(ctx: ToolContext, task_id: str) -> dict:
    """Get the status, plan, schedule and latest run result of one task by its task_id."""
    svc = _service(ctx)
    if svc is None:
        return {"status": "failure", "error": "The task system is not available."}
    from sentient.tasks.service import TaskNotFound

    try:
        task = await svc.get(task_id)
    except TaskNotFound:
        return {"status": "failure", "error": f"No task with id {task_id}."}
    return {"status": "success", "task": _compact(task)}


@tool("update_task", risk=Risk.write, internal=True)
async def update_task(
    ctx: ToolContext,
    task_id: str,
    name: str | None = None,
    description: str | None = None,
    enabled: bool | None = None,
    schedule: dict | None = None,
    script_code: str | None = None,
    script_condition: str | None = None,
    script_then: str | None = None,
) -> dict:
    """Change an existing task directly: rename it, edit its description, turn it on or off (`enabled`),
    change its `schedule` (same shape as in search_tasks results; every N minutes is
    {"type": "recurring", "frequency": "interval", "interval_minutes": N}), or edit a watch job's check
    script (`script_code`, `script_condition` "alert"|"changed", `script_then` "notify"|"run").
    Changed script code waits for the user's approval before it runs again. To change what a normal
    task does, use request_task_change instead."""
    svc = _service(ctx)
    if svc is None:
        return {"status": "failure", "error": "The task system is not available."}
    from sentient.tasks.service import TaskConflict, TaskNotFound

    fields: dict[str, Any] = {}
    for key, value in (("name", name), ("description", description), ("enabled", enabled), ("schedule", schedule)):
        if value is not None:
            fields[key] = value
    script = {k: v for k, v in (("code", script_code), ("condition", script_condition), ("then", script_then)) if v}
    if script:
        fields["script"] = script
    if not fields:
        return {"status": "failure", "error": "Nothing to change: pass at least one field."}
    try:
        task = await svc.update(task_id, fields, reapprove_script=True)
    except TaskNotFound:
        return {"status": "failure", "error": f"No task with id {task_id}."}
    except (TaskConflict, ValueError) as exc:
        return {"status": "failure", "error": str(exc)}
    await svc.publish(task_id)
    note = " The new script is waiting for the user's approval in Tasks." if task["status"] == "approval_pending" and script else ""
    return {"status": "success", "task": _compact(task), "result": f"Task updated.{note}"}


@tool("request_task_change", risk=Risk.write, internal=True)
async def request_task_change(ctx: ToolContext, task_id: str, message: str) -> dict:
    """Ask for a change to what a task does ("also include the weather", "send it to Slack instead").
    The task is re-planned with the request and its previous result, then waits for the user's approval."""
    svc = _service(ctx)
    if svc is None:
        return {"status": "failure", "error": "The task system is not available."}
    from sentient.tasks.service import TaskConflict, TaskNotFound

    try:
        task = await svc.chat(task_id, message)
    except TaskNotFound:
        return {"status": "failure", "error": f"No task with id {task_id}."}
    except (TaskConflict, ValueError) as exc:
        return {"status": "failure", "error": str(exc)}
    return {"status": "success", "task_id": task["task_id"], "result": "The task is being re-planned with your change."}


class TasksPlugin(ToolPlugin):
    id = "tasks"
    display_name = "Tasks"
    description = (
        "Create and track long-running background tasks: scheduled, recurring, triggered, swarm and watch jobs "
        "(check scripts that alert you when something happens)."
    )
    category = "core"
    icon = "IconChecklist"
    selection_hint = (
        "delegating scheduled, recurring, triggered or long background work; watching for something and alerting; "
        "checking on or changing existing tasks"
    )
    tools = [create_task_from_prompt, search_tasks, get_task_status, update_task, request_task_change]
