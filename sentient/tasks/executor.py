"""Execution of a single task run and structured result generation.

Uses ``Agent.run_loop`` (the shared tool-calling engine) with the ``executor`` role
and maps its typed events to v2 ProgressUpdates. The transcript is checkpointed to
``task_runs.messages`` after every tool result so a run can resume after a restart.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from sentient.agent.loop import LoopResult, history_to_openai
from sentient.llm.events import TextDelta, ThinkingDelta, ToolCallEvent, ToolResultEvent
from sentient.tasks.jsonio import complete_json_object
from sentient.tasks.prompts import (
    CORE_HELPER_PLUGINS,
    EXCLUDED_PLUGINS,
    EXECUTOR_KICKOFF,
    RESULT_GENERATOR_SYSTEM_PROMPT,
    RESUME_NOTE,
    build_executor_prompt,
)
from sentient.tasks.schedule import get_tz
from sentient.tools.base import Risk

if TYPE_CHECKING:  # pragma: no cover
    from sentient.tasks.service import TaskService
    from sentient.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# Literal fallback text Agent.run_loop leaves when it runs out of rounds.
STEP_LIMIT_TEXT = "I reached the step limit before finishing. Tell me how to continue."
MAX_PROGRESS_RESULT_CHARS = 6000
# Writing skills is the evolution reviewer's job; inside a run it only distracts small models.
EXECUTOR_EXCLUDED_TOOLS = {"skill_save"}
MAX_CONTINUE_NUDGES = 2
CONTINUE_NUDGE = (
    "You described what you will do next but did not do it. Carry out the remaining plan steps now by calling "
    "the tools. When everything is done, reply with a short summary of what you actually did."
)
_ANNOUNCE_RE = re.compile(
    r"\b(?:I will|I'll|I am going to|I'm going to|let me|let's|now I will|next,? I will|I shall)\s+"
    r"(?:now\s+|then\s+|next\s+)?(?:proceed|create|write|save|start|begin|go ahead|draft|compose|generate|"
    r"make|put|store|send|update|add|fetch|search|check)\b",
    re.IGNORECASE,
)
_DONE_RE = re.compile(
    r"\b(?:saved|created|written|wrote|done|completed|finished|sent|updated|added|here is|here's)\b", re.IGNORECASE
)


def announces_unfinished_work(text: str) -> bool:
    """True for a short final answer that only announces the next step ("I will now write the poem...")."""
    if not text or len(text) > 1200:
        return False
    return bool(_ANNOUNCE_RE.search(text[:500])) and not _DONE_RE.search(text)


class RunFailed(RuntimeError):
    """A run ended without a usable final answer."""


def is_first_retry_attempt(run: dict) -> bool:
    """A run created by 'retry' that has not been resumed after a restart yet."""
    return bool(run.get("retry_of")) and not int(run.get("resume_count") or 0)


def truncate(value: Any, limit: int = MAX_PROGRESS_RESULT_CHARS) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + " ... [truncated]"
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) <= limit:
        return value
    return text[:limit] + " ... [truncated]"


def select_tools(
    registry: ToolRegistry, requested: list[str], *, include_core: bool = True
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Tool names for the plugins named in a plan (plugin ids, or tool names by mistake).

    Returns ``(tool_names, {plugin_id: [tool names]}, missing)``. Core helpers (memory,
    time, files, skills) are added with read/write tools only. The tasks plugin is
    never included, so a run cannot spawn more tasks.
    """
    plugins = {p.id: p for p in registry.plugins()}
    by_tool = {t.name: t.plugin for t in registry.tools()}
    wanted: list[str] = []
    missing: list[str] = []
    for raw in requested:
        key = str(raw or "").strip()
        if not key or key.lower() in {"none", "null"}:
            continue
        pid = key if key in plugins else key.lower() if key.lower() in plugins else by_tool.get(key)
        if pid is None:
            if key not in missing:
                missing.append(key)
            continue
        if pid in EXCLUDED_PLUGINS or pid in wanted:
            continue
        wanted.append(pid)
    tool_map: dict[str, list[str]] = {}

    def usable(t) -> bool:  # registered, not excluded and not behind a "never" rule (ADR 0016)
        return registry.get(t.name) is not None and t.name not in EXECUTOR_EXCLUDED_TOOLS and not registry.is_blocked(t)

    for pid in wanted:
        names = [t.name for t in plugins[pid].tools if usable(t)]
        if names:
            tool_map[pid] = names
    if include_core:
        for pid in CORE_HELPER_PLUGINS:
            if pid in plugins and pid not in tool_map:
                names = [t.name for t in plugins[pid].tools if usable(t) and t.risk <= Risk.write]
                if names:
                    tool_map[pid] = names
    tool_names = [n for names in tool_map.values() for n in names]
    return tool_names, tool_map, missing


class ProgressMapper:
    """Turns streamed agent events into persisted ProgressUpdates."""

    def __init__(self, svc: TaskService, task_id: str, run_id: str):
        self.svc = svc
        self.task_id = task_id
        self.run_id = run_id
        self.thought = ""
        self.text = ""

    async def _emit(self, message: dict) -> None:
        await self.svc.progress(self.task_id, self.run_id, message)

    async def flush_thought(self) -> None:
        text, self.thought = self.thought.strip(), ""
        if text:
            await self._emit({"type": "thought", "content": text})

    async def handle(self, event: Any, messages: list[dict]) -> None:
        if isinstance(event, ThinkingDelta):
            self.thought += event.text
        elif isinstance(event, TextDelta):
            self.text += event.text
        elif isinstance(event, ToolCallEvent):
            await self.flush_thought()
            narration, self.text = self.text.strip(), ""
            if narration:  # text the model wrote before calling tools is its reasoning
                await self._emit({"type": "thought", "content": narration})
            await self._emit({"type": "tool_call", "tool_name": event.name, "parameters": event.arguments})
        elif isinstance(event, ToolResultEvent):
            await self._emit(
                {"type": "tool_result", "tool_name": event.name, "result": truncate(event.result), "is_error": event.is_error}
            )
            # run_loop appends the tool message right after yielding this event; include it now
            tool_msg = {
                "role": "tool",
                "tool_call_id": event.call_id,
                "name": event.name,
                "content": json.dumps(event.result, ensure_ascii=False, default=str),
            }
            await self.svc.repo.update_run(self.run_id, {"messages": [*messages, tool_msg]})
        # Usage is ignored; Error is reported once by the service when the run finishes.


async def _executor_system_prompt(
    svc: TaskService, task: dict, run: dict, plan: list[dict], tool_map: dict[str, list[str]]
) -> str:
    app = svc.app
    cfg = app.config.assistant
    tz = get_tz(svc.tz_name())
    memories: list[str] = []
    if app.memory is not None:
        query = " ".join(
            str(x) for x in (task.get("name"), task.get("description"), (run.get("trigger_data") or {}).get("subject")) if x
        )[:500]
        try:
            facts = await app.memory.recall(query)
            memories = [f["content"] for f in facts if f.get("content")]
        except Exception as exc:
            log.debug("memory recall for task %s skipped: %s", task["id"], exc)
    return build_executor_prompt(
        assistant_name=cfg.name or "Sentient",
        user_name=cfg.user_name or "User",
        user_location=cfg.location or "Not specified",
        current_time=svc.now().astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z (%A)"),
        task_id=task["id"],
        run_id=run["id"],
        name=task.get("name") or "",
        description=task.get("description") or "",
        plan=plan,
        original_context=task.get("original_context"),
        trigger_event_data=run.get("trigger_data"),
        memories=memories,
        tool_map=tool_map,
    )


async def execute_single(svc: TaskService, task: dict, run: dict, *, resume: bool = False) -> LoopResult:
    app = svc.app
    assert app.agent is not None
    max_rounds = app.config.tasks.max_tool_rounds
    task_id, run_id = task["id"], run["id"]
    plan = run.get("plan") or task.get("plan") or []
    requested = [str(s.get("tool", "")) for s in plan if isinstance(s, dict)]
    tool_names, tool_map, missing = select_tools(app.registry, requested)
    if missing:
        await svc.progress(task_id, run_id, {
            "type": "info",
            "content": f"The plan mentions tools that are not available: {', '.join(missing)}. The executor will work around them.",
        })

    checkpoint = run.get("messages") if resume else None
    if isinstance(checkpoint, list) and checkpoint:
        messages = history_to_openai(checkpoint)
        if not is_first_retry_attempt(run):  # a retry's checkpoint already ends with the retry note
            messages.append({"role": "user", "content": RESUME_NOTE})
    else:
        system = await _executor_system_prompt(svc, task, run, plan, tool_map)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": EXECUTOR_KICKOFF}]
        await svc.repo.update_run(run_id, {"messages": messages})

    ctx = app.agent.tool_context(None, "task")
    ctx.extra.update({"task_id": task_id, "run_id": run_id})
    result = LoopResult()
    mapper = ProgressMapper(svc, task_id, run_id)
    rounds = max_rounds
    for attempt in range(MAX_CONTINUE_NUDGES + 1):
        async for event in app.agent.run_loop(
            messages,
            ctx,
            result=result,
            role="executor",
            model=task.get("model") or None,
            tool_names=tool_names,
            max_rounds=rounds,
            use_approvals=False,  # v2: approving the plan is the approval
            source="task",
        ):
            await mapper.handle(event, messages)
        await mapper.flush_thought()
        announced = (result.text or "").strip()
        if attempt >= MAX_CONTINUE_NUDGES or result.error or result.hit_step_limit or not announces_unfinished_work(announced):
            break
        # small local models sometimes stop after announcing the next step: ask once more to actually do it
        messages.append({"role": "assistant", "content": announced})
        messages.append({"role": "user", "content": CONTINUE_NUDGE})
        await svc.progress(task_id, run_id, {
            "type": "info",
            "content": "The executor described its next step without doing it, so I asked it to carry on.",
        })
        rounds = max(4, max_rounds // 2)
    await svc.repo.update_run(run_id, {"messages": messages})

    if result.stopped_by_rule:  # an "ask" rule stopped the run; say which and how to change it (ADR 0016)
        raise RunFailed(result.stopped_by_rule)
    if result.error:
        raise RunFailed(f"Executor agent failed: {result.error}")
    final = (result.text or "").strip()
    if not final:
        raise RunFailed(
            "Agent finished execution without providing a final answer as required by its instructions. "
            "The task may be incomplete."
        )
    if final == STEP_LIMIT_TEXT:
        raise RunFailed(f"The executor used all {max_rounds} tool rounds (tasks.max_tool_rounds) before finishing.")
    await svc.progress(task_id, run_id, {"type": "final_answer", "content": final})
    return result


# ---------------------------------------------------------------------------- results
def _link_list(value: Any) -> list[dict]:
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str) and item.strip():
            out.append({"url": item.strip(), "description": ""})
        elif isinstance(item, dict) and item.get("url"):
            out.append({"url": str(item["url"]), "description": str(item.get("description") or "")})
    return out


def _file_list(value: Any) -> list[dict]:
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str) and item.strip():
            out.append({"filename": item.strip(), "description": ""})
        elif isinstance(item, dict) and (item.get("filename") or item.get("name")):
            out.append({
                "filename": str(item.get("filename") or item.get("name")),
                "description": str(item.get("description") or ""),
            })
    return out


def normalize_result(data: Any, fallback_summary: str) -> dict:
    data = data if isinstance(data, dict) else {}
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = fallback_summary
    return {
        "summary": summary,
        "links_created": _link_list(data.get("links_created")),
        "links_found": _link_list(data.get("links_found")),
        "files_created": _file_list(data.get("files_created")),
        "tools_used": [str(t) for t in data.get("tools_used") or [] if isinstance(t, str)],
    }


def _written_files(messages: list[dict]) -> list[str]:
    names: list[str] = []
    for m in messages or []:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") != "file_write":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(args, dict) and args.get("name") and args["name"] not in names:
                names.append(str(args["name"]))
    return names


async def generate_result(
    svc: TaskService,
    task: dict,
    run_id: str,
    *,
    loop_result: LoopResult | None = None,
    aggregated: list[Any] | None = None,
) -> dict:
    """Port of v2 generate_task_result: the ``fast`` role writes the structured report."""
    app = svc.app
    run = await svc.repo.get_run(run_id) or {}
    updates = await svc.repo.events(run_id)
    execution_log: list[dict] = []
    final_answer = ""
    for u in updates:
        msg = dict(u.get("message") or {})
        if msg.get("type") == "thought":
            continue
        if msg.get("type") == "final_answer":
            final_answer = str(msg.get("content") or "")
        for key in ("content", "result"):
            if key in msg:
                msg[key] = truncate(msg[key], 1500)
        execution_log.append({"timestamp": u.get("timestamp"), **msg})
    context = {
        "goal": task.get("name") if not task.get("description") else f"{task.get('name')}: {task.get('description')}",
        "plan": run.get("plan") or task.get("plan") or [],
        "execution_log": execution_log[-80:],
        "aggregated_results": truncate(aggregated, 12000) if aggregated is not None else None,
    }
    data: Any = None
    try:
        data = await complete_json_object(app.llm,
            "fast",
            [
                {"role": "system", "content": RESULT_GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
            ],
            keys=("summary",),
        )
    except Exception as exc:
        log.warning("result generator failed for task %s: %s", task.get("id"), exc)
    if aggregated is not None:
        failed = sum(1 for r in aggregated if isinstance(r, dict) and "error" in r)
        fallback = f"{len(aggregated)} agents finished" + (f", {failed} with errors." if failed else ".")
    else:
        fallback = final_answer or "The task finished."
    result = normalize_result(data, fallback)

    # ground truth from the transcript beats what a small model remembers
    plugin_ids = {p.id for p in app.registry.plugins()}
    tools_used = [t for t in result["tools_used"] if t in plugin_ids]
    for pid in (loop_result.tools_used if loop_result else []):
        if pid not in tools_used:
            tools_used.append(pid)
    result["tools_used"] = tools_used
    known = {f["filename"] for f in result["files_created"]}
    for name in _written_files(loop_result.messages if loop_result else run.get("messages") or []):
        if name not in known:
            result["files_created"].append({"filename": name, "description": ""})
    return result
