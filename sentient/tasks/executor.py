"""Execution of a single task run and structured result generation.

Uses ``Agent.run_loop`` (the shared tool-calling engine) with the ``executor`` role
and maps its typed events to v2 ProgressUpdates. The transcript is checkpointed to
``task_runs.messages`` after every tool result so a run can resume after a restart.
A run that calls ``ask_user`` stops after that round and raises ``RunPaused``; the
service parks it as ``waiting_for_user`` until the answer arrives (``tasks/ask.py``).
A run that read outside content (a tool result, or the event that started it) and then tries
to send something pauses the same way and asks first (ADR 0018).
A run that gets stuck (``tasks/stuck.py``) pauses the same way with a plain reason.
The memories a run had in mind (facts in its prompt, facts memory tools returned that the model read) are kept on
the run as ``memory_sources``, merged across pauses, resumes and restarts (``sentient/memory/sources.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from sentient.agent.loop import Budget, LoopResult, history_to_openai
from sentient.llm.events import TextDelta, ThinkingDelta, ToolCallEvent, ToolResultEvent
from sentient.llm.provider import ToolCall
from sentient.memory.sources import MemorySources
from sentient.tasks import ask, limits, stuck
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
from sentient.tools.rules import never_message, untrusted_in

if TYPE_CHECKING:  # pragma: no cover
    from sentient.tasks.service import TaskService
    from sentient.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_PROGRESS_RESULT_CHARS = 6000
# A run's time limit is checked before each model call. A tool or model call still running this long after the
# limit is cancelled (a hung call must not run forever) and the run asks whether to keep going, as at the limit.
HARD_DEADLINE_GRACE_S = 120.0
# While a run works, its ``last_activity_at`` is saved (and ``task.run_activity`` published) at most this often.
HEARTBEAT_S = 10.0
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


def run_budget(config: Any) -> Budget:
    """The token and cost limits of one swarm run (``tasks.max_tokens_per_run``, ``tasks.max_cost_per_run_usd``)."""
    return Budget(max_tokens=config.tasks.max_tokens_per_run, max_cost_usd=config.tasks.max_cost_per_run_usd)


class RunPaused(Exception):
    """The run asked the user a question (``ask_user``) and waits for the answer."""

    def __init__(self, question: dict):
        super().__init__(question.get("question") or "")
        self.question = question


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
) -> tuple[str, list[dict]]:
    """The executor's system prompt and the recalled facts put into it."""
    app = svc.app
    cfg = app.config.assistant
    tz = get_tz(svc.tz_name())
    facts: list[dict] = []
    if app.memory is not None:
        query = " ".join(
            str(x) for x in (task.get("name"), task.get("description"), (run.get("trigger_data") or {}).get("subject")) if x
        )[:500]
        try:
            facts = [f for f in await app.memory.recall(query) if f.get("content")]
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
        memories=[f["content"] for f in facts],
        tool_map=tool_map,
    ), facts


def _trigger_source(app: Any, task: dict, run: dict) -> str:
    """The app whose event started this run ("Gmail"), or "" for runs nobody outside started (ADR 0018)."""
    if not run.get("trigger_data"):
        return ""
    source = str((task.get("schedule") or {}).get("source") or "")
    plugin = app.registry.plugin(source) if source else None
    return getattr(plugin, "display_name", None) or source or "the event that started it"


async def _run_approved_call(svc: TaskService, task_id: str, run_id: str, ctx: Any, checkpoint: list[dict]) -> list[dict]:
    """Run the held call the user said yes to (ADR 0018), once: the mark is saved away before the call starts, so a
    restart in the middle never repeats it, and the saved placeholder then says the outcome is unknown
    (``ask.INTERRUPTED_NOTE``). The call's result replaces that placeholder."""
    taken = ask.take_approved(checkpoint)
    if taken is None:
        return checkpoint
    messages, call = taken
    await svc.repo.update_run(run_id, {"messages": messages})
    if call is None:
        return messages
    await svc.progress(task_id, run_id, {"type": "tool_call", "tool_name": call["name"], "parameters": call["arguments"]})
    res, is_error, content = await svc.app.agent.run_tool(ToolCall(**call), ctx)
    await svc.progress(
        task_id, run_id, {"type": "tool_result", "tool_name": call["name"], "result": truncate(res), "is_error": is_error}
    )
    messages = ask.fill_result(messages, call["id"], content)
    await svc.repo.update_run(run_id, {"messages": messages})
    return messages


def _spent(state: dict, budget: Budget, started: float) -> dict:
    """Add this segment's steps, tokens, cost and active seconds to the run's stored limits."""
    used = state["used"]
    used.update(
        steps=budget.steps, tokens=budget.tokens, cost_usd=budget.cost_usd,
        seconds=used["seconds"] + (time.monotonic() - started),
    )
    return state


async def execute_single(
    svc: TaskService, task: dict, run: dict, *, resume: bool = False, answered: bool = False
) -> LoopResult:
    """Run (or continue) one task run. ``answered``: the checkpoint already holds the user's answer to
    ``ask_user`` (or to a limit question), so it continues without the restart note. Raises ``RunPaused``
    when the run asks a question or reaches one of its limits (``tasks/limits.py``)."""
    app = svc.app
    assert app.agent is not None
    task_id, run_id = task["id"], run["id"]
    state = limits.load(run, app.config)
    used, limit = state["used"], state["max"]
    started = time.monotonic()
    remaining_s = max(limit["seconds"] - used["seconds"], 0)
    # One budget for the whole run (continue nudges included), carried across pauses in ``state``. Its deadline is
    # checked before each model call, so a call in flight may finish; the hard deadline below catches a hung one.
    budget = Budget(
        max_tokens=int(limit["tokens"]), max_cost_usd=float(limit["cost_usd"]),
        tokens=int(used["tokens"]), cost_usd=float(used["cost_usd"]), steps=int(used["steps"]),
        deadline=started + remaining_s,
    )
    result = LoopResult()
    messages: list[dict] = []
    asking: dict[str, Any] = {}
    sources = MemorySources(run.get("memory_sources"))  # what the run had in mind so far (before a pause or restart)
    # active time only (waiting for an answer is not counted), plus a grace so a slow call can finish first
    hard = asyncio.timeout(remaining_s + HARD_DEADLINE_GRACE_S)
    cfg = app.config.tasks
    watch = stuck.Watch(
        app.registry, stall_s=float(cfg.stuck_after_minutes) * 60, error_limit=int(cfg.stuck_after_repeated_errors)
    )
    # no activity for stall_s cancels the call in flight and the run pauses as stuck; every streamed event resets it
    stall = asyncio.timeout(watch.stall_s or None)
    clock = asyncio.get_running_loop()
    beat = {"at": time.monotonic()}

    async def alive(event: Any) -> None:
        watch.see(event)
        if watch.stall_s:
            stall.reschedule(clock.time() + watch.stall_s)
        if time.monotonic() - beat["at"] >= HEARTBEAT_S:
            beat["at"] = time.monotonic()
            await svc.heartbeat(task_id, run_id)

    try:
        async with hard, stall:
            plan = run.get("plan") or task.get("plan") or []
            requested = [str(s.get("tool", "")) for s in plan if isinstance(s, dict)]
            tool_names, tool_map, missing = select_tools(app.registry, requested)
            if missing:
                await svc.progress(task_id, run_id, {
                    "type": "info",
                    "content": f"The plan mentions tools that are not available: {', '.join(missing)}. "
                    "The executor will work around them.",
                })

            ctx = app.agent.tool_context(None, "task")
            ctx.extra.update({"task_id": task_id, "run_id": run_id, ask.STATE_KEY: asking})
            if task.get("browser_profile"):  # the browser tools use the task's profile (docs/API.md section 12)
                ctx.extra["browser_profile"] = task["browser_profile"]
            checkpoint = run.get("messages") if resume else None
            if isinstance(checkpoint, list) and checkpoint:
                checkpoint = await _run_approved_call(svc, task_id, run_id, ctx, checkpoint)
                messages = history_to_openai(checkpoint)
                if not answered and not is_first_retry_attempt(run):  # a retry's checkpoint already has its note
                    messages.append({"role": "user", "content": RESUME_NOTE})
            else:
                system, facts = await _executor_system_prompt(svc, task, run, plan, tool_map)
                sources.add_facts(facts)
                messages = [{"role": "system", "content": system}, {"role": "user", "content": EXECUTOR_KICKOFF}]
                await svc.repo.update_run(run_id, {
                    "messages": messages, "memory_sources": await sources.resolve(app.store),
                })

            asking["asked"] = ask.count_questions(messages)
            # outside content in play: the event that started the run, or a tool result earlier in it (ADR 0018)
            ctx.untrusted = _trigger_source(app, task, run) or untrusted_in(messages, app.registry)
            if app.registry.get(ask.ASK_TOOL) is not None:
                tool_names = [*tool_names, ask.ASK_TOOL]
            mapper = ProgressMapper(svc, task_id, run_id)
            for attempt in range(MAX_CONTINUE_NUDGES + 1):
                rounds = int(limit["steps"]) - budget.steps
                if rounds <= 0:  # every step of this run is used
                    result.hit_step_limit = True
                    break
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
                    stop=lambda: (
                        bool(asking.get("question")) or result.needs_ok is not None or watch.reason is not None
                    ),
                    budget=budget,
                ):
                    await alive(event)
                    await mapper.handle(event, messages)
                    if isinstance(event, ToolResultEvent) and not event.is_error:
                        seen = len(sources)
                        sources.add_tool_result(event.name, app.agent.delivered_rows(event.result))
                        if len(sources) > seen:  # saved at once, so a restart keeps it
                            await svc.repo.update_run(run_id, {"memory_sources": await sources.resolve(app.store)})
                await mapper.flush_thought()
                if result.paused or result.stopped_by_budget:
                    break
                announced = (result.text or "").strip()
                if (
                    attempt >= MAX_CONTINUE_NUDGES or result.error or result.hit_step_limit
                    or not announces_unfinished_work(announced)
                ):
                    break
                # small local models sometimes stop after announcing the next step: ask once more to actually do it
                messages.append({"role": "assistant", "content": announced})
                messages.append({"role": "user", "content": CONTINUE_NUDGE})
                await svc.progress(task_id, run_id, {
                    "type": "info",
                    "content": "The executor described its next step without doing it, so I asked it to carry on.",
                })
    except TimeoutError:
        if stall.expired():
            watch.stalled()
            log.warning("task run %s made no progress for %ss; the call in flight was cancelled", run_id, watch.stall_s)
        elif not hard.expired():
            raise
        else:
            log.warning("task run %s passed its hard time limit; the call in flight was cancelled", run_id)
        # The transcript keeps every finished step; a tool call left without its result is dropped when the run
        # resumes (history_to_openai), so the model simply makes it again.
    except asyncio.CancelledError:
        # shutdown or Cancel: keep what this segment used so a resumed run still counts it. The write is shielded
        # and awaited, so it lands before the run's task ends even if it is cancelled again.
        save = asyncio.ensure_future(svc.repo.update_run(run_id, {"limits": _spent(state, budget, started)}))
        try:
            await asyncio.shield(save)
        except asyncio.CancelledError:
            await save
        raise
    _spent(state, budget, started)
    await svc.repo.update_run(run_id, {"limits": state, **({"messages": messages} if messages else {})})

    if result.stopped_by_rule:  # an "ask" rule stopped the run; say which and how to change it (ADR 0016)
        raise RunFailed(result.stopped_by_rule)
    # the loop breaker: the same call kept getting the same result. A repeated error is stuck; anything else fails.
    if result.stopped_by_repeat and not watch.repeated():
        raise RunFailed(f"{result.stopped_by_repeat} Edit the task to add what it needs, or retry it.")
    # One question at a time, each with its own pending_question keys: ask_user's own question first, then a call
    # held after outside content (its yes runs that exact call), then stuck (seen again later if it still is).
    if result.paused and result.needs_ok is not None and not asking.get("question"):
        raise RunPaused(ask.untrusted_pending(result.needs_ok))  # it read outside content: ask before sending
    if watch.reason and not asking.get("question"):  # stuck: ask what to do (tasks/stuck.py)
        raise RunPaused(stuck.pending(watch.kind or "stalled", watch.reason))
    if result.paused:
        raise RunPaused({
            "question": asking["question"],
            "options": asking.get("options") or [],
            "tool_call_id": asking.get("tool_call_id") or ask.last_call_id(messages),
        })
    reached = (
        "seconds" if hard.expired()
        else budget.over() if result.stopped_by_budget
        else "steps" if result.hit_step_limit
        else None
    )
    if reached:  # a limit: ask whether to keep going (tasks/limits.py)
        raise RunPaused(limits.pending(state, reached))
    if result.error:
        raise RunFailed(f"Executor agent failed: {result.error}")
    final = (result.text or "").strip()
    if not final:
        raise RunFailed(
            "Agent finished execution without providing a final answer as required by its instructions. "
            "The task may be incomplete."
        )
    await svc.progress(task_id, run_id, {"type": "final_answer", "content": final})
    return result


def fixed_call_of(task: dict) -> dict | None:
    """``original_context.fixed_call`` of a task the user approved as one exact tool call, else None."""
    call = (task.get("original_context") or {}).get("fixed_call")
    return call if isinstance(call, dict) and call.get("tool") else None


async def execute_fixed_call(svc: TaskService, task: dict, run: dict, *, resume: bool = False) -> LoopResult:
    """Run the one tool call the user approved, with exactly the stored arguments: no planner, no executor model,
    so nothing can change them. Lasting "never" rules still apply (ADR 0016): the run fails with the rule's
    message instead of calling the tool. A run interrupted mid-call is never repeated on resume."""
    app = svc.app
    call = fixed_call_of(task) or {}
    name = str(call.get("tool") or "")
    arguments = dict(call.get("arguments") or {})
    task_id, run_id = task["id"], run["id"]
    if resume and any((e.get("message") or {}).get("type") == "tool_call" for e in await svc.repo.events(run_id)):
        raise RunFailed("Sentient restarted while doing this, so it did not do it again. Check whether it went through.")
    tool = app.registry.get(name)
    if tool is None:
        raise RunFailed(f"The tool {name} is not available, so nothing was done.")
    if app.approvals.rule(tool) == "never":
        raise RunFailed(never_message(app.approvals.label(tool, app.registry)))
    await svc.progress(task_id, run_id, {"type": "tool_call", "tool_name": name, "parameters": arguments})
    ctx = app.agent.tool_context(None, "task") if app.agent is not None else None
    if ctx is not None:
        ctx.extra.update({"task_id": task_id, "run_id": run_id})
        if task.get("browser_profile"):
            ctx.extra["browser_profile"] = task["browser_profile"]
    try:
        result = await tool.call(ctx, arguments)
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    is_error = isinstance(result, dict) and bool(result.get("error"))
    await svc.progress(
        task_id, run_id, {"type": "tool_result", "tool_name": name, "result": truncate(result), "is_error": is_error}
    )
    if is_error:
        raise RunFailed(str(result["error"]))
    final = str(call.get("done_text") or "Done.")
    await svc.progress(task_id, run_id, {"type": "final_answer", "content": final})
    return LoopResult(text=final, tool_calls=1, tools_used=[tool.plugin])


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
    tools_used = [t for t in result["tools_used"] if t in plugin_ids and t != ask.PLUGIN_ID]
    for pid in (loop_result.tools_used if loop_result else []):
        if pid not in tools_used and pid != ask.PLUGIN_ID:
            tools_used.append(pid)
    result["tools_used"] = tools_used
    known = {f["filename"] for f in result["files_created"]}
    for name in _written_files(loop_result.messages if loop_result else run.get("messages") or []):
        if name not in known:
            result["files_created"].append({"filename": name, "description": ""})
    return result
