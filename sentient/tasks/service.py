"""Long-running tasks: refine -> plan -> approve -> schedule -> execute -> result.

Types: single (one-off / scheduled once), recurring, triggered, swarm. The v2 state
machine, prompts and field names are preserved; Celery/Redis/Mongo are replaced by
asyncio tasks and one SQLite file.

Status flow (single tasks)::

    planning -> [clarification_pending -> planning] -> approval_pending
      approve: recurring -> active, triggered -> active,
               once in the future -> pending, otherwise -> processing
    active|pending --scheduler/trigger/run-now--> processing
    processing -> recurring/triggered: active (next run computed)
               -> once: completed | error | cancelled
    processing --ask_user--> waiting_for_user --answer--> processing   (the run pauses; survives restarts)
    processing --stuck--> waiting_for_user   (no progress, the same error again and again, or a step only the user
               can do: Try again / Skip this step / Cancel, tasks/stuck.py)
    active|pending, missed while the computer was off or asleep --> run once now or skipped (tasks/catchup.py)
    decline -> declined, archive -> archived

Swarm tasks skip approval (v2): planning -> processing -> completed | completed_with_errors | error.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

from sentient.llm.jobs import detached
from sentient.llm.provider import ModelRefused, ProviderError
from sentient.services import Service, cancel_tasks
from sentient.tasks import ask, catchup, executor, limits, scripts, stuck, swarm
from sentient.tasks.delivery import from_stored, stored
from sentient.tasks.executor import RunFailed, RunPaused
from sentient.tasks.jsonio import complete_json_object
from sentient.tasks.prompts import (
    EXCLUDED_PLUGINS,
    PLANNER_SYSTEM_PROMPT,
    RETRY_NOTE,
    TASK_CREATION_PROMPT,
    build_tool_catalog,
)
from sentient.tasks.repo import TaskRepo
from sentient.tasks.schedule import (
    calculate_next_run,
    get_tz,
    iso,
    normalize_schedule,
    parse_iso,
    parse_run_at,
    user_timezone_name,
)
from sentient.tasks.scripts import ScriptInvalid, normalize_script, validate_code
from sentient.tasks.triggers import event_matches_filter
from sentient.tools.base import Risk

log = logging.getLogger(__name__)

STATUSES = {
    "planning", "clarification_pending", "approval_pending", "pending", "active", "processing", "waiting_for_user",
    "completed", "completed_with_errors", "error", "declined", "cancelled", "archived",
}
UPDATABLE_FIELDS = {
    "name", "description", "priority", "schedule", "plan", "enabled", "status", "model", "assignee", "script",
    "browser_profile", "deliver_to",
}
SANDBOX_RESULT = {
    "ok": False, "backend": None, "stdout": "", "stderr": "", "result": None, "files_created": [],
    "tool_calls": 0, "duration_ms": 0, "error": None,
}
PROVIDER_DOWN = "Sorry, the AI model is unavailable right now. Check Settings > Models and try again."
MAX_RESUMES = 2
BUSY = ("processing", "waiting_for_user")
STOPPED_NOTE = "Run stopped by Stop everything."
WAITING_CONFLICT = "This task is waiting for your answer. Answer the question or cancel the run first."
_KEEP: Any = object()


def _provider_down(exc: ProviderError, *, detail: bool = False) -> str:
    """What a task shows when its model failed: the reason itself when a model refused the job (it says what to
    change), else the general sentence, with the error when ``detail``."""
    if isinstance(exc, ModelRefused):
        return str(exc)
    return f"{PROVIDER_DOWN} ({exc})" if detail else PROVIDER_DOWN


class TaskNotFound(LookupError):
    pass


class TaskConflict(ValueError):
    """The request is valid but not allowed in the task's current state."""


# ---------------------------------------------------------------------------- helpers
def _title(prompt: str) -> str:
    text = " ".join(prompt.split())
    return text if len(text) <= 120 else text[:117] + "..."


def _priority(value: Any) -> int:
    try:
        return min(2, max(0, int(value)))
    except (TypeError, ValueError):
        return 1


def normalize_plan(steps: Any, registry: Any) -> list[dict]:
    """``[{tool, description}]`` with ``tool`` mapped to a plugin id where possible."""
    if isinstance(steps, dict):
        steps = steps.get("steps") or []
    if not isinstance(steps, list):
        return []
    plugins = {p.id for p in registry.plugins()}
    by_tool = {t.name: t.plugin for t in registry.tools()}
    out: list[dict] = []
    for step in steps:
        if isinstance(step, str):
            tool, desc = "", step.strip()
        elif isinstance(step, dict):
            tool = str(step.get("tool") or step.get("service") or "").strip()
            desc = str(step.get("description") or step.get("step") or "").strip()
        else:
            continue
        if not tool and not desc:
            continue
        key = tool if tool in plugins else tool.lower()
        if key not in plugins:
            key = by_tool.get(tool) or by_tool.get(tool.lower()) or key
        out.append({"tool": key, "description": desc})
    return out


def normalize_questions(raw: Any, start: int = 0) -> list[dict]:
    out: list[dict] = []
    n = start
    for q in raw if isinstance(raw, list) else []:
        text = q if isinstance(q, str) else (q.get("text") or q.get("question")) if isinstance(q, dict) else None
        if not isinstance(text, str) or not text.strip():
            continue
        n += 1
        out.append({"question_id": f"q{n}", "text": text.strip(), "answer": None})
    return out[:5]


class TaskService(Service):
    name = "tasks"
    model_kind = "task"

    def __init__(self, app: Any):
        super().__init__(app)
        self.repo = TaskRepo(app.store)
        # Injectable clock (tests freeze time). Must return an aware datetime.
        self.clock: Callable[[], datetime] = lambda: datetime.now(UTC)
        self._background: set[asyncio.Task] = set()
        self._runs: dict[str, asyncio.Task] = {}
        self._cancel_requested: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        self._sem: asyncio.Semaphore | None = None
        self._bus_subscription: Any = None
        self._items: asyncio.Queue[dict] = asyncio.Queue()
        self._last_tick: datetime | None = None  # wall clock of the last scheduler tick (a big jump means sleep)
        self._catch_up_reason: str | None = "start"  # what the next catch-up notice says it caught up after

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        cfg = self.app.config.tasks
        self._sem = asyncio.Semaphore(cfg.max_concurrent_runs)
        await self.repo.ensure_schema()
        from sentient.tasks.tools import TasksPlugin

        if not any(p.id == TasksPlugin.id for p in self.app.registry.plugins()):
            self.app.registry.register(TasksPlugin())
        if not any(p.id == ask.PLUGIN_ID for p in self.app.registry.plugins()):
            self.app.registry.register(ask.TaskQuestionsPlugin())
        # source.items is event-driven (feeds, polls, webhooks), so it is consumed even without timers.
        # Subscribe synchronously here so an event published right after start() is never missed.
        self._bus_subscription = self.app.bus.subscribe()
        bus_queue = await self._bus_subscription.__aenter__()
        self._loops.append(asyncio.create_task(self._watch_bus(bus_queue), name="tasks:bus"))
        self._loops.append(asyncio.create_task(self._consume_items(), name="tasks:source-items"))
        if self.app.enable_background:
            if not self.app.stopped:  # stopped: interrupted work is picked up on resume (app.resume)
                await self.recover_interrupted()
            self.run_every(cfg.tick_seconds, self._tick_job, name="scheduler", initial_delay=min(5, cfg.tick_seconds))

    async def stop(self) -> None:
        await super().stop()
        if self._bus_subscription is not None:
            await self._bus_subscription.__aexit__(None, None, None)
            self._bus_subscription = None
        pending = [t for t in (*self._runs.values(), *self._background) if not t.done()]
        for t in pending:
            t.cancel()  # runs stay 'processing' with their checkpoint and resume on next start
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._runs.clear()
        self._background.clear()

    async def drain(self, timeout: float = 60) -> None:
        """Wait for background planning and runs (tests, graceful shutdown)."""
        async with asyncio.timeout(timeout):
            while True:
                pending = [t for t in (*self._background, *self._runs.values()) if not t.done()]
                if not pending:
                    return
                await asyncio.gather(*pending, return_exceptions=True)

    async def halt(self) -> int:
        """Stop everything: cancel running runs (waiting questions keep waiting), planning and checks.

        Cancelled runs can be retried from where they stopped; tasks left planning are planned again on resume."""
        self._catch_up_reason = "resume"
        runs = [t for t in self._runs.values() if not t.done()]
        cancelled = 0
        for run in await self.repo.processing_runs():
            try:
                await self.cancel_run(run["task_id"], run["id"], note=STOPPED_NOTE)
                cancelled += 1
            except (TaskNotFound, TaskConflict):
                continue  # finished or cancelled meanwhile
        if runs:
            await asyncio.wait(runs, timeout=5)
        return cancelled + await cancel_tasks(self._background)

    def _job_running(self, kind: str, task_id: str) -> bool:
        """True while a background job (``plan``, ``swarm``, ``script``) runs for this task."""
        name = f"tasks:{kind}:{task_id}"
        return any(t.get_name() == name and not t.done() for t in self._background)

    # ------------------------------------------------------------------ small helpers
    def now(self) -> datetime:
        dt = self.clock()
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)

    def now_iso(self) -> str:
        return iso(self.now()) or ""

    def tz_name(self) -> str:
        return user_timezone_name(self.app.config.assistant.timezone)

    def _local_now_str(self, tz_name: str) -> str:
        return self.now().astimezone(get_tz(tz_name)).strftime("%Y-%m-%d %H:%M:%S %Z (%A)")

    def _spawn(self, coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task:
        # planning and runs are task work even when a chat or a button started them (#149)
        t = asyncio.create_task(coro, name=f"tasks:{name}", context=detached("task"))
        self._background.add(t)
        t.add_done_callback(self._background.discard)
        return t

    def _lock(self, task_id: str) -> asyncio.Lock:
        return self._locks.setdefault(task_id, asyncio.Lock())

    async def _require(self, task_id: str) -> dict:
        task = await self.repo.get_task(task_id)
        if task is None:
            raise TaskNotFound(task_id)
        return task

    async def _set(self, task_id: str, fields: dict) -> None:
        await self.repo.update_task(task_id, {**fields, "updated_at": self.now_iso()})

    async def publish(self, task_id: str) -> dict | None:
        data = await self.repo.serialize(task_id)
        if data is not None:
            self.app.bus.publish("task.updated", data)
        return data

    async def progress(self, task_id: str, run_id: str, message: dict) -> dict:
        clean = {k: v for k, v in message.items() if v is not None}
        update = await self.repo.add_event(run_id, clean, self.now_iso())
        await self.repo.update_run(run_id, {"last_activity_at": update["timestamp"]})
        self.app.bus.publish("task.run_progress", {"task_id": task_id, "run_id": run_id, "update": update})
        return update

    async def heartbeat(self, task_id: str, run_id: str) -> None:
        """A working run is alive (the model is writing): save and publish ``last_activity_at``."""
        now = self.now_iso()
        await self.repo.update_run(run_id, {"last_activity_at": now})
        self.app.bus.publish("task.run_activity", {"task_id": task_id, "run_id": run_id, "last_activity_at": now})

    @staticmethod
    def _runnable(task: dict) -> bool:
        if task.get("task_type") == "script":
            return bool((task.get("script") or {}).get("code"))
        return bool(task.get("plan"))

    async def _notify(self, task: dict, message: str, title: str, event: str, extra: dict | None = None) -> None:
        try:
            payload = {"task_id": task["id"], "event": event, **(extra or {})}
            await self.app.notify("task", message, title=title, payload=payload)
        except Exception:
            log.exception("task notification failed")

    # ------------------------------------------------------------------ public API
    async def create_task(
        self,
        prompt: str,
        *,
        is_swarm: bool = False,
        source: str = "user",
        original_context: dict | None = None,
        model: str | None = None,
        auto_approve: bool = False,
        browser_profile: str | None = None,
    ) -> dict:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("A prompt is required.")
        browser_profile = self._browser_profile(browser_profile)
        context = dict(original_context or {})
        context.setdefault("source", "manual_creation" if source == "user" else source)
        now = self.now_iso()
        fields: dict[str, Any] = {
            "name": _title(prompt),
            "description": prompt,
            "status": "planning",
            "priority": 1,
            "assignee": "ai",
            "original_prompt": prompt,
            "source": source,
            "enabled": True,
            "model": model or None,
            "browser_profile": browser_profile,
            "original_context": context,
            "plan": [],
            "chat_history": [],
            "clarifying_questions": [],
            "created_at": now,
            "updated_at": now,
        }
        if is_swarm:
            fields.update(
                task_type="swarm",
                description=f"Swarm task to achieve the goal: {prompt}",
                swarm_details=swarm.empty_swarm_details(prompt),
            )
        else:
            fields.update(task_type="single", schedule=None)
        task_id = await self.repo.insert_task(fields)
        data = await self.publish(task_id)
        if is_swarm:
            self._spawn(self._orchestrate_swarm(task_id), f"swarm:{task_id}")
        else:
            self._spawn(self._refine_and_plan(task_id, auto_approve=auto_approve), f"plan:{task_id}")
        assert data is not None
        return data

    async def create_approved_call(
        self,
        prompt: str,
        tool: str,
        arguments: dict,
        *,
        step: str,
        description: str | None = None,
        source: str = "user",
        original_context: dict | None = None,
        done_text: str = "Done.",
        schedule: dict | None = None,
        quiet: bool = False,
    ) -> dict:
        """A task the user has already approved as one exact tool call (a follow-up's "Send reply").

        No planner and no executor model: the run calls ``tool`` with exactly ``arguments`` right away, whatever
        ``tasks.require_plan_approval`` says, because the user approved this exact call. Lasting "never" rules
        still stop it (the run fails with the rule's message). With a recurring ``schedule`` the task is active
        and runs at its times instead of now (the Daily Brief). ``quiet`` skips the "Task completed" notification
        for a tool that sends its own."""
        prompt = (prompt or "").strip()
        t = self.app.registry.get(tool)
        if not prompt or t is None:
            raise ValueError("A prompt and a known tool are required.")
        context = dict(original_context or {})
        context.setdefault("source", source)
        context["fixed_call"] = {"tool": tool, "arguments": dict(arguments), "done_text": done_text}
        if quiet:
            context["fixed_call"]["quiet"] = True
        recurring = normalize_schedule(schedule, self.tz_name(), override_timezone=False) if schedule else None
        if recurring is not None and recurring["type"] != "recurring":
            raise ValueError("Only a recurring schedule can be given here.")
        now = self.now_iso()
        task_id = await self.repo.insert_task({
            "name": _title(prompt),
            "description": (description or prompt).strip(),
            "status": "active" if recurring else "pending",
            "next_execution_at": iso(calculate_next_run(recurring, self.now())) if recurring else None,
            "priority": 1,
            "assignee": "ai",
            "original_prompt": prompt,
            "source": source,
            "enabled": True,
            "model": None,
            "original_context": context,
            "plan": [{"tool": t.plugin, "description": step}],
            "chat_history": [],
            "clarifying_questions": [],
            "task_type": "single",
            "schedule": recurring,
            "created_at": now,
            "updated_at": now,
        })
        if recurring is None:
            await self._start_run(await self._require(task_id))
        data = await self.publish(task_id)
        assert data is not None
        return data

    async def create_imported(
        self, *, name: str, prompt: str, schedule: dict, script: dict | None = None, context: dict | None = None,
        deliver_to: Any = None,
    ) -> dict:
        """A paused task brought over from another assistant (Hermes' scheduled jobs, ``sentient/migrate``).

        It has a schedule but no plan, so it never runs as it is: resuming it plans it and asks for approval like
        any new task (``_start_imported``). ``script`` makes it a script task (code checked, not yet approved)."""
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("A prompt is required.")
        sched = normalize_schedule(schedule, self.tz_name(), override_timezone=False)
        now = self.now_iso()
        fields: dict[str, Any] = {
            "name": (name or "").strip()[:200] or _title(prompt),
            "description": prompt,
            "status": "active" if sched["type"] == "recurring" else "pending",
            "priority": 1,
            "assignee": "ai",
            "original_prompt": prompt,
            "source": "import",
            "enabled": False,
            "model": None,
            "original_context": {**(context or {}), "imported_from": (context or {}).get("imported_from") or "import"},
            "plan": [],
            "chat_history": [],
            "clarifying_questions": [],
            "task_type": "script" if script else "single",
            "script": normalize_script(script) if script else None,
            "deliver_to": stored(deliver_to),
            "schedule": sched,
            "next_execution_at": None,
            "created_at": now,
            "updated_at": now,
        }
        task_id = await self.repo.insert_task(fields)
        data = await self.publish(task_id)
        assert data is not None
        return data

    @staticmethod
    def _unplanned_import(task: dict) -> bool:
        return bool((task.get("original_context") or {}).get("imported_from")) and not task.get("plan")

    async def _start_imported(self, task: dict, changes: dict) -> dict:
        """First Resume of an imported task: a script that only notifies goes straight to approval (its code is the
        plan); anything else is planned now and then waits for approval, like a new task."""
        task_id = task["id"]
        script = task.get("script") if task.get("task_type") == "script" else None
        base = {**changes, "enabled": True, "next_execution_at": None, "error": None}
        if script and script.get("then") == "notify":
            plan = [{"tool": "", "description": scripts.describe_for_approval(script)}]
            if not self.app.config.tasks.require_plan_approval:
                await self._set(task_id, {**base, "plan": plan})
                return await self.approve(task_id)
            await self._set(task_id, {**base, "plan": plan, "status": "approval_pending"})
            await self.publish(task_id)
            await self._notify(
                task, f"Check the script for '{task.get('name')}' and approve it to turn the job on.",
                "Plan ready for approval", "approval_needed",
            )
            return await self.get(task_id)
        await self._set(task_id, {**base, "status": "planning"})
        await self.publish(task_id)
        self._spawn(self._plan_job(task_id), f"plan:{task_id}")
        return await self.get(task_id)

    async def delivery_for(self, task_id: str) -> str | list[dict]:
        """Where this task's notifications go besides the app (``tasks/delivery.py``); the default when unknown."""
        task = await self.repo.get_task(task_id)
        return from_stored((task or {}).get("deliver_to"))

    async def preview(self, prompt: str) -> dict:
        """v2 generate-plan: ``{name, description, priority, schedule}`` without creating a task."""
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("A prompt is required.")
        return await self._refine(prompt)

    async def get(self, task_id: str) -> dict:
        data = await self.repo.serialize(task_id)
        if data is None:
            raise TaskNotFound(task_id)
        return data

    async def list(self) -> list[dict]:
        return await self.repo.serialize_many(await self.repo.list_tasks())

    async def search(self, query: str = "", status: str | None = None, limit: int = 20) -> list[dict]:
        q = (query or "").strip().lower()
        rows = [
            t for t in await self.repo.list_tasks()
            if (not status or t["status"] == status)
            and (not q or q in f"{t.get('name') or ''} {t.get('description') or ''}".lower())
        ]
        rows.sort(key=lambda t: t.get("priority", 1))  # stable: newest first within a priority (v2)
        return await self.repo.serialize_many(rows[:limit])

    async def update(self, task_id: str, fields: dict, *, reapprove_script: bool = False) -> dict:
        """PATCH semantics. With ``reapprove_script`` (chat tools) a changed check script goes back to
        ``approval_pending`` when plans need approval, because script code runs without a model."""
        task = await self._require(task_id)
        data = {k: v for k, v in (fields or {}).items() if k in UPDATABLE_FIELDS}
        changes: dict[str, Any] = {}
        needs_approval = False
        if "script" in data:
            if task.get("task_type") == "swarm" and data["script"] is not None:
                raise ValueError("Swarm tasks cannot have a check script.")
            if data["script"] is None:
                if task.get("task_type") == "script":
                    changes.update(task_type="single", script=None)
            else:
                previous = task.get("script") if task.get("task_type") == "script" else None
                script = normalize_script(data["script"], previous)
                changes.update(task_type="script", script=script)
                code_changed = (previous or {}).get("code") != script["code"]
                if reapprove_script and code_changed and self.app.config.tasks.require_plan_approval:
                    if task["status"] in BUSY:
                        raise TaskConflict("This task is running right now. Try again when the run has finished.")
                    if task["status"] not in {"planning", "clarification_pending", "declined", "archived"}:
                        changes.update(status="approval_pending", next_execution_at=None)
                        data.pop("status", None)
                        needs_approval = True
        if "name" in data:
            changes["name"] = str(data["name"] or "").strip() or task["name"]
        if "description" in data:
            changes["description"] = str(data["description"] or "")
        if "priority" in data:
            changes["priority"] = _priority(data["priority"])
        if "model" in data:
            changes["model"] = data["model"] or None
        if "browser_profile" in data:
            changes["browser_profile"] = self._browser_profile(data["browser_profile"])
        if "deliver_to" in data:
            changes["deliver_to"] = stored(data["deliver_to"])
        if "assignee" in data:
            changes["assignee"] = data["assignee"] or "ai"
        if "plan" in data:
            changes["plan"] = normalize_plan(data["plan"], self.app.registry)
        if "enabled" in data:
            changes["enabled"] = bool(data["enabled"])
        if "status" in data:
            if data["status"] not in STATUSES:
                raise ValueError(f"Unknown status '{data['status']}'.")
            changes["status"] = data["status"]
        schedule = task.get("schedule")
        if "schedule" in data:
            schedule = (
                normalize_schedule(data["schedule"], self.tz_name(), override_timezone=False)
                if data["schedule"] else None
            )
            changes["schedule"] = schedule
        status = changes.get("status", task["status"])
        enabled = changes.get("enabled", task["enabled"])
        if enabled and not task["enabled"] and status in {"active", "pending"} and self._unplanned_import(task):
            changes.pop("plan", None)  # an imported task's plan only comes from the planner or its own script
            return await self._start_imported(task, changes)
        reschedule = "schedule" in changes or "status" in changes or (enabled and not task["enabled"])
        if reschedule and status in {"active", "pending"}:
            kind = (schedule or {}).get("type")
            if kind == "recurring":
                changes["status"] = "active"
                changes["next_execution_at"] = iso(calculate_next_run(schedule, self.now()))
            elif kind == "triggered":
                changes["status"] = "active"
                changes["next_execution_at"] = None
            else:
                changes["status"] = "pending"
                changes["next_execution_at"] = iso(parse_run_at(schedule) or self.now())
        await self._set(task_id, changes)
        if needs_approval:
            await self.publish(task_id)
            name = changes.get("name") or task.get("name") or "Untitled task"
            await self._notify(
                task, f"The check script for '{name}' was changed. Review the code and approve it to turn the job back on.",
                "Plan ready for approval", "approval_needed",
            )
        return await self.get(task_id)

    async def delete(self, task_id: str) -> dict:
        await self._require(task_id)
        for run in await self.repo.processing_runs(task_id):
            t = self._runs.get(run["id"])
            if t is not None and not t.done():
                self._cancel_requested.add(run["id"])
                t.cancel()
        await self.repo.delete_task(task_id)
        await self._delete_notifications(task_id)
        self._locks.pop(task_id, None)
        self.app.bus.publish("task.deleted", {"task_id": task_id})
        return {"ok": True}

    async def approve(self, task_id: str) -> dict:
        task = await self._require(task_id)
        if task.get("task_type") == "swarm":
            raise TaskConflict("Swarm tasks start automatically and do not need approval.")
        if task["status"] == "waiting_for_user":
            raise TaskConflict(WAITING_CONFLICT)
        if task["status"] == "processing":
            raise TaskConflict("This task is already running.")
        is_script = task.get("task_type") == "script"
        if not self._runnable(task):
            raise TaskConflict("Task has no check script to approve." if is_script else "Task has no plan to approve.")
        if is_script:
            try:
                validate_code((task.get("script") or {}).get("code"))
            except ScriptInvalid as exc:
                raise TaskConflict(str(exc)) from exc
        schedule = task.get("schedule") or {}
        kind = schedule.get("type")
        now = self.now()
        if kind == "recurring":
            nxt = calculate_next_run(schedule, now)
            if nxt:
                await self._set(task_id, {"status": "active", "enabled": True, "next_execution_at": iso(nxt), "error": None})
            else:
                await self._set(task_id, {"status": "error", "error": "Could not calculate next run time for recurring task."})
        elif kind == "triggered":
            await self._set(task_id, {"status": "active", "enabled": True, "next_execution_at": None, "error": None})
        else:
            run_at = parse_run_at(schedule)
            if run_at and run_at > now:
                await self._set(task_id, {"status": "pending", "next_execution_at": iso(run_at), "error": None})
            elif is_script:
                await self._set(task_id, {"next_execution_at": None})
                await self._start_script(task)
            else:
                await self._start_run(task, next_execution_at=None)
        await self._resolve_plan_notifications(task_id, "approved")
        return await self.get_and_publish(task_id)

    async def decline(self, task_id: str) -> dict:
        await self._require(task_id)
        await self._set(task_id, {"status": "declined"})
        await self._resolve_plan_notifications(task_id, "declined")
        return await self.get_and_publish(task_id)

    def _browser_profile(self, value: Any) -> str | None:
        """A task's browser profile (``None``: the default one); it must be a profile in Settings > Browser."""
        name = str(value or "").strip()
        if not name or name == "default":
            return None
        if name not in self.app.config.browser.profiles:
            raise ValueError(f"There is no browser profile named '{name}'. Add it in Settings > Browser first.")
        return name

    async def rename_browser_profile(self, old: str, new: str) -> None:
        """Keep tasks on a renamed browser profile (called by the browser service)."""
        rows = await self.repo.store.fetchall("SELECT id FROM tasks WHERE browser_profile = ?", (old,))
        await self.repo.store.execute("UPDATE tasks SET browser_profile = ? WHERE browser_profile = ?", (new, old))
        for row in rows:
            await self.publish(row["id"])

    async def archive(self, task_id: str) -> dict:
        await self._require(task_id)
        await self._set(task_id, {"status": "archived"})
        return await self.get_and_publish(task_id)

    async def rerun(self, task_id: str) -> dict:
        """v2: duplicate the task and send the copy back to planning."""
        task = await self._require(task_id)
        now = self.now_iso()
        keep = ("name", "description", "priority", "task_type", "schedule", "original_prompt", "source",
                "assignee", "model", "browser_profile", "deliver_to", "original_context", "chat_history")
        fields = {k: task.get(k) for k in keep}
        fields.update(status="planning", enabled=True, plan=[], clarifying_questions=[], error=None,
                      created_at=now, updated_at=now)
        if task.get("task_type") == "script" and task.get("script"):
            fields["script"] = {**task["script"], **dict.fromkeys(scripts.SCRIPT_STATE_FIELDS)}
        is_swarm = task.get("task_type") == "swarm"
        if is_swarm:
            old = task.get("swarm_details") or {}
            fields["swarm_details"] = swarm.empty_swarm_details(
                old.get("goal") or task.get("original_prompt") or "", old.get("items")
            )
        new_task_id = await self.repo.insert_task(fields)
        data = await self.get_and_publish(new_task_id)
        if is_swarm:
            self._spawn(self._orchestrate_swarm(new_task_id), f"swarm:{new_task_id}")
        else:
            self._spawn(self._plan_job(new_task_id), f"plan:{new_task_id}")
        return data

    async def run_now(self, task_id: str) -> dict:
        task = await self._require(task_id)
        if task["status"] in {"planning", "clarification_pending"}:
            raise TaskConflict("This task is still being planned.")
        if self._unplanned_import(task):
            raise TaskConflict("Resume this task first, so Sentient can plan it and you can approve the plan.")
        kind = (task.get("schedule") or {}).get("type")
        if task.get("task_type") == "swarm":
            if task["status"] == "processing":
                raise TaskConflict("This task is already running.")
            await self._set(task_id, {"status": "planning", "error": None})
            self._spawn(self._orchestrate_swarm(task_id), f"swarm:{task_id}")
            return await self.get_and_publish(task_id)
        if task.get("task_type") == "script":
            if not self._runnable(task):
                raise TaskConflict("Task has no check script to run.")
            await self._start_script(task)
            return await self.get_and_publish(task_id)
        if not task.get("plan"):
            raise TaskConflict("Task has no plan to run.")
        if kind not in {"recurring", "triggered"}:
            if await self.repo.waiting_runs(task_id):
                raise TaskConflict(WAITING_CONFLICT)
            if await self.repo.processing_runs(task_id):
                raise TaskConflict("This task is already running.")
        await self._start_run(task, next_execution_at=_KEEP if kind == "recurring" else None)
        return await self.get_and_publish(task_id)

    async def chat(self, task_id: str, message: str) -> dict:
        """Change request: append to chat_history and replan with the previous plan/result (v2)."""
        message = (message or "").strip()
        if not message:
            raise ValueError("A message is required.")
        task = await self._require(task_id)
        if task["status"] == "waiting_for_user":
            raise TaskConflict(WAITING_CONFLICT)
        if task["status"] == "processing":
            raise TaskConflict("This task is running right now. Cancel the run before requesting changes.")
        if task.get("task_type") == "swarm":
            raise TaskConflict("Change requests are not supported for swarm tasks; re-run the task instead.")
        history = [*(task.get("chat_history") or []), {"role": "user", "content": message, "timestamp": self.now_iso()}]
        await self._set(task_id, {"chat_history": history, "status": "planning", "error": None})
        data = await self.get_and_publish(task_id)
        self._spawn(self._plan_job(task_id), f"plan:{task_id}")
        return data

    async def answer_clarifications(self, task_id: str, answers: list[Any]) -> dict:
        task = await self._require(task_id)
        questions = [dict(q) for q in task.get("clarifying_questions") or []]
        if not questions:
            raise TaskConflict("This task has no clarifying questions.")
        answer_map: dict[str, str] = {}
        for a in answers or []:
            if isinstance(a, dict):
                qid, text = a.get("question_id"), a.get("answer_text", a.get("answer"))
            else:
                qid, text = getattr(a, "question_id", None), getattr(a, "answer_text", None)
            if qid and text is not None and str(text).strip():
                answer_map[str(qid)] = str(text).strip()
        for q in questions:
            if q.get("question_id") in answer_map:
                q["answer"] = answer_map[q["question_id"]]
        resume = task["status"] == "clarification_pending" and all(q.get("answer") for q in questions)
        fields: dict[str, Any] = {"clarifying_questions": questions}
        if resume:
            fields["status"] = "planning"
        await self._set(task_id, fields)
        data = await self.get_and_publish(task_id)
        if resume:
            self._spawn(self._plan_job(task_id), f"plan:{task_id}")
        return data

    async def cancel_run(self, task_id: str, run_id: str, *, note: str = "Run cancelled by user.") -> dict:
        await self._require(task_id)
        run = await self.repo.get_run(run_id)
        if run is None or run["task_id"] != task_id:
            raise TaskNotFound(run_id)
        if run["status"] not in BUSY:
            raise TaskConflict("This run is not in progress.")
        t = self._runs.get(run_id)
        if t is not None and not t.done():
            self._cancel_requested.add(run_id)
            t.cancel()
        if await self.repo.finish_run(run_id, "cancelled", error=None, now=self.now_iso(), from_statuses=BUSY):
            if run["status"] == "waiting_for_user":
                # the question is withdrawn; a retry sees that the user cancelled instead of answering
                call_id = (run.get("pending_question") or {}).get("tool_call_id")
                messages = ask.fill_result(run.get("messages") or [], call_id, ask.cancelled_content())
                await self.repo.update_run(run_id, {"messages": messages})
                await self._resolve_question_notifications(run_id, "cancelled")
            await self.progress(task_id, run_id, {"type": "info", "content": note})
            await self._after_run(task_id, "cancelled", None)
            await self._publish_run_finished(task_id, run_id, "cancelled")
        return await self.get_and_publish(task_id)

    async def retry_run(self, task_id: str, run_id: str) -> dict:
        """Start a new run that continues a failed or cancelled run from its transcript checkpoint.

        Steps whose tool results are already in the transcript are not repeated; the model is told
        what went wrong and tries the failing step again. Without a checkpoint it is a fresh run
        with the same plan and trigger data."""
        task = await self._require(task_id)
        run = await self.repo.get_run(run_id)
        if run is None or run["task_id"] != task_id:
            raise TaskNotFound(run_id)
        if run["status"] not in {"error", "cancelled"}:
            raise TaskConflict("Only a failed or cancelled run can be retried.")
        if task.get("task_type") == "swarm":
            raise TaskConflict("Swarm runs cannot be retried step by step. Use Run now to start the swarm again.")
        if task["status"] in {"planning", "clarification_pending"}:
            raise TaskConflict("This task is still being planned.")
        kind = (task.get("schedule") or {}).get("type")
        if kind not in {"recurring", "triggered"}:
            if await self.repo.waiting_runs(task_id):
                raise TaskConflict(WAITING_CONFLICT)
            if await self.repo.processing_runs(task_id):
                raise TaskConflict("This task is already running.")
        now = self.now_iso()
        new_run_id = await self.repo.insert_run(
            task_id, now=now, plan=run.get("plan") or task.get("plan") or [], trigger_data=run.get("trigger_data")
        )
        checkpoint = run.get("messages") if isinstance(run.get("messages"), list) else []
        fields: dict[str, Any] = {"retry_of": run_id}
        if checkpoint:
            problem = run.get("error") or "the run was cancelled before it finished"
            fields["messages"] = [*checkpoint, {"role": "user", "content": RETRY_NOTE.format(error=problem)}]
            fields["memory_sources"] = run.get("memory_sources")  # the transcript it continues had these in mind
        await self.repo.update_run(new_run_id, fields)
        await self._set(task_id, {"status": "processing", "last_execution_at": now, "error": None})
        self._dispatch(task_id, new_run_id, resume=bool(checkpoint))
        return await self.get_and_publish(task_id)

    async def answer_question(self, task_id: str, run_id: str, answer: str) -> dict:
        """Resume a run that is ``waiting_for_user``: the answer becomes the result of its ``ask_user`` call."""
        text = str(answer or "").strip()[: ask.MAX_ANSWER_CHARS]
        if not text:
            raise ValueError("An answer is required.")
        await self._require(task_id)
        run = await self.repo.get_run(run_id)
        if run is None or run["task_id"] != task_id:
            raise TaskNotFound(run_id)
        if run["status"] != "waiting_for_user":
            raise TaskConflict("This run is not waiting for an answer.")
        pending = run.get("pending_question") or {}
        if pending.get("limit") in limits.KINDS:
            return await self._answer_limit(task_id, run, pending, text)
        if pending.get("stuck"):
            return await self._answer_stuck(task_id, run, pending, text)
        call_id = pending.get("tool_call_id")
        if pending.get("untrusted_call"):  # a held call after outside content (ADR 0018): only a clear yes runs it
            if not ask.approves(text):
                return await self._stop_at_question(task_id, run_id, pending, text)
            messages = ask.approve_call(run.get("messages") or [], call_id)
        else:
            messages = ask.fill_result(run.get("messages") or [], call_id, ask.answer_content(text))
        if not await self.repo.resume_run(run_id, messages):
            raise TaskConflict("This run is not waiting for an answer.")  # answered or cancelled meanwhile
        await self.progress(task_id, run_id, {"type": "info", "content": f"You answered: {text}"})
        await self._set(task_id, {"status": "processing", "error": None})
        await self._resolve_question_notifications(run_id, "answered", answer=text)
        self._dispatch(task_id, run_id, resume=True, answered=True)
        return await self.get_and_publish(task_id)

    async def _answer_limit(self, task_id: str, run: dict, pending: dict, text: str) -> dict:
        """A run waiting at a limit: "Keep going" raises that limit for this run; anything else fails the run."""
        run_id = run["id"]
        if not limits.keeps_going(text):
            return await self._stop_at_question(task_id, run_id, pending, text)
        state = limits.raise_limit(limits.load(run, self.app.config), pending["limit"])
        # one guarded write: a run cancelled meanwhile keeps its limits; the raised one survives a restart
        if not await self.repo.resume_run(run_id, run.get("messages") or [], limits=state):
            raise TaskConflict("This run is not waiting for an answer.")  # answered or cancelled meanwhile
        await self.progress(task_id, run_id, {"type": "info", "content": f"You answered: {text}"})
        await self._set(task_id, {"status": "processing", "error": None})
        await self._resolve_question_notifications(run_id, "answered", answer=text)
        self._dispatch(task_id, run_id, resume=True, answered=True)
        return await self.get_and_publish(task_id)

    async def _stop_at_question(self, task_id: str, run_id: str, pending: dict, text: str) -> dict:
        """The answer ends a waiting run: it fails with the question's ``stop_error``."""
        await self.progress(task_id, run_id, {"type": "info", "content": f"You answered: {text}"})
        await self._resolve_question_notifications(run_id, "answered", answer=text)
        error = str(pending.get("stop_error") or "Stopped at a limit without finishing.")
        await self._finish_run(task_id, run_id, "error", error=error, from_statuses=("waiting_for_user",))
        return await self.get_and_publish(task_id)

    async def _answer_stuck(self, task_id: str, run: dict, pending: dict, text: str) -> dict:
        """A stuck run: "Cancel" cancels it; "Try again", "Skip this step" or the user's own words carry it on."""
        run_id = run["id"]
        if stuck.choice(text) == "cancel":
            await self._resolve_question_notifications(run_id, "answered", answer=text)
            return await self.cancel_run(task_id, run_id, note="Cancelled after getting stuck.")
        reason = str(pending.get("reason") or "something went wrong")
        messages = [*(run.get("messages") or []), {"role": "user", "content": stuck.note(reason, text)}]
        if not await self.repo.resume_run(run_id, messages):
            raise TaskConflict("This run is not waiting for an answer.")  # answered or cancelled meanwhile
        await self.progress(task_id, run_id, {"type": "info", "content": f"You answered: {text}"})
        await self._set(task_id, {"status": "processing", "error": None})
        await self._resolve_question_notifications(run_id, "answered", answer=text)
        self._dispatch(task_id, run_id, resume=True, answered=True)
        return await self.get_and_publish(task_id)

    async def waiting_questions(self) -> list[dict]:
        """Every question a run is waiting on, oldest first: ``[{task_id, task_name, run_id, question, options, asked_at}]``."""
        out: list[dict] = []
        for run in await self.repo.waiting_runs():
            question = run.get("pending_question") or {}
            if not question.get("question"):
                continue
            task = await self.repo.get_task(run["task_id"])
            if task is None:
                continue
            out.append({
                "task_id": task["id"],
                "task_name": task.get("name") or "Untitled task",
                "run_id": run["id"],
                "question": question["question"],
                "options": list(question.get("options") or []),
                "asked_at": question.get("asked_at"),
            })
        return out

    async def run_events(self, task_id: str, run_id: str) -> list[dict]:
        run = await self.repo.get_run(run_id)
        if run is None or run["task_id"] != task_id:
            raise TaskNotFound(run_id)
        return await self.repo.events(run_id)

    async def handle_event(
        self, source: str, event_type: str, event_data: dict, event_id: str | None = None
    ) -> list[str]:
        """Fire matching triggered tasks for an external event. Returns the new run ids.

        Idempotent per (task, item id): a task handles an item at most once, whichever path
        (change feed, poll, webhook, a direct call) delivers it. Script jobs start a check
        instead of a run and add no run id. Nothing starts while Sentient is stopped (Stop everything)."""
        if self.app.stopped:
            return []
        source = (source or "").strip().lower()
        event_data = event_data if isinstance(event_data, dict) else {}
        if event_id is None and event_data.get("id") is not None:
            event_id = str(event_data["id"])
        match_data = event_data
        if source == "webhook" and isinstance(event_data.get("body"), dict):
            match_data = {**event_data["body"], **event_data}  # filters may name body fields directly
        run_ids: list[str] = []
        for task in await self.repo.list_tasks("enabled = 1 AND status IN ('active', 'processing', 'waiting_for_user')"):
            schedule = task.get("schedule") or {}
            if schedule.get("type") != "triggered" or str(schedule.get("source") or "").lower() != source:
                continue
            wanted_event = str(schedule.get("event") or "")
            if (wanted_event and wanted_event != event_type) or not self._runnable(task):
                continue
            if not event_matches_filter(match_data, schedule.get("filter") or {}, source):
                continue
            if event_id and not await self.repo.mark_task_seen(task["id"], source, str(event_id), self.now_iso()):
                continue
            if task.get("task_type") == "script":
                await self._start_script(task, trigger_data=event_data)
            else:
                run_ids.append(await self._start_run(task, trigger_data=event_data, next_execution_at=None))
            await self.publish(task["id"])
        return run_ids

    async def handle_source_items(self, data: Any) -> list[str]:
        """Consumer of the ``source.items`` domain event (docs/API.md section 16), for every origin."""
        if not isinstance(data, dict):
            return []
        source = str(data.get("source") or "")
        event = str(data.get("event") or "")
        run_ids: list[str] = []
        for item in data.get("items") or []:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            try:
                run_ids += await self.handle_event(source, event, item, None if item_id is None else str(item_id))
            except Exception:
                log.exception("triggered tasks failed for %s item %s", source, item_id)
        return run_ids

    async def _watch_bus(self, queue: asyncio.Queue) -> None:
        # never block here: the bus drops events for a full queue, so hand items to the worker
        while True:
            event = await queue.get()
            if isinstance(event, dict) and event.get("type") == "source.items":
                self._items.put_nowait(event.get("data"))

    async def _consume_items(self) -> None:
        while True:
            data = await self._items.get()
            try:
                await self.handle_source_items(data)
            except Exception:
                log.exception("source.items handling failed")
            finally:
                self._items.task_done()

    async def test_script(self, task_id: str, code: str | None = None) -> dict:
        """Run a script job's code (or ``code``) once and return the SandboxResult. Stored state is untouched."""
        task = await self._require(task_id)
        if code is None:
            code = (task.get("script") or {}).get("code") if task.get("task_type") == "script" else None
            if not code:
                raise TaskConflict("This task has no check script.")
        validate_code(code)
        return await self._sandbox_run(code)

    async def disable_tasks_for_plugin(self, plugin_id: str) -> int:
        """Called when an integration is disconnected: disable dependent tasks and notify."""
        pid = (plugin_id or "").strip().lower()
        if not pid:
            return 0
        display = next((p.display_name for p in self.app.registry.plugins() if p.id == pid), plugin_id)
        count = 0
        for task in await self.repo.list_tasks("enabled = 1 AND status NOT IN ('declined', 'archived')"):
            uses = any(
                isinstance(s, dict) and str(s.get("tool") or "").lower() == pid for s in task.get("plan") or []
            )
            schedule = task.get("schedule") or {}
            if schedule.get("type") == "triggered" and str(schedule.get("source") or "").lower() == pid:
                uses = True
            if not uses:
                continue
            await self._set(task["id"], {"enabled": False})
            await self.publish(task["id"])
            await self._notify(
                task,
                f"Task '{task.get('name')}' was disabled because {display} was disconnected. "
                f"Reconnect {display} and turn the task back on to resume it.",
                "Task disabled",
                "disabled",
            )
            count += 1
        return count

    async def get_and_publish(self, task_id: str) -> dict:
        data = await self.publish(task_id)
        if data is None:
            raise TaskNotFound(task_id)
        return data

    # ------------------------------------------------------------------ scheduler
    async def tick(self) -> list[str]:
        """Claim due tasks atomically and start their runs. Returns the new run ids.

        While Sentient is stopped nothing is claimed; due tasks start on the first tick after resume."""
        run_ids: list[str] = []
        if self.app.stopped:
            self._catch_up_reason = "resume"  # missed runs are caught up on the first tick after Resume
            return run_ids
        await self._catch_up()
        for task_id in await self.repo.claim_due(self.now_iso()):
            task = await self.repo.get_task(task_id)
            if task is None:
                continue
            kind = (task.get("schedule") or {}).get("type")
            if task.get("task_type") == "swarm":
                await self._set(task_id, {"status": "planning", "next_execution_at": None})
                self._spawn(self._orchestrate_swarm(task_id), f"swarm:{task_id}")
                await self.publish(task_id)
                continue
            if task.get("task_type") == "script":
                if self._runnable(task):
                    await self._start_script(task)
                else:
                    await self._set(task_id, {"status": "error", "error": "Task has no check script to run.", "next_execution_at": None})
                await self.publish(task_id)
                continue
            if not task.get("plan"):
                await self._set(task_id, {"status": "error", "error": "Task has no plan to run.", "next_execution_at": None})
                await self.publish(task_id)
                continue
            run_id = await self._start_run(task, next_execution_at=_KEEP if kind == "recurring" else None)
            await self.publish(task_id)
            run_ids.append(run_id)
        return run_ids

    async def _tick_job(self) -> None:
        await self.tick()

    async def _catch_up(self) -> dict[str, list[dict]]:
        """Handle scheduled runs missed while the computer was off or asleep (tasks/catchup.py).

        Runs every tick, so it covers startup, waking from sleep (a big jump of the wall clock between ticks) and
        Resume. A missed task either stays due, so this tick's claim starts it once, or moves on without running."""
        cfg = self.app.config.tasks
        now = self.now()
        reason = self._catch_up_reason
        gap = (now - self._last_tick).total_seconds() if self._last_tick is not None else 0.0
        if reason is None and gap > cfg.tick_seconds + catchup.WAKE_GAP_S:
            reason = "sleep"
        self._last_tick, self._catch_up_reason = now, None
        report: dict[str, list[dict]] = {"ran": [], "skipped": []}
        cutoff = now - timedelta(seconds=catchup.grace_seconds(cfg.tick_seconds))
        for task in await self.repo.missed(iso(cutoff) or ""):
            schedule = task.get("schedule") or {}
            due = parse_iso(task.get("next_execution_at")) or now
            action = catchup.decide(schedule, (now - due).total_seconds(), cfg.catch_up_window_hours)
            if action == "quiet":
                continue  # an interval check: this tick's claim runs it once
            # a brief (quiet fixed call) reports for itself and is about its own day: never in the notice, and
            # a missed one from an earlier day is skipped
            silent = bool((executor.fixed_call_of(task) or {}).get("quiet"))
            if silent and action == "run" and catchup.day_over(due, now, get_tz(schedule.get("timezone") or self.tz_name())):
                action = "skip"
            entry = catchup.item(task, task.get("next_execution_at"))
            if action == "run":
                if not silent:
                    report["ran"].append(entry)
                continue
            if schedule.get("type") == "recurring":
                fields: dict[str, Any] = {"next_execution_at": iso(calculate_next_run(schedule, now))}
            else:
                local = due.astimezone(get_tz(schedule.get("timezone") or self.tz_name())).strftime("%b %d, %H:%M")
                fields = {
                    "status": "error", "next_execution_at": None,
                    "error": f"Skipped: it was due {local}, while the computer was off or asleep. "
                    "Choose Run now if you still want it.",
                }
            if await self.repo.update_task_if_status(task["id"], {"active", "pending"}, {**fields, "updated_at": self.now_iso()}):
                if not silent:
                    report["skipped"].append(entry)
                await self.publish(task["id"])
        if report["ran"] or report["skipped"]:
            title, message = catchup.summary(reason or "start", report["ran"], report["skipped"])
            items = [*report["ran"], *report["skipped"]]
            payload: dict[str, Any] = {"event": "caught_up", "reason": reason or "start", **report}
            if len(items) == 1:
                payload["task_id"] = items[0]["task_id"]
            try:
                await self.app.notify("task", message, title=title, payload=payload)
            except Exception:
                log.exception("catch-up notification failed")
        return report

    async def recover_interrupted(self) -> dict:
        """Resume runs left 'processing' by a crash or restart (or fail them), and restart planning."""
        cfg = self.app.config.tasks
        report: dict[str, list[str]] = {"resumed": [], "failed": [], "replanned": []}
        for run in await self.repo.processing_runs():
            if run["id"] in self._runs:
                continue
            task = await self.repo.get_task(run["task_id"])
            if task is None:
                continue
            resumes = int(run.get("resume_count") or 0)
            if cfg.resume_interrupted_runs and task.get("task_type") != "swarm" and resumes < MAX_RESUMES:
                await self.repo.update_run(run["id"], {"resume_count": resumes + 1})
                if task["status"] != "processing":
                    await self._set(task["id"], {"status": "processing"})
                self._dispatch(task["id"], run["id"], resume=True)
                report["resumed"].append(run["id"])
                continue
            error = "Interrupted by restart"
            if await self.repo.finish_run(run["id"], "error", error=error, now=self.now_iso()):
                await self.progress(task["id"], run["id"], {"type": "error", "content": error})
                await self._after_run(task["id"], "error", error)
                await self.publish(task["id"])
                await self._publish_run_finished(task["id"], run["id"], "error")
                await self._notify(
                    task, f"Task '{task.get('name')}' was interrupted by a restart and could not be resumed.",
                    "Task interrupted", "run_failed",
                )
                report["failed"].append(run["id"])
        # tasks left 'processing' with no live run: claimed right before a crash (no run row), or
        # stopped after the run finished but before the task was settled
        for task in await self.repo.list_tasks("status = 'processing'"):
            if self._job_running("script", task["id"]) or await self.repo.processing_runs(task["id"]):
                continue
            last = await self.repo.latest_run(task["id"])
            if last is not None and last["status"] in {"completed", "completed_with_errors", "error", "cancelled"}:
                await self._after_run(task["id"], last["status"], last.get("error"))
                if last["status"] in {"completed", "completed_with_errors"} and not last.get("result"):
                    self._spawn(
                        self._regenerate_result(task["id"], last["id"], notify_status=last["status"]),
                        f"result:{last['id']}",
                    )
            else:
                await self._after_run(task["id"], "error", "Interrupted by restart")
            await self.publish(task["id"])
        await self._resolve_stale_plan_notifications()
        for task in await self.repo.list_tasks("status = 'planning'"):
            if self._job_running("plan", task["id"]) or self._job_running("swarm", task["id"]):
                continue  # being planned right now (resume after Stop everything)
            if task.get("task_type") == "swarm":
                self._spawn(self._orchestrate_swarm(task["id"]), f"swarm:{task['id']}")
            elif task.get("schedule") or task.get("chat_history") or task.get("clarifying_questions"):
                self._spawn(self._plan_job(task["id"]), f"plan:{task['id']}")
            else:
                self._spawn(self._refine_and_plan(task["id"]), f"plan:{task['id']}")
            report["replanned"].append(task["id"])
        return report

    # ------------------------------------------------------------------ planning pipeline
    async def _refine(self, prompt: str) -> dict:
        tz = self.tz_name()
        system = TASK_CREATION_PROMPT.format(
            user_name=self.app.config.assistant.user_name or "User",
            user_timezone=tz,
            current_time=self._local_now_str(tz),
        )
        data = await complete_json_object(self.app.llm,
            "planner", [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            keys=("name", "description", "schedule"),
        )
        if not isinstance(data, dict):
            raise TypeError("The model did not return task details.")
        return {
            "name": str(data.get("name") or "").strip()[:200] or _title(prompt),
            "description": str(data.get("description") or "").strip() or prompt,
            "priority": _priority(data.get("priority", 1)),
            "schedule": normalize_schedule(data.get("schedule"), tz),
        }

    async def _refine_and_plan(self, task_id: str, *, auto_approve: bool = False) -> None:
        task = await self.repo.get_task(task_id)
        if task is None or task["status"] != "planning":
            return
        prompt = task.get("original_prompt") or task.get("description") or task.get("name") or ""
        try:
            details = await self._refine(prompt)
        except Exception as exc:  # v2: proceed with the raw description
            log.warning("could not refine task %s, planning from the raw prompt: %s", task_id, exc)
            details = {"schedule": normalize_schedule(None, self.tz_name())}
        await self._set(task_id, details)
        await self.publish(task_id)
        await self._plan_job(task_id, auto_approve=auto_approve)

    async def _plan_job(self, task_id: str, *, auto_approve: bool = False) -> None:
        try:
            await self._plan(task_id, auto_approve=auto_approve)
        except ProviderError as exc:
            log.warning("planner unavailable for task %s: %s", task_id, exc)
            await self._plan_failed(task_id, _provider_down(exc))
        except Exception as exc:
            log.exception("planning failed for task %s", task_id)
            await self._plan_failed(task_id, f"Planning failed: {exc}")

    async def _plan_failed(self, task_id: str, error: str) -> None:
        task = await self.repo.get_task(task_id)
        if task is None or task["status"] != "planning":
            return  # the user declined/archived/edited it meanwhile
        if not await self.repo.update_task_if_status(
            task_id, {"planning"}, {"status": "error", "error": error, "updated_at": self.now_iso()}
        ):
            return  # changed while we were failing
        await self.publish(task_id)
        await self._notify(task, f"I couldn't create a plan for '{task.get('name')}'. {error}", "Planning failed", "planning_failed")

    async def _plan(self, task_id: str, *, auto_approve: bool = False) -> None:
        task = await self.repo.get_task(task_id)
        if task is None or task["status"] != "planning" or task.get("task_type") == "swarm":
            return
        cfg = self.app.config
        tz = self.tz_name()
        catalog = await build_tool_catalog(self.app)
        system = PLANNER_SYSTEM_PROMPT.format(
            user_name=cfg.assistant.user_name or "User",
            user_location=cfg.assistant.location or "Not specified",
            current_time=self._local_now_str(tz),
            available_tools_json=json.dumps(catalog, indent=2, ensure_ascii=False),
        )
        history = task.get("chat_history") or []
        is_change = any(m.get("role") == "user" for m in history if isinstance(m, dict))
        context = dict(task.get("original_context") or {})
        if is_change:
            latest = await self.repo.latest_run(task_id)
            context["chat_history"] = history
            context["previous_plan"] = task.get("plan") or []
            context["previous_result"] = (latest or {}).get("result")
        questions = task.get("clarifying_questions") or []
        answered = [q for q in questions if q.get("answer")]

        parts = ["Please create a plan for the following action items:\n- " + (task.get("description") or task.get("name") or "")]
        schedule = task.get("schedule") or {}
        if schedule.get("type") in {"recurring", "triggered"} or schedule.get("run_at"):
            parts.append(
                "Schedule (handled automatically by the scheduler; plan a single occurrence):\n"
                + json.dumps(schedule, ensure_ascii=False)
            )
        if set(context) - {"source"}:
            parts.append("Context:\n" + json.dumps(context, indent=2, ensure_ascii=False, default=str)[:12000])
        if answered:
            parts.append("Answers to your clarifying questions:\n" + "\n".join(
                f"- Q: {q.get('text')}\n  A: {q.get('answer')}" for q in answered
            ))
        data = await complete_json_object(self.app.llm,
            "planner",
            [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}],
            keys=("plan", "steps", "clarifying_questions"),
        )
        if not isinstance(data, dict):
            raise TypeError(f"Planner agent returned invalid JSON: {str(data)[:200]}")
        script, data = await self._planned_script(task, data, system, parts)
        raw_plan = data.get("plan") or data.get("steps")
        if raw_plan is None and data.get("tool"):  # a lone step object
            raw_plan = [data]
        plan = normalize_plan(raw_plan, self.app.registry)
        new_questions = normalize_questions(data.get("clarifying_questions"), start=len(questions) if answered else 0)

        current = await self.repo.get_task(task_id)
        if current is None or current["status"] != "planning":
            return  # deleted or changed while the model was thinking
        name = str(data.get("name") or "").strip()[:200] if not is_change else ""
        display_name = name or task.get("name") or "Untitled task"

        if new_questions and (not answered or not plan):
            merged = [*questions, *new_questions] if answered else new_questions
            if not await self.repo.update_task_if_status(
                task_id, {"planning"},
                {"status": "clarification_pending", "clarifying_questions": merged, "updated_at": self.now_iso()},
            ):
                return  # declined, archived or edited while the planner was running
            await self.publish(task_id)
            await self._notify(
                task, f"I need a bit more information to plan '{display_name}'. Please answer the questions in the task.",
                "Clarification needed", "clarification_needed",
            )
            return
        if not plan and script is None:
            raise ValueError(f"Planner agent returned no plan steps: {json.dumps(data, default=str)[:300]}")

        fields: dict[str, Any] = {"plan": plan, "status": "approval_pending", "error": None}
        if script is not None:
            fields.update(task_type="script", script=script)
            if schedule.get("type") not in {"recurring", "triggered"}:
                proposed = data.get("schedule")
                sched = normalize_schedule(proposed, tz) if isinstance(proposed, dict) and proposed else None
                if sched is None or (sched["type"] == "once" and not sched.get("run_at")):
                    # a watch job that would check only once is almost never what was meant
                    sched = normalize_schedule({"type": "recurring", "frequency": "interval", "interval_minutes": 60}, tz)
                fields["schedule"] = sched
        if is_change:
            fields["chat_history"] = [*history, {
                "role": "assistant",
                "content": f"I've updated the plan ({len(plan)} steps) based on your request. Review it and approve to run it.",
                "timestamp": self.now_iso(),
            }]
        else:  # v2 update_task_with_plan: the first plan names the task
            if name:
                fields["name"] = name
            if str(data.get("description") or "").strip():
                fields["description"] = str(data["description"]).strip()
        if not await self.repo.update_task_if_status(task_id, {"planning"}, {**fields, "updated_at": self.now_iso()}):
            return  # declined, archived or edited while the planner was running
        if auto_approve or not cfg.tasks.require_plan_approval:
            await self.approve(task_id)
            return
        await self.publish(task_id)
        short = display_name[:50] + ("..." if len(display_name) > 50 else "")
        await self._notify(task, f"I've created a new plan for you: '{short}'", "Plan ready for approval", "approval_needed")

    async def _planned_script(self, task: dict, data: dict, system: str, parts: list[str]) -> tuple[dict | None, dict]:
        """The check script the planner proposed (validated), else the task's existing one.

        Code that does not compile gets one correction round; if it still does not compile,
        planning fails with a readable message (ScriptInvalid is a ValueError)."""
        previous = task.get("script") if task.get("task_type") == "script" else None
        raw = data.get("script")
        if not isinstance(raw, dict) or raw.get("code") is None:
            return previous, data
        try:
            return normalize_script(raw, previous), data
        except ScriptInvalid as exc:
            log.info("planner script for task %s rejected, asking for a fix: %s", task["id"], exc)
            retry = [*parts, (
                f"Your previous reply included a `script` that cannot be used: {exc}. "
                "Reply again with the complete JSON object and corrected script code."
            )]
            fixed = await complete_json_object(self.app.llm,
                "planner",
                [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(retry)}],
                keys=("plan", "steps", "clarifying_questions"),
            )
        if not isinstance(fixed, dict):
            raise TypeError(f"Planner agent returned invalid JSON: {str(fixed)[:200]}")
        raw = fixed.get("script")
        if not isinstance(raw, dict) or raw.get("code") is None:
            return previous, fixed
        return normalize_script(raw, previous), fixed

    # ------------------------------------------------------------------ swarm
    async def _orchestrate_swarm(self, task_id: str) -> None:
        task = await self.repo.get_task(task_id)
        if task is None or task.get("task_type") != "swarm":
            return
        try:
            items, configs, used_fallback = await swarm.plan_swarm(self, task)
        except ProviderError as exc:
            log.warning("swarm planning unavailable for %s: %s", task_id, exc)
            await self._swarm_failed(task, _provider_down(exc))
            return
        except Exception as exc:
            log.exception("swarm orchestration failed for %s", task_id)
            await self._swarm_failed(task, str(exc))
            return
        now = self.now_iso()
        total = sum(len(c["item_indices"]) for c in configs)
        details = {
            **(task.get("swarm_details") or {}),
            "items": items, "total_agents": total, "completed_agents": 0,
            "progress_updates": [], "aggregated_results": [],
        }
        run_id = await self.repo.insert_run(task_id, now=now, plan=configs)
        await self._set(task_id, {"swarm_details": details, "status": "processing", "last_execution_at": now, "error": None})
        if used_fallback:
            await self.progress(task_id, run_id, {
                "type": "info",
                "content": "The resource manager did not return a usable plan, so every item gets the goal as its instruction.",
            })
        await self.progress(task_id, run_id, {"type": "info", "content": f"Resource manager created a plan for {total} agents."})
        await self.publish(task_id)
        self._dispatch(task_id, run_id)

    async def _swarm_failed(self, task: dict, error: str) -> None:
        await self._set(task["id"], {"status": "error", "error": error})
        await self.publish(task["id"])
        await self._notify(task, f"Swarm task '{task.get('name')}' has finished with status: error.\n\n{error}", "Task failed", "run_failed")

    async def swarm_update(
        self, task_id: str, worker_id: str, status: str, message: str, *, aggregated_results: list | None = None
    ) -> None:
        async with self._lock(task_id):
            task = await self.repo.get_task(task_id)
            if task is None:
                return
            details = dict(task.get("swarm_details") or {})
            details["progress_updates"] = [*(details.get("progress_updates") or []), {
                "worker_id": worker_id, "timestamp": self.now_iso(), "status": status, "message": message,
            }]
            if status in {"completed", "error"}:
                details["completed_agents"] = int(details.get("completed_agents") or 0) + 1
            if aggregated_results is not None:
                details["aggregated_results"] = aggregated_results
            await self._set(task_id, {"swarm_details": details})
        await self.publish(task_id)

    # ------------------------------------------------------------------ execution
    async def _start_run(self, task: dict, *, trigger_data: dict | None = None, next_execution_at: Any = _KEEP) -> str:
        now = self.now_iso()
        run_id = await self.repo.insert_run(task["id"], now=now, plan=task.get("plan") or [], trigger_data=trigger_data)
        fields: dict[str, Any] = {"status": "processing", "last_execution_at": now, "error": None}
        if next_execution_at is not _KEEP:
            fields["next_execution_at"] = next_execution_at
        await self._set(task["id"], fields)
        self._dispatch(task["id"], run_id)
        return run_id

    def _dispatch(self, task_id: str, run_id: str, *, resume: bool = False, answered: bool = False) -> None:
        t = asyncio.create_task(
            self._execute(task_id, run_id, resume=resume, answered=answered), name=f"tasks:run:{run_id}",
            context=detached("task"),
        )
        self._runs[run_id] = t
        t.add_done_callback(lambda _t, rid=run_id: self._runs.pop(rid, None))

    async def _execute(self, task_id: str, run_id: str, *, resume: bool = False, answered: bool = False) -> None:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.app.config.tasks.max_concurrent_runs)
        try:
            async with self._sem:
                await self._execute_locked(task_id, run_id, resume=resume, answered=answered)
        except asyncio.CancelledError:
            if run_id not in self._cancel_requested:
                raise  # shutdown: leave the run 'processing' so it resumes on next start
            self._cancel_requested.discard(run_id)
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            run = await self.repo.get_run(run_id)
            if run is not None and run["status"] != "processing":
                await self._after_run(task_id, run["status"], run.get("error"))
                await self.publish(task_id)
        except Exception:
            log.exception("task run %s crashed", run_id)
            try:  # never fail silently: end the run with a message and the usual notification
                await self._finish_run(
                    task_id, run_id, "error", error="Something went wrong inside Sentient while running this task."
                )
            except Exception:
                log.exception("could not record the crash of task run %s", run_id)

    async def _execute_locked(self, task_id: str, run_id: str, *, resume: bool, answered: bool = False) -> None:
        run = await self.repo.get_run(run_id)
        task = await self.repo.get_task(task_id)
        if run is None or task is None or run["status"] != "processing":
            return
        cfg = self.app.config.tasks
        opening = "Resuming the run after a restart." if resume else "Executor has picked up the task and is starting execution."
        if executor.is_first_retry_attempt(run):
            opening = (
                "Retrying the failed run, continuing from where it stopped." if resume
                else "Retrying the failed run from the beginning."
            )
        if answered:
            opening = "Got your answer. Carrying on with the task."
        await self.progress(task_id, run_id, {"type": "info", "content": opening})
        await self.publish(task_id)
        status, error = "completed", None
        loop_result = None
        aggregated: list | None = None
        # a single run keeps its own clock of active time and asks before going over (tasks/limits.py)
        single = task.get("task_type") != "swarm" and executor.fixed_call_of(task) is None
        try:
            async with asyncio.timeout(None if single else cfg.run_timeout_minutes * 60):
                if task.get("task_type") == "swarm":
                    status, aggregated = await swarm.execute_swarm(self, task, run)
                elif executor.fixed_call_of(task) is not None:
                    loop_result = await executor.execute_fixed_call(self, task, run, resume=resume)
                else:
                    loop_result = await executor.execute_single(self, task, run, resume=resume, answered=answered)
        except RunPaused as paused:
            await self._pause_run(task_id, run_id, paused.question)
            return
        except TimeoutError:
            status, error = "error", f"Stopped after {cfg.run_timeout_minutes} minutes without finishing. {limits.LIMITS_HINT}"
        except RunFailed as exc:
            status, error = "error", str(exc)
        except ProviderError as exc:
            status, error = "error", _provider_down(exc, detail=True)
        except Exception as exc:
            log.exception("executor failed for task %s run %s", task_id, run_id)
            status, error = "error", f"Executor agent failed: {exc}"
        await self._finish_run(task_id, run_id, status, error=error, loop_result=loop_result, aggregated=aggregated)

    async def _finish_run(
        self, task_id: str, run_id: str, status: str, *, error: str | None,
        loop_result: Any = None, aggregated: list | None = None, from_statuses: tuple[str, ...] = ("processing",),
    ) -> None:
        if not await self.repo.finish_run(run_id, status, error=error, now=self.now_iso(), from_statuses=from_statuses):
            return  # cancelled meanwhile
        task = await self.repo.get_task(task_id)
        if task is None:
            return
        is_swarm = task.get("task_type") == "swarm"
        if error:
            await self.progress(task_id, run_id, {"type": "error", "content": error})
        succeeded = status in {"completed", "completed_with_errors"}
        if succeeded and not is_swarm:
            await self.progress(task_id, run_id, {"type": "info", "content": "Execution finished. Generating final report..."})
        # settle the task first: if the app stops during the slow report call, the status is still right
        await self._after_run(task_id, status, error)
        await self.publish(task_id)
        await self._publish_run_finished(task_id, run_id, status)
        quiet = bool((executor.fixed_call_of(task) or {}).get("quiet"))
        if succeeded:
            if quiet:  # the tool reported for itself (the Daily Brief): no model call for a report
                result = executor.normalize_result(
                    {"tools_used": loop_result.tools_used if loop_result else []},
                    (loop_result.text if loop_result else "") or "Done.",
                )
            else:
                result = await executor.generate_result(self, task, run_id, loop_result=loop_result, aggregated=aggregated)
            await self.repo.update_run(run_id, {"result": result})
            await self.publish(task_id)
        name = task.get("name") or "Untitled task"
        if is_swarm and status in {"completed", "completed_with_errors"}:
            await self._notify(task, f"Swarm task '{name}' has completed.", "Swarm task completed", "run_completed")
        elif status in {"completed", "completed_with_errors"}:
            if quiet:
                return  # the tool delivered its own notification (the Daily Brief)
            await self._notify(task, f"Task '{name}' has finished with status: {status}.", "Task completed", "run_completed")
        elif status == "error":
            detail = f"\n\n{error}" if error else ""
            if not is_swarm:
                detail += "\n\nOpen the task and choose Retry to continue from where it stopped."
            await self._notify(
                task, f"Task '{name}' has finished with status: error.{detail}", "Task failed", "run_failed",
                {"run_id": run_id},
            )

    async def _pause_run(self, task_id: str, run_id: str, question: dict) -> None:
        """Park a run that called ``ask_user``: store the question, flag the task, tell the user."""
        asked = {**question, "asked_at": self.now_iso()}
        if not await self.repo.pause_run(run_id, asked):
            return  # cancelled while the last round was running
        text = str(question.get("question") or "")
        reason = str(question.get("reason") or "") if question.get("stuck") else ""
        await self.progress(task_id, run_id, {"type": "info", "content": f"Waiting for your answer: {text}"})
        await self._after_run(task_id, "waiting_for_user", None)
        task = await self.repo.get_task(task_id)
        await self.publish(task_id)
        if task is None:
            return
        name = task.get("name") or "Untitled task"
        short = name[:60] + ("..." if len(name) > 60 else "")
        extra: dict[str, Any] = {"run_id": run_id, "question": text, "options": list(question.get("options") or [])}
        if reason:  # stuck (tasks/stuck.py): the same answerable question, with the reason up front
            extra.update(stuck=True, reason=reason)
            await self._notify(task, stuck.notice(name, reason), f"{short} is stuck", "question", extra)
            return
        await self._notify(task, text, f"{short} needs your answer", "question", extra)

    async def _resolve_question_notifications(self, run_id: str, status: str, *, answer: str | None = None) -> None:
        """Mark the 'needs your answer' notification of a run as answered or cancelled (channels settle their buttons)."""
        try:
            for note in await self.app.notifications.list(limit=1000):
                payload = note.get("payload") or {}
                if (
                    note.get("kind") == "task"
                    and payload.get("event") == "question"
                    and payload.get("run_id") == run_id
                    and not payload.get("status")
                ):
                    extra = {"answer": answer} if answer is not None else {}
                    await self.app.notifications.update_payload(note["id"], {**payload, "status": status, **extra})
                    if not note.get("read"):
                        await self.app.notifications.mark_read(note["id"])
        except Exception:
            log.exception("could not resolve question notifications for run %s", run_id)

    async def _publish_run_finished(self, task_id: str, run_id: str, status: str) -> None:
        """``task.run_finished`` for the self-improvement reviewer (docs/API.md section 10)."""
        try:
            tool_errors = 0
            skills_viewed: list[str] = []
            for update in await self.repo.events(run_id):
                msg = (update.get("message") or {}) if isinstance(update, dict) else {}
                if msg.get("type") == "tool_result" and msg.get("is_error"):
                    tool_errors += 1
                elif msg.get("type") == "tool_call" and msg.get("tool_name") == "skill_view":
                    params = msg.get("parameters") if isinstance(msg.get("parameters"), dict) else {}
                    skill = params.get("name")
                    if isinstance(skill, str) and skill and skill not in skills_viewed:
                        skills_viewed.append(skill)
            self.app.bus.publish("task.run_finished", {
                "task_id": task_id, "run_id": run_id, "status": status,
                "tool_errors": tool_errors, "skills_viewed": skills_viewed,
            })
        except Exception:
            log.exception("could not publish task.run_finished for run %s", run_id)

    # ------------------------------------------------------------------ script jobs
    async def _start_script(self, task: dict, *, trigger_data: dict | None = None) -> None:
        await self._set(task["id"], {"status": "processing", "last_execution_at": self.now_iso(), "error": None})
        self._spawn(self._script_job(task["id"], trigger_data), f"script:{task['id']}")

    def _script_tool_names(self) -> list[str]:
        return [
            t.name for t in self.app.registry.tools()
            if t.risk == Risk.read and t.plugin not in EXCLUDED_PLUGINS
        ]

    async def _sandbox_run(self, code: str, trigger_data: dict | None = None) -> dict:
        """``app.sandbox.run`` with a time limit; always returns a SandboxResult-shaped dict."""
        runner = getattr(getattr(self.app, "sandbox", None), "run", None)
        if not callable(runner):
            return {**SANDBOX_RESULT, "error": "code execution is not available in this version of Sentient"}
        full = scripts.trigger_prelude(trigger_data) + code if trigger_data else code
        limit = self.app.config.tasks.run_timeout_minutes
        try:
            async with asyncio.timeout(limit * 60):
                out = await runner(full, session_id=None, channel="task", allowed_tools=self._script_tool_names())
        except TimeoutError:
            return {**SANDBOX_RESULT, "error": f"it ran for more than {limit} minutes (tasks.run_timeout_minutes)"}
        except Exception as exc:
            log.warning("sandbox run failed: %s", exc)
            return {**SANDBOX_RESULT, "error": str(exc) or type(exc).__name__}
        if not isinstance(out, dict):
            return {**SANDBOX_RESULT, "error": "the code runner returned an unexpected result"}
        return {**SANDBOX_RESULT, **out}

    async def _script_job(self, task_id: str, trigger_data: dict | None) -> None:
        try:
            async with self._lock(task_id):
                await self._script_check(task_id, trigger_data)
        except asyncio.CancelledError:
            if self.app.stopped:  # Stop everything: settle the task now (a shutdown leaves it to recovery)
                await self._after_run(task_id, "cancelled", None)
                await self.publish(task_id)
            raise
        except Exception:
            log.exception("script job %s crashed", task_id)
            await self._after_run(task_id, "error", "The check script could not be run.")
            await self.publish(task_id)

    async def _save_script_state(self, task_id: str, code: str, state: dict) -> None:
        """Store check state unless the code was edited while the check ran (that resets the state)."""
        current = await self.repo.get_task(task_id)
        if current is None or current.get("task_type") != "script":
            return
        script = dict(current.get("script") or {})
        if script.get("code") != code:
            return
        script.update(state)
        await self._set(task_id, {"script": script})

    async def _script_check(self, task_id: str, trigger_data: dict | None) -> None:
        """One check: run the code with no model, act on the condition, record state, settle the task."""
        task = await self.repo.get_task(task_id)
        if task is None or task.get("task_type") != "script":
            return
        script = dict(task.get("script") or {})
        code = str(script.get("code") or "")
        name = task.get("name") or "Untitled task"
        outcome = await self._sandbox_run(code, trigger_data)
        now = self.now_iso()
        previous_error = script.get("last_error")
        error = scripts.outcome_error(outcome)
        if error:
            await self._save_script_state(task_id, code, {"last_run_at": now, "last_error": error})
            await self._after_run(task_id, "error", error)
            await self.publish(task_id)
            if not previous_error:  # tell once; recovery is announced below
                await self._notify(
                    task, f"The check for '{name}' failed. {error}\n\nI'll keep checking on schedule and tell you when it works again.",
                    "Check failed", "script_failed",
                )
            return
        value = scripts.outcome_value(outcome)
        condition = script.get("condition") or "alert"
        act = scripts.should_act(condition, value, script.get("last_result"))
        state: dict[str, Any] = {"last_run_at": now, "last_error": None}
        if value is not None or condition != "changed":  # an empty result keeps the baseline of a 'changed' job
            state["last_result"] = value
        await self._save_script_state(task_id, code, state)
        if previous_error:
            await self._notify(task, f"The check for '{name}' is working again.", "Check recovered", "script_recovered")
        if act:
            message = scripts.alert_message(name, condition, value)
            if script.get("then") == "run":
                context: dict[str, Any] = {"script_result": value, "message": message}
                if trigger_data:
                    context["trigger_event"] = trigger_data
                fresh = await self.repo.get_task(task_id) or task
                await self._start_run(fresh, trigger_data=context, next_execution_at=_KEEP)
            else:
                await self._notify(task, message, name, "script_alert", {"result": value})
        await self._after_run(task_id, "completed", None)
        await self.publish(task_id)

    async def _regenerate_result(self, task_id: str, run_id: str, *, notify_status: str | None = None) -> None:
        """Produce the structured result for a completed run whose report was never written.

        With ``notify_status`` the completion notification that the interrupted run never sent goes out too.
        """
        task = await self.repo.get_task(task_id)
        if task is None:
            return
        try:
            result = await executor.generate_result(self, task, run_id)
        except Exception:
            log.exception("regenerating result for run %s failed", run_id)
        else:
            await self.repo.update_run(run_id, {"result": result})
            await self.publish(task_id)
        if notify_status:
            name = task.get("name") or "Untitled task"
            await self._notify(
                task, f"Task '{name}' has finished with status: {notify_status}.", "Task completed", "run_completed"
            )

    async def _resolve_plan_notifications(self, task_id: str, status: str) -> None:
        """Retire 'Plan ready for approval' cards once the plan was approved or declined anywhere."""
        try:
            for note in await self.app.notifications.list(limit=1000):
                payload = note.get("payload") or {}
                if (
                    note.get("kind") == "task"
                    and note.get("task_id") == task_id
                    and payload.get("event") == "approval_needed"
                    and not payload.get("status")
                ):
                    await self.app.notifications.update_payload(note["id"], {**payload, "status": status})
                    if not note.get("read"):
                        await self.app.notifications.mark_read(note["id"])
        except Exception:
            log.exception("could not resolve plan notifications for task %s", task_id)

    async def _resolve_stale_plan_notifications(self) -> None:
        """Startup sweep for plan cards whose task already moved on (approved via another screen or older builds)."""
        try:
            notes = await self.app.notifications.list(limit=1000)
        except Exception:
            log.exception("could not list notifications")
            return
        for note in notes:
            payload = note.get("payload") or {}
            tid = note.get("task_id")
            if note.get("kind") != "task" or payload.get("event") != "approval_needed" or payload.get("status") or not tid:
                continue
            task = await self.repo.get_task(tid)
            if task is None or task["status"] == "approval_pending":
                continue
            await self._resolve_plan_notifications(tid, "declined" if task["status"] == "declined" else "approved")

    async def _after_run(self, task_id: str, run_status: str, error: str | None) -> None:
        """Move the task out of 'processing' once no runs remain (v2 executor rescheduling)."""
        task = await self.repo.get_task(task_id)
        if task is None or task["status"] not in BUSY:
            return
        if await self.repo.processing_runs(task_id):
            return
        if await self.repo.waiting_runs(task_id):  # a run still waits for the user's answer
            if task["status"] != "waiting_for_user":
                await self._set(task_id, {"status": "waiting_for_user"})
            return
        schedule = task.get("schedule") or {}
        kind = schedule.get("type")
        if kind == "recurring":
            fields: dict[str, Any] = {"status": "active", "next_execution_at": iso(calculate_next_run(schedule, self.now()))}
        elif kind == "triggered":
            fields = {"status": "active", "next_execution_at": None}
        else:
            fields = {"status": run_status, "next_execution_at": None}
            if run_status == "error":
                fields["error"] = error
        await self._set(task_id, fields)

    async def _delete_notifications(self, task_id: str) -> None:
        try:
            for note in await self.app.notifications.list(limit=1000):
                if note.get("task_id") == task_id and note.get("id"):
                    await self.app.notifications.delete(note["id"])
        except Exception:
            log.exception("could not delete notifications for task %s", task_id)
