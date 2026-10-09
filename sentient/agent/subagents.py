"""Chat subagents: background and parallel workers spawned by the assistant (docs/API.md section 10).

A subagent is one ``Agent.run_loop`` with its own message list, a focused system
prompt, the ``subagents.role`` model, ``subagents.max_rounds`` rounds and a
timeout. It cannot talk to the user, so it cannot run anything that would need
an approval, anything of effective risk ``send``/``exec``, or start subagents:
those calls come back as refusals the model can relay.

- Foreground (``delegate_task``): the parent tool call waits; every step is
  streamed to the chat as ``tool_progress`` (kind ``subagent``).
- Background: the tool returns at once; when the subagent finishes, its summary
  is added to the chat as an assistant message and a notification is created.

Every state change is persisted in the ``subagents`` table and published as
``subagent.updated``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sentient import paths
from sentient.agent.loop import Budget, LoopResult
from sentient.llm.events import Error, TextDelta, ToolCallEvent, ToolResultEvent
from sentient.services import Service, cancel_tasks
from sentient.store.db import new_id, now_iso
from sentient.tools.base import Risk, Tool

log = logging.getLogger(__name__)

MAX_EVENTS = 200
MAX_FILES_SCANNED = 5000
PLUGIN_ID = "subagents"
UpdateFn = Callable[[str, str], Any]  # (subagent_id, message)

SUBAGENT_SYSTEM = """You are a subagent of {assistant}, the personal assistant of {user}. The main assistant gave you one goal. \
Work on it on your own with your tools, then report back.

Rules:
- You cannot talk to {user} and nobody will answer questions. Make reasonable assumptions and say which.
- You cannot send, post, buy, delete or run code, and you cannot start other subagents. If the goal needs that, \
stop and say exactly what the main assistant should do.
- Never invent results. If a tool fails, try one sensible alternative, then report what failed.
- For long output, save a file with file_write when that tool is available and mention its name.
- Finish with a concise report for the main assistant: what you found or did, key facts with their sources, \
files created, and anything left open. No greetings.

Now: {now}"""

_VERBS = {Risk.send: "send, delete or spend", Risk.exec: "run code or commands"}


def _loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _row(r: Any, *, events: bool = True) -> dict:
    d = dict(r)
    out = {
        "subagent_id": d["id"],
        "session_id": d.get("session_id"),
        "parent_call_id": d.get("parent_call_id"),
        "goal": d["goal"],
        "status": d["status"],
        "background": bool(d.get("background")),
        "summary": d.get("summary"),
        "error": d.get("error"),
        "tool_calls": int(d.get("tool_calls") or 0),
        "files_created": _loads(d.get("files_created"), []),
        "started_at": d.get("started_at"),
        "finished_at": d.get("finished_at"),
    }
    if events:
        out["events"] = _loads(d.get("events"), [])
    return out


def _snapshot_files() -> dict[str, float]:
    root = paths.files_dir()
    out: dict[str, float] = {}
    if not root.exists():
        return out
    for i, p in enumerate(root.rglob("*")):
        if i >= MAX_FILES_SCANNED:
            break
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel.startswith("outputs/tool-"):  # cut tool results are bookkeeping, not work products
                continue
            with contextlib.suppress(OSError):
                out[rel] = p.stat().st_mtime
    return out


def _short(value: Any, limit: int = 1000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


class SubagentManager(Service):
    name = "subagents"

    def __init__(self, app: Any):
        super().__init__(app)
        self._tasks: dict[str, asyncio.Task] = {}
        self._active = 0
        self._slot_free: asyncio.Condition | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        await self.app.store.execute(
            "UPDATE subagents SET status = 'error', error = ?, finished_at = ? WHERE status = 'running'",
            ("Sentient restarted before this subagent finished.", now_iso()),
        )
        self.app.registry.set_hidden(PLUGIN_ID, not self.app.config.subagents.enabled)

    async def stop(self) -> None:
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await super().stop()

    async def halt(self) -> int:
        """Stop everything: cancel every running helper (they end as ``cancelled``)."""
        return await cancel_tasks(self._tasks.values(), timeout=10)

    # ------------------------------------------------------------------ queries
    async def get(self, subagent_id: str, *, events: bool = True) -> dict | None:
        r = await self.app.store.fetchone("SELECT * FROM subagents WHERE id = ?", (subagent_id,))
        return _row(r, events=events) if r else None

    async def list_for_session(self, session_id: str, limit: int = 100) -> list[dict]:
        rows = await self.app.store.fetchall(
            "SELECT * FROM subagents WHERE session_id = ? ORDER BY started_at DESC, rowid DESC LIMIT ?",
            (session_id, limit),
        )
        return [_row(r) for r in rows]

    def is_running(self, subagent_id: str) -> bool:
        t = self._tasks.get(subagent_id)
        return bool(t and not t.done())

    # ------------------------------------------------------------------ actions
    async def cancel(self, subagent_id: str) -> dict | None:
        task = self._tasks.get(subagent_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=10)
        else:
            sub = await self.get(subagent_id, events=False)
            if sub and sub["status"] == "running":  # orphaned row: nothing is running it any more
                await self._update(subagent_id, status="cancelled", finished_at=now_iso())
                self._publish(await self.get(subagent_id, events=False))
        return await self.get(subagent_id)

    async def delegate(
        self,
        goal: str,
        *,
        context: str = "",
        tools: list[str] | None = None,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        background: bool = False,
        on_update: UpdateFn | None = None,
        origin: str = "user",
    ) -> dict:
        """Start a subagent. Foreground: waits and returns ``{subagent_id, status, summary, files_created}``
        (plus ``error``). Background: returns ``{subagent_id, status: "running"}`` right away. ``origin`` is the
        parent run's ``ToolContext.origin``: a subagent of work nobody asked for may only read too (ADR 0017)."""
        cfg = self.app.config.subagents
        if not cfg.enabled:
            return {"status": "error", "error": "Subagents are turned off in Settings."}
        goal = (goal or "").strip()
        if not goal:
            return {"status": "error", "error": "A subagent needs a goal."}
        if self.app.agent is None:
            return {"status": "error", "error": "The assistant is not ready yet."}
        sub_id = new_id()
        await self.app.store.execute(
            "INSERT INTO subagents(id, session_id, parent_call_id, goal, context, tools, status, background, events, started_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                sub_id, session_id, parent_call_id, goal, context or None, json.dumps(tools) if tools else None,
                "running", int(bool(background)), "[]", now_iso(),
            ),
        )
        self._publish(await self.get(sub_id, events=False))
        task = asyncio.create_task(
            self._run(sub_id, goal, context or "", tools, session_id, bool(background), on_update, origin),
            name=f"subagent:{sub_id}",
        )
        self._tasks[sub_id] = task
        task.add_done_callback(lambda _t, sid=sub_id: self._tasks.pop(sid, None))
        if background:
            return {"subagent_id": sub_id, "status": "running"}
        await task  # cancelling the parent tool call cancels the subagent too
        sub = await self.get(sub_id, events=False) or {}
        out = {
            "subagent_id": sub_id,
            "status": sub.get("status"),
            "summary": sub.get("summary") or "",
            "files_created": sub.get("files_created") or [],
        }
        if sub.get("error"):
            out["error"] = sub["error"]
        return out

    async def delegate_many(
        self,
        tasks: list[dict],
        *,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        on_update: UpdateFn | None = None,
        origin: str = "user",
    ) -> list[dict]:
        """Run several foreground subagents; at most ``subagents.max_concurrent`` at a time."""
        jobs = [
            self.delegate(
                str(t.get("goal") or ""),
                context=str(t.get("context") or ""),
                tools=t.get("tools") if isinstance(t.get("tools"), list) else None,
                session_id=session_id,
                parent_call_id=parent_call_id,
                on_update=on_update,
                origin=origin,
            )
            for t in tasks
        ]
        return list(await asyncio.gather(*jobs))

    # ------------------------------------------------------------------ internals
    def _publish(self, sub: dict | None) -> None:
        if sub is not None:
            sub = {k: v for k, v in sub.items() if k != "events"}
            self.app.bus.publish("subagent.updated", sub)

    async def _update(self, subagent_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        await self.app.store.execute(f"UPDATE subagents SET {cols} WHERE id = ?", (*fields.values(), subagent_id))

    @contextlib.asynccontextmanager
    async def _slot(self):
        if self._slot_free is None:
            self._slot_free = asyncio.Condition()
        async with self._slot_free:
            await self._slot_free.wait_for(lambda: self._active < self.app.config.subagents.max_concurrent)
            self._active += 1
        try:
            yield
        finally:
            async with self._slot_free:
                self._active -= 1
                self._slot_free.notify_all()

    def policy(self, session_id: str | None) -> Callable[[Tool, Risk, dict], str | None]:
        approvals = self.app.approvals

        def check(tool: Tool, risk: Risk, arguments: dict) -> str | None:
            if tool.plugin == PLUGIN_ID or tool.name.startswith("delegate_"):
                return "Subagents cannot start other subagents. Do this part yourself or report it back."
            if risk >= Risk.send:
                return (
                    f"A subagent cannot run '{tool.name}' because it can {_VERBS.get(risk, 'make changes')} "
                    f"(risk {risk.name}). Stop and tell the main assistant exactly what to run so it can ask the user."
                )
            if approvals.needs_approval(tool, session_id, risk):
                return (
                    f"'{tool.name}' needs the user's approval, which a subagent cannot ask for. "
                    "Stop and tell the main assistant exactly what to run."
                )
            return None

        return check

    async def _tool_names(self, goal: str, context: str, tools: list[str] | None, role: str) -> list[str]:
        reg = self.app.registry
        usable = [
            t for t in reg.tools()
            if t.plugin != PLUGIN_ID and not (t.risk >= Risk.send and t.risk_fn is None)
        ]
        usable_names = {t.name for t in usable}
        if tools:
            wanted: list[str] = []
            for n in tools:
                if n in usable_names:
                    wanted.append(n)
                else:  # a plugin id expands to its tools
                    wanted.extend(t.name for t in usable if t.plugin == n)
            if wanted:
                return list(dict.fromkeys(wanted))
        try:
            selected = await self.app.agent.tool_selector.select(
                f"{goal}\n{context}".strip(), model=self.app.llm.model_for(role)
            )
        except Exception as exc:
            log.debug("subagent tool selection failed: %s", exc)
            selected = None
        if selected is None:
            return [t.name for t in usable]
        return [n for n in selected if n in usable_names]

    def _system_prompt(self) -> str:
        from sentient.tools.builtin.time_tool import resolve_tz

        cfg = self.app.config.assistant
        now = datetime.now(resolve_tz(cfg.timezone))
        return SUBAGENT_SYSTEM.format(
            assistant=cfg.name or "Sentient",
            user=cfg.user_name or "the user",
            now=f"{now.strftime('%A %Y-%m-%d %H:%M')} ({now.tzinfo})",
        )

    async def _run(
        self,
        sub_id: str,
        goal: str,
        context: str,
        tools: list[str] | None,
        session_id: str | None,
        background: bool,
        on_update: UpdateFn | None,
        origin: str = "user",
    ) -> None:
        cfg = self.app.config.subagents
        agent = self.app.agent
        events: list[dict] = []
        result = LoopResult()
        status, error = "completed", None
        before: dict[str, float] = {}

        def note(message: dict, text: str | None = None) -> None:
            events.append({"timestamp": now_iso(), "message": message})
            del events[:-MAX_EVENTS]
            if text and on_update is not None:
                try:
                    on_update(sub_id, text)
                except Exception as exc:
                    log.debug("subagent update listener failed: %s", exc)

        try:
            async with self._slot():
                before = await asyncio.to_thread(_snapshot_files)
                note({"type": "info", "content": f"Started: {goal[:200]}"}, "Started")
                messages = [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": f"Goal: {goal}" + (f"\n\nContext:\n{context}" if context else "")},
                ]
                ctx = agent.tool_context(session_id, "subagent", origin=origin)
                ctx.extra["subagent_id"] = sub_id
                names = await self._tool_names(goal, context, tools, cfg.role)
                narration = ""
                async with asyncio.timeout(cfg.timeout_minutes * 60):
                    async for event in agent.run_loop(
                        messages, ctx, result=result, role=cfg.role, tool_names=names, max_rounds=cfg.max_rounds,
                        use_approvals=False, policy=self.policy(session_id), source="subagent",
                        budget=Budget(max_tokens=cfg.max_tokens, max_cost_usd=cfg.max_cost_usd),
                    ):
                        if isinstance(event, TextDelta):
                            narration += event.text
                        elif isinstance(event, ToolCallEvent):
                            if narration.strip():
                                note({"type": "thought", "content": narration.strip()[:1000]})
                            narration = ""
                            note(
                                {"type": "tool_call", "tool_name": event.name, "parameters": event.arguments},
                                f"Using {event.name}",
                            )
                        elif isinstance(event, ToolResultEvent):
                            note(
                                {"type": "tool_result", "tool_name": event.name, "result": _short(event.result),
                                 "is_error": event.is_error},
                                f"{event.name} failed" if event.is_error else f"{event.name} finished",
                            )
                            await self._update(sub_id, events=json.dumps(events, default=str), tool_calls=result.tool_calls)
                        elif isinstance(event, Error):
                            note({"type": "error", "content": event.message})
            if result.error:
                status, error = "error", result.error
            elif result.hit_step_limit:
                status, error = "error", f"Stopped after {cfg.max_rounds} steps without finishing."
        except TimeoutError:
            status, error = "error", f"Stopped after {cfg.timeout_minutes} minutes without finishing."
        except asyncio.CancelledError:
            status, error = "cancelled", None
            await asyncio.shield(self._finish(sub_id, goal, session_id, background, status, error, result, events, before))
            raise
        except Exception as exc:
            log.exception("subagent %s failed", sub_id)
            status, error = "error", f"{type(exc).__name__}: {exc}"
        await self._finish(sub_id, goal, session_id, background, status, error, result, events, before)

    async def _finish(
        self,
        sub_id: str,
        goal: str,
        session_id: str | None,
        background: bool,
        status: str,
        error: str | None,
        result: LoopResult,
        events: list[dict],
        before: dict[str, float],
    ) -> None:
        summary = (result.text or "").strip()
        if result.hit_step_limit and summary.startswith("I reached the step limit"):
            summary = ""
        try:
            after = await asyncio.to_thread(_snapshot_files) if before is not None else {}
        except Exception:
            after = {}
        files = sorted(n for n, m in after.items() if before and (n not in before or before[n] != m)) if before else []
        if status == "completed":
            events.append({"timestamp": now_iso(), "message": {"type": "final_answer", "content": summary}})
        elif status == "error":
            events.append({"timestamp": now_iso(), "message": {"type": "error", "content": error or "failed"}})
        else:
            events.append({"timestamp": now_iso(), "message": {"type": "info", "content": "Cancelled."}})
        try:
            await self._update(
                sub_id,
                status=status,
                summary=summary or None,
                error=error,
                tool_calls=result.tool_calls,
                files_created=json.dumps(files),
                events=json.dumps(events[-MAX_EVENTS:], default=str),
                finished_at=now_iso(),
            )
            sub = await self.get(sub_id, events=False)
            self._publish(sub)
            if background and session_id and status != "cancelled":
                await self._report_to_chat(sub_id, goal, session_id, status, summary, error, files)
        except Exception:
            log.exception("could not record the end of subagent %s", sub_id)

    async def _report_to_chat(
        self, sub_id: str, goal: str, session_id: str, status: str, summary: str, error: str | None, files: list[str]
    ) -> None:
        short_goal = goal if len(goal) <= 120 else goal[:117] + "..."
        if status == "completed":
            body = f"**Background work finished:** {short_goal}\n\n{summary or '(no summary)'}"
            title = "Background work finished"
        else:
            body = f"**Background work stopped:** {short_goal}\n\n{error or 'It failed.'}"
            if summary:
                body += f"\n\n{summary}"
            title = "Background work stopped"
        if files:
            body += "\n\nFiles: " + ", ".join(files[:20])
        if await self.app.store.get_session(session_id) is None:
            return
        await self.app.store.add_message(session_id, "assistant", body)
        with contextlib.suppress(Exception):
            await self.app.notify(
                "info",
                f"{title}: {short_goal}",
                title=title,
                payload={"subagent_id": sub_id, "session_id": session_id},
            )
