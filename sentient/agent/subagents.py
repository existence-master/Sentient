"""Chat subagents: background and parallel workers spawned by the assistant (docs/API.md section 10).

A subagent is one ``Agent.run_loop`` with its own message list, a focused system
prompt, the ``subagents.role`` model (the chat's own model where that is the primary one, #312),
``subagents.max_rounds`` rounds and a timeout. It cannot talk to the user, so it cannot run anything that would need
an approval, anything of effective risk ``send``/``exec``, or start subagents:
those calls come back as refusals the model can relay.

- Foreground (``delegate_task``): the parent tool call waits; every step is
  streamed to the chat as ``tool_progress`` (kind ``subagent``).
- Background: the tool returns at once; when the subagent finishes, its summary
  is added to the chat as an assistant message and a notification is created.
- Reader (``read``, the ``read_with_reader`` tool, #144, ADR 0031): no loop and no tools at all. One model call
  reads an outside document and code keeps only the requested fields (``sentient.agent.reader``). Its row has
  ``kind = "reader"``, ``tool_calls = 0`` and a trace of the fields asked for and the fields returned.

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
from pathlib import Path
from typing import Any

from sentient import paths
from sentient.agent import reader
from sentient.agent.loop import Budget, LoopResult
from sentient.llm.events import Error, TextDelta, ToolCallEvent, ToolResultEvent
from sentient.llm.jobs import detached
from sentient.services import Service, cancel_tasks
from sentient.store.db import new_id, now_iso
from sentient.tools.base import MEMORY_TOOLS, Risk, Tool

log = logging.getLogger(__name__)

MAX_EVENTS = 200
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
        "kind": d.get("kind") or "worker",
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


def _files_from(result: Any) -> list[str]:
    """The files one of the subagent's own tool calls saved, relative to the files folder: any result whose ``path``
    is inside the files folder, plus a script run's ``files_created``. Read from its results rather than by
    comparing the folder before and after, so helpers running at once never report each other's files (#94).
    It also includes browser screenshots and downloaded files if their paths are provided as relative paths."""
    if not isinstance(result, dict):
        return []
    names: list[str] = []
    path = result.get("path")
    if not path and isinstance(result.get("file"), str):
        file_path = Path(result["file"])
        if not file_path.is_absolute() and ".." not in file_path.parts:
            resolved_path = (paths.files_dir() / file_path).resolve()
            if resolved_path.exists() and resolved_path.is_relative_to(paths.files_dir().resolve()):
                names.append(resolved_path.relative_to(paths.files_dir().resolve()).as_posix())
    # Process browser downloads
    if "downloads" in result:
        for dl in result["downloads"]:
            dl_path = Path(dl) if isinstance(dl, str) else None
            if dl_path and not dl_path.is_absolute() and ".." not in dl_path.parts:
                resolved_dl = (paths.files_dir() / dl_path).resolve()
                if resolved_dl.exists() and resolved_dl.is_relative_to(paths.files_dir().resolve()):
                    names.append(resolved_dl.relative_to(paths.files_dir().resolve()).as_posix())
    if isinstance(path, str) and Path(path).is_absolute():
        with contextlib.suppress(OSError, ValueError):
            names.append(Path(path).resolve().relative_to(paths.files_dir().resolve()).as_posix())
    names += [n for n in result.get("files_created") or [] if isinstance(n, str)]
    return names


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
    async def cancel_for_session(self, session_id: str) -> int:
        """Cancel the helpers a chat started (before the chat is deleted). Returns how many were running."""
        rows = await self.app.store.fetchall(
            "SELECT id FROM subagents WHERE session_id = ? AND status = 'running'", (session_id,)
        )
        tasks = [t for r in rows if (t := self._tasks.get(r["id"])) is not None and not t.done()]
        return await cancel_tasks(tasks, timeout=10) if tasks else 0

    async def cancel(self, subagent_id: str) -> dict | None:
        task = self._tasks.get(subagent_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=10)
        elif subagent_id not in self._tasks:  # still listed: it is saving its own result right now
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
        untrusted: str = "",
        memory_allowed: bool = True,
    ) -> dict:
        """Start a subagent. Foreground: waits and returns ``{subagent_id, status, summary, files_created}``
        (plus ``error``). Background: returns ``{subagent_id, status: "running"}`` right away. ``origin`` is the
        parent run's ``ToolContext.origin``: a subagent of work nobody asked for may only read too (ADR 0017).
        ``untrusted`` is the parent's ``ToolContext.untrusted``: outside content it read still counts (ADR 0018).
        ``memory_allowed`` is the parent's ``ToolContext.memory_allowed`` (#525): False, the helper gets no memory
        tools, no user name and nothing Sentient knows, like the turn that started it."""
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
                sub_id,
                session_id,
                parent_call_id,
                goal,
                context or None,
                json.dumps(tools) if tools else None,
                "running",
                int(bool(background)),
                "[]",
                now_iso(),
            ),
        )
        self._publish(await self.get(sub_id, events=False))
        task = asyncio.create_task(
            self._run(
                sub_id,
                goal,
                context or "",
                tools,
                session_id,
                bool(background),
                on_update,
                origin,
                untrusted,
                memory_allowed,
            ),
            name=f"subagent:{sub_id}",
            # a foreground helper is part of whatever waits on it (a chat reply); a background one is task work (#149)
            context=detached("task") if background else None,
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

    async def read(
        self,
        document: str,
        fields: list[reader.Field],
        *,
        source: str,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        attended: bool = False,
    ) -> dict:
        """The reader (#144): one tool-free model call reads ``document`` (outside content from ``source``) and only
        the requested ``fields`` come back, checked in code. Recorded as a ``kind = "reader"`` subagent whose trace
        shows the fields asked for and returned, and no tool calls. Returns ``{subagent_id, status, fields}`` or
        ``{subagent_id, status: "error", error}``."""
        sub_id = new_id()
        names = ", ".join(f.name for f in fields)
        goal = f"Read {source} for: {names}"[:500]
        events: list[dict] = [
            {
                "timestamp": now_iso(),
                "message": {
                    "type": "info",
                    "content": "Reader started with no tools. Fields asked for: "
                    + ", ".join(f"{f.name} ({f.describe()})" for f in fields),
                },
            }
        ]
        await self.app.store.execute(
            "INSERT INTO subagents(id, session_id, parent_call_id, goal, kind, tools, status, background, events,"
            " started_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                sub_id,
                session_id,
                parent_call_id,
                goal,
                "reader",
                "[]",
                "running",
                0,
                json.dumps(events),
                now_iso(),
            ),
        )
        self._publish(await self.get(sub_id, events=False))
        status, error, values = "completed", None, None
        # its own task, so ``cancel(sub_id)`` (POST /api/subagents/{id}/cancel) stops the model call too
        task = asyncio.create_task(
            reader.read(self.app.llm, document, fields, source=source, attended=attended),
            name=f"reader:{sub_id}",
        )
        self._tasks[sub_id] = task
        try:
            values = await task
            events.append(
                {
                    "timestamp": now_iso(),
                    "message": {
                        "type": "final_answer",
                        "content": json.dumps(values, ensure_ascii=False),
                    },
                }
            )
        except asyncio.CancelledError:
            status, error = "cancelled", "The reader was stopped."
            if not task.cancelled() or (
                asyncio.current_task() and asyncio.current_task().cancelling()
            ):
                task.cancel()
                raise  # the parent was stopped: pass it on
            events.append(
                {"timestamp": now_iso(), "message": {"type": "info", "content": "Cancelled."}}
            )
        except reader.ReaderError as exc:
            status, error = "error", str(exc)
            events.append({"timestamp": now_iso(), "message": {"type": "error", "content": error}})
        finally:
            self._tasks.pop(sub_id, None)
            await asyncio.shield(
                self._update(
                    sub_id,
                    status=status,
                    error=error,
                    tool_calls=0,
                    finished_at=now_iso(),
                    summary=json.dumps(values, ensure_ascii=False) if values is not None else None,
                    events=json.dumps(events, default=str),
                )
            )
            self._publish(await self.get(sub_id, events=False))
        if values is None:
            return {
                "subagent_id": sub_id,
                "status": status if status == "cancelled" else "error",
                "error": error or "The reader failed.",
            }
        return {"subagent_id": sub_id, "status": "completed", "fields": values}

    async def delegate_many(
        self,
        tasks: list[dict],
        *,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        on_update: UpdateFn | None = None,
        origin: str = "user",
        untrusted: str = "",
        memory_allowed: bool = True,
    ) -> list[dict]:
        """Run several foreground subagents; at most ``subagents.max_concurrent`` at a time. ``origin``,
        ``untrusted`` and ``memory_allowed`` are the parent's, as for ``delegate``."""
        jobs = [
            self.delegate(
                str(t.get("goal") or ""),
                context=str(t.get("context") or ""),
                tools=t.get("tools") if isinstance(t.get("tools"), list) else None,
                session_id=session_id,
                parent_call_id=parent_call_id,
                on_update=on_update,
                origin=origin,
                untrusted=untrusted,
                memory_allowed=memory_allowed,
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
        await self.app.store.execute(
            f"UPDATE subagents SET {cols} WHERE id = ?", (*fields.values(), subagent_id)
        )

    @contextlib.asynccontextmanager
    async def _slot(self):
        if self._slot_free is None:
            self._slot_free = asyncio.Condition()
        async with self._slot_free:
            await self._slot_free.wait_for(
                lambda: self._active < self.app.config.subagents.max_concurrent
            )
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

    async def _tool_names(
        self, goal: str, context: str, tools: list[str] | None, role: str, model: str | None = None,
        *, memory_allowed: bool = True,
    ) -> list[str]:
        """The helper's tools: never wider than its parent turn's, so without memory (#525) no memory tools, even
        when the goal names them."""
        reg = self.app.registry
        usable = [
            t
            for t in reg.tools()
            if t.plugin != PLUGIN_ID and not (t.risk >= Risk.send and t.risk_fn is None)
            and (memory_allowed or t.name not in MEMORY_TOOLS)
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
                f"{goal}\n{context}".strip(), model=model or self.app.llm.model_for(role)
            )
        except Exception as exc:
            log.debug("subagent tool selection failed: %s", exc)
            selected = None
        if selected is None:
            return [t.name for t in usable]
        return [n for n in selected if n in usable_names]

    def _system_prompt(self, *, private: bool = False) -> str:
        """``private``: started from a turn without memory (an incognito chat that doesn't use it, #154, #525), so the
        user's name stays out."""
        from sentient.tools.builtin.time_tool import resolve_tz

        cfg = self.app.config.assistant
        now = datetime.now(resolve_tz(cfg.timezone))
        return SUBAGENT_SYSTEM.format(
            assistant=cfg.name or "Sentient",
            user=(None if private else cfg.user_name) or "the user",
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
        untrusted: str = "",
        memory_allowed: bool = True,
    ) -> None:
        cfg = self.app.config.subagents
        agent = self.app.agent
        events: list[dict] = []
        result = LoopResult()
        status, error = "completed", None
        files: list[str] = []

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
                note({"type": "info", "content": f"Started: {goal[:200]}"}, "Started")
                incognito = await self.app.store.is_incognito(
                    session_id
                )  # kept if the chat is deleted (#154)
                # never wider than the parent turn (#525); the chat's own setting still holds for a caller that
                # didn't pass the parent's decision
                allowed = memory_allowed and not (incognito and not self.app.config.chat.incognito_uses_memory)
                private = not allowed
                messages = [
                    {"role": "system", "content": self._system_prompt(private=private)},
                    {
                        "role": "user",
                        "content": f"Goal: {goal}"
                        + (f"\n\nContext:\n{context}" if context else ""),
                    },
                ]
                ctx = agent.tool_context(session_id, "subagent", origin=origin)
                ctx.incognito = incognito
                ctx.memory_allowed = allowed
                ctx.untrusted = untrusted
                ctx.extra["subagent_id"] = sub_id
                # where it would use the primary model, a helper uses the chat's own model (#312)
                model = await agent.helper_model(session_id, cfg.role)
                names = await self._tool_names(goal, context, tools, cfg.role, model, memory_allowed=allowed)
                narration = ""
                async with asyncio.timeout(cfg.timeout_minutes * 60):
                    async for event in agent.run_loop(
                        messages,
                        ctx,
                        result=result,
                        role=cfg.role,
                        model=model,
                        tool_names=names,
                        max_rounds=cfg.max_rounds,
                        use_approvals=False,
                        policy=self.policy(session_id),
                        source="subagent",
                        budget=Budget(max_tokens=cfg.max_tokens, max_cost_usd=cfg.max_cost_usd),
                    ):
                        if isinstance(event, TextDelta):
                            narration += event.text
                        elif isinstance(event, ToolCallEvent):
                            if narration.strip():
                                note({"type": "thought", "content": narration.strip()[:1000]})
                            narration = ""
                            note(
                                {
                                    "type": "tool_call",
                                    "tool_name": event.name,
                                    "parameters": event.arguments,
                                },
                                f"Using {event.name}",
                            )
                        elif isinstance(event, ToolResultEvent):
                            if not event.is_error:
                                files.extend(_files_from(event.result))
                            note(
                                {
                                    "type": "tool_result",
                                    "tool_name": event.name,
                                    "result": _short(event.result),
                                    "is_error": event.is_error,
                                },
                                f"{event.name} failed"
                                if event.is_error
                                else f"{event.name} finished",
                            )
                            await self._update(
                                sub_id,
                                events=json.dumps(events, default=str),
                                tool_calls=result.tool_calls,
                            )
                        elif isinstance(event, Error):
                            note({"type": "error", "content": event.message})
            if result.error:
                status, error = "error", result.error
            elif result.hit_step_limit:
                status, error = "error", f"Stopped after {cfg.max_rounds} steps without finishing."
        except TimeoutError:
            status, error = (
                "error",
                f"Stopped after {cfg.timeout_minutes} minutes without finishing.",
            )
        except asyncio.CancelledError:
            status, error = "cancelled", None
            await asyncio.shield(
                self._finish(
                    sub_id, goal, session_id, background, status, error, result, events, files
                )
            )
            raise
        except Exception as exc:
            log.exception("subagent %s failed", sub_id)
            status, error = "error", f"{type(exc).__name__}: {exc}"
        await self._finish(
            sub_id, goal, session_id, background, status, error, result, events, files
        )

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
        files: list[str],
    ) -> None:
        summary = (result.text or "").strip()
        if result.hit_step_limit and summary.startswith("I reached the step limit"):
            summary = ""
        files = sorted(set(files))
        if status == "completed":
            events.append(
                {"timestamp": now_iso(), "message": {"type": "final_answer", "content": summary}}
            )
        elif status == "error":
            events.append(
                {"timestamp": now_iso(), "message": {"type": "error", "content": error or "failed"}}
            )
        else:
            events.append(
                {"timestamp": now_iso(), "message": {"type": "info", "content": "Cancelled."}}
            )
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
        self,
        sub_id: str,
        goal: str,
        session_id: str,
        status: str,
        summary: str,
        error: str | None,
        files: list[str],
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
