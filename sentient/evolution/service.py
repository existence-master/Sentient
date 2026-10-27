"""Self-evolution: background skill review of finished chats and task runs, the skill
curator lifecycle, USER.md/MEMORY.md upkeep, and the memory housekeeping jobs
(episodic summarization of old conversations, hourly purge of expired facts).

OWNER: MEMORY/PROACTIVITY AGENT.

Hermes-inspired loop:
- reviewer: after a chat has been idle ``evolution.review_idle_minutes`` and used at least
  ``min_tool_calls_for_review`` tool calls (or a task run completed), the fast model reads
  the transcript plus the skill index and answers none | create | patch with a complete
  SKILL.md body. Proposals always land in ``skills/pending`` with a ``skill`` notification.
- repair: a skill viewed in a chat turn or task run that then hit tool errors, a failed run, or a user
  correction on the next message gets a fix proposal (pending, origin "repair"); every use is counted
  as a success or a failure in ``skill_stats``.
- curator: unused skills go active -> stale -> archived; near-duplicates get a merge proposal.
- profile: MEMORY.md is regenerated from facts + summaries; stable new facts are appended
  under "## Learned" in USER.md without touching the user's own text.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import re
import shutil
from datetime import UTC, datetime, timedelta
from typing import Any

from sentient.evolution import prompts
from sentient.evolution.log import log_event
from sentient.memory import review as memory_review
from sentient.memory import scopes as memory_scopes
from sentient.memory.episodic import CLEAN as CLEAN_SUMMARY
from sentient.memory.topics import TOPIC_NAMES
from sentient.memory.vectors import cosine
from sentient.services import Service, cancel_tasks
from sentient.skills import checks as skill_checks
from sentient.skills.loader import render_skill, slugify, valid_name
from sentient.store.db import after_watermark, next_watermark, now_iso

log = logging.getLogger(__name__)

SECTIONS = ["When to use", "Procedure", "Pitfalls", "Verification"]
TRANSCRIPT_BUDGET = 14_000
REVIEW_LOOKBACK_DAYS = 3
STABLE_TOPICS = set(TOPIC_NAMES) - {"Miscellaneous"}
# the memory scopes the profile files (MEMORY.md, USER.md) are built from (#155); work and project chats leave them out
PROFILE_SCOPES = "COALESCE(scope, 'personal') IN ('personal', 'always')"
DONE_STATUSES = {"completed", "completed_with_errors"}
PLACEHOLDER_TAGS = {"short", "labels", "learned"}
FAILED_RUN_STATUSES = {"error", "failed", "completed_with_errors"}
REPAIR_TRANSCRIPT_BUDGET = 6_000
MAX_TRACKED_TURNS = 200
PROFILE_FACTS_MARK = "evolution.profile_facts_mark"  # what profile upkeep read last (store.db.next_watermark)
PROFILE_SUMMARIES_MARK = "evolution.profile_summaries_mark"

# cheap first pass before one fast-model confirmation; false positives are fine, misses are not
_CORRECTION_RE = re.compile(
    r"(^\s*(no|nope|wrong)\b"
    r"|\b(that|this|it)\s*(is|'s)?\s*(wrong|incorrect|not right|not correct|not it)\b"
    r"|\b(that|this|it)\s+(didn't|did not|doesn't|does not|isn't|is not)\s+(work|right|correct|what)"
    r"|\bnot what i (asked|wanted|meant|said)"
    r"|\byou (forgot|missed|ignored|skipped|misunderstood|got it wrong|didn't|did not|messed)"
    r"|\b(still|again) (wrong|broken|failing|not working)"
    r"|\b(that|it) failed\b|\btry again\b|\bwrong (file|person|date|time|email|address|one|answer|account)\b)",
    re.IGNORECASE,
)


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def ensure_sections(body: str) -> str:
    out = body.strip()
    for sec in SECTIONS:
        if not re.search(rf"^#+\s*{re.escape(sec)}\b", out, flags=re.IGNORECASE | re.MULTILINE):
            out += f"\n\n## {sec}\n- (none noted yet)"
    return out


def _count(value: Any) -> int:
    """``tool_errors`` may arrive as a number, a bool or a list of errors."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return max(0, int(value))
    if isinstance(value, list | tuple | set | dict):
        return len(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _names(value: Any) -> list[str]:
    """``skills_viewed`` as unique lowercase names (accepts a string, names or dicts with ``name``)."""
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for v in value if isinstance(value, list | tuple | set) else []:
        name = str(v.get("name") if isinstance(v, dict) else v or "").strip().lower()
        if name and name not in out:
            out.append(name)
    return out


def _error_lines(value: Any, limit: int = 5) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    lines = []
    for e in list(value)[:limit]:
        if isinstance(e, dict):
            label = e.get("name") or e.get("tool") or e.get("tool_name") or "tool"
            detail = e.get("error") or e.get("result") or e.get("message") or ""
            lines.append(f"{label}: {str(detail)[:300]}")
        else:
            lines.append(str(e)[:300])
    return lines


def looks_like_correction(text: str) -> bool:
    return bool(_CORRECTION_RE.search((text or "").replace(chr(0x2019), "'")[:600]))


class EvolutionService(Service):
    name = "evolution"
    model_kind = "skills"
    pause_on_stop = True  # Stop everything pauses reviews, the curator and profile updates until resume

    def __init__(self, app):
        super().__init__(app)
        self._dirty_sessions: dict[str, dict[str, Any]] = {}
        self._task_runs: dict[tuple[str, str | None], datetime] = {}
        self._review_lock = asyncio.Lock()
        self._repair_lock = asyncio.Lock()
        self._last_skill_turn: dict[str, dict[str, Any]] = {}  # session_id -> last turn that viewed skills
        self._repairing: set[str] = set()
        self._background: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        app = self.app
        app.skills.store = app.store
        if app.memory is not None:
            app.memory.bus = app.bus
        with contextlib.suppress(Exception):
            await app.skills.sync_stats(app.store)
        if not app.enable_background:
            return
        self._loops.append(asyncio.create_task(self._listen(), name="evolution:bus"))
        self.run_every(60, self._tick, name="tick", initial_delay=60)

    async def stop(self) -> None:
        for t in list(self._background):
            t.cancel()
        await super().stop()

    async def halt(self) -> int:
        return await super().halt() + await cancel_tasks(self._background)

    async def _listen(self) -> None:
        async with self.app.bus.subscribe() as q:
            while True:
                event = await q.get()
                try:
                    self.on_event(event)
                except Exception:
                    log.exception("evolution failed to handle %s", event.get("type"))
                if event.get("type") in {"chat.turn_completed", "task.run_finished"} and not self.app.stopped:
                    # model calls must not hold up the bus queue; the repair lock keeps them in order
                    task = asyncio.create_task(self.handle_repair_event(event), name="evolution:repair")
                    self._background.add(task)
                    task.add_done_callback(self._background.discard)

    def on_event(self, event: dict) -> None:
        """Bus hook: remember chats that used tools and task runs that completed."""
        kind, data = event.get("type"), event.get("data") or {}
        if not isinstance(data, dict):
            return
        if kind == "chat.turn_completed" and data.get("session_id"):
            entry = self._dirty_sessions.setdefault(data["session_id"], {"tool_calls": 0})
            entry["tool_calls"] += _count(data.get("tool_calls"))
            entry["last_at"] = datetime.now(UTC)
        elif kind == "task.run_finished" and data.get("status") in DONE_STATUSES and data.get("task_id"):
            self._task_runs.setdefault((data["task_id"], data.get("run_id")), datetime.now(UTC))
        elif kind == "task.updated" and data.get("status") in DONE_STATUSES and data.get("task_id"):
            runs = data.get("runs") or []
            run_id = next((r.get("run_id") for r in reversed(runs)
                           if r.get("status") in DONE_STATUSES | {"completed"} and not r.get("dry_run")), None)
            self._task_runs.setdefault((data["task_id"], run_id), datetime.now(UTC))

    async def _tick(self) -> None:
        now = datetime.now(UTC)
        cfg = self.app.config
        if cfg.evolution.review_enabled:
            await self.review_due(now)
        if self.app.memory is not None:
            if await self._due("memory.purge_last_run", timedelta(minutes=cfg.memory.purge_interval_minutes), now):
                try:
                    await self.app.memory.purge_expired()
                finally:
                    # A purge failure must not block the pending-erase retries (#462):
                    # the drain still runs — its own failure is logged rather than
                    # raised, so review expiry still happens — then the purge error
                    # keeps propagating.
                    try:
                        await self.app.memory.drain_pending_erases()
                    except Exception:
                        log.exception("pending-erase retry failed during upkeep")
                    await memory_review.expire(self.app, now)
            if cfg.memory.summaries_enabled and await self._due(
                "memory.summaries_last_run", timedelta(minutes=cfg.memory.summarize_interval_minutes), now
            ):
                await self.summarize_tick(now)
        if cfg.evolution.curator_enabled and await self._due(
            "evolution.curator_last_run", timedelta(hours=cfg.evolution.curator_interval_hours), now, mark=False
        ):
            await self.run_curator(now)
        if cfg.evolution.user_profile_updates and await self._due(
            "evolution.profile_last_run", timedelta(hours=cfg.evolution.profile_update_hours), now, mark=False
        ):
            await self.update_profile(now)

    async def _due(self, key: str, every: timedelta, now: datetime, *, mark: bool = True) -> bool:
        last = _parse_ts(await self.app.store.get_meta(key))
        if last is not None and now - last < every:
            return False
        if mark:
            await self.app.store.set_meta(key, now.isoformat())
        return True

    # ------------------------------------------------------------------ memory jobs
    async def summarize_tick(self, now: datetime | None = None) -> list[dict]:
        mem = self.app.memory
        if mem is None:
            return []
        created = await mem.episodic.summarize_pending(user_name=self.app.config.assistant.user_name, now=now)
        for s in created:
            await log_event(
                self.app.store, "summary_created",
                {"id": s["id"], "session_id": s["session_id"], "start_at": s["start_at"], "end_at": s["end_at"]},
            )
        return created

    # ------------------------------------------------------------------ reviewer
    async def sessions_due(self, now: datetime, idle_minutes: int | None = None) -> list[str]:
        cfg = self.app.config.evolution
        idle = cfg.review_idle_minutes if idle_minutes is None else idle_minutes
        cutoff = (now - timedelta(minutes=idle)).isoformat()
        lookback = (now - timedelta(days=REVIEW_LOOKBACK_DAYS)).isoformat()
        # "<=": a message saved in the same clock tick as the last one reviewed still counts (#297, #236)
        rows = await self.app.store.fetchall(
            "SELECT s.id, s.updated_at, r.activity_at, r.activity_mark FROM sessions s"
            " LEFT JOIN evolution_reviews r ON r.target = 'session:' || s.id"
            " WHERE s.updated_at <= ? AND s.updated_at >= ? AND (r.activity_at IS NULL OR r.activity_at <= s.updated_at)"
            " AND s.incognito = 0"  # skills are never learned from an incognito chat (#154)
            " ORDER BY s.updated_at",
            (cutoff, lookback),
        )
        due = []
        for r in rows:
            after, params = after_watermark(r["activity_mark"] or r["activity_at"] or "", "created_at", "id")
            n = await self.app.store.fetchone(
                f"SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND role = 'tool' AND {after}",
                (r["id"], *params),
            )
            if n and int(n["n"]) >= cfg.min_tool_calls_for_review:
                due.append(r["id"])
        return due

    async def review_due(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now(UTC)
        proposed: list[str] = []
        for sid in await self.sessions_due(now):
            self._dirty_sessions.pop(sid, None)
            res = await self.review_session(sid)
            if res.get("skill"):
                proposed.append(res["skill"])
        for key in list(self._task_runs):
            self._task_runs.pop(key, None)
            res = await self.review_task_run(*key)
            if res.get("skill"):
                proposed.append(res["skill"])
        return proposed

    async def review_now(self, session_id: str | None = None) -> dict:
        now = datetime.now(UTC)
        reviewed, proposed = 0, []
        if session_id:
            targets = [session_id]
        else:
            targets = await self.sessions_due(now, idle_minutes=0)
        for sid in targets:
            res = await self.review_session(sid, force=bool(session_id))
            reviewed += int(res.get("reviewed", False))
            if res.get("skill"):
                proposed.append(res["skill"])
        if not session_id:
            for key in list(self._task_runs):
                self._task_runs.pop(key, None)
                res = await self.review_task_run(*key)
                reviewed += int(res.get("reviewed", False))
                if res.get("skill"):
                    proposed.append(res["skill"])
        return {"reviewed": reviewed, "proposed": proposed}

    @staticmethod
    def _transcript_from_messages(rows: list[dict]) -> tuple[str, int]:
        lines: list[str] = []
        tool_calls = 0
        for m in rows:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role == "user":
                lines.append(f"USER: {content[:1500]}")
            elif role == "assistant":
                calls = m.get("tool_calls")
                if isinstance(calls, str):
                    try:
                        calls = json.loads(calls)
                    except json.JSONDecodeError:
                        calls = []
                if content:
                    lines.append(f"ASSISTANT: {content[:1500]}")
                for c in calls or []:
                    fn = c.get("function", {}) if isinstance(c, dict) else {}
                    lines.append(f"TOOL CALL {fn.get('name')}: {str(fn.get('arguments'))[:400]}")
            elif role == "tool":
                tool_calls += 1
                lines.append(f"TOOL RESULT {m.get('name') or ''}: {content[:500]}")
        text = "\n".join(lines)
        if len(text) > TRANSCRIPT_BUDGET:
            half = TRANSCRIPT_BUDGET // 2
            text = text[:half] + "\n[... middle of the transcript omitted ...]\n" + text[-half:]
        return text, tool_calls

    async def _record_review(
        self, target: str, activity_at: str | None, tool_calls: int, decision: str, skill: str | None,
        mark: str | None = None,
    ) -> None:
        await self.app.store.execute(
            "INSERT INTO evolution_reviews(target, reviewed_at, activity_at, activity_mark, tool_calls, decision, skill)"
            " VALUES(?,?,?,?,?,?,?) ON CONFLICT(target) DO UPDATE SET reviewed_at = excluded.reviewed_at,"
            " activity_at = excluded.activity_at, activity_mark = excluded.activity_mark,"
            " tool_calls = excluded.tool_calls, decision = excluded.decision, skill = COALESCE(excluded.skill, skill)",
            (target, now_iso(), activity_at, mark, tool_calls, decision, skill),
        )

    async def review_session(self, session_id: str, *, force: bool = False) -> dict:
        store = self.app.store
        if await store.is_incognito(session_id):  # never learned from (#154)
            return {"reviewed": False, "decision": "skipped"}
        target = f"session:{session_id}"
        prev = await store.fetchone(
            "SELECT activity_at, activity_mark FROM evolution_reviews WHERE target = ?", (target,)
        )
        # what the last review read: messages saved in the same clock tick as its last one are still new (#297)
        since = (prev["activity_mark"] or prev["activity_at"] if prev else None) or ""
        after, params = after_watermark(since, "created_at", "id")
        rows = [
            dict(r)
            for r in await store.fetchall(
                "SELECT id, role, content, tool_calls, name, created_at FROM messages"
                f" WHERE session_id = ? AND {after} ORDER BY created_at, rowid",
                (session_id, *params),
            )
        ]
        if force and not rows:
            rows = [
                dict(r)
                for r in await store.fetchall(
                    "SELECT id, role, content, tool_calls, name, created_at FROM messages WHERE session_id = ?"
                    " ORDER BY created_at, rowid",
                    (session_id,),
                )
            ]
        if not rows:
            return {"reviewed": False, "decision": "skipped"}
        transcript, tool_calls = self._transcript_from_messages(rows)
        activity_at = rows[-1]["created_at"]
        mark = next_watermark(since, rows, "created_at", "id")
        if tool_calls < self.app.config.evolution.min_tool_calls_for_review:
            await self._record_review(target, activity_at, tool_calls, "skipped", None, mark)
            return {"reviewed": False, "decision": "skipped"}
        async with self._review_lock:
            decision, skill = await self._review(transcript, label="Conversation transcript", origin={"session_id": session_id})
        await self._record_review(target, activity_at, tool_calls, decision, skill, mark)
        return {"reviewed": True, "decision": decision, "skill": skill}

    async def _task_run_log(self, task_id: str, run_id: str | None) -> tuple[str, int, str | None, str | None]:
        """(transcript, tool_calls, run_id, finished_at) from the tasks service, or its tables as a fallback."""
        tasks = self.app.tasks
        task = None
        for getter in ("get_task", "get"):
            if hasattr(tasks, getter):
                with contextlib.suppress(Exception):
                    task = await _maybe_await(getattr(tasks, getter)(task_id))
                if task:
                    break
        if isinstance(task, dict):
            runs = task.get("runs") or []
            run = next((r for r in runs if r.get("run_id") == run_id), None) if run_id else (runs[-1] if runs else None)
            if run:
                lines = [f"TASK: {task.get('name')}: {(task.get('description') or '')[:800]}"]
                calls = 0
                for u in run.get("progress_updates") or []:
                    msg = u.get("message") or {}
                    t = msg.get("type")
                    if t == "tool_call":
                        calls += 1
                        lines.append(f"TOOL CALL {msg.get('tool_name')}: {json.dumps(msg.get('parameters'), default=str)[:400]}")
                    elif t == "tool_result":
                        lines.append(f"TOOL RESULT {msg.get('tool_name') or ''}: {str(msg.get('result'))[:500]}")
                    elif msg.get("content"):
                        lines.append(f"{str(t).upper()}: {str(msg.get('content'))[:800]}")
                summary = (run.get("result") or {}).get("summary")
                if summary:
                    lines.append(f"RESULT: {summary[:1500]}")
                text = "\n".join(lines)
                if calls == 0 and run.get("messages"):
                    text, calls = self._transcript_from_messages(run["messages"])
                return text[-TRANSCRIPT_BUDGET:], calls, run.get("run_id"), run.get("finished_at")
        # fallback: the tasks package's checkpointed transcript
        with contextlib.suppress(Exception):
            if run_id:
                row = await self.app.store.fetchone("SELECT id, messages, finished_at FROM task_runs WHERE id = ?", (run_id,))
            else:
                row = await self.app.store.fetchone(
                    "SELECT id, messages, finished_at FROM task_runs WHERE task_id = ? ORDER BY created_at DESC LIMIT 1",
                    (task_id,),
                )
            if row and row["messages"]:
                text, calls = self._transcript_from_messages(json.loads(row["messages"]))
                return text, calls, row["id"], row["finished_at"]
        return "", 0, run_id, None

    async def review_task_run(self, task_id: str, run_id: str | None = None) -> dict:
        text, calls, run_id, finished = await self._task_run_log(task_id, run_id)
        target = f"run:{task_id}:{run_id or 'latest'}"
        done = await self.app.store.fetchone("SELECT decision FROM evolution_reviews WHERE target = ?", (target,))
        if done or not text:
            return {"reviewed": False, "decision": "skipped"}
        if calls < self.app.config.evolution.min_tool_calls_for_review:
            await self._record_review(target, finished, calls, "skipped", None)
            return {"reviewed": False, "decision": "skipped"}
        async with self._review_lock:
            decision, skill = await self._review(text, label="Task run log", origin={"task_id": task_id, "run_id": run_id})
        await self._record_review(target, finished, calls, decision, skill)
        return {"reviewed": True, "decision": decision, "skill": skill}

    async def _review(self, transcript: str, *, label: str, origin: dict) -> tuple[str, str | None]:
        lib = self.app.skills
        skills = "\n".join(f"- {s.name}: {s.description}" for s in lib.list()) or "(none)"
        pending = "\n".join(f"- {s.name}: {s.description}" for s in lib.list_pending()) or "(none)"
        try:
            raw = await self.app.llm.complete_json(
                "fast",
                [
                    {"role": "system", "content": prompts.REVIEW_SYSTEM},
                    {
                        "role": "user",
                        "content": prompts.REVIEW_USER.format(
                            skills=skills, pending=pending, label=label, transcript=transcript
                        ),
                    },
                ],
            )
        except Exception as exc:
            log.warning("skill review failed: %s", exc)
            return "error", None
        if not isinstance(raw, dict):
            return "none", None
        name = await self.propose_skill(raw, origin=origin)
        if name is None:
            return "none", None
        return ("patch" if lib.get_active_file(name) else "create"), name

    async def propose_skill(self, raw: dict, *, origin: dict) -> str | None:
        """Validate a reviewer/curator answer and stage it in skills/pending. Returns the skill name."""
        lib = self.app.skills
        decision = str(raw.get("decision") or raw.get("action") or "none").strip().lower()
        if decision not in {"create", "patch"}:
            return None
        name = str(raw.get("name") or "").strip().lower()
        if not valid_name(name):
            name = slugify(name)
        body = str(raw.get("body") or "").strip()
        description = " ".join(str(raw.get("description") or "").split())
        if not valid_name(name) or len(body) < 80 or "procedure" not in body.lower() or not description:
            log.info("reviewer proposal rejected as incomplete: %r", raw.get("name"))
            return None
        existing = lib.get_active_file(name)
        if decision == "patch" and existing is None:
            decision = "create"
        if decision == "create" and existing is not None:
            decision = "patch"
        plugin_ids = {p.id for p in self.app.registry.plugins()}
        requires = [str(t) for t in raw.get("requires_tools") or [] if str(t) in plugin_ids]
        tags = [str(t) for t in raw.get("tags") or []][:8]
        validate = skill_checks.merge(existing.validate if existing else [], skill_checks.parse(raw.get("validate"))[0])
        body = ensure_sections(body)
        report = await self.check_skill(name, validate, origin=origin, description=description, body=body)
        if skill_checks.failure(report):
            return None  # a skill whose own check fails isn't proposed (#386)
        lib.write(
            name, description, body,
            author=existing.author if existing else "assistant",
            tags=tags or (existing.tags if existing else []),
            requires_tools=requires or (existing.requires_tools if existing else []),
            pending=True, created_by_review=True, validate=validate,
        )
        await lib.record_patch(name, "pending_review", self.app.store)
        await lib.record_check(name, report, self.app.store, pending=True)
        user_edited = existing is not None and (await lib.edit_state(name, self.app.store))["user_edited"]
        kind = "skill_patched" if existing else "skill_created"
        detail = {"name": name, "pending": True, "reason": raw.get("reason"), **origin}
        if user_edited:
            detail["user_edited"] = True
        await log_event(self.app.store, kind, detail)
        self.app.bus.publish("skill.updated", {"name": name, "state": "pending_review"})
        if user_edited:
            msg = (
                f"You changed **{name}** yourself, so I left it as it is and drafted an update for you to compare. "
                "Review the change in Skills."
            )
            title = "Skill update to review"
        elif existing:
            msg = f"I found a better way to do **{name}** and drafted an update. Review the change in Skills."
            title = "Skill update to review"
        else:
            msg = f"I learned a repeatable procedure and saved it as **{name}**: {description} Review it in Skills."
            title = "New skill to review"
        await self.app.notify(
            "skill", msg, title=title, payload={"skill": name, "action": decision, "origin": origin}
        )
        return name

    async def check_skill(
        self, name: str, validate: list[dict], *, origin: dict, description: str, body: str
    ) -> dict | None:
        """Run a drafted skill's checks before it is proposed (#386): its file checks now, in the active skill's folder
        and Sentient's files folder. This is work Sentient started itself, so a code check never runs here (ADR 0017):
        it waits for the user's approval. A failure is logged as ``skill_check_failed`` and the draft is dropped."""
        if not validate:
            return None
        current = self.app.skills.get_active_file(name)
        draft = render_skill(name, description, body, author="assistant", version=1, tags=[], requires_tools=[],
                             validate=validate)  # what a check of SKILL.md itself reads
        report = await skill_checks.run(self.app, name, validate, [current.dir] if current else [], run_code=False,
                                        candidate=draft)
        if reason := skill_checks.failure(report):
            log.info("skill %s not proposed: its check failed (%s)", name, reason)
            clean = {k: v for k, v in origin.items() if v is not None}
            await log_event(self.app.store, "skill_check_failed", {"name": name, "reason": reason, **clean})
        return report

    # ------------------------------------------------------------------ learning from docs (#167)
    async def learn_skill(self, source: str, *, via: str = "app", transport: Any = None) -> dict:
        """``/learn <link|file|folder>``: draft a skill from docs or a written workflow and stage it in pending.

        A short source is drafted from in full. A long one becomes a knowledge skill: its text is kept in
        ``references/`` files that ``skill_view`` loads only when the skill is used, and the model drafts the steps
        from an outline. What was read is outside content (ADR 0018): the model sees it marked as data, and the draft
        waits with the safety check of skills from outside (``skill_imports`` source ``learn``, ADR 0027), so a
        high-risk finding makes declining the default and its code never runs. Raises ``LearnError`` (a plain
        sentence) when the source can't be read or no skill could be drafted. Returns ``{name, description, verdict,
        references, kind, where}``."""
        from sentient.agent.prompt import mark_outside
        from sentient.skills import learn, vet

        lib = self.app.skills
        src = await learn.collect(source, transport=transport)
        refs = learn.references(src) if src.long else []
        if refs:
            content = learn.outline(refs)
            mode = prompts.LEARN_OUTLINE
        else:
            content = "\n\n".join(f"## {d.title}\n{d.text.strip()}" for d in src.docs)[: learn.SMALL_SOURCE]
            mode = prompts.LEARN_WHOLE
        try:
            raw = await self.app.llm.complete_json(
                "fast",
                [
                    {"role": "system", "content": prompts.LEARN_SYSTEM.format(mode=mode)},
                    {"role": "user", "content": prompts.LEARN_USER.format(
                        where=src.where, content=mark_outside(content, src.label))},
                ],
            )
        except Exception as exc:
            log.warning("drafting a skill from %s failed: %s", src.label, exc)
            raise learn.LearnError(f"Sentient couldn't draft the skill: {str(exc).strip() or type(exc).__name__}") from exc
        raw = raw if isinstance(raw, dict) else {}
        body = str(raw.get("body") or "").strip()
        description = " ".join(str(raw.get("description") or "").split())[:300]
        if len(body) < 80 or "procedure" not in body.lower() or not description:
            log.info("learned skill draft from %s was incomplete: %r", src.label, raw.get("name"))
            raise learn.LearnError("Sentient read it but couldn't turn it into clear steps. Try again, or point it at "
                                   "the part that describes the workflow.")
        base = slugify(str(raw.get("name") or ""))
        if not valid_name(base):
            base = slugify(src.docs[0].title) if valid_name(slugify(src.docs[0].title)) else "learned-skill"
        name, n = base, 1
        while lib.get_any(name) or (lib.pending_dir / name).exists():
            n += 1
            name = f"{base[:58]}-{n}"
        body = ensure_sections(body)
        if refs:
            body = f"{body.rstrip()}\n\n{learn.reference_section(refs)}"
        # how to check it: a knowledge skill's reference files are all there and intact (#386). Sentient drafts this
        # itself; checks a model drafts from outside text aren't kept (their code would never run anyway, ADR 0027)
        validate = [{"file": r.path, "expect": f"# {r.text.splitlines()[0].lstrip('# ').strip()}"}
                    for r in refs[: skill_checks.MAX_CHECKS]]
        tags = [str(t).strip().lower() for t in raw.get("tags") or [] if str(t).strip()]
        tags = [*[t for t in tags if t not in PLACEHOLDER_TAGS][:7], "learned"]  # small models copy the example
        path = lib.write(name, description, body, author="assistant", tags=tags, pending=True, validate=validate)
        folder = path.parent
        try:
            for r in refs:
                target = folder / r.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(r.text, encoding="utf-8")
            report = await skill_checks.run(self.app, name, validate, [folder], run_code=False, outside=True)
            vetting = vet.vet_folder(folder, source="learn", origin=src.where)
            await vet.record_import(self.app.store, name, vetting)
        except BaseException:
            # never leave a draft without its safety record: approving it would skip the check of skills from outside
            shutil.rmtree(folder, ignore_errors=True)
            with contextlib.suppress(Exception):
                await vet.forget(self.app.store, name)
            raise
        await lib.record_patch(name, "pending_review", self.app.store)
        await lib.record_check(name, report, self.app.store, pending=True)
        reason = f"Learned from {src.where}"
        if src.skipped:
            reason += f" ({src.skipped} file{'s were' if src.skipped != 1 else ' was'} left out)"
        await log_event(self.app.store, "skill_learned", {
            "name": name, "pending": True, "reason": reason, "source": src.kind, "origin": src.where,
            "verdict": vetting["verdict"], "references": [r.path for r in refs], "via": via,
        })
        log.info("learned the skill %s from %s (%s, %d reference files, safety check: %s)",
                 name, src.label, src.kind, len(refs), vetting["verdict"])
        self.app.bus.publish("skill.updated", {"name": name, "state": "pending_review"})
        risky = " The safety check found high-risk problems in it, so read it carefully." if vetting["verdict"] == "high" else ""
        await self.app.notify(
            "skill",
            f"I read {src.label} and drafted the skill **{name}**: {description}{risky} Review it in Skills.",
            title="New skill to review",
            payload={"skill": name, "action": "create", "origin": {"learned": src.where}},
        )
        return {"name": name, "description": description, "verdict": vetting["verdict"],
                "references": [r.path for r in refs], "kind": src.kind, "where": src.where}

    # ------------------------------------------------------------------ repair during use
    async def handle_repair_event(self, event: dict) -> list[str]:
        """Record skill outcomes for a finished chat turn or task run and propose fixes. Returns proposed names."""
        kind, data = event.get("type"), event.get("data")
        if not isinstance(data, dict):
            return []
        async with self._repair_lock:
            try:
                if kind == "chat.turn_completed":
                    return await self._on_turn(data)
                if kind == "task.run_finished":
                    return await self._on_run(data)
            except Exception:
                log.exception("skill repair failed for %s", kind)
        return []

    async def _on_turn(self, data: dict) -> list[str]:
        sid = data.get("session_id")
        if not sid:
            return []
        lib, store = self.app.skills, self.app.store
        if await store.is_incognito(sid):  # an incognito chat never shapes skills (#154)
            return []
        proposed: list[str] = []
        prev = self._last_skill_turn.pop(sid, None)
        user_text = str(data.get("user_text") or "")
        if prev and user_text and prev.get("turn_id") != data.get("turn_id"):
            what = await self.detect_correction(str(prev.get("reply") or ""), user_text)
            if what is not None:
                excerpt = await self._session_excerpt(sid)
                for name in prev["skills"]:
                    await lib.record_outcome(name, False, store, revert_success=True)
                    done = await self.repair_skill(
                        name, reason=f"You corrected the result: {what}",
                        failure=f"The user corrected the reply on the next message.\nUser said: {user_text[:600]}",
                        transcript=excerpt, origin={"session_id": sid, "turn_id": prev.get("turn_id")},
                        failure_kind="user_correction",
                    )
                    if done:
                        proposed.append(done)
        skills = _names(data.get("skills_viewed"))
        if not skills:
            return proposed
        errors = _count(data.get("tool_errors"))
        if errors:
            excerpt = await self._session_excerpt(sid)
            lines = _error_lines(data.get("tool_errors"))
            for name in skills:
                await lib.record_outcome(name, False, store)
                done = await self.repair_skill(
                    name, reason=f"{errors} tool call{'s' if errors != 1 else ''} failed while following it"
                    + (f" ({lines[0][:120]})" if lines else ""),
                    failure="Tool errors in this turn:\n" + ("\n".join(f"- {x}" for x in lines) or f"- {errors} errors"),
                    transcript=excerpt, origin={"session_id": sid, "turn_id": data.get("turn_id")},
                    failure_kind="tool_errors",
                )
                if done and done not in proposed:
                    proposed.append(done)
            return proposed
        for name in skills:
            await lib.record_outcome(name, True, store)
        self._last_skill_turn[sid] = {
            "turn_id": data.get("turn_id"), "skills": skills, "reply": str(data.get("reply") or "")[:1500],
        }
        while len(self._last_skill_turn) > MAX_TRACKED_TURNS:
            self._last_skill_turn.pop(next(iter(self._last_skill_turn)))
        return proposed

    async def _on_run(self, data: dict) -> list[str]:
        task_id = data.get("task_id")
        skills = _names(data.get("skills_viewed"))
        status = str(data.get("status") or "").strip().lower()
        if not task_id or not skills or status in {"cancelled", "running", ""}:
            return []
        errors = _count(data.get("tool_errors"))
        failed = status in FAILED_RUN_STATUSES or errors > 0
        lib, store = self.app.skills, self.app.store
        for name in skills:
            await lib.record_outcome(name, not failed, store)
        if not failed:
            return []
        run_id = data.get("run_id")
        if status in {"error", "failed"}:
            reason, kind = "the task run failed", "run_failed"
        elif errors:
            reason, kind = f"{errors} tool call{'s' if errors != 1 else ''} failed during the task run", "tool_errors"
        else:
            reason, kind = "the task run finished with errors", "run_failed"
        lines = _error_lines(data.get("tool_errors"))
        failure = f"Task run status: {status}."
        if data.get("error"):
            failure += f"\nRun error: {str(data['error'])[:500]}"
        if lines:
            failure += "\nTool errors:\n" + "\n".join(f"- {x}" for x in lines)
        text, _calls, run_id, _finished = await self._task_run_log(task_id, run_id)
        proposed = []
        for name in skills:
            done = await self.repair_skill(
                name, reason=reason, failure=failure, transcript=text[-REPAIR_TRANSCRIPT_BUDGET:],
                origin={"task_id": task_id, "run_id": run_id}, failure_kind=kind,
            )
            if done:
                proposed.append(done)
        return proposed

    async def detect_correction(self, reply: str, user_text: str) -> str | None:
        """What went wrong when ``user_text`` corrects the previous reply, else None (regex, then one fast call)."""
        if not looks_like_correction(user_text):
            return None
        try:
            raw = await self.app.llm.complete_json(
                "fast",
                [
                    {"role": "system", "content": prompts.CORRECTION_SYSTEM},
                    {"role": "user", "content": prompts.CORRECTION_USER.format(reply=reply[-1500:] or "(unknown)", user_text=user_text[:1000])},
                ],
            )
        except Exception as exc:
            log.debug("correction check failed: %s", exc)
            return None
        if not isinstance(raw, dict):
            return None
        flag = raw.get("correction", raw.get("is_correction"))
        if flag is True or str(flag).strip().lower() in {"true", "yes"}:
            what = " ".join(str(raw.get("what_went_wrong") or raw.get("reason") or "").split())
            return (what or " ".join(user_text.split()))[:200]
        return None

    async def _session_excerpt(self, session_id: str) -> str:
        rows = await self.app.store.fetchall(
            "SELECT role, content, tool_calls, name, created_at FROM messages WHERE session_id = ?"
            " ORDER BY created_at DESC, rowid DESC LIMIT 24",
            (session_id,),
        )
        text, _ = self._transcript_from_messages([dict(r) for r in reversed(rows)])
        return text[-REPAIR_TRANSCRIPT_BUDGET:]

    async def _repair_cooling_down(self, name: str, now: datetime) -> bool:
        hours = self.app.config.evolution.repair_cooldown_hours
        if hours <= 0:
            return False
        rows = await self.app.store.fetchall(
            "SELECT ts, detail FROM evolution_log WHERE kind = 'skill_repair_proposed' AND ts >= ? ORDER BY ts DESC",
            ((now - timedelta(hours=hours)).isoformat(),),
        )
        for r in rows:
            with contextlib.suppress(Exception):
                if json.loads(r["detail"] or "{}").get("name") == name:
                    return True
        return False

    async def repair_skill(
        self, name: str, *, reason: str, failure: str, transcript: str, origin: dict, failure_kind: str,
    ) -> str | None:
        """Ask the fast model for a fix and stage it as a pending proposal. Never activates anything."""
        if not self.app.config.evolution.skill_repair:
            return None
        lib = self.app.skills
        name = name.strip().lower()
        current = lib.get_active_file(name)
        if current is None:  # only skills in the writable library can be patched (and diffed)
            return None
        if lib.get_pending(name) is not None or name in self._repairing:
            log.info("skill %s already has a proposal waiting for review; not proposing a repair", name)
            return None
        now = datetime.now(UTC)
        if await self._repair_cooling_down(name, now):
            return None
        user_edited = (await lib.edit_state(name, self.app.store))["user_edited"]
        skill_text = f"description: {current.description}\n\n{current.body}"[:6000]
        if user_edited:  # small models keep the user's wording when told plainly
            skill_text += "\n\n(The user edited this skill by hand. Keep their changes; change only what fixes the problem.)"
        self._repairing.add(name)
        try:
            raw = await self.app.llm.complete_json(
                "fast",
                [
                    {"role": "system", "content": prompts.REPAIR_SYSTEM},
                    {
                        "role": "user",
                        "content": prompts.REPAIR_USER.format(
                            name=name, skill=skill_text,
                            failure=failure[:2000], transcript=transcript or "(not available)",
                        ),
                    },
                ],
            )
        except Exception as exc:
            log.warning("skill repair for %s failed: %s", name, exc)
            return None
        finally:
            self._repairing.discard(name)
        if not isinstance(raw, dict):
            return None
        decision = str(raw.get("decision") or raw.get("action") or ("patch" if raw.get("body") else "none")).strip().lower()
        body = str(raw.get("body") or "").strip()
        if decision not in {"patch", "repair", "update"} or len(body) < 80:
            log.info("no usable repair for %s (%s: %s)", name, decision, str(raw.get("reason") or "")[:200])
            return None
        body = ensure_sections(body)  # adds any missing standard section; a fix need not repeat the headings
        if " ".join(body.split()) == " ".join(current.body.split()):
            return None
        description = " ".join(str(raw.get("description") or "").split()) or current.description
        fix = " ".join(str(raw.get("reason") or "").split())
        full_reason = (reason.rstrip(". ") + (f". {fix}" if fix else ""))[:400]
        validate = skill_checks.merge(current.validate, skill_checks.parse(raw.get("validate"))[0])
        report = await self.check_skill(name, validate, origin={"repair": True, **origin}, description=description,
                                        body=body)
        if skill_checks.failure(report):
            return None  # the fix doesn't pass the skill's own check (#386)
        lib.write(
            name, description, body, author=current.author, tags=current.tags,
            requires_tools=current.requires_tools, pending=True, created_by_review=current.created_by_review,
            validate=validate,
        )
        await lib.record_patch(name, "pending_review", self.app.store)
        await lib.record_check(name, report, self.app.store, pending=True)
        clean_origin = {k: v for k, v in origin.items() if v is not None}
        detail = {"name": name, "pending": True, "origin": "repair", "failure": failure_kind, "reason": full_reason, **clean_origin}
        if user_edited:
            detail["user_edited"] = True
        await log_event(self.app.store, "skill_repair_proposed", detail)
        self.app.bus.publish("skill.updated", {"name": name, "state": "pending_review"})
        kept = " You edited it yourself, so I left your version in place." if user_edited else ""
        await self.app.notify(
            "skill",
            f"**{name}** didn't work as planned: {reason.rstrip('.')}.{kept} I drafted a fix. Review the change in Skills.",
            title="Skill fix to review",
            payload={"skill": name, "action": "patch", "origin": "repair", "reason": full_reason, **clean_origin},
        )
        return name

    # ------------------------------------------------------------------ curator
    async def run_curator(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(UTC)
        cfg = self.app.config.evolution
        lib = self.app.skills
        store = self.app.store
        lib.reload()
        await lib.sync_stats(store)
        stats = await lib.stats(store)
        staled: list[str] = []
        archived: list[str] = []
        for skill in lib._list_dir(lib.root):
            st = stats.get(skill.name) or {}
            last_used = _parse_ts(st.get("last_used_at"))
            ref = last_used or _parse_ts(st.get("created_at"))
            if ref is None:
                ref = datetime.fromtimestamp(skill.path.stat().st_mtime, tz=UTC)
            idle_days = (now - ref).total_seconds() / 86400
            state = st.get("state") or "active"
            # skills the user wrote and has actually used get twice the grace period before archiving
            recently_used_by_user = (
                skill.author == "user" and last_used is not None and (now - last_used).days < 2 * cfg.archive_after_days
            )
            if state == "stale" and idle_days >= cfg.archive_after_days and not recently_used_by_user:
                lib.archive(skill.name)
                await lib.set_state(skill.name, "archived", store)
                archived.append(skill.name)
                await log_event(store, "skill_archived", {"name": skill.name, "by": "curator", "idle_days": int(idle_days)})
                self.app.bus.publish("skill.updated", {"name": skill.name, "state": "archived"})
            elif state == "active" and idle_days >= cfg.stale_after_days:
                await lib.set_state(skill.name, "stale", store)
                staled.append(skill.name)
                self.app.bus.publish("skill.updated", {"name": skill.name, "state": "stale"})
        merges: list[str] = []
        if cfg.curator_merge_suggestions:
            with contextlib.suppress(Exception):
                merges = await self._merge_suggestions(stats)
        lib.reload()
        with contextlib.suppress(Exception):
            self.app.skills.reload(self.app._available_tools())
        await store.set_meta("evolution.curator_last_run", now.isoformat())
        detail = {"staled": staled, "archived": archived, "merge_proposals": merges}
        await log_event(store, "curator_run", detail, ts=now.isoformat())
        return detail

    async def _merge_suggestions(self, stats: dict, threshold: float = 0.9, limit: int = 1) -> list[str]:
        lib = self.app.skills
        pending = {s.name for s in lib.list_pending()}
        skills = [s for s in lib._list_dir(lib.root) if s.name not in pending]
        if len(skills) < 2:
            return []
        vecs = await self.app.llm.embed([f"{s.name.replace('-', ' ')}: {s.description}" for s in skills])
        pairs = []
        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                sim = cosine(vecs[i], vecs[j])
                if sim >= threshold:
                    pairs.append((sim, skills[i], skills[j]))
        proposed: list[str] = []
        used: set[str] = set()
        for _sim, a, b in sorted(pairs, key=lambda p: -p[0]):
            if len(proposed) >= limit or a.name in used or b.name in used:
                continue
            # keep the more reliable skill (recorded successes vs failures), then the more used one
            def rank(s, _stats=stats):
                st = _stats.get(s.name) or {}
                return (lib.reliability(st), int(st.get("use_count") or 0))

            keep, other = (a, b) if rank(a) >= rank(b) else (b, a)
            raw = await self.app.llm.complete_json(
                "fast",
                [
                    {"role": "system", "content": prompts.MERGE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Skill A ({keep.name}): {keep.description}\n{keep.body}\n\n"
                        f"Skill B ({other.name}): {other.description}\n{other.body}",
                    },
                ],
            )
            if not isinstance(raw, dict):
                continue
            raw = {**raw, "decision": "patch", "name": keep.name, "reason": f"merge of {keep.name} and {other.name}"}
            name = await self.propose_skill(raw, origin={"curator": True, "merged_from": other.name})
            if name:
                proposed.append(name)
                used.update({a.name, b.name})
        return proposed

    # ------------------------------------------------------------------ profile upkeep
    async def update_profile(self, now: datetime | None = None, *, force: bool = False) -> dict:
        now = now or datetime.now(UTC)
        app = self.app
        cfg = app.config
        if not cfg.evolution.user_profile_updates or app.memory is None:
            return {"updated": False}
        store = app.store
        last = _parse_ts(await store.get_meta("evolution.profile_last_run"))
        if not force and last is not None and now - last < timedelta(hours=cfg.evolution.profile_update_hours):
            return {"updated": False}
        since = (last or now - timedelta(days=30)).isoformat()
        nowiso = now.isoformat()
        # what upkeep read last time: rows saved in the same clock tick as that read are still picked up (#236)
        facts_mark = await store.get_meta(PROFILE_FACTS_MARK) or since
        summaries_mark = await store.get_meta(PROFILE_SUMMARIES_MARK) or since
        after, params = after_watermark(facts_mark, "updated_at", "id")
        # MEMORY.md and USER.md are personal: work and project memories never go in them (#155)
        new_facts = await store.fetchall(
            "SELECT id, content, topics, source, updated_at FROM facts WHERE status = 'active' AND state = 'current'"
            f" AND memory_type = 'long-term' AND {after} AND (expires_at IS NULL OR expires_at > ?)"
            f" AND {PROFILE_SCOPES} ORDER BY updated_at, id",
            (*params, nowiso),
        )
        # chats that read outside content never feed MEMORY.md (ADR 0021): their facts wait for review and their
        # summaries are left out
        after, params = after_watermark(summaries_mark, "created_at", "id")
        personal, personal_params = memory_scopes.personal_summary_sql(cfg)  # no work or project chat's (#155)
        new_summaries = await store.fetchall(
            f"SELECT id, content, created_at FROM summaries WHERE {after} AND {CLEAN_SUMMARY} AND {personal}"
            " ORDER BY created_at, id LIMIT 8",  # in watermark order, so the limit never jumps over a summary
            (*params, *personal_params),
        )
        await store.set_meta("evolution.profile_last_run", nowiso)
        # saved after every read, also an empty one: the next read starts where this one did, never at this run's time
        await store.set_meta(PROFILE_FACTS_MARK, next_watermark(facts_mark, new_facts, "updated_at", "id"))
        await store.set_meta(
            PROFILE_SUMMARIES_MARK, next_watermark(summaries_mark, new_summaries, "created_at", "id")
        )
        if not new_facts and not new_summaries:
            return {"updated": False}

        learned = [
            r["content"] for r in new_facts
            if r["source"] != "onboarding" and STABLE_TOPICS & set(json.loads(r["topics"] or "[]"))
        ][:20]
        appended = app.workspace.append_learned(learned) if learned else []

        budget = cfg.memory.workspace_budget_chars
        all_facts = await store.fetchall(
            "SELECT content FROM facts WHERE status = 'active' AND state = 'current'"
            " AND (expires_at IS NULL OR expires_at > ?)"
            f" AND {PROFILE_SCOPES} ORDER BY updated_at DESC, id DESC LIMIT 80",
            (nowiso,),
        )
        summaries = await store.fetchall(
            f"SELECT content FROM summaries WHERE {CLEAN_SUMMARY} AND {personal}"
            " ORDER BY end_at DESC, rowid DESC LIMIT 8",
            personal_params,
        )
        current = app.workspace.read_full()["memory"]
        memory_chars = 0
        try:
            text = await app.llm.complete_text(
                "fast",
                [
                    {
                        "role": "system",
                        "content": prompts.PROFILE_SYSTEM.format(
                            user=cfg.assistant.user_name or "the user", budget=budget
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompts.PROFILE_USER.format(
                            memory=current[:budget] or "(empty)",
                            facts="\n".join(f"- {r['content']}" for r in all_facts) or "(none)",
                            summaries="\n\n".join(r["content"][:1200] for r in summaries) or "(none)",
                        ),
                    },
                ],
            )
            text = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
            if len(text) >= 40:
                if not text.startswith("#"):
                    text = "# Long-term memory\n\n" + text
                text = text[:budget]
                app.workspace.write("memory", text.rstrip() + "\n")
                memory_chars = len(text)
        except Exception as exc:
            log.warning("MEMORY.md refresh failed: %s", exc)
        detail = {
            "learned_appended": len(appended),
            "facts_considered": len(new_facts),
            "summaries_considered": len(new_summaries),
            "memory_chars": memory_chars,
        }
        await log_event(store, "profile_updated", detail, ts=nowiso)
        return {"updated": True, **detail}
