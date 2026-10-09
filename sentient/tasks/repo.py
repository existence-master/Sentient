"""SQLite persistence for tasks, runs, progress events and trigger dedupe.

Serializes to the v2-compatible JSON shapes in docs/API.md section 4.
"""

from __future__ import annotations

import json
from typing import Any

from sentient.store.db import Store, new_id

TASK_JSON_FIELDS = {
    "schedule", "plan", "chat_history", "clarifying_questions", "swarm_details", "original_context", "script",
}
TASK_COLUMNS = {
    "name", "description", "status", "priority", "task_type", "schedule", "plan", "original_prompt",
    "source", "enabled", "assignee", "model", "chat_history", "clarifying_questions", "swarm_details",
    "original_context", "script", "error", "next_execution_at", "last_execution_at", "created_at", "updated_at",
}
RUN_JSON_FIELDS = {"plan", "trigger_data", "messages", "result", "pending_question", "limits"}
RUN_COLUMNS = {
    "status", "plan", "trigger_data", "messages", "result", "error", "resume_count", "retry_of",
    "pending_question", "limits", "last_activity_at", "started_at", "finished_at", "created_at",
}
# Columns added after the first stub schema; ensured on start for older databases.
_ADDED_TASK_COLUMNS = {
    "assignee": "TEXT NOT NULL DEFAULT 'ai'",
    "model": "TEXT",
    "chat_history": "TEXT",
    "clarifying_questions": "TEXT",
    "swarm_details": "TEXT",
    "original_context": "TEXT",
    "error": "TEXT",
    "next_execution_at": "TEXT",
    "last_execution_at": "TEXT",
    "script": "TEXT",
}
_ADDED_RUN_COLUMNS = {
    "plan": "TEXT", "resume_count": "INTEGER NOT NULL DEFAULT 0", "retry_of": "TEXT", "pending_question": "TEXT",
    "limits": "TEXT", "last_activity_at": "TEXT",
}

# Progress updates embedded in each run of a serialized Task; the full log is at
# GET /api/tasks/{id}/runs/{run_id}/events.
PROGRESS_IN_TASK = 200


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _question_to_api(question: Any) -> dict | None:
    if not isinstance(question, dict) or not question.get("question"):
        return None
    options = question.get("options")
    kind = "stuck" if question.get("stuck") else "limit" if question.get("limit") else "question"
    return {
        "question": str(question["question"]),
        "options": [str(o) for o in options] if isinstance(options, list) else [],
        "asked_at": question.get("asked_at"),
        "kind": kind,
        "reason": (str(question.get("reason") or "") or None) if kind == "stuck" else None,
    }


class TaskRepo:
    def __init__(self, store: Store):
        self.store = store

    # ------------------------------------------------------------------ schema
    async def ensure_schema(self) -> None:
        for col, decl in _ADDED_TASK_COLUMNS.items():
            await self.store.ensure_column("tasks", col, decl)
        for col, decl in _ADDED_RUN_COLUMNS.items():
            await self.store.ensure_column("task_runs", col, decl)
        await self.store.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_next_exec ON tasks(status, enabled, next_execution_at)"
        )
        await self.store.db.execute(
            "CREATE TABLE IF NOT EXISTS task_trigger_seen ("
            " task_id TEXT NOT NULL, item_key TEXT NOT NULL, seen_at TEXT NOT NULL,"
            " PRIMARY KEY (task_id, item_key))"
        )
        await self.store.db.commit()

    # ------------------------------------------------------------------ tasks
    @staticmethod
    def _decode_task(row: Any) -> dict:
        d = dict(row)
        for f in TASK_JSON_FIELDS:
            d[f] = _loads(d.get(f))
        d["enabled"] = bool(d.get("enabled", 1))
        return d

    async def insert_task(self, fields: dict) -> str:
        task_id = fields.get("id") or new_id()
        data = {k: v for k, v in fields.items() if k in TASK_COLUMNS}
        data["id"] = task_id
        for f in TASK_JSON_FIELDS & data.keys():
            data[f] = _dumps(data[f])
        if "enabled" in data:
            data["enabled"] = 1 if data["enabled"] else 0
        cols = ", ".join(data)
        marks = ", ".join("?" for _ in data)
        await self.store.execute(f"INSERT INTO tasks({cols}) VALUES({marks})", tuple(data.values()))
        return task_id

    async def get_task(self, task_id: str) -> dict | None:
        row = await self.store.fetchone("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return self._decode_task(row) if row else None

    async def list_tasks(self, where: str = "", params: tuple = ()) -> list[dict]:
        sql = "SELECT * FROM tasks"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY created_at DESC, rowid DESC"
        return [self._decode_task(r) for r in await self.store.fetchall(sql, params)]

    async def update_task(self, task_id: str, fields: dict) -> None:
        data = {k: v for k, v in fields.items() if k in TASK_COLUMNS}
        if not data:
            return
        for f in TASK_JSON_FIELDS & data.keys():
            data[f] = _dumps(data[f])
        if "enabled" in data:
            data["enabled"] = 1 if data["enabled"] else 0
        sets = ", ".join(f"{k} = ?" for k in data)
        await self.store.execute(f"UPDATE tasks SET {sets} WHERE id = ?", (*data.values(), task_id))

    async def update_task_if_status(self, task_id: str, statuses: set[str] | list[str], fields: dict) -> bool:
        """Update only while the task is still in one of ``statuses`` (guards background jobs
        against user actions that happened meanwhile). Returns True when a row changed."""
        data = {k: v for k, v in fields.items() if k in TASK_COLUMNS}
        if not data:
            return False
        for f in TASK_JSON_FIELDS & data.keys():
            data[f] = _dumps(data[f])
        if "enabled" in data:
            data["enabled"] = 1 if data["enabled"] else 0
        wanted = list(statuses)
        sets = ", ".join(f"{k} = ?" for k in data)
        marks = ", ".join("?" for _ in wanted)
        cur = await self.store.execute(
            f"UPDATE tasks SET {sets} WHERE id = ? AND status IN ({marks})", (*data.values(), task_id, *wanted)
        )
        return bool(getattr(cur, "rowcount", 1))

    async def delete_task(self, task_id: str) -> None:
        # explicit child deletes so it also works if foreign keys are off
        await self.store.execute(
            "DELETE FROM run_events WHERE run_id IN (SELECT id FROM task_runs WHERE task_id = ?)", (task_id,)
        )
        await self.store.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
        await self.store.execute("DELETE FROM task_trigger_seen WHERE task_id = ?", (task_id,))
        await self.store.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

    async def missed(self, cutoff: str) -> list[dict]:
        """Scheduled tasks that were due at or before ``cutoff`` and have not started (the computer was off or asleep)."""
        return await self.list_tasks(
            "status IN ('active', 'pending') AND enabled = 1 AND next_execution_at IS NOT NULL"
            " AND next_execution_at <= ?",
            (cutoff,),
        )

    async def claim_due(self, now: str) -> list[str]:
        """Atomically move due active/pending tasks to processing and return their ids."""
        db = self.store.db
        async with db.execute(
            "UPDATE tasks SET status = 'processing', last_execution_at = ?, updated_at = ?"
            " WHERE status IN ('active', 'pending') AND enabled = 1"
            " AND next_execution_at IS NOT NULL AND next_execution_at <= ?"
            " RETURNING id",
            (now, now, now),
        ) as cur:
            rows = await cur.fetchall()
        await db.commit()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------ runs
    @staticmethod
    def _decode_run(row: Any) -> dict:
        d = dict(row)
        for f in RUN_JSON_FIELDS:
            d[f] = _loads(d.get(f))
        return d

    async def insert_run(
        self, task_id: str, *, now: str, plan: Any = None, trigger_data: Any = None, status: str = "processing"
    ) -> str:
        run_id = new_id()
        await self.store.execute(
            "INSERT INTO task_runs(id, task_id, status, plan, trigger_data, started_at, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (run_id, task_id, status, _dumps(plan), _dumps(trigger_data), now, now),
        )
        return run_id

    async def get_run(self, run_id: str) -> dict | None:
        row = await self.store.fetchone("SELECT * FROM task_runs WHERE id = ?", (run_id,))
        return self._decode_run(row) if row else None

    async def runs_for(self, task_id: str) -> list[dict]:
        rows = await self.store.fetchall(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY created_at, rowid", (task_id,)
        )
        return [self._decode_run(r) for r in rows]

    async def latest_run(self, task_id: str) -> dict | None:
        row = await self.store.fetchone(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1", (task_id,)
        )
        return self._decode_run(row) if row else None

    async def update_run(self, run_id: str, fields: dict) -> None:
        data = {k: v for k, v in fields.items() if k in RUN_COLUMNS}
        if not data:
            return
        for f in RUN_JSON_FIELDS & data.keys():
            data[f] = _dumps(data[f])
        sets = ", ".join(f"{k} = ?" for k in data)
        await self.store.execute(f"UPDATE task_runs SET {sets} WHERE id = ?", (*data.values(), run_id))

    async def finish_run(
        self, run_id: str, status: str, *, error: str | None, now: str, from_statuses: tuple[str, ...] = ("processing",)
    ) -> bool:
        """Compare-and-set a processing (or ``from_statuses``) run to a final status. False if it already finished."""
        marks = ", ".join("?" for _ in from_statuses)
        cur = await self.store.execute(
            "UPDATE task_runs SET status = ?, error = ?, finished_at = ?, pending_question = NULL"
            f" WHERE id = ? AND status IN ({marks})",
            (status, error, now, run_id, *from_statuses),
        )
        return (cur.rowcount or 0) > 0

    async def pause_run(self, run_id: str, question: dict) -> bool:
        """Compare-and-set a processing run to ``waiting_for_user`` with its question. False if it changed meanwhile."""
        cur = await self.store.execute(
            "UPDATE task_runs SET status = 'waiting_for_user', pending_question = ? WHERE id = ? AND status = 'processing'",
            (_dumps(question), run_id),
        )
        return (cur.rowcount or 0) > 0

    async def resume_run(self, run_id: str, messages: list[dict], *, limits: dict | None = None) -> bool:
        """Compare-and-set a waiting run back to processing with the answered transcript (and, for "Keep going",
        its raised ``limits``, in the same write). False if not waiting."""
        cur = await self.store.execute(
            "UPDATE task_runs SET status = 'processing', pending_question = NULL, messages = ?,"
            " limits = COALESCE(?, limits) WHERE id = ? AND status = 'waiting_for_user'",
            (_dumps(messages), _dumps(limits), run_id),
        )
        return (cur.rowcount or 0) > 0

    async def waiting_runs(self, task_id: str | None = None) -> list[dict]:
        sql = "SELECT * FROM task_runs WHERE status = 'waiting_for_user'"
        params: tuple = ()
        if task_id:
            sql += " AND task_id = ?"
            params = (task_id,)
        rows = await self.store.fetchall(sql + " ORDER BY created_at, rowid", params)
        return [self._decode_run(r) for r in rows]

    async def processing_runs(self, task_id: str | None = None) -> list[dict]:
        if task_id:
            rows = await self.store.fetchall(
                "SELECT * FROM task_runs WHERE status = 'processing' AND task_id = ? ORDER BY created_at", (task_id,)
            )
        else:
            rows = await self.store.fetchall("SELECT * FROM task_runs WHERE status = 'processing' ORDER BY created_at")
        return [self._decode_run(r) for r in rows]

    # ------------------------------------------------------------------ progress
    async def add_event(self, run_id: str, message: dict, now: str) -> dict:
        update = {"timestamp": now, "message": message}
        await self.store.execute(
            "INSERT INTO run_events(run_id, kind, payload, created_at) VALUES(?,?,?,?)",
            (run_id, str(message.get("type", "info")), _dumps(update), now),
        )
        return update

    async def events(self, run_id: str, limit: int | None = None) -> list[dict]:
        if limit:
            rows = await self.store.fetchall(
                "SELECT payload FROM (SELECT payload, id FROM run_events WHERE run_id = ? ORDER BY id DESC LIMIT ?)"
                " ORDER BY id",
                (run_id, limit),
            )
        else:
            rows = await self.store.fetchall("SELECT payload FROM run_events WHERE run_id = ? ORDER BY id", (run_id,))
        return [_loads(r["payload"]) for r in rows]

    # ------------------------------------------------------------------ triggers
    async def mark_seen(self, source: str, event_id: str, now: str) -> bool:
        cur = await self.store.execute(
            "INSERT OR IGNORE INTO seen_events(source, event_id, seen_at) VALUES(?,?,?)", (source, event_id, now)
        )
        return (cur.rowcount or 0) > 0

    async def mark_task_seen(self, task_id: str, source: str, item_id: str, now: str) -> bool:
        """Claim one item for one triggered task. False when that task already handled it."""
        cur = await self.store.execute(
            "INSERT OR IGNORE INTO task_trigger_seen(task_id, item_key, seen_at) VALUES(?,?,?)",
            (task_id, f"{source}:{item_id}", now),
        )
        return (cur.rowcount or 0) > 0

    # ------------------------------------------------------------------ serialization
    @staticmethod
    def run_to_api(run: dict, progress: list[dict]) -> dict:
        return {
            "run_id": run["id"],
            "status": run["status"],
            "created_at": run.get("created_at"),
            "execution_start_time": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "plan": run.get("plan") or [],
            "trigger_event_data": run.get("trigger_data"),
            "progress_updates": progress,
            "result": run.get("result"),
            "error": run.get("error"),
            "retry_of": run.get("retry_of"),
            "pending_question": _question_to_api(run.get("pending_question")) if run["status"] == "waiting_for_user" else None,
            "last_activity_at": run.get("last_activity_at") or run.get("started_at"),
        }

    @staticmethod
    def task_to_api(task: dict, runs: list[dict]) -> dict:
        return {
            "task_id": task["id"],
            "name": task.get("name"),
            "description": task.get("description") or "",
            "status": task.get("status"),
            "priority": task.get("priority", 1),
            "assignee": task.get("assignee") or "ai",
            "task_type": task.get("task_type") or "single",
            "schedule": task.get("schedule"),
            "plan": task.get("plan") or [],
            "runs": runs,
            "chat_history": task.get("chat_history") or [],
            "clarifying_questions": task.get("clarifying_questions") or [],
            "swarm_details": task.get("swarm_details"),
            "enabled": bool(task.get("enabled", True)),
            "model": task.get("model"),
            "original_context": task.get("original_context") or {"source": "manual_creation"},
            "script": task.get("script") if task.get("task_type") == "script" else None,
            "error": task.get("error"),
            "next_execution_at": task.get("next_execution_at"),
            "last_execution_at": task.get("last_execution_at"),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
        }

    async def _progress_by_run(self, task_ids: list[str]) -> dict[str, list[dict]]:
        if not task_ids:
            return {}
        marks = ", ".join("?" for _ in task_ids)
        rows = await self.store.fetchall(
            "SELECT run_id, payload FROM ("
            "  SELECT e.run_id, e.payload, e.id,"
            "         ROW_NUMBER() OVER (PARTITION BY e.run_id ORDER BY e.id DESC) AS rn"
            "  FROM run_events e JOIN task_runs r ON r.id = e.run_id"
            f"  WHERE r.task_id IN ({marks})"
            ") WHERE rn <= ? ORDER BY id",
            (*task_ids, PROGRESS_IN_TASK),
        )
        out: dict[str, list[dict]] = {}
        for r in rows:
            out.setdefault(r["run_id"], []).append(_loads(r["payload"]))
        return out

    async def serialize_many(self, tasks: list[dict]) -> list[dict]:
        if not tasks:
            return []
        ids = [t["id"] for t in tasks]
        marks = ", ".join("?" for _ in ids)
        run_rows = await self.store.fetchall(
            f"SELECT * FROM task_runs WHERE task_id IN ({marks}) ORDER BY created_at, rowid", tuple(ids)
        )
        progress = await self._progress_by_run(ids)
        runs_by_task: dict[str, list[dict]] = {}
        for row in run_rows:
            run = self._decode_run(row)
            runs_by_task.setdefault(run["task_id"], []).append(self.run_to_api(run, progress.get(run["id"], [])))
        return [self.task_to_api(t, runs_by_task.get(t["id"], [])) for t in tasks]

    async def serialize(self, task_id: str) -> dict | None:
        task = await self.get_task(task_id)
        if task is None:
            return None
        return (await self.serialize_many([task]))[0]
