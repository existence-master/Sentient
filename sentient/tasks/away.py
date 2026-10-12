"""Pause routines while the user is away (#163).

Recurring and triggered tasks (the Daily Brief is one) keep delivering results to someone who may have stopped
reading them. When the user has not been seen for ``tasks.away_pause_days`` days, or
``tasks.away_pause_after_results`` results were delivered since they were last seen, Sentient asks once (a
``task`` notification with ``payload.event = "away_check"``) and pauses those routines if
``tasks.away_ask_hours`` pass with no sign of the user. Any activity (a chat message on any surface, the app
window in use, a message or button in a paired chat) resets the timer and cancels the question. Pausing
sends one notification (``event = "away_paused"``) with a Resume action. Tasks marked ``keep_running`` are never
paused.

State lives in store meta:

- ``user.last_active_at``: ISO time of the last sign of the user, written at most once a minute
- ``tasks.away``: ``{state: "asked"|"paused", since, notification_id, task_ids, reason}`` or absent
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sentient.tasks.schedule import parse_iso

log = logging.getLogger(__name__)

ACTIVE_KEY = "user.last_active_at"
STATE_KEY = "tasks.away"
ROUTINE_KINDS = ("recurring", "triggered")
CHECK_EVERY_S = 600  # the check runs every 10 minutes, the first one 10 minutes after start
WRITE_EVERY_S = 60  # activity is written at most this often


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


class AwayWatch:
    def __init__(self, tasks: Any):
        self.tasks = tasks
        self.app = tasks.app
        self._last_written: datetime | None = None

    # ------------------------------------------------------------------ state
    async def last_active(self) -> datetime | None:
        return parse_iso(await self.app.store.get_meta(ACTIVE_KEY))

    async def _state(self) -> dict:
        try:
            raw = json.loads(await self.app.store.get_meta(STATE_KEY) or "null")
        except ValueError:
            raw = None
        return raw if isinstance(raw, dict) and raw.get("state") in {"asked", "paused"} else {}

    async def _save_state(self, state: dict) -> None:
        await self.app.store.set_meta(STATE_KEY, json.dumps(state) if state else "")

    async def status(self) -> dict:
        """Shape of ``GET /api/tasks/away``."""
        cfg = self.app.config.tasks
        state = await self._state()
        last = await self.last_active()
        out: dict[str, Any] = {
            "state": state.get("state") or "active",
            "last_active_at": last.isoformat() if last else None,
            "since": state.get("since"),
            "reason": state.get("reason"),
            "task_ids": list(state.get("task_ids") or []),
            "notification_id": state.get("notification_id"),
            "away_pause_days": cfg.away_pause_days,
            "away_pause_after_results": cfg.away_pause_after_results,
            "away_ask_hours": cfg.away_ask_hours,
        }
        if out["state"] == "asked" and state.get("since"):
            since = parse_iso(state["since"])
            if since is not None:
                out["pause_at"] = (since + timedelta(hours=cfg.away_ask_hours)).isoformat()
        return out

    # ------------------------------------------------------------------ activity
    async def seen(self, surface: str = "app") -> None:
        """The user did something. Resets the away timer and cancels a pending question (paused routines stay
        paused until the user resumes them)."""
        now = self.tasks.now()
        if self._last_written is None or (now - self._last_written).total_seconds() >= WRITE_EVERY_S:
            self._last_written = now
            await self.app.store.set_meta(ACTIVE_KEY, now.isoformat())
        state = await self._state()
        if state.get("state") == "asked":
            await self._save_state({})
            await self._settle(state.get("notification_id"), "kept")
            log.info("away check cancelled: the user is back (%s)", surface)
            self.app.bus.publish("tasks.away", await self.status())

    async def _settle(self, notification_id: str | None, status: str) -> None:
        if not notification_id:
            return
        note = await self.app.notifications.get(notification_id)
        if note is None:
            return
        await self.app.notifications.update_payload(notification_id, {**(note.get("payload") or {}), "status": status})

    # ------------------------------------------------------------------ the check
    async def routines(self) -> list[dict]:
        """Running routines that would pause: enabled recurring or triggered tasks not marked keep-running."""
        out = []
        for task in await self.tasks.repo.list_tasks("enabled = 1 AND status = 'active'"):
            kind = (task.get("schedule") or {}).get("type")
            if kind in ROUTINE_KINDS and not task.get("keep_running"):
                out.append(task)
        return out

    async def results_since(self, since: datetime) -> int:
        """Task results and briefs delivered after ``since`` (the user hasn't been around to see them)."""
        row = await self.app.store.fetchone(
            "SELECT COUNT(*) AS n FROM notifications WHERE kind IN ('task', 'brief') AND created_at > ?"
            " AND payload LIKE '%\"task_id\"%'",
            (since.isoformat(),),
        )
        return int(row["n"]) if row else 0

    async def check(self, now: datetime | None = None) -> dict:
        """Ask, then pause, when the user has been away. Returns ``status()``."""
        cfg = self.app.config.tasks
        now = now or self.tasks.now()
        if self.app.stopped or (cfg.away_pause_days <= 0 and cfg.away_pause_after_results <= 0):
            return await self.status()
        last = await self.last_active()
        if last is None:  # first start (or an older install): count from now
            await self.app.store.set_meta(ACTIVE_KEY, now.isoformat())
            return await self.status()
        state = await self._state()
        if state.get("state") == "paused":
            return await self.status()
        idle_days = (now - last).total_seconds() / 86400
        results = await self.results_since(last) if cfg.away_pause_after_results > 0 else 0
        reason = None
        if cfg.away_pause_days > 0 and idle_days >= cfg.away_pause_days:
            reason = "days"
        elif cfg.away_pause_after_results > 0 and results >= cfg.away_pause_after_results:
            reason = "results"
        if reason is None:
            if state.get("state") == "asked":  # the settings changed meanwhile
                await self._save_state({})
                await self._settle(state.get("notification_id"), "kept")
            return await self.status()
        routines = await self.routines()
        if not routines:
            return await self.status()
        if state.get("state") == "asked":
            since = parse_iso(state.get("since")) or now
            if now - since >= timedelta(hours=cfg.away_ask_hours):
                await self._pause(routines, reason, now, ask_note=state.get("notification_id"))
            return await self.status()
        if cfg.away_ask_hours <= 0:
            await self._pause(routines, reason, now)
        else:
            await self._ask(routines, reason, now, idle_days, results)
        return await self.status()

    async def check_job(self) -> None:
        await self.check()

    def _why(self, reason: str, idle_days: float, results: int) -> str:
        if reason == "days":
            return f"You haven't used Sentient for {_plural(int(idle_days), 'day', 'days')}"
        return f"{_plural(results, 'result', 'results')} arrived since you were last around"

    async def _ask(self, routines: list[dict], reason: str, now: datetime, idle_days: float, results: int) -> None:
        hours = self.app.config.tasks.away_ask_hours
        names = ", ".join(t.get("name") or "Untitled task" for t in routines[:3])
        more = f" and {len(routines) - 3} more" if len(routines) > 3 else ""
        message = (
            f"{self._why(reason, idle_days, results)}. I'll pause {_plural(len(routines), 'routine', 'routines')} "
            f"({names}{more}) in {_plural(hours, 'hour', 'hours')} unless you're back. "
            "Open Sentient or send me any message to keep them running."
        )
        note = await self.app.notify(
            "task", message, title="Keep your routines running?",
            payload={"event": "away_check", "reason": reason, "routines": len(routines)},
        )
        await self._save_state({"state": "asked", "since": now.isoformat(), "notification_id": note["id"], "reason": reason})
        log.info("away check: asked before pausing %d routines (%s)", len(routines), reason)
        self.app.bus.publish("tasks.away", await self.status())

    async def _pause(self, routines: list[dict], reason: str, now: datetime, ask_note: str | None = None) -> None:
        paused: list[str] = []
        for task in routines:
            try:
                await self.tasks.update(task["id"], {"enabled": False})
            except Exception:
                log.exception("could not pause task %s", task.get("id"))
                continue
            paused.append(task["id"])  # recorded as soon as it is off, so Resume finds it whatever happens next
            with contextlib.suppress(Exception):
                await self.tasks.publish(task["id"])
        if not paused:
            return
        if ask_note:  # the question is answered by the pause itself
            await self._settle(ask_note, "paused")
        names = ", ".join(t.get("name") or "Untitled task" for t in routines[:3])
        more = f" and {len(routines) - 3} more" if len(routines) > 3 else ""
        message = (
            f"You've been away, so I paused {_plural(len(paused), 'routine', 'routines')} ({names}{more}). "
            "Nothing is lost. Resume them whenever you like."
        )
        note = await self.app.notify(
            "task", message, title="Routines paused while you were away",
            payload={"event": "away_paused", "reason": reason, "task_ids": paused},
        )
        await self._save_state({
            "state": "paused", "since": now.isoformat(), "notification_id": note["id"], "reason": reason, "task_ids": paused,
        })
        log.info("away: paused %d routines (%s)", len(paused), reason)
        self.app.bus.publish("tasks.away", await self.status())

    async def resume(self) -> dict:
        """Turn the routines paused for absence back on. Returns ``{resumed: n, status}``."""
        state = await self._state()
        resumed = 0
        failed: list[str] = []
        if state.get("state") == "paused":
            for task_id in state.get("task_ids") or []:
                task = await self.tasks.repo.get_task(task_id)
                if task is None or task.get("enabled"):
                    continue  # deleted, or resumed by hand already
                try:
                    await self.tasks.update(task_id, {"enabled": True})
                except Exception:
                    log.exception("could not resume task %s", task_id)
                    failed.append(task_id)
                    continue
                resumed += 1
                with contextlib.suppress(Exception):
                    await self.tasks.publish(task_id)
            if failed:  # keep what is still off for another try; the notification keeps its Resume button
                await self._save_state({**state, "task_ids": failed})
                self.app.bus.publish("tasks.away", await self.status())
                return {"resumed": resumed, "status": await self.status()}
            await self._settle(state.get("notification_id"), "resumed")
        await self._save_state({})
        await self.seen("resume")
        self.app.bus.publish("tasks.away", await self.status())
        return {"resumed": resumed, "status": await self.status()}

    async def keep(self) -> dict:
        """The user chose Keep them running."""
        await self.seen("keep")
        return await self.status()
