"""Timers and one-off reminders (#150): kept in SQLite, fired on time as notifications.

A timer or reminder is set by a quick command (``sentient.agent.fastpath``, no model), by the model through the
``reminders`` tools, or over REST. One clock task sleeps until the next one is due (at most ``MAX_SLEEP_S``, so a
computer waking from sleep or a clock change is noticed) and fires it with ``app.notify("reminder", ...)``: the
window shows it (a native notification when the window is not in front), and a reminder set in a Telegram, Discord or
WhatsApp chat goes back to that chat (``ChannelService``). One that came due while Sentient was not running fires
as soon as it starts, saying when it was due.

Firing never asks the model, and it is not paused by Stop everything: a timer the user set still rings.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sentient.services import Service
from sentient.store.db import new_id, now_iso

log = logging.getLogger(__name__)

KINDS = ("timer", "reminder")
STATUSES = ("scheduled", "fired", "cancelled")
MAX_SLEEP_S = 30.0
LATE_AFTER_S = 90.0      # fired this much after it was due: the notification says when it was due
MAX_TIMER_S = 24 * 3600
MAX_AHEAD = timedelta(days=366)
KEEP_DAYS = 30           # fired and cancelled ones are forgotten after this long
LIST_LIMIT = 100


class ReminderError(ValueError):
    """A timer or reminder that can't be set, in a plain sentence."""


def _row(r: Any) -> dict:
    return {
        "id": r["id"], "kind": r["kind"], "message": r["message"], "connector": r["connector"], "label": r["label"],
        "duration_s": r["duration_s"], "fire_at": r["fire_at"], "status": r["status"], "session_id": r["session_id"],
        "channel": r["channel"], "language": r["language"], "source": r["source"], "created_at": r["created_at"],
        "fired_at": r["fired_at"], "cancelled_at": r["cancelled_at"], "notification_id": r["notification_id"],
    }


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class ReminderService(Service):
    name = "reminders"

    def __init__(self, app):
        super().__init__(app)
        self._wake = asyncio.Event()
        self._fire_lock = asyncio.Lock()

    async def start(self) -> None:
        from sentient.reminders.tools import RemindersPlugin

        if not any(p.id == RemindersPlugin.id for p in self.app.registry.plugins()):
            self.app.registry.register(RemindersPlugin())
        if self.app.enable_background:
            self._loops.append(asyncio.create_task(self._clock(), name="reminders:clock"))

    # ------------------------------------------------------------------ reading
    async def list(self, status: str | None = "scheduled", *, kind: str | None = None, limit: int = LIST_LIMIT) -> list[dict]:
        """Soonest first for ``scheduled``; newest first otherwise. ``status=None``: all."""
        where, params = [], []
        if status:
            where.append("status = ?")
            params.append(status)
        if kind in KINDS:
            where.append("kind = ?")
            params.append(kind)
        order = "fire_at ASC" if status == "scheduled" else "COALESCE(fired_at, cancelled_at, created_at) DESC"
        rows = await self.app.store.fetchall(
            f"SELECT * FROM reminders {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY {order} LIMIT ?",
            [*params, max(1, min(int(limit), 500))],
        )
        return [_row(r) for r in rows]

    async def get(self, reminder_id: str) -> dict | None:
        r = await self.app.store.fetchone("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
        return _row(r) if r else None

    # ------------------------------------------------------------------ changing
    async def add(
        self, kind: str, fire_at: datetime, *, message: str = "", connector: str = "", label: str = "",
        duration_s: float | None = None, session_id: str | None = None, channel: str | None = None,
        language: str = "en", source: str = "chat",
    ) -> dict:
        if kind not in KINDS:
            raise ReminderError("A reminder is a timer or a reminder.")
        if fire_at.tzinfo is None:
            raise ReminderError("The time needs a timezone.")
        now = datetime.now(UTC)
        if kind == "timer" and (duration_s is None or duration_s < 1 or duration_s > MAX_TIMER_S):
            raise ReminderError("A timer can run from 1 second to 24 hours.")
        if fire_at < now - timedelta(seconds=5):
            raise ReminderError("That time has already passed.")
        if fire_at > now + MAX_AHEAD:
            raise ReminderError("Reminders can be set up to a year ahead.")
        connector = connector if connector in {"to", "about", "that"} else ""
        item = {
            "id": new_id(), "kind": kind, "message": " ".join(str(message or "").split())[:500], "connector": connector,
            "label": " ".join(str(label or "").split())[:60], "duration_s": duration_s,
            "fire_at": fire_at.astimezone(UTC).isoformat(), "status": "scheduled", "session_id": session_id,
            "channel": channel, "language": language or "en", "source": source, "created_at": now_iso(),
            "fired_at": None, "cancelled_at": None, "notification_id": None,
        }
        await self.app.store.execute(
            "INSERT INTO reminders(id, kind, message, connector, label, duration_s, fire_at, status, session_id, channel,"
            " language, source, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (item["id"], kind, item["message"], connector, item["label"], duration_s, item["fire_at"], "scheduled",
             session_id, channel, item["language"], source, item["created_at"]),
        )
        log.info("%s %s set for %s (%s, from %s)", kind, item["id"][:8], item["fire_at"], source, channel or "?")
        self._changed(item)
        return item

    async def cancel(self, reminder_id: str) -> tuple[dict | None, bool]:
        """Cancel a scheduled one. Returns ``(it, cancelled_now)``: None when unknown, and False when it had already
        gone off or been cancelled."""
        cur = await self.app.store.execute(
            "UPDATE reminders SET status = 'cancelled', cancelled_at = ? WHERE id = ? AND status = 'scheduled'",
            (now_iso(), reminder_id),
        )
        item = await self.get(reminder_id)
        changed = item is not None and bool(cur.rowcount)
        if changed:
            log.info("%s %s cancelled", item["kind"], reminder_id[:8])
            self._changed(item)
        return item, changed

    def _changed(self, item: dict) -> None:
        self.app.bus.publish("reminder.updated", item)
        self._wake.set()

    # ------------------------------------------------------------------ the clock
    async def _clock(self) -> None:
        await self._forget_old()
        while True:
            self._wake.clear()
            try:
                await self.fire_due()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("firing reminders failed")
            nxt = await self._next_due()
            delay = MAX_SLEEP_S
            if nxt is not None:
                delay = min(MAX_SLEEP_S, max(0.0, (nxt - datetime.now(UTC)).total_seconds()))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=delay)

    async def _next_due(self) -> datetime | None:
        row = await self.app.store.fetchone(
            "SELECT fire_at FROM reminders WHERE status = 'scheduled' ORDER BY fire_at ASC LIMIT 1"
        )
        return _parse(row["fire_at"]) if row else None

    async def fire_due(self, now: datetime | None = None) -> int:
        """Fire every scheduled one that is due. Returns how many fired."""
        now = now or datetime.now(UTC)
        fired = 0
        async with self._fire_lock:
            for item in await self.list("scheduled", limit=500):
                due = _parse(item["fire_at"])
                if due is None or due > now:
                    continue
                if await self._fire(item, due, now):
                    fired += 1
        return fired

    async def _fire(self, item: dict, due: datetime, now: datetime) -> bool:
        cur = await self.app.store.execute(
            "UPDATE reminders SET status = 'fired', fired_at = ? WHERE id = ? AND status = 'scheduled'",
            (now.isoformat(), item["id"]),
        )
        if not cur.rowcount:
            return False  # cancelled a moment ago
        from sentient.agent import fastpath
        from sentient.tasks.schedule import get_tz, user_timezone_name

        pack = fastpath.language(item.get("language"))
        late = (now - due).total_seconds() > LATE_AFTER_S
        tz = get_tz(user_timezone_name(self.app.config.assistant.timezone))
        if item["kind"] == "timer":
            title, body = pack.timer_done(item)
        else:
            title, body = pack.reminder_due(item, due.astimezone(tz) if late else None)
        payload = {"event": item["kind"], "reminder_id": item["id"], "fire_at": item["fire_at"], "late": late,
                   "channel": item.get("channel")}
        if item.get("session_id"):
            payload["session_id"] = item["session_id"]
        note = await self.app.notify("reminder", body, title=title, payload=payload)
        await self.app.store.execute(
            "UPDATE reminders SET notification_id = ? WHERE id = ?", (note.get("id"), item["id"])
        )
        log.info("%s %s fired%s", item["kind"], item["id"][:8], " late" if late else "")
        self.app.bus.publish("reminder.updated", {**item, "status": "fired", "fired_at": now.isoformat(),
                                                  "notification_id": note.get("id")})
        return True

    async def _forget_old(self) -> None:
        cutoff = (datetime.now(UTC) - timedelta(days=KEEP_DAYS)).isoformat()
        with contextlib.suppress(Exception):
            await self.app.store.execute(
                "DELETE FROM reminders WHERE status != 'scheduled' AND COALESCE(fired_at, cancelled_at, created_at) < ?",
                (cutoff,),
            )
