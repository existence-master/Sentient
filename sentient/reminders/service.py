"""Timers, one-off reminders and alarms (#150, #359): kept in SQLite, fired on time as notifications.

A timer or reminder is set by a quick command (``sentient.agent.fastpath``, no model), by the model through the
``reminders`` tools, or over REST. One clock task sleeps until the next one is due (at most ``MAX_SLEEP_S``, so a
computer waking from sleep or a clock change is noticed) and fires it with ``app.notify("reminder", ...)``: the
window shows it (a native notification when the window is not in front), and a reminder set in a Telegram, Discord or
WhatsApp chat goes back to that chat (``ChannelService``), and with ``channels.deliver_reminders`` a reminder or alarm
set anywhere else goes to the delivery chats too. An alarm's notification makes the window ring. One that came due while Sentient was not running fires
as soon as it starts, saying when it was due.

An alarm counts as ringing for ``RING_S`` after it goes off (as long as the window rings), until it is stopped or
snoozed (#390): from the banner, over REST, or by saying "stop" or "snooze" (a quick command). A snooze stops it and
sets a new alarm with the same name, ``snoozed_from`` the one that rang.

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

KINDS = ("timer", "reminder", "alarm")
STATUSES = ("scheduled", "fired", "cancelled")
MAX_SLEEP_S = 30.0
LATE_AFTER_S = 90.0      # fired this much after it was due: the notification says when it was due
MAX_TIMER_S = 24 * 3600
MAX_AHEAD = timedelta(days=366)
KEEP_DAYS = 30           # fired and cancelled ones are forgotten after this long
LIST_LIMIT = 100
RETRY_S = 10.0          # a notification that could not be made is tried again after this long
RING_S = 180.0          # an alarm rings this long at most (the window's chime, desktop stores/alarm.ts)
SNOOZE_MIN, SNOOZE_MAX = 1, 180  # minutes a snooze can be


class ReminderError(ValueError):
    """A timer or reminder that can't be set, in a plain sentence."""


def _row(r: Any) -> dict:
    return {
        "id": r["id"], "kind": r["kind"], "message": r["message"], "connector": r["connector"], "label": r["label"],
        "duration_s": r["duration_s"], "fire_at": r["fire_at"], "status": r["status"], "session_id": r["session_id"],
        "channel": r["channel"], "language": r["language"], "source": r["source"], "created_at": r["created_at"],
        "fired_at": r["fired_at"], "cancelled_at": r["cancelled_at"], "notification_id": r["notification_id"],
        "stopped_at": r["stopped_at"], "snoozed_from": r["snoozed_from"],
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

        await self.app.store.ensure_column("reminders", "stopped_at", "TEXT")  # #390
        await self.app.store.ensure_column("reminders", "snoozed_from", "TEXT")

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
        language: str = "en", source: str = "chat", snoozed_from: str | None = None,
    ) -> dict:
        if kind not in KINDS:
            raise ReminderError("A reminder is a timer, a reminder or an alarm.")
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
            "fired_at": None, "cancelled_at": None, "notification_id": None, "stopped_at": None,
            "snoozed_from": snoozed_from,
        }
        await self.app.store.execute(
            "INSERT INTO reminders(id, kind, message, connector, label, duration_s, fire_at, status, session_id, channel,"
            " language, source, created_at, snoozed_from) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (item["id"], kind, item["message"], connector, item["label"], duration_s, item["fire_at"], "scheduled",
             session_id, channel, item["language"], source, item["created_at"], snoozed_from),
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

    # ------------------------------------------------------------------ a ringing alarm (#390)
    async def ringing(self, now: datetime | None = None) -> list[dict]:
        """The alarms ringing now, newest first: gone off in the last ``RING_S`` and not stopped or snoozed."""
        now = now or datetime.now(UTC)
        rows = await self.app.store.fetchall(
            "SELECT * FROM reminders WHERE kind = 'alarm' AND status = 'fired' AND stopped_at IS NULL"
            " ORDER BY fired_at DESC LIMIT 20"
        )
        out = []
        for r in rows:
            fired = _parse(r["fired_at"])
            if fired is not None and 0 <= (now - fired).total_seconds() <= RING_S:
                out.append(_row(r))
        return out

    async def _claim(self, item: dict, now: datetime) -> dict | None:
        """Mark one ringing alarm stopped; None when something else stopped it first."""
        cur = await self.app.store.execute(
            "UPDATE reminders SET stopped_at = ? WHERE id = ? AND stopped_at IS NULL", (now.isoformat(), item["id"])
        )
        return {**item, "stopped_at": now.isoformat()} if cur.rowcount else None

    async def stop(self, reminder_id: str | None = None, *, now: datetime | None = None) -> list[dict]:
        """Stop the ringing alarms (every one, or only ``reminder_id``). Returns the ones stopped now."""
        now = now or datetime.now(UTC)
        stopped = []
        for item in await self.ringing(now):
            if reminder_id and item["id"] != reminder_id:
                continue
            done = await self._claim(item, now)
            if done is not None:
                log.info("alarm %s stopped", item["id"][:8])
                self._changed(done)
                stopped.append(done)
        return stopped

    async def snooze(
        self, minutes: float, reminder_id: str | None = None, *, source: str = "chat", now: datetime | None = None
    ) -> tuple[dict, dict] | None:
        """Snooze the ringing alarm (the newest, or ``reminder_id``): it stops, and a new alarm with the same name rings
        ``minutes`` from now. Without ``reminder_id`` every other ringing alarm stops too. Returns ``(the one that
        rang, the new alarm)``, or None when nothing is ringing."""
        if not SNOOZE_MIN <= float(minutes) <= SNOOZE_MAX:
            raise ReminderError(f"A snooze can be {SNOOZE_MIN} minute to {SNOOZE_MAX // 60} hours.")
        now = now or datetime.now(UTC)
        ringing = [r for r in await self.ringing(now) if not reminder_id or r["id"] == reminder_id]
        for item in ringing:
            rang = await self._claim(item, now)
            if rang is None:
                continue
            seconds = float(minutes) * 60
            new = await self.add(
                "alarm", now + timedelta(seconds=seconds), label=item["label"], duration_s=seconds,
                session_id=item["session_id"], channel=item["channel"], language=item["language"], source=source,
                snoozed_from=item["id"],
            )
            log.info("alarm %s snoozed for %s minutes", item["id"][:8], minutes)
            self._changed(rang)  # after the new alarm, so the window shows "snoozed" rather than nothing
            if not reminder_id:
                await self.stop(now=now)
            return rang, new
        return None

    def _changed(self, item: dict) -> None:
        self.app.bus.publish("reminder.updated", item)
        self._wake.set()

    # ------------------------------------------------------------------ the clock
    async def _clock(self) -> None:
        await self._forget_old()
        while True:
            self._wake.clear()
            failed = False
            try:
                failed = (await self._fire_all())[1] > 0
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("firing reminders failed")
                failed = True
            nxt = await self._next_due()
            delay = MAX_SLEEP_S
            if nxt is not None:
                delay = min(MAX_SLEEP_S, max(0.0, (nxt - datetime.now(UTC)).total_seconds()))
            if failed:  # never a busy loop on a notification that can't be made
                delay = max(delay, RETRY_S)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=delay)

    async def _next_due(self) -> datetime | None:
        row = await self.app.store.fetchone(
            "SELECT fire_at FROM reminders WHERE status = 'scheduled' ORDER BY fire_at ASC LIMIT 1"
        )
        return _parse(row["fire_at"]) if row else None

    async def fire_due(self, now: datetime | None = None) -> int:
        """Fire every scheduled one that is due. Returns how many fired."""
        return (await self._fire_all(now))[0]

    async def _fire_all(self, now: datetime | None = None) -> tuple[int, int]:
        """``(fired, failed)``. One that fails stays scheduled and is tried again; the others still fire."""
        now = now or datetime.now(UTC)
        fired = failed = 0
        async with self._fire_lock:
            for item in await self.list("scheduled", limit=500):
                due = _parse(item["fire_at"])
                if due is None or due > now:
                    continue
                try:
                    if await self._fire(item, due, now):
                        fired += 1
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("could not fire %s %s; trying again soon", item["kind"], item["id"][:8])
                    failed += 1
        return fired, failed

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
        elif item["kind"] == "alarm":
            title, body = pack.alarm_due(item, due.astimezone(tz), late)
        else:
            title, body = pack.reminder_due(item, due.astimezone(tz) if late else None)
        payload = {"event": item["kind"], "reminder_id": item["id"], "fire_at": item["fire_at"], "late": late,
                   "channel": item.get("channel")}
        if item.get("session_id"):
            payload["session_id"] = item["session_id"]
        try:
            note = await self.app.notify("reminder", body, title=title, payload=payload)
        except Exception:
            made = await self.app.store.fetchone(
                "SELECT id FROM notifications WHERE json_valid(payload) AND json_extract(payload, '$.reminder_id') = ?",
                (item["id"],),
            )
            if made is None:  # nothing reached the user: it goes back to waiting and is tried again
                await self.app.store.execute(
                    "UPDATE reminders SET status = 'scheduled', fired_at = NULL WHERE id = ? AND status = 'fired'"
                    " AND notification_id IS NULL", (item["id"],),
                )
                raise
            note = {"id": made["id"]}  # it was made; only telling the window failed
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
