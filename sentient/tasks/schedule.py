"""Schedule math: timezones, one-off run times and recurring next-run calculation.

Ported from v2 ``workers/tasks.py::calculate_next_run`` and the approve-task route.
All stored timestamps are UTC ISO-8601 strings with seconds precision so they can
be compared lexicographically inside SQLite.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import rrule

log = logging.getLogger(__name__)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_RRULE_DAYS = [rrule.MO, rrule.TU, rrule.WE, rrule.TH, rrule.FR, rrule.SA, rrule.SU]
_OFFSET_RE = re.compile(r"^(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)
_TIME_RE = re.compile(r"^\s*(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------- timezones
def get_tz(name: str | None) -> tzinfo:
    """IANA name, 'UTC', 'auto', or a fixed offset like 'UTC+05:30'. Unknown names fall back to UTC."""
    if not name:
        return UTC
    key = name.strip()
    if key.upper() in {"UTC", "GMT", "Z"}:
        return UTC
    if key.lower() == "auto":
        key = local_timezone_name()
        if key == "UTC":
            return UTC
    try:
        return ZoneInfo(key)
    except (ZoneInfoNotFoundError, ValueError):
        pass
    m = _OFFSET_RE.match(key)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
        return timezone(sign * delta)
    log.warning("unknown timezone %r, using UTC", name)
    return UTC


def _valid_tz(name: str) -> bool:
    if name.upper() in {"UTC", "GMT"} or _OFFSET_RE.match(name):
        return True
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


def local_timezone_name() -> str:
    try:  # optional dependency; gives a real IANA name on every OS
        import tzlocal  # type: ignore[import-not-found]

        return str(tzlocal.get_localzone_name())
    except Exception:
        pass
    offset = datetime.now(UTC).astimezone().utcoffset() or timedelta(0)
    minutes = int(offset.total_seconds() // 60)
    if minutes == 0:
        return "UTC"
    sign = "+" if minutes > 0 else "-"
    minutes = abs(minutes)
    return f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def user_timezone_name(configured: str | None) -> str:
    if configured and configured != "auto" and _valid_tz(configured):
        return configured
    return local_timezone_name()


# ---------------------------------------------------------------------------- timestamps
def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def parse_iso(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def parse_run_at(schedule: dict | None) -> datetime | None:
    """A once-schedule's ``run_at`` in UTC. Naive values are in the schedule's timezone (v2)."""
    if not isinstance(schedule, dict):
        return None
    run_at = schedule.get("run_at")
    if not isinstance(run_at, str) or not run_at.strip():
        return None
    text = run_at.strip().replace("Z", "+00:00")
    if len(text) == 16:
        text += ":00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=get_tz(schedule.get("timezone")))
    return dt.astimezone(UTC)


# ---------------------------------------------------------------------------- recurring
def parse_time(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        m = _TIME_RE.match(value)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2) or 0)
            ampm = (m.group(3) or "").lower()
            if ampm == "pm" and hour < 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            if 0 <= hour < 24 and 0 <= minute < 60:
                return hour, minute
    return 9, 0


def normalize_days(days: Any) -> list[str]:
    if isinstance(days, str):
        days = [d for d in re.split(r"[,\s]+", days) if d]
    if not isinstance(days, list):
        return []
    out: list[str] = []
    for d in days:
        key = str(d).strip().lower()
        if key in {"weekday", "weekdays"}:
            picks = DAY_NAMES[:5]
        elif key in {"weekend", "weekends"}:
            picks = DAY_NAMES[5:]
        elif key in {"daily", "everyday", "all"}:
            picks = DAY_NAMES
        else:
            picks = [n for n in DAY_NAMES if len(key) >= 2 and n.lower().startswith(key[:3])]
        for p in picks:
            if p not in out:
                out.append(p)
    return sorted(out, key=DAY_NAMES.index)


MIN_INTERVAL_MINUTES = 5


def interval_minutes(schedule: dict) -> int:
    """Minutes between runs of an ``interval`` schedule (``interval_minutes``; ``every_minutes``/``hours`` accepted)."""
    raw: Any = schedule.get("interval_minutes") or schedule.get("every_minutes") or schedule.get("minutes")
    try:
        minutes = int(float(raw)) if raw is not None else 0
        if not minutes and schedule.get("hours") is not None:
            minutes = int(float(schedule["hours"]) * 60)
    except (TypeError, ValueError):
        minutes = 0
    return max(MIN_INTERVAL_MINUTES, minutes or 60)


def calculate_next_run(schedule: dict | None, after: datetime | None = None) -> datetime | None:
    """Next execution (UTC) strictly after ``after`` for a recurring schedule, or None."""
    if not isinstance(schedule, dict):
        return None
    tz = get_tz(schedule.get("timezone"))
    now = after or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    if str(schedule.get("frequency") or "").lower() == "interval":
        return now.replace(second=0, microsecond=0) + timedelta(minutes=interval_minutes(schedule))
    start_local = now.astimezone(tz)
    # rrule arithmetic on naive wall-clock times; the zone is re-applied afterwards so
    # daylight-saving transitions keep the user's local HH:MM.
    naive_start = start_local.replace(tzinfo=None)
    hour, minute = parse_time(schedule.get("time"))
    dtstart = naive_start.replace(hour=hour, minute=minute, second=0, microsecond=0)
    until = naive_start + timedelta(days=366)
    frequency = str(schedule.get("frequency") or "").lower()
    try:
        if frequency == "daily":
            rule = rrule.rrule(rrule.DAILY, dtstart=dtstart, until=until)
        elif frequency in {"weekly", "weekdays", "weekday"}:
            days = normalize_days(schedule.get("days")) or (
                ["Monday"] if frequency == "weekly" else DAY_NAMES[:5]
            )
            byweekday = [_RRULE_DAYS[DAY_NAMES.index(d)] for d in days]
            rule = rrule.rrule(rrule.WEEKLY, dtstart=dtstart, byweekday=byweekday, until=until)
        else:
            return None
        nxt = rule.after(naive_start)
    except Exception as exc:
        log.error("could not calculate next run for %s: %s", schedule, exc)
        return None
    if nxt is None:
        return None
    return nxt.replace(tzinfo=tz).astimezone(UTC)


# ---------------------------------------------------------------------------- normalization
def normalize_schedule(schedule: Any, tz_name: str, *, override_timezone: bool = True) -> dict:
    """Coerce a model- or user-provided schedule into one of the three documented shapes."""
    if isinstance(schedule, str):
        try:
            schedule = json.loads(schedule)
        except json.JSONDecodeError:
            schedule = None
    if not isinstance(schedule, dict):
        return {"type": "once", "run_at": None, "timezone": tz_name}
    out = dict(schedule)
    kind = str(out.get("type") or "").lower()
    if kind not in {"once", "recurring", "triggered"}:
        if out.get("frequency"):
            kind = "recurring"
        elif out.get("source") or out.get("event"):
            kind = "triggered"
        else:
            kind = "once"
    out["type"] = kind
    catch_up = str(out.pop("catch_up", None) or "").strip().lower()
    if kind != "triggered" and catch_up in {"run", "skip"}:  # missed while asleep (tasks/catchup.py); default auto
        out["catch_up"] = catch_up
    if kind == "once":
        run_at = out.get("run_at")
        if not isinstance(run_at, str) or run_at.strip().lower() in {"", "null", "none", "now"}:
            out["run_at"] = None
    elif kind == "recurring":
        freq = str(out.get("frequency") or "daily").lower()
        if freq in {"weekday", "weekdays"}:
            freq, out["days"] = "weekly", DAY_NAMES[:5]
        if freq == "hourly":
            freq, out["interval_minutes"] = "interval", out.get("interval_minutes") or 60
        if freq not in {"daily", "weekly", "interval"}:
            freq = "daily"
        out["frequency"] = freq
        if freq == "interval":  # additive: every N minutes (used by script jobs)
            out["interval_minutes"] = interval_minutes(out)
            for key in ("days", "time", "every_minutes", "minutes", "hours"):
                out.pop(key, None)
            if override_timezone or not out.get("timezone"):
                out["timezone"] = tz_name
            return out
        hour, minute = parse_time(out.get("time"))
        out["time"] = f"{hour:02d}:{minute:02d}"
        if freq == "weekly":
            out["days"] = normalize_days(out.get("days")) or ["Monday"]
        else:
            out.pop("days", None)
    else:
        out["source"] = str(out.get("source") or "").strip().lower()
        out["event"] = str(out.get("event") or "").strip()
        if not isinstance(out.get("filter"), dict):
            out["filter"] = {}
    if override_timezone or not out.get("timezone"):
        out["timezone"] = tz_name
    return out
