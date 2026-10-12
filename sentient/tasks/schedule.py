"""Schedule math: timezones, one-off run times and recurring next-run calculation.

Ported from v2 ``workers/tasks.py::calculate_next_run`` and the approve-task route.
All stored timestamps are UTC ISO-8601 strings with seconds precision so they can
be compared lexicographically inside SQLite.
"""

from __future__ import annotations

import json
import logging
import random
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

# Fuzzy windows (#157): a daily or weekly schedule with ``window`` runs at a random minute inside that part of the
# day instead of at ``time``, local time, ``[start, end)`` in minutes after midnight.
WINDOWS: dict[str, tuple[int, int]] = {
    "morning": (7 * 60, 10 * 60),
    "afternoon": (12 * 60, 16 * 60),
    "evening": (17 * 60, 20 * 60),
    "night": (20 * 60, 23 * 60),
}
_window_random = random.Random()  # tests may seed it


# Words that name each window in a request; a model-proposed window the user never said is dropped.
WINDOW_WORDS: dict[str, tuple[str, ...]] = {
    "morning": ("morning", "mornings"),
    "afternoon": ("afternoon", "afternoons"),
    "evening": ("evening", "evenings"),
    "night": ("night", "nights", "tonight"),
}


def spoken_window(schedule: dict, prompt: str) -> dict:
    """Keep a model-proposed ``window`` only when the request names that part of the day (small models add one to
    requests like "every 5 minutes")."""
    window = schedule.get("window") if isinstance(schedule, dict) else None
    if not window:
        return schedule
    words = set(re.findall(r"[a-z]+", str(prompt or "").lower()))
    if words & set(WINDOW_WORDS.get(window, ())):
        return schedule
    return {k: v for k, v in schedule.items() if k != "window"}


# "every 10 minutes", "every hour", "every 2 hours", "every 30 min" (#305): a request that repeats within a day is
# an interval, whatever a small model wrote (they turn these into a daily 09:00 schedule).
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
    "forty five": 45, "sixty": 60, "ninety": 90,
}
_UNIT_MINUTES = {"s": 1 / 60, "m": 1, "h": 60}
_EVERY_N_RE = re.compile(
    r"\b(?:every|each)\s+(\d+(?:\.\d+)?|" + "|".join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True))
    + r")[\s-]*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b"
)
_EVERY_UNIT_RE = re.compile(r"\b(?:every|each)\s+(?:(other|single)\s+)?(minute|hour)\b")  # not "every second Friday"
_EVERY_PART_RE = re.compile(r"\bevery\s+(half|quarter)[\s-]+(?:of\s+)?(?:an\s+)?hour\b")
_HOURLY_RE = re.compile(r"\bhourly\b")
_DAY_LEVEL_RE = re.compile(r"\b(?:daily|weekly|every\s+(?:day|week|weekday|morning|evening|night)|weekdays?)\b")


def spoken_interval(prompt: str) -> int | None:
    """Minutes between runs when the request says so ("every 10 minutes", "every hour", "hourly"), else None.
    At least ``MIN_INTERVAL_MINUTES``."""
    text = str(prompt or "").lower()
    minutes: float | None = None
    if m := _EVERY_N_RE.search(text):
        raw = m.group(1)
        count = float(raw) if raw[0].isdigit() else _NUMBER_WORDS[raw]
        minutes = count * _UNIT_MINUTES[m.group(2)[0]]
    elif m := _EVERY_PART_RE.search(text):
        minutes = 30 if m.group(1) == "half" else 15
    elif m := _EVERY_UNIT_RE.search(text):
        minutes = _UNIT_MINUTES[m.group(2)[0]] * (2 if m.group(1) == "other" else 1)
    elif _HOURLY_RE.search(text) and not _DAY_LEVEL_RE.search(text):  # "weekly hourly" is weekly
        minutes = 60
    if minutes is None or minutes <= 0:
        return None
    return max(MIN_INTERVAL_MINUTES, round(minutes))


def spoken_schedule(schedule: dict, prompt: str) -> dict:
    """The refined schedule checked against the request's own words: an interval the user said wins over the
    model's daily or one-off guess, and a ``window`` the user never named is dropped. Triggered schedules stay."""
    minutes = spoken_interval(prompt)
    if minutes is not None and isinstance(schedule, dict) and schedule.get("type") != "triggered":
        keep = {k: schedule[k] for k in ("timezone", "catch_up") if k in schedule}
        return {"type": "recurring", "frequency": "interval", "interval_minutes": minutes, **keep}
    return spoken_window(schedule, prompt)


def window_of(schedule: dict | None) -> tuple[int, int] | None:
    """``(start, end)`` minutes of a daily/weekly schedule's ``window``, or None."""
    if not isinstance(schedule, dict) or str(schedule.get("frequency") or "").lower() == "interval":
        return None
    return WINDOWS.get(str(schedule.get("window") or "").strip().lower())


def interval_minutes(schedule: dict) -> int:
    """Minutes between runs of an ``interval`` schedule (``interval_minutes``; ``every_minutes``/``hours`` accepted)."""
    raw: Any = (
        schedule.get("interval_minutes") or schedule.get("every_minutes") or schedule.get("minutes")
        or schedule.get("interval")
    )
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
    window = window_of(schedule)
    hour, minute = divmod(window[0], 60) if window else parse_time(schedule.get("time"))
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
        if nxt is not None and window:
            # the next window that has not started yet (so a run inside today's window is not repeated today),
            # at a random minute inside it
            nxt += timedelta(minutes=_window_random.randrange(window[1] - window[0]))
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
            for key in ("days", "time", "window", "every_minutes", "minutes", "hours", "interval"):
                out.pop(key, None)
            if override_timezone or not out.get("timezone"):
                out["timezone"] = tz_name
            return out
        hour, minute = parse_time(out.get("time"))
        out["time"] = f"{hour:02d}:{minute:02d}"
        window = str(out.pop("window", None) or "").strip().lower()
        if window in WINDOWS:  # a fuzzy window replaces the exact time (``time`` is kept for the editor)
            out["window"] = window
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
