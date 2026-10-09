"""Catching up on scheduled runs missed while the computer was off or asleep (issue #135).

The scheduler claims due tasks every ``tasks.tick_seconds``, so a run whose time passed more than ``grace_seconds`` ago
without starting was missed: the computer was off or asleep, Sentient was not running, or Stop everything was on.
Each missed task runs once or is skipped, by its schedule's ``catch_up``:

- ``auto`` (the default, stored as no key): run once now if it is less than ``tasks.catch_up_window_hours`` late,
  otherwise skip it;
- ``run``: always run once now, however late;
- ``skip``: never catch up.

Never more than once: a recurring task's next time is computed from now when the run ends, so missed occurrences are
not replayed one by one. ``interval`` schedules (every N minutes) run once without a notice, since their next check
is due anyway, unless they say ``skip``. Deterministic: no model decides.

Quiet fixed-call tasks (the Daily and Evening Brief, whose tool sends its own notification) follow the same window but
are never listed in the catch-up notice, and only run if it is still the day they were due: a brief is about its day,
so a missed one from an earlier day is skipped quietly.
"""

from __future__ import annotations

from datetime import datetime, tzinfo
from typing import Any

POLICIES = ("run", "skip")  # plus the default, "auto"
WAKE_GAP_S = 120  # scheduler ticks further apart than tick_seconds plus this mean the computer was asleep
TITLES = {
    "start": "Caught up after Sentient was off",
    "sleep": "Caught up after sleep",
    "resume": "Caught up after resuming",
}
MAX_NAMES = 5


def grace_seconds(tick_seconds: float) -> float:
    """How late a run may start and still be on time (a few scheduler ticks)."""
    return max(300.0, 3.0 * tick_seconds)


def policy(schedule: dict | None) -> str:
    value = str((schedule or {}).get("catch_up") or "").strip().lower()
    return value if value in POLICIES else "auto"


def decide(schedule: dict | None, late_s: float, window_hours: float) -> str:
    """``run`` (once, now), ``skip``, or ``quiet`` (run once without telling: an interval check)."""
    schedule = schedule or {}
    chosen = policy(schedule)
    if schedule.get("type") == "recurring" and str(schedule.get("frequency") or "") == "interval":
        return "skip" if chosen == "skip" else "quiet"  # an explicit skip still wins
    if chosen == "auto":
        chosen = "run" if window_hours > 0 and late_s < window_hours * 3600 else "skip"
    return chosen


def day_over(due: datetime, now: datetime, tz: tzinfo) -> bool:
    """True when ``now`` is a later local day than ``due`` (a missed brief from an earlier day is not worth running)."""
    return now.astimezone(tz).date() != due.astimezone(tz).date()


def _names(items: list[dict]) -> str:
    names = [f"'{i['name']}'" for i in items[:MAX_NAMES]]
    more = len(items) - len(names)
    return ", ".join(names) + (f" and {more} more" if more > 0 else "")


def summary(reason: str, ran: list[dict], skipped: list[dict]) -> tuple[str, str]:
    """``(title, message)`` of the one notification a catch-up sends."""
    title = f"{TITLES.get(reason, TITLES['start'])}: ran {len(ran)}, skipped {len(skipped)}"
    lines: list[str] = []
    if ran:
        lines.append(f"Ran once now: {_names(ran)}.")
    if skipped:
        lines.append(f"Skipped: {_names(skipped)}. Open one and choose Run now if you still want it.")
    return title, "\n".join(lines)


def item(task: dict, due_at: Any) -> dict:
    return {"task_id": task["id"], "name": task.get("name") or "Untitled task", "due_at": due_at}
