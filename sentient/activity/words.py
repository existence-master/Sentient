"""Plain sentences for the activity timeline (#147): what happened and why, in words a non-engineer reads at a
glance. No jargon and no em-dashes (UI copy rules)."""

from __future__ import annotations

import re
from datetime import tzinfo
from typing import Any

from sentient.agent.fastpath import en as english
from sentient.tasks.schedule import DAY_NAMES, WINDOWS, interval_minutes, normalize_days, parse_iso

# Apps that are not tool plugins: messaging channels and trigger sources.
APP_LABELS = {
    "telegram": "Telegram", "discord": "Discord", "whatsapp": "WhatsApp", "gmail": "Gmail",
    "gcalendar": "Google Calendar", "email_imap": "Email", "webhook": "Webhook", "slack": "Slack",
    "github": "GitHub", "notion": "Notion", "trello": "Trello", "web": "Web", "voice": "Voice",
}
EVENT_PHRASES = {
    "new_email": "a new email arrives", "new_event": "a new calendar event is added",
    "updated_event": "a calendar event changes", "new_message": "a new message arrives",
    "new_issue": "a new issue is opened", "new_pull_request": "a new pull request is opened",
}
# a notification sent to a paired chat (#345): why it went there, and what it was
DELIVERY_REASONS = {
    "chosen": "The task sends its updates there",
    "default": "That chat gets Sentient's notifications",
    "chat": "A helper you started in that chat finished",
    "reminder": "You set it in that chat",
    "reminder_copy": "Your reminders and alarms go to this chat too",
}
DELIVERY_WHAT = {
    "task": "A task update", "question": "A question from a task", "approval": "An approval request from a task",
    "plan": "A plan waiting for your OK", "brief": "Your brief", "suggestion": "A suggestion",
    "subagent": "Background work that finished", "reminder": "A timer, reminder or alarm you set",
}
WINDOW_PHRASES = {"morning": "in the morning", "afternoon": "in the afternoon", "evening": "in the evening",
                  "night": "at night"}


def clock(hour: int, minute: int) -> str:
    """ "7:30 AM"."""
    return f"{hour % 12 or 12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"


def clock_text(value: Any) -> str:
    """ "07:30" -> "7:30 AM"; anything unreadable as it is."""
    try:
        hour, minute = (int(p) for p in str(value or "09:00").split(":")[:2])
    except ValueError:
        return str(value)
    return clock(hour, minute)


def when_text(value: Any, tz: tzinfo) -> str:
    """ "Tue 14 Oct at 9:00 AM" in the user's timezone."""
    dt = parse_iso(value)
    if dt is None:
        return ""
    local = dt.astimezone(tz)
    return f"{local:%a} {local.day} {local:%b} at {clock(local.hour, local.minute)}"


def every_minutes(minutes: int) -> str:
    if minutes % 1440 == 0:
        return "once a day" if minutes == 1440 else f"every {minutes // 1440} days"
    if minutes % 60 == 0:
        return "every hour" if minutes == 60 else f"every {minutes // 60} hours"
    return f"every {minutes} minutes"


def days_text(schedule: dict) -> str:
    if str(schedule.get("frequency") or "").lower() == "daily":
        return "every day"
    days = normalize_days(schedule.get("days")) or (
        ["Monday"] if str(schedule.get("frequency") or "").lower() == "weekly" else DAY_NAMES[:5]
    )
    if len(days) == 7:
        return "every day"
    if days == DAY_NAMES[:5]:
        return "every weekday"
    if days == DAY_NAMES[5:]:
        return "every weekend"
    if len(days) == 1:
        return f"every {days[0]}"
    return "every " + ", ".join(days[:-1]) + f" and {days[-1]}"


def app_label(app: str | None, registry: Any = None) -> str | None:
    if not app:
        return None
    plugin = registry.plugin(app) if registry is not None else None
    return getattr(plugin, "display_name", None) or APP_LABELS.get(app) or app.replace("_", " ").title()


def trigger_text(schedule: dict, registry: Any = None) -> str:
    source = str(schedule.get("source") or "")
    if source == "webhook":
        return "your webhook is called"
    event = str(schedule.get("event") or "")
    what = EVENT_PHRASES.get(event) or (f"{event.replace('_', ' ')} happens" if event else "something happens")
    return f"{what} in {app_label(source, registry) or 'an app'}"


def schedule_why(schedule: Any, tz: tzinfo, *, verb: str = "runs", registry: Any = None) -> str:
    """Why a task ran or will run, from its schedule: "It runs every weekday at 7:30 AM"."""
    if not isinstance(schedule, dict):
        return "You asked for it"
    kind = str(schedule.get("type") or "")
    if kind == "recurring":
        if str(schedule.get("frequency") or "").lower() == "interval":
            return f"It {verb} {every_minutes(interval_minutes(schedule))}"
        window = str(schedule.get("window") or "").strip().lower()
        if window in WINDOWS:
            return f"It {verb} {days_text(schedule)} {WINDOW_PHRASES[window]}"
        return f"It {verb} {days_text(schedule)} at {clock_text(schedule.get('time'))}"
    if kind == "triggered":
        return f"It {verb} when {trigger_text(schedule, registry)}"
    run_at = schedule.get("run_at")
    if run_at:
        return f"You set it for {when_text(run_at, tz) or run_at}"
    return "You asked for it"


_LINE_MARK = re.compile(r"^\s*(#{1,6}|>|[-*+]|\d+\.)\s+", re.MULTILINE)  # headings, quotes, list bullets
_TABLE = re.compile(r"\||:?-{3,}:?")  # table pipes and rules


def short(text: Any, limit: int = 160) -> str:
    """One line of plain text (markdown marks dropped), at most ``limit`` characters, cut at a word."""
    plain = _TABLE.sub(" ", _LINE_MARK.sub("", str(text or ""))).replace("*", "").replace("`", "")
    flat = " ".join(plain.split())
    if len(flat) <= limit:
        return flat
    cut = flat.rfind(" ", 0, limit)
    return flat[: cut if cut > limit // 2 else limit].rstrip(" ,.;:") + "..."


def reminder_name(reminder: dict) -> str:
    """ "pasta timer", "10 minute timer", "reminder to call Ravi" (#150)."""
    return english.name(reminder)


def money(usd: float) -> str:
    return f"${usd:,.2f}" if usd >= 0.01 else "less than $0.01"
