"""Tools for timers, one-off reminders and alarms (#150, #359). Registered by ``ReminderService.start()``.

Quick commands (``sentient.agent.fastpath``) call these same tools without the model, so a timer set either way shows
the same in the chat and on the Activity timeline. Everything here stays inside Sentient (``internal``): it never
sends, buys or deletes anything.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool


def _service(ctx: ToolContext) -> Any:
    return getattr(ctx.extra.get("app"), "reminders", None)


def _where(ctx: ToolContext) -> dict:
    """Where it was set: the chat (none for an incognito chat, which is deleted when closed) and the surface."""
    quick = bool(ctx.extra.get("quick"))
    return {
        "session_id": None if ctx.incognito else ctx.session_id,
        "channel": ctx.channel,
        "language": str(ctx.extra.get("quick_language") or getattr(ctx.config.assistant, "language", "en") or "en"),
        "source": "quick" if quick else "chat",
    }


def _local_tz(ctx: ToolContext):
    from sentient.tasks.schedule import get_tz, user_timezone_name

    return get_tz(user_timezone_name(ctx.config.assistant.timezone))


def _shown(item: dict, ctx: ToolContext) -> dict:
    at = datetime.fromisoformat(item["fire_at"]).astimezone(_local_tz(ctx))
    left = max(0, int((at - datetime.now(at.tzinfo)).total_seconds())) if item["status"] == "scheduled" else 0
    return {**item, "fire_at_local": at.strftime("%Y-%m-%d %H:%M:%S"), "seconds_left": left}


@tool("set_timer", risk=Risk.write, internal=True)
async def set_timer(ctx: ToolContext, minutes: float = 0, seconds: int = 0, hours: float = 0, label: str = "") -> dict:
    """Start a countdown timer that notifies the user when it ends (1 second to 24 hours). `label` names it
    ("pasta"). For "remind me to ..." use set_reminder."""
    svc = _service(ctx)
    if svc is None:
        return {"error": "Timers are not available right now."}
    from sentient.reminders.service import ReminderError

    length = float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(seconds or 0)
    try:
        item = await svc.add("timer", datetime.now(UTC) + timedelta(seconds=length), label=label, duration_s=length,
                             **_where(ctx))
    except ReminderError as exc:
        return {"error": str(exc)}
    return {"status": "success", "reminder": _shown(item, ctx)}


@tool("set_reminder", risk=Risk.write, internal=True)
async def set_reminder(
    ctx: ToolContext, message: str = "", at: str = "", in_minutes: float = 0, connector: str = "to"
) -> dict:
    """Remind the user once, with a notification, at a time. Give `at` as a local date and time "YYYY-MM-DD HH:MM"
    (check current_datetime first) or `in_minutes` from now. `message` is what to remind about, in the user's words
    ("call Ravi"). For repeating reminders, or work to do at that time, use create_task_from_prompt instead."""
    svc = _service(ctx)
    if svc is None:
        return {"error": "Reminders are not available right now."}
    from sentient.reminders.service import ReminderError

    duration: float | None = None
    if in_minutes and in_minutes > 0:
        duration = float(in_minutes) * 60
        when = datetime.now(UTC) + timedelta(seconds=duration)
    elif at:
        try:
            when = datetime.fromisoformat(at.strip().replace("Z", "+00:00").replace(" ", "T", 1))
        except ValueError:
            return {"error": f"I couldn't read the time {at!r}: use YYYY-MM-DD HH:MM."}
        if when.tzinfo is None:
            when = when.replace(tzinfo=_local_tz(ctx))
    else:
        return {"error": "Say when: `at` (YYYY-MM-DD HH:MM) or `in_minutes`."}
    try:
        item = await svc.add("reminder", when, message=message, connector=connector if message else "",
                             duration_s=duration, **_where(ctx))
    except ReminderError as exc:
        return {"error": str(exc)}
    return {"status": "success", "reminder": _shown(item, ctx)}


@tool("set_alarm", risk=Risk.write, internal=True)
async def set_alarm(ctx: ToolContext, at: str = "", in_minutes: float = 0, label: str = "") -> dict:
    """Set an alarm that rings on the desktop with a sound at a time. Give `at` as a local date and time
    "YYYY-MM-DD HH:MM" (check current_datetime first) or `in_minutes` from now. `label` names it ("gym"). For a
    message at a time use set_reminder; for a countdown use set_timer."""
    svc = _service(ctx)
    if svc is None:
        return {"error": "Alarms are not available right now."}
    from sentient.reminders.service import ReminderError

    duration: float | None = None
    if in_minutes and in_minutes > 0:
        duration = float(in_minutes) * 60
        when = datetime.now(UTC) + timedelta(seconds=duration)
    elif at:
        try:
            when = datetime.fromisoformat(at.strip().replace("Z", "+00:00").replace(" ", "T", 1))
        except ValueError:
            return {"error": f"I couldn't read the time {at!r}: use YYYY-MM-DD HH:MM."}
        if when.tzinfo is None:
            when = when.replace(tzinfo=_local_tz(ctx))
    else:
        return {"error": "Say when: `at` (YYYY-MM-DD HH:MM) or `in_minutes`."}
    try:
        item = await svc.add("alarm", when, label=label, duration_s=duration, **_where(ctx))
    except ReminderError as exc:
        return {"error": str(exc)}
    return {"status": "success", "reminder": _shown(item, ctx)}


@tool("list_reminders", risk=Risk.read)
async def list_reminders(ctx: ToolContext, kind: str = "any") -> dict:
    """The timers, one-off reminders and alarms that have not gone off yet, soonest first, with the time left.
    `kind`: "timer", "reminder", "alarm" or "any"."""
    svc = _service(ctx)
    if svc is None:
        return {"error": "Reminders are not available right now."}
    items = await svc.list("scheduled", kind=kind if kind in {"timer", "reminder", "alarm"} else None)
    return {"status": "success", "reminders": [_shown(i, ctx) for i in items]}


@tool("cancel_reminder", risk=Risk.write, internal=True)
async def cancel_reminder(ctx: ToolContext, reminder_id: str) -> dict:
    """Cancel a timer, reminder or alarm that has not gone off yet, by its id from list_reminders."""
    svc = _service(ctx)
    if svc is None:
        return {"error": "Reminders are not available right now."}
    item, changed = await svc.cancel(reminder_id)
    if item is None:
        return {"error": f"There is no timer, reminder or alarm with id {reminder_id}."}
    if not changed:
        return {"error": "It already went off." if item["status"] == "fired" else "It was already cancelled."}
    return {"status": "success", "reminder": _shown(item, ctx)}


@tool("stop_alarm", risk=Risk.write, internal=True)
async def stop_alarm(ctx: ToolContext, label: str = "") -> dict:
    """Stop the Sentient alarm that is ringing right now (one set with set_alarm). `label`: its name if the user said
    one ("gym"), else "". Not for a house alarm."""
    svc = _service(ctx)
    if svc is None:
        return {"error": "Alarms are not available right now."}
    # a tool with at least one argument: qwen3 on Ollama often says it will call an argument-free tool and then doesn't
    wanted = " ".join(str(label or "").lower().split())
    ringing = [r for r in await svc.ringing() if not wanted or wanted in str(r.get("label") or "").lower()]
    stopped = [item for r in ringing for item in await svc.stop(r["id"])]
    if not stopped:
        return {"error": f"No Sentient alarm{f' called {label}' if wanted else ''} is ringing right now."}
    return {"status": "success", "stopped": [_shown(i, ctx) for i in stopped]}


@tool("snooze_alarm", risk=Risk.write, internal=True)
async def snooze_alarm(ctx: ToolContext, minutes: float = 0) -> dict:
    """Snooze the Sentient alarm that is ringing right now: it stops and rings again `minutes` from now (0: the usual
    snooze length). Not for a house alarm."""
    svc = _service(ctx)
    if svc is None:
        return {"error": "Alarms are not available right now."}
    from sentient.reminders.service import ReminderError

    length = float(minutes or 0) or float(ctx.config.chat.alarm_snooze_minutes)
    try:
        done = await svc.snooze(length, source="quick" if ctx.extra.get("quick") else "chat")
    except ReminderError as exc:
        return {"error": str(exc)}
    if done is None:
        return {"error": "No Sentient alarm is ringing right now."}
    rang, new = done
    return {"status": "success", "stopped": _shown(rang, ctx), "reminder": _shown(new, ctx)}


class RemindersPlugin(ToolPlugin):
    id = "reminders"
    display_name = "Timers & reminders"
    description = "Countdown timers, one-off reminders and alarms that notify you at the right time."
    category = "core"
    icon = "IconAlarm"
    selection_hint = ("timers, countdowns, alarms, wake me up, remind me once at a time or in some minutes, cancel or list "
                      "them, stop or snooze a ringing alarm")
    tools = [set_timer, set_reminder, set_alarm, list_reminders, cancel_reminder, stop_alarm, snooze_alarm]
