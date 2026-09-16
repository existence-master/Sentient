from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool


def system_tz_name() -> str | None:
    """IANA name of the OS timezone (e.g. Asia/Kolkata), or None if it cannot be determined."""
    try:
        from tzlocal import get_localzone_name

        return get_localzone_name()
    except Exception:
        return None


def resolve_tz(name: str) -> ZoneInfo | None:
    if not name or name == "auto":
        iana = system_tz_name()
        if iana:
            try:
                return ZoneInfo(iana)
            except ZoneInfoNotFoundError:
                pass
        try:
            return datetime.now().astimezone().tzinfo  # type: ignore[return-value]
        except Exception:
            return None
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return None


@tool("current_datetime", risk=Risk.read)
async def current_datetime(ctx: ToolContext, timezone: str | None = None) -> dict:
    """Get the current date, time, weekday and timezone. Use before any date arithmetic
    ("next Friday", "in 3 days") so you never guess the date."""
    tz = resolve_tz(timezone or ctx.config.assistant.timezone)
    now = datetime.now(tz)
    return {
        "iso": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M"),
        "weekday": now.strftime("%A"),
        "timezone": str(now.tzinfo),
    }


class TimePlugin(ToolPlugin):
    id = "time"
    display_name = "Date & time"
    description = "Knows what time it is."
    category = "core"
    icon = "IconClock"
    selection_hint = "dates, times, scheduling arithmetic"
    tools = [current_datetime]


PLUGIN = TimePlugin()
