"""Google Calendar over the REST API."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from sentient.integrations.base import FeedBatch, IntegrationError, itool, manager_from
from sentient.integrations.common import event_blocked
from sentient.integrations.google import gapi
from sentient.integrations.plugins._google_base import GooglePlugin
from sentient.tools.base import Risk, ToolContext
from sentient.tools.builtin.time_tool import resolve_tz

if TYPE_CHECKING:  # pragma: no cover
    from sentient.integrations.service import IntegrationManager

API = "https://www.googleapis.com/calendar/v3"
PID = "gcalendar"


def _tz(ctx_or_mgr: Any, override: str | None = None) -> Any:
    cfg = ctx_or_mgr.app.config if hasattr(ctx_or_mgr, "app") else manager_from(ctx_or_mgr).app.config
    if override:
        try:
            return ZoneInfo(override)
        except Exception:
            pass
    return resolve_tz(cfg.assistant.timezone) or UTC


def _tz_name(tz: Any) -> str:
    return getattr(tz, "key", None) or "UTC"


def parse_when(value: str, tz: Any) -> tuple[datetime | None, date | None]:
    """'2026-09-16' -> (None, date); '2026-09-16T10:00' / with offset -> (aware datetime, None)."""
    v = value.strip().replace("Z", "+00:00")
    if len(v) == 10:
        return None, date.fromisoformat(v)
    try:
        dt = datetime.fromisoformat(v.replace(" ", "T"))
    except ValueError as exc:
        raise IntegrationError(f"'{value}' isn't a date/time I understand. Use ISO format like 2026-09-16T15:00.") from exc
    return (dt if dt.tzinfo else dt.replace(tzinfo=tz)), None


def rfc3339(value: str | None, tz: Any, *, end_of_day: bool = False) -> str | None:
    if not value:
        return None
    dt, d = parse_when(value, tz)
    if d is not None:
        dt = datetime.combine(d, time.max if end_of_day else time.min, tz)
    assert dt is not None
    return dt.isoformat()


def normalize_event(e: dict) -> dict:
    start = e.get("start") or {}
    end = e.get("end") or {}
    return {
        "id": e.get("id"),
        "summary": e.get("summary") or "(no title)",
        "description": e.get("description") or "",
        "start": start.get("dateTime") or start.get("date"),
        "end": end.get("dateTime") or end.get("date"),
        "all_day": "date" in start and "dateTime" not in start,
        "location": e.get("location"),
        "attendees": [a.get("email") for a in e.get("attendees") or [] if a.get("email")],
        "organizer_email": (e.get("organizer") or {}).get("email"),
        "url": e.get("htmlLink"),
        "status": e.get("status"),
        "created": e.get("created"),
        "updated": e.get("updated"),
        "meet_link": e.get("hangoutLink"),
    }


async def _list(ctx: ToolContext | None, mgr: IntegrationManager, calendar_id: str, params: dict) -> list[dict]:
    res = await gapi(ctx, PID, "GET", f"{API}/calendars/{quote(calendar_id, safe='@')}/events", params=params, mgr=mgr)
    return [normalize_event(e) for e in res.get("items") or [] if e.get("status") != "cancelled"]


def _time_body(start: str, end: str | None, duration_minutes: int, all_day: bool, tz: Any) -> dict:
    s_dt, s_d = parse_when(start, tz)
    if all_day or s_d is not None:
        s_day = s_d or (s_dt.date() if s_dt else None)
        assert s_day is not None
        if end:
            e_dt, e_d = parse_when(end, tz)
            e_day = (e_d or (e_dt.date() if e_dt else s_day)) + timedelta(days=1 if e_d is None else 0)
            e_day = max(e_day, s_day + timedelta(days=1))
        else:
            e_day = s_day + timedelta(days=1)
        return {"start": {"date": s_day.isoformat()}, "end": {"date": e_day.isoformat()}}
    assert s_dt is not None
    if end:
        e_dt, e_d = parse_when(end, tz)
        e_dt = e_dt or datetime.combine(e_d, time.min, tz)  # type: ignore[arg-type]
    else:
        e_dt = s_dt + timedelta(minutes=max(5, int(duration_minutes or 60)))
    if e_dt <= s_dt:
        raise IntegrationError("The event must end after it starts.")
    name = _tz_name(tz)
    return {"start": {"dateTime": s_dt.isoformat(), "timeZone": name}, "end": {"dateTime": e_dt.isoformat(), "timeZone": name}}


# ---------------------------------------------------------------------------- tools
@itool(PID, "gcal_list_events")
async def gcal_list_events(ctx: ToolContext, time_min: str | None = None, time_max: str | None = None,
                           query: str | None = None, calendar_id: str = "primary", max_results: int = 25) -> dict:
    """List or search calendar events in a time range. Dates like "2026-09-16" or "2026-09-16T09:00" (user's timezone).
    Defaults: from now to 7 days ahead; with only time_min, 30 days after it. `query` searches titles, descriptions, places, attendees."""
    mgr = manager_from(ctx)
    tz = _tz(mgr)
    now = datetime.now(tz)
    t_min = rfc3339(time_min, tz) or now.isoformat()
    if time_max:
        t_max = rfc3339(time_max, tz, end_of_day=True)
    else:
        base = datetime.fromisoformat(t_min)
        t_max = (base + timedelta(days=30 if time_min else 7)).isoformat()
    params: dict[str, Any] = {"timeMin": t_min, "timeMax": t_max, "singleEvents": "true", "orderBy": "startTime",
                              "maxResults": max(1, min(int(max_results or 25), 250))}
    if query:
        params["q"] = query
    items = await _list(ctx, mgr, calendar_id, params)
    kept = await mgr.filter_items(PID, items, "event")
    out: dict[str, Any] = {"time_min": t_min, "time_max": t_max, "timezone": _tz_name(tz), "count": len(kept), "events": kept}
    if len(kept) < len(items):
        out["hidden_by_privacy_filters"] = len(items) - len(kept)
    return out


def invites_others(arguments: dict, ctx: Any) -> bool:
    """True when the call puts people other than the calendar's owner on the event (or emails its guests):
    they see its details, so after outside content it asks first (ADR 0018)."""
    owner = str(arguments.get("calendar_id") or "").strip().lower()
    guests = [str(a or "").strip().lower() for a in arguments.get("attendees") or []]
    return any(g and g != owner for g in guests) or bool(arguments.get("send_updates"))


@itool(PID, "gcal_create_event", risk=Risk.write, exfiltrates=invites_others)
async def gcal_create_event(ctx: ToolContext, summary: str, start: str, end: str | None = None,
                            duration_minutes: int = 60, description: str | None = None, location: str | None = None,
                            attendees: list[str] | None = None, all_day: bool = False, add_meet_link: bool = False,
                            send_invites: bool = False, calendar_id: str = "primary") -> dict:
    """Create a calendar event. `start`/`end` like "2026-09-16T15:00" in the user's timezone (or a date for all-day).
    Without `end`, lasts `duration_minutes`. Guests only get email invitations when send_invites=true."""
    tz = _tz(ctx)
    body: dict[str, Any] = {"summary": summary, **_time_body(start, end, duration_minutes, all_day, tz)}
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    if attendees:
        body["attendees"] = [{"email": a.strip()} for a in attendees if a.strip()]
    params: dict[str, Any] = {"sendUpdates": "all" if send_invites else "none"}
    if add_meet_link:
        body["conferenceData"] = {"createRequest": {"requestId": f"sentient-{datetime.now(UTC).timestamp()}",
                                                    "conferenceSolutionKey": {"type": "hangoutsMeet"}}}
        params["conferenceDataVersion"] = 1
    res = await gapi(ctx, PID, "POST", f"{API}/calendars/{quote(calendar_id, safe='@')}/events", params=params, json=body)
    return {"created": True, "event": normalize_event(res)}


@itool(PID, "gcal_update_event", risk=Risk.write, exfiltrates=invites_others)
async def gcal_update_event(ctx: ToolContext, event_id: str, summary: str | None = None, start: str | None = None,
                            end: str | None = None, description: str | None = None, location: str | None = None,
                            attendees: list[str] | None = None, send_updates: bool = False,
                            calendar_id: str = "primary") -> dict:
    """Change an existing event (only the fields you pass). Moving `start` without `end` keeps the event's length."""
    tz = _tz(ctx)
    path = f"{API}/calendars/{quote(calendar_id, safe='@')}/events/{event_id}"
    body: dict[str, Any] = {}
    if summary is not None:
        body["summary"] = summary
    if description is not None:
        body["description"] = description
    if location is not None:
        body["location"] = location
    if attendees is not None:
        body["attendees"] = [{"email": a.strip()} for a in attendees if a.strip()]
    if start is not None:
        current = await gapi(ctx, PID, "GET", path)
        cur = normalize_event(current)
        duration = 60
        if cur["start"] and cur["end"] and not cur["all_day"]:
            duration = int((datetime.fromisoformat(cur["end"]) - datetime.fromisoformat(cur["start"])).total_seconds() // 60)
        body.update(_time_body(start, end, duration, cur["all_day"] and len(start.strip()) == 10, tz))
    elif end is not None:
        e_dt, e_d = parse_when(end, tz)
        body["end"] = {"date": e_d.isoformat()} if e_d else {"dateTime": e_dt.isoformat(), "timeZone": _tz_name(tz)}  # type: ignore[union-attr]
    if not body:
        raise IntegrationError("Nothing to change: pass at least one field.")
    res = await gapi(ctx, PID, "PATCH", path, params={"sendUpdates": "all" if send_updates else "none"}, json=body)
    return {"updated": True, "event": normalize_event(res)}


@itool(PID, "gcal_delete_event", risk=Risk.send)
async def gcal_delete_event(ctx: ToolContext, event_id: str, notify_guests: bool = False,
                            calendar_id: str = "primary") -> dict:
    """Delete a calendar event."""
    await gapi(ctx, PID, "DELETE", f"{API}/calendars/{quote(calendar_id, safe='@')}/events/{event_id}",
               params={"sendUpdates": "all" if notify_guests else "none"})
    return {"deleted": True, "event_id": event_id}


@itool(PID, "gcal_quick_add", risk=Risk.write)
async def gcal_quick_add(ctx: ToolContext, text: str, calendar_id: str = "primary") -> dict:
    """Create an event from a short sentence, e.g. "Lunch with Priya at Cafe Mocha tomorrow 1pm"."""
    res = await gapi(ctx, PID, "POST", f"{API}/calendars/{quote(calendar_id, safe='@')}/events/quickAdd",
                     params={"text": text, "sendUpdates": "none"})
    return {"created": True, "event": normalize_event(res)}


def free_slots(busy: list[tuple[datetime, datetime]], start: datetime, end: datetime, duration: timedelta,
               work_start: time, work_end: time, tz: Any, limit: int = 20) -> list[dict]:
    slots: list[dict] = []
    busy = sorted(busy)
    day = start.astimezone(tz).date()
    while datetime.combine(day, time.min, tz) < end and len(slots) < limit:
        w0 = max(datetime.combine(day, work_start, tz), start)
        w1 = min(datetime.combine(day, work_end, tz), end)
        cursor = w0
        for b0, b1 in busy:
            if b1 <= cursor or b0 >= w1:
                continue
            if b0 - cursor >= duration:
                slots.append({"start": cursor.isoformat(), "end": b0.isoformat()})
            cursor = max(cursor, b1)
        if w1 - cursor >= duration:
            slots.append({"start": cursor.isoformat(), "end": w1.isoformat()})
        day += timedelta(days=1)
    return slots[:limit]


@itool(PID, "gcal_find_free_slots")
async def gcal_find_free_slots(ctx: ToolContext, time_min: str | None = None, time_max: str | None = None,
                               duration_minutes: int = 30, work_start: str = "09:00", work_end: str = "18:00",
                               calendar_ids: list[str] | None = None) -> dict:
    """Find free time slots of at least `duration_minutes` within working hours (default next 7 days, 09:00-18:00,
    user's timezone). Also returns the busy blocks. Pass other people's calendar_ids (emails) to find a shared gap."""
    mgr = manager_from(ctx)
    tz = _tz(mgr)
    start = datetime.fromisoformat(rfc3339(time_min, tz) or datetime.now(tz).isoformat())
    end = datetime.fromisoformat(rfc3339(time_max, tz, end_of_day=True) or (start + timedelta(days=7)).isoformat())
    ids = calendar_ids or ["primary"]
    res = await gapi(ctx, PID, "POST", f"{API}/freeBusy", json={
        "timeMin": start.isoformat(), "timeMax": end.isoformat(), "timeZone": _tz_name(tz),
        "items": [{"id": i} for i in ids]})
    busy: list[tuple[datetime, datetime]] = []
    errors = {}
    for cid, cal in (res.get("calendars") or {}).items():
        if cal.get("errors"):
            errors[cid] = cal["errors"][0].get("reason")
        for b in cal.get("busy") or []:
            busy.append((datetime.fromisoformat(b["start"].replace("Z", "+00:00")),
                         datetime.fromisoformat(b["end"].replace("Z", "+00:00"))))
    try:
        ws, we = time.fromisoformat(work_start), time.fromisoformat(work_end)
    except ValueError as exc:
        raise IntegrationError("work_start and work_end must look like 09:00 and 18:00.") from exc
    slots = free_slots(busy, start, end, timedelta(minutes=max(5, int(duration_minutes or 30))), ws, we, tz)
    out: dict[str, Any] = {"timezone": _tz_name(tz), "free_slots": slots,
                           "busy": [{"start": b0.astimezone(tz).isoformat(), "end": b1.astimezone(tz).isoformat()} for b0, b1 in sorted(busy)]}
    if errors:
        out["calendar_errors"] = errors
    return out


@itool(PID, "gcal_list_calendars")
async def gcal_list_calendars(ctx: ToolContext) -> dict:
    """List the calendars the user can see (ids for other calendar tools)."""
    res = await gapi(ctx, PID, "GET", f"{API}/users/me/calendarList")
    return {"calendars": [{"id": c.get("id"), "name": c.get("summary"), "primary": bool(c.get("primary")),
                           "access": c.get("accessRole")} for c in res.get("items") or []]}


# ---------------------------------------------------------------------------- change feed (sync tokens)
NEW_EVENT_WINDOW = timedelta(minutes=2)
FEED_PAGES = 20
FULL_SYNC_PAGES = 40


def _ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or "T" not in value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def event_kind(e: dict) -> str:
    """``new_event`` when the event was created just now, else ``updated_event``."""
    created, updated = _ts(e.get("created")), _ts(e.get("updated"))
    if created and updated and updated - created > NEW_EVENT_WINDOW:
        return "updated_event"
    return "new_event"


def _ended(e: dict, now: datetime) -> bool:
    end = e.get("end") or {}
    if end.get("dateTime"):
        dt = _ts(end["dateTime"])
        return dt is not None and dt < now
    if end.get("date"):
        try:
            return date.fromisoformat(end["date"]) <= now.date()  # all-day end dates are exclusive
        except ValueError:
            return False
    return False


async def _event_pages(mgr: IntegrationManager, params: dict, max_pages: int) -> tuple[list[dict], str | None]:
    items: list[dict] = []
    page_token: str | None = None
    for _ in range(max_pages):
        p = dict(params)
        if page_token:
            p["pageToken"] = page_token
        res = await gapi(None, PID, "GET", f"{API}/calendars/primary/events", params=p, mgr=mgr)
        items.extend(res.get("items") or [])
        if res.get("nextSyncToken"):
            return items, str(res["nextSyncToken"])
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return items, None


async def _full_sync_token(mgr: IntegrationManager) -> str:
    _, token = await _event_pages(mgr, {"maxResults": 2500}, FULL_SYNC_PAGES)
    if not token:
        raise IntegrationError("Google Calendar has too many events to watch for changes, so it is checked on a timer instead.")
    return token


async def gcal_change_feed(mgr: IntegrationManager, cursor: str | None) -> FeedBatch:
    """Events created or changed since the stored sync token.

    No cursor: full sync to get a token (nothing emitted). 410 (token expired): full resync without
    re-emitting old events. Cancelled and already-ended events are skipped.
    """
    if not cursor:
        return FeedBatch(cursor=await _full_sync_token(mgr), rebaselined=True)
    try:
        raw, token = await _event_pages(mgr, {"syncToken": cursor, "maxResults": 250}, FEED_PAGES)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 410:
            return FeedBatch(cursor=await _full_sync_token(mgr), rebaselined=True,
                             note="Google Calendar asked for a full resync, so watching restarted from now.")
        raise
    now = datetime.now(UTC)
    items: list[dict] = []
    for e in raw:
        if not e.get("id") or e.get("status") == "cancelled" or _ended(e, now):
            continue
        item = normalize_event(e)
        item["_key"] = f"{e['id']}:{e.get('updated')}"
        item["_event"] = event_kind(e)
        items.append(item)
    return FeedBatch(cursor=token or cursor, items=items)


class GCalendarPlugin(GooglePlugin):
    id = PID
    display_name = "Google Calendar"
    description = (
        "See and manage your schedule. Sentient can list and search events, create, move and cancel them, find "
        "free time for meetings, and watch for new or changed events to prepare you or trigger tasks. Privacy "
        "filters hide events with chosen keywords or attendees."
    )
    category = "productivity"
    icon = "google-calendar"
    api_name = "Google Calendar API"
    api_slug = "calendar-json.googleapis.com"
    selection_hint = "Use to check the schedule, create/update/delete events, or find free time in Google Calendar."
    privacy_fields = ["keywords", "emails"]
    privacy_kind = "event"
    feed_kind = "calendar_sync_token"
    triggers = [{"event": "new_event", "label": "New event"}, {"event": "updated_event", "label": "Changed event"}]
    tools = [gcal_list_events, gcal_create_event, gcal_update_event, gcal_delete_event, gcal_quick_add,
             gcal_find_free_slots, gcal_list_calendars]

    async def change_feed(self, mgr: IntegrationManager, cursor: str | None) -> FeedBatch:
        return await gcal_change_feed(mgr, cursor)

    async def poll(self, mgr: IntegrationManager, since: datetime) -> list[dict]:
        """Upcoming (next 14 days) events created or changed since ``since``."""
        now = datetime.now(UTC)
        items = await _list(None, mgr, "primary", {
            "updatedMin": since.astimezone(UTC).isoformat(), "timeMin": now.isoformat(),
            "timeMax": (now + timedelta(days=14)).isoformat(), "singleEvents": "true", "orderBy": "startTime",
            "maxResults": 50})
        for i in items:
            i["_key"] = f"{i['id']}:{i.get('updated')}"
            i["_event"] = event_kind(i)
        return items


__all__ = ["GCalendarPlugin", "event_blocked", "free_slots", "normalize_event"]

PLUGIN = GCalendarPlugin()
