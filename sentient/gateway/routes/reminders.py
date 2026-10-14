"""REST routes for timers and one-off reminders (owner: reminders, #150). Contract: docs/API.md section 23."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from sentient.gateway.deps import AUTH, get_core
from sentient.reminders.service import STATUSES, ReminderError

router = APIRouter(prefix="/api/reminders", tags=["reminders"], dependencies=AUTH)


class AddBody(BaseModel):
    kind: str = "reminder"
    minutes: float | None = None
    at: str | None = None
    message: str = ""
    label: str = ""


@router.get("")
async def list_reminders(request: Request, status: str = "scheduled", limit: int = 100):
    wanted = None if status == "all" else status
    if wanted is not None and wanted not in STATUSES:
        raise HTTPException(400, "status is scheduled, fired, cancelled or all.")
    return {"reminders": await get_core(request).reminders.list(wanted, limit=limit)}


@router.post("")
async def add_reminder(request: Request, body: AddBody):
    core = get_core(request)
    duration = None
    if body.minutes is not None and body.minutes > 0:
        duration = body.minutes * 60
        when = datetime.now(UTC) + timedelta(seconds=duration)
    elif body.at:
        try:
            when = datetime.fromisoformat(body.at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(400, "at is an ISO date and time with a timezone.") from exc
    else:
        raise HTTPException(400, "Give minutes or at.")
    try:
        return await core.reminders.add(body.kind, when, message=body.message, label=body.label, duration_s=duration,
                                        channel="desktop", language=core.config.assistant.language, source="api")
    except ReminderError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{reminder_id}/cancel")
async def cancel_reminder(request: Request, reminder_id: str):
    item, changed = await get_core(request).reminders.cancel(reminder_id)
    if item is None:
        raise HTTPException(404, "There is no timer or reminder with that id.")
    if not changed:
        raise HTTPException(409, "It already went off." if item["status"] == "fired" else "It was already cancelled.")
    return item
