"""#389: a check script that reads the web runs on a timer. A `triggered` schedule it never uses (a small model's guess
for "tell me when a new event is added") is repaired to a recurring one in planning."""

from __future__ import annotations

import pytest

from sentient.tasks.scripts import trigger_is_real
from tests.conftest import FakeProvider
from tests.tasks.conftest import PLAN_TIME

FETCH = "from sentient_tools import tools, result\npage = tools.web_fetch(url='http://192.0.2.1/events.html')\n"
WHOLE = FETCH + "result(page.get('content', ''))\n"
WATCH = "Track http://192.0.2.1/events.html and tell me when a new event is added"
CALENDAR = {"type": "triggered", "source": "gcalendar", "event": "new_event", "filter": {}}
HOURLY = {"type": "recurring", "frequency": "interval", "interval_minutes": 60}


def refined(schedule: dict) -> dict:
    return {"name": "Events watch", "description": WATCH, "priority": 1, "schedule": schedule}


def planned(code: str = WHOLE, **extra) -> dict:
    return {**PLAN_TIME, "script": {"code": code, "condition": "noteworthy", "then": "notify"}, **extra}


async def plan(make_app, prompt: str, *replies: dict) -> dict:
    app = await make_app(FakeProvider(json_replies=list(replies)))
    created = await app.tasks.create_task(prompt)
    await app.tasks.drain()
    return await app.tasks.get(created["task_id"])


@pytest.mark.parametrize(("schedule", "code", "request_text", "real"), [
    (CALENDAR, WHOLE, WATCH, False),
    ({"type": "triggered", "source": "custom", "event": "", "filter": {}}, WHOLE, WATCH, False),
    ({"type": "triggered", "source": "", "event": "", "filter": {}}, WHOLE, WATCH, False),
    (CALENDAR, WHOLE, "When a meeting is added to my calendar, check http://192.0.2.1/room.html", True),
    ({"type": "triggered", "source": "gmail", "event": "new_email", "filter": {}}, WHOLE,
     "Whenever I get an email from the bank, check http://192.0.2.1/status.html", True),
    (CALENDAR, FETCH + "result([trigger_event.get('summary'), page['content']])\n", WATCH, True),
])
def test_a_trigger_is_real_when_the_script_reads_it_or_the_request_names_its_app(schedule, code, request_text, real):
    assert trigger_is_real(schedule, code, request_text) is real


async def test_a_calendar_trigger_on_a_page_watch_becomes_hourly(make_app):
    task = await plan(make_app, WATCH, refined(CALENDAR), planned())
    assert task["task_type"] == "script" and task["status"] == "approval_pending"
    assert task["schedule"]["type"] == "recurring"
    assert (task["schedule"]["frequency"], task["schedule"]["interval_minutes"]) == ("interval", 60)


async def test_the_repair_keeps_the_interval_the_user_said(make_app):
    prompt = "Check http://192.0.2.1/events.html every 30 minutes and tell me when a new event is added"
    task = await plan(make_app, prompt, refined(CALENDAR), planned())
    assert (task["schedule"]["type"], task["schedule"]["interval_minutes"]) == ("recurring", 30)


async def test_the_repair_takes_the_planners_daily_schedule_or_else_hourly(make_app):
    once = {"type": "once", "run_at": None}
    task = await plan(make_app, WATCH, refined(once), planned(schedule={**CALENDAR, "source": "custom"}))
    assert task["schedule"]["type"] == "recurring" and task["schedule"]["interval_minutes"] == 60

    unasked = {"type": "recurring", "frequency": "interval", "interval_minutes": 5}  # what qwen3:8b proposed
    task = await plan(make_app, WATCH, refined(CALENDAR), planned(schedule=unasked))
    assert task["schedule"]["type"] == "recurring" and task["schedule"]["interval_minutes"] == 60

    daily = {"type": "recurring", "frequency": "daily", "time": "08:00"}
    task = await plan(make_app, WATCH, refined(CALENDAR), planned(schedule=daily))
    assert (task["schedule"]["type"], task["schedule"]["frequency"], task["schedule"]["time"]) == (
        "recurring", "daily", "08:00")


async def test_a_trigger_the_user_asked_for_stays(make_app):
    prompt = "Whenever I get an email from the bank, check http://192.0.2.1/status.html and tell me what it says"
    gmail = {"type": "triggered", "source": "gmail", "event": "new_email", "filter": {"from": "bank@example.com"}}
    task = await plan(make_app, prompt, refined(gmail), planned())
    assert task["schedule"]["type"] == "triggered" and task["schedule"]["source"] == "gmail"


async def test_a_script_that_reads_no_web_keeps_its_trigger(make_app):
    code = "from sentient_tools import tools, result\nresult(trigger_event.get('summary'))\n"
    task = await plan(make_app, "Tell me when a new event is added", refined(CALENDAR), planned(code=code))
    assert task["schedule"]["type"] == "triggered"
