"""#305: "every 10 minutes" is saved as an interval, and model-written check scripts that cannot work are rejected
and re-planned once."""

from __future__ import annotations

import pytest

from sentient.tasks.schedule import spoken_interval, spoken_schedule
from sentient.tasks.scripts import ScriptInvalid, check_rules, normalize_script
from tests.conftest import FakeProvider
from tests.tasks.conftest import PLAN_TIME, json_calls

WATCH = "from sentient_tools import tools, result\npage = tools.web_fetch(url='https://example.com')\nresult(page.get('title'))\n"


@pytest.mark.parametrize(("prompt", "minutes"), [
    ("Every 10 minutes watch the library page for new hours", 10),
    ("every hour check the train status", 60),
    ("Every 2 hours, look at my inbox", 120),
    ("check the price every 30 min", 30),
    ("every 1.5 hours remind me to stretch", 90),
    ("every ten minutes check the score", 10),
    ("every half an hour check the queue", 30),
    ("every other hour check the oven", 120),
    ("hourly news check", 60),
    ("every minute check the page", 5),  # the 5-minute minimum
    ("every 2 minutes check the page", 5),
    ("every 90 seconds check the page", 5),
])
def test_spoken_interval(prompt, minutes):
    assert spoken_interval(prompt) == minutes


@pytest.mark.parametrize("prompt", [
    "every day at 8am give me the weather",
    "every weekday morning summarize my email",
    "every second Friday send the report",
    "remind me in 10 minutes to call Maya",
    "weekly hourly report",
    "summarize this document",
])
def test_no_spoken_interval(prompt):
    assert spoken_interval(prompt) is None


def test_spoken_schedule_overrides_a_daily_guess_and_keeps_triggers():
    daily = {"type": "recurring", "frequency": "daily", "time": "09:00", "window": "morning", "timezone": "UTC"}
    assert spoken_schedule(daily, "Every 10 minutes watch the page") == {
        "type": "recurring", "frequency": "interval", "interval_minutes": 10, "timezone": "UTC",
    }
    triggered = {"type": "triggered", "source": "gmail", "event": "new_email", "filter": {}, "timezone": "UTC"}
    assert spoken_schedule(triggered, "on every new email, check every 10 minutes") == triggered
    assert spoken_schedule(daily, "every day at 9am")["frequency"] == "daily"


async def test_every_10_minutes_is_saved_as_an_interval(make_app, clock):
    # what qwen3:8b wrote in the real run: a daily 09:00 schedule with a morning window
    refine = {"name": "Watch library page", "description": "Watch the library page", "priority": 1,
              "schedule": {"type": "recurring", "frequency": "daily", "time": "09:00", "window": "morning"}}
    plan = {**PLAN_TIME, "schedule": {"type": "recurring", "frequency": "interval", "interval_minutes": 10}}
    llm = FakeProvider(json_replies=[refine, plan])
    app = await make_app(llm, clock=clock)
    created = await app.tasks.create_task("Every 10 minutes watch the library page for new opening hours")
    await app.tasks.drain()
    await app.tasks.approve(created["task_id"])
    task = await app.tasks.get(created["task_id"])
    assert task["schedule"] == {"type": "recurring", "frequency": "interval", "interval_minutes": 10,
                                "timezone": "Asia/Kolkata"}
    assert task["next_execution_at"] == "2026-09-15T02:10:00+00:00"
    assert "interval" in json_calls(llm, "planner")[0]["messages"][0]["content"]  # the refine prompt knows intervals


async def test_refine_failure_still_keeps_the_spoken_interval(make_app):
    llm = FakeProvider(json_replies=["not json", "still not json", PLAN_TIME])
    app = await make_app(llm)
    created = await app.tasks.create_task("every hour check the train status")
    await app.tasks.drain()
    task = await app.tasks.get(created["task_id"])
    assert task["schedule"]["frequency"] == "interval" and task["schedule"]["interval_minutes"] == 60


# ---------------------------------------------------------------------------- check script rules
@pytest.mark.parametrize(("code", "condition", "reason"), [
    ("from sentient_tools import tools, result\nwhile True:\n    result(1)\n", "changed", "loops forever"),
    ("import time\nfrom sentient_tools import tools, result\ntime.sleep(60)\nresult(1)\n", "changed", "time.sleep"),
    ("from sentient_tools import tools, result\ntools.time_sleep(seconds=5)\nresult(1)\n", "changed", "tools.time_sleep"),
    ("from time import sleep\nsleep(5)\n", "every_run", "sleep"),
    ("from sentient_tools import tools, result\nold = tools.memory_recall(query='price')\nresult(old)\n", "changed",
     "memory_recall"),
    ("from sentient_tools import tools, result\ntools.memory_remember(fact='42')\nresult(42)\n", "alert",
     "memory_remember"),
    ("from sentient_tools import tools, result\np = 41\nresult({'alert': p < 50, 'message': 'cheap'})\n", "changed",
     "returns an alert"),
    ("from sentient_tools import tools, result\nresult({'alert': True})\n", "noteworthy", "returns an alert"),
    ("from bs4 import BeautifulSoup\nimport requests\nresult(1)\n", "changed", "imports `bs4`"),
    ("import requests.adapters\nresult(1)\n", "changed", "imports `requests`"),
])
def test_bad_model_scripts_are_rejected_with_a_reason(code, condition, reason):
    with pytest.raises(ScriptInvalid) as err:
        check_rules(code, condition)
    assert reason in str(err.value)
    with pytest.raises(ScriptInvalid):
        normalize_script({"code": code, "condition": condition}, model_written=True)
    normalize_script({"code": code, "condition": condition})  # a person may write what they like


@pytest.mark.parametrize(("code", "condition"), [
    (WATCH, "changed"),
    ("from sentient_tools import tools, result\nresult({'alert': False})\n", "alert"),
    ("from sentient_tools import tools, result\nfor i in range(3):\n    pass\nmemory_used = 1\nresult(i)\n", "changed"),
    ("from sentient_tools import tools, result\nn = 0\nwhile n < 3:\n    n += 1\nresult(n)\n", "changed"),
    ("import json, re\nfrom datetime import datetime\nfrom sentient_tools import tools, result\nresult(1)\n", "changed"),
])
def test_good_scripts_pass(code, condition):
    check_rules(code, condition)


async def test_planner_re_plans_once_with_the_reason(make_app):
    refine = {"name": "Price watch", "description": "Tell me when the lamp is under 40 euros", "priority": 1,
              "schedule": {"type": "once", "run_at": None}}
    good = {**PLAN_TIME, "script": {"code": WATCH, "condition": "changed", "then": "notify"},
            "schedule": {"type": "recurring", "frequency": "interval", "interval_minutes": 30}}
    looping = {**good, "script": {**good["script"], "code": "from sentient_tools import tools, result\nwhile True:\n    pass\n"}}
    llm = FakeProvider(json_replies=[refine, looping, good])
    app = await make_app(llm)
    created = await app.tasks.create_task("Tell me when the lamp is under 40 euros")
    await app.tasks.drain()
    task = await app.tasks.get(created["task_id"])
    assert task["status"] == "approval_pending" and task["script"]["code"] == WATCH
    retry = json_calls(llm, "planner")[2]["messages"][1]["content"]
    assert "loops forever" in retry and "while True:" in retry  # the reason and the code it rejected
    assert "time_sleep" in json_calls(llm, "planner")[1]["messages"][0]["content"]  # the planner prompt says so


async def test_chat_tool_script_edits_follow_the_rules_but_ui_edits_do_not(make_app):
    app = await make_app(FakeProvider())
    created = await app.tasks.repo.insert_task({
        "name": "Lamp watch", "description": "Lamp watch", "status": "active", "plan": PLAN_TIME["plan"],
        "task_type": "script", "script": normalize_script({"code": WATCH, "condition": "changed"}),
        "schedule": {"type": "recurring", "frequency": "interval", "interval_minutes": 60, "timezone": "UTC"},
        "created_at": app.tasks.now_iso(), "updated_at": app.tasks.now_iso(),
    })
    sleepy = WATCH + "import time\ntime.sleep(1)\n"
    with pytest.raises(ValueError, match="must not sleep"):
        await app.tasks.update(created, {"script": {"code": sleepy}}, reapprove_script=True)
    edited = await app.tasks.update(created, {"script": {"code": sleepy}})
    assert edited["script"]["code"] == sleepy
