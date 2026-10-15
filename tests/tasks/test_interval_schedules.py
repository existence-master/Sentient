"""#305: "every 10 minutes" is saved as an interval, and model-written check scripts that cannot work are rejected
and re-planned once."""

from __future__ import annotations

import pytest

from sentient.tasks import service as service_mod
from sentient.tasks.schedule import spoken_interval, spoken_schedule
from sentient.tasks.scripts import ScriptInvalid, check_rules, normalize_script, tools_named
from tests.conftest import FakeProvider
from tests.tasks.conftest import PLAN_TIME, json_calls

TOOLS = "from sentient_tools import tools, result\n"
WATCH = "from sentient_tools import tools, result\npage = tools.web_fetch(url='https://example.com')\nresult(page.get('content'))\n"


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
    # #366: a key the tool never returns
    (f"{TOOLS}page = tools.web_fetch(url='https://example.com')\nresult(page['html'])\n", "changed",
     "reads `html` from the result of `tools.web_fetch`, which has no `html`: it has only `url`, `title`, `content` "
     "and `truncated`. The page's text is in `content`."),
    (f"{TOOLS}result(tools.web_fetch(url='https://example.com').get('text', ''))\n", "noteworthy", "no `text`"),
    (f"{TOOLS}hits = tools.web_search(query='x')\nfor h in hits['results']:\n    print(h['body'])\n", "every_run",
     "reads `body` from each item of `results` from `tools.web_search`, which has no `body`"),
    (f"{TOOLS}news = tools.news_search(query='x')\nresult([a['headline'] for a in news.get('articles') or []])\n",
     "changed", "no `headline`: it has only `title`, `source`, `url`, `published_at` and `summary`"),
    (f"{TOOLS}rows = tools.news_top_headlines()['articles']\ntop = rows[0]\nresult(top['link'])\n", "changed",
     "no `link`"),
    (f"{TOOLS}res = tools.web_search(query='x')\nresult(res['results']['title'])\n", "changed", "which is a list"),
    (f"{TOOLS}page = tools.web_fetch(url='https://x')\na = page\nb = a\nc = b\nd = c\ne = d\nresult(e.get('html', ''))\n",
     "changed", "no `html`"),  # any number of aliases
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
    (f"{TOOLS}page = tools.web_fetch(url='https://example.com')\nresult(page.get('content', '')[:200])\n", "changed"),
    (f"{TOOLS}import json\ndata = tools.web_fetch(url='https://x.org/a.json')\ndata = json.loads(data['content'])\n"
     "result(data['events'])\n", "changed"),  # the name now holds something else
    (f"{TOOLS}def read(url):\n    page = tools.web_fetch(url=url)\n    return page['content']\nresult(read('https://x'))\n",
     "changed"),
    (f"{TOOLS}hits = tools.web_search(query='x')\nresult([h['title'] for h in hits['results'][:3]])\n", "changed"),
    (f"{TOOLS}page = tools.web_fetch(url='https://x')\nresult(None if page.get('error') else page['title'])\n",
     "changed"),
    (f"{TOOLS}w = tools.weather_current()\nresult(w['temp'])\n", "changed"),  # shapes only for the common tools
    (f"{TOOLS}page = tools.web_fetch(url='https://x')\npage['price'] = '40'\nresult(page['price'])\n", "changed"),
    (f"{TOOLS}hits = tools.web_search(query='x')\nfor h in hits['results']:\n    h.update(seen=True)\n"
     "result([h['seen'] for h in hits['results']])\n", "changed"),  # keys the script adds itself
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


async def test_planner_re_plans_a_script_reading_a_key_web_fetch_never_returns(make_app):
    refine = {"name": "Events watch", "description": "Watch the events page for new events", "priority": 1,
              "schedule": {"type": "once", "run_at": None}}
    reads = "from sentient_tools import tools, result\npage = tools.web_fetch(url='https://example.com/events')\n"
    good = {**PLAN_TIME, "script": {"code": reads + "result(page['content'])\n", "condition": "noteworthy",
                                    "then": "notify"}}
    html = {**good, "script": {**good["script"], "code": reads + "result(page.get('html', ''))\n"}}
    llm = FakeProvider(json_replies=[refine, html, good])
    app = await make_app(llm)
    created = await app.tasks.create_task("Watch https://example.com/events for new events")
    await app.tasks.drain()
    task = await app.tasks.get(created["task_id"])
    assert task["status"] == "approval_pending" and task["script"]["code"] == good["script"]["code"]
    retry = json_calls(llm, "planner")[2]["messages"][1]["content"]
    assert "has no `html`" in retry and "The page's text is in `content`." in retry
    prompt = json_calls(llm, "planner")[1]["messages"][0]["content"]
    assert '`tools.web_fetch(url=...)` -> {"url", "title", "content"' in prompt  # the planner knows the shape


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


# ---------------------------------------------------------------------------- no script after two tries (#310)
MEMORY_SCRIPT = ("from sentient_tools import tools, result\npage = tools.web_fetch(url='http://127.0.0.1:8000/sapiens')\n"
                 "old = tools.memory_recall(query='price')\nresult(old)\n")
BROKEN_SCRIPT = "from sentient_tools import tools, result\nwhile True:\n    pass\n"


async def test_a_script_rejected_twice_becomes_an_ai_run_recurring_task(make_app):
    """What qwen3:8b did in the real run: memory_recall again, then broken code. The task is saved, not failed."""
    prompt = "Monitor Sapiens book price on http://127.0.0.1:8000/sapiens every 30 minutes and remind me what it was last time"
    refine = {"name": "Sapiens price", "description": prompt, "priority": 1, "schedule": {"type": "once", "run_at": None}}
    first = {**PLAN_TIME, "plan": [{"tool": "web_search", "description": "Search the internet for the Sapiens price"}],
             "script": {"code": MEMORY_SCRIPT, "condition": "changed", "then": "notify"}}
    second = {"name": "Sapiens price", "clarifying_questions": [], "script": {"code": BROKEN_SCRIPT, "condition": "changed", "then": "notify"}}
    result = {"summary": "Sapiens costs 499. Last time it was 549.", "links_created": [], "links_found": [],
              "files_created": [], "tools_used": []}
    llm = FakeProvider(replies=["Sapiens costs 499. Last time it was 549."], json_replies=[refine, first, second, result])
    app = await make_app(llm)
    created = await app.tasks.create_task(prompt)
    await app.tasks.drain()
    task = await app.tasks.get(created["task_id"])
    assert task["status"] == "approval_pending" and task["error"] is None
    assert task["task_type"] == "single" and task["script"] is None
    # the request itself (it has the page address), with the app the scripts used rather than the planner's guess
    assert task["plan"] == [{"tool": "web", "description": prompt}]
    assert task["schedule"]["frequency"] == "interval" and task["schedule"]["interval_minutes"] == 30
    assert [m["content"] for m in task["chat_history"]] == [service_mod.NO_SCRIPT_NOTE]
    assert len(json_calls(llm, "planner")) == 3  # refine, plan, one re-plan

    await app.tasks.approve(task["task_id"])
    await app.tasks.run_now(task["task_id"])
    await app.tasks.drain()
    ran = await app.tasks.get(task["task_id"])
    assert ran["runs"][-1]["status"] == "completed" and ran["status"] == "active"
    assert ran["runs"][-1]["result"]["summary"].startswith("Sapiens costs 499")


REFINE_LAMP = {"name": "Lamp watch", "description": "Check the lamp price every hour", "priority": 1,
               "schedule": {"type": "once", "run_at": None}}


async def test_no_plan_at_all_still_gives_a_one_step_ai_task(make_app):
    refine = REFINE_LAMP
    bad = {"name": "Lamp watch", "clarifying_questions": [], "script": {"code": BROKEN_SCRIPT, "condition": "changed", "then": "notify"}}
    llm = FakeProvider(json_replies=[refine, bad, bad])
    app = await make_app(llm)
    created = await app.tasks.create_task("Check the lamp price every hour")
    await app.tasks.drain()
    task = await app.tasks.get(created["task_id"])
    assert task["status"] == "approval_pending" and task["task_type"] == "single"
    assert task["plan"] == [{"tool": "", "description": "Check the lamp price every hour"}]
    assert task["schedule"]["interval_minutes"] == 60


async def test_a_change_request_keeps_the_re_plans_steps_without_a_script(make_app):
    llm = FakeProvider(json_replies=[REFINE_LAMP, {**PLAN_TIME}])
    app = await make_app(llm)
    created = await app.tasks.create_task("Check the lamp price every hour")
    await app.tasks.drain()
    steps = [{"tool": "web", "description": "Read the lamp page at http://shop.example/lamp"}]
    bad = {"clarifying_questions": [], "plan": steps,
           "script": {"code": BROKEN_SCRIPT, "condition": "changed", "then": "notify"}}
    llm.json_replies += [bad, bad]
    await app.tasks.chat(created["task_id"], "Also read the lamp page at http://shop.example/lamp")
    await app.tasks.drain()
    task = await app.tasks.get(created["task_id"])
    assert task["status"] == "approval_pending" and task["task_type"] == "single" and task["plan"] == steps
    assert service_mod.NO_SCRIPT_NOTE in [m["content"] for m in task["chat_history"]]


def test_tools_named_reads_broken_scripts_too():
    code = "tools.web_fetch(url=x)\nmytools.nope()\nif (:\n tools.web_fetch(); tools.gmail_search(q=1)"
    assert tools_named(code) == ["web_fetch", "gmail_search"]


async def test_a_correction_without_a_script_also_falls_back_and_keeps_the_first_schedule(make_app):
    refine = {"name": "Lamp watch", "description": "Tell me when the lamp is cheap", "priority": 1,
              "schedule": {"type": "once", "run_at": None}}
    first = {**PLAN_TIME, "schedule": {"type": "recurring", "frequency": "interval", "interval_minutes": 45},
             "script": {"code": MEMORY_SCRIPT, "condition": "changed", "then": "notify"}}
    llm = FakeProvider(json_replies=[refine, first, {"clarifying_questions": []}])
    app = await make_app(llm)
    created = await app.tasks.create_task("Tell me when the lamp is cheap")
    await app.tasks.drain()
    task = await app.tasks.get(created["task_id"])
    assert task["status"] == "approval_pending" and task["task_type"] == "single"
    assert task["plan"] == [{"tool": "web", "description": "Tell me when the lamp is cheap"}]
    assert task["schedule"]["interval_minutes"] == 45  # the first reply's schedule, not the hourly default
    assert service_mod.NO_SCRIPT_NOTE in [m["content"] for m in task["chat_history"]]
