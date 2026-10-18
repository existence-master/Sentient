"""#375: for a watch request the planner may not report on every run, and its script must watch the whole page
unless the request names the part to watch."""

from __future__ import annotations

import pytest

from sentient.tasks.scripts import ScriptInvalid, check_rules, is_watch_request, normalize_script
from tests.conftest import FakeProvider
from tests.tasks.conftest import PLAN_TIME, json_calls

FETCH = "from sentient_tools import tools, result\npage = tools.web_fetch(url='http://192.0.2.1/events.html')\n"
WHOLE = FETCH + "result(page.get('content', ''))\n"
WATCH_EVENTS = "Watch http://192.0.2.1/events.html for new events."


@pytest.mark.parametrize("request_text", [
    WATCH_EVENTS,
    "Track the price of the blue lamp",
    "monitor the school's news page",
    "Keep an eye on the train status page",
    "Tell me when the library adds a new event",
    "let me know if the shop restocks the lamp",
    "Notify me whenever a new post appears on the blog",
    "Send me a report every hour on the page\nNo, only tell me when a new event is added",  # a later change wins
])
def test_watch_requests(request_text):
    assert is_watch_request(request_text)


@pytest.mark.parametrize("request_text", [
    "Every morning tell me the weather",
    "Summarize the news every day at 8",
    "Watch the events page and send me a report every hour",
    "Monitor the server page and give me a daily summary",
    "track the queue and tell me its length each time you check",
    "watch the score page, even if nothing changed",
    "Watch the page for new events\nActually, just send me a report every hour",  # a later change wins
])
def test_not_watch_requests(request_text):
    assert not is_watch_request(request_text)


def test_every_run_is_rejected_for_a_watch_request_with_a_plain_reason():
    with pytest.raises(ScriptInvalid) as err:
        check_rules(WHOLE, "every_run", WATCH_EVENTS)
    assert '"every_run" would send a notification after every check' in str(err.value)
    assert '"noteworthy"' in str(err.value)
    check_rules(WHOLE, "every_run", "Check the events page and report every hour")  # a report was asked for
    check_rules(WHOLE, "every_run")  # chat tools pass no request
    check_rules(WHOLE, "noteworthy", WATCH_EVENTS)


@pytest.mark.parametrize(("body", "reason"), [
    ("result(page['content'].split('\\n')[0])\n", "keeps only the first line of the page"),
    ("text = page.get('content', '')\nresult(text.strip().splitlines()[0])\n", "first line"),
    ("text = page.get('content') or ''\nlines = text.splitlines()\nresult(lines[0])\n", "`lines[0]`"),
    ("lines = page['content'].lower().split('\\n')\nresult(lines[:1])\n", "first line"),
    ("result(page['content'].partition('\\n')[0])\n", "first line"),
    ("result(page['content'][:120])\n", "keeps only the first 120 characters"),
    ("result(page.get('title', ''))\n", "reads only the page's `title`"),
    ("title = page['title']\nresult({'title': title})\n", "only the page's `title`"),
    ("body = page['content']\nresult(page['title'])\n", "only the page's `title`"),  # content read, not returned
    ("title: str = page.get('title', '')\nprint(title)\n", "only the page's `title`"),
])
def test_a_script_that_cuts_the_page_down_is_rejected(body, reason):
    code = FETCH + body
    with pytest.raises(ScriptInvalid) as err:
        check_rules(code, "noteworthy", WATCH_EVENTS)
    assert reason in str(err.value) and "Return the page's whole `content` instead." in str(err.value)
    with pytest.raises(ScriptInvalid):
        normalize_script({"code": code, "condition": "noteworthy"}, model_written=True, request=WATCH_EVENTS)
    check_rules(code, "noteworthy")  # without the request (chat tools) the rule does not apply
    check_rules(code, "changed", "Tell me when the first line of http://192.0.2.1/events.html changes")
    check_rules(code, "changed", "Watch the headline of http://192.0.2.1/events.html")


@pytest.mark.parametrize("body", [
    "result(page.get('content', ''))\n",
    "result(page['content'].strip())\n",
    "result({'title': page['title'], 'content': page['content']})\n",
    "lines = page['content'].splitlines()\nresult([line for line in lines if line.startswith('- ')])\n",
    "for line in page['content'].splitlines():\n    if 'Price' in line:\n        result(line.split(':')[0])\n",
    "result(page['content'][:8000])\n",
    "result(page['content'].split('Events')[1])\n",
])
def test_scripts_that_keep_the_whole_page_or_pick_a_value_pass(body):
    check_rules(FETCH + body, "noteworthy", WATCH_EVENTS)


async def test_the_planner_re_plans_every_run_and_a_heading_with_the_reasons(make_app):
    refine = {"name": "Events watch", "description": "Watch the events page for new events", "priority": 1,
              "schedule": {"type": "once", "run_at": None}}
    good = {**PLAN_TIME, "script": {"code": WHOLE, "condition": "noteworthy", "then": "notify"},
            "schedule": {"type": "recurring", "frequency": "interval", "interval_minutes": 60}}
    every = {**good, "script": {**good["script"], "condition": "every_run"}}
    llm = FakeProvider(json_replies=[refine, every, good])
    app = await make_app(llm)
    created = await app.tasks.create_task(WATCH_EVENTS)
    await app.tasks.drain()
    task = await app.tasks.get(created["task_id"])
    assert task["status"] == "approval_pending" and task["script"]["condition"] == "noteworthy"
    retry = json_calls(llm, "planner")[2]["messages"][1]["content"]
    assert '"every_run" would send a notification after every check' in retry
    prompt = json_calls(llm, "planner")[1]["messages"][0]["content"]
    assert 'never to watch for something new ("watch", "track", "monitor", "tell me when")' in prompt
    assert "return its whole `content`" in prompt

    heading = {**good, "script": {**good["script"], "code": FETCH + "result(page['content'].split('\\n')[0])\n"}}
    llm2 = FakeProvider(json_replies=[refine, heading, good])
    app2 = await make_app(llm2, db_name="tasks2.db")
    created = await app2.tasks.create_task(WATCH_EVENTS)
    await app2.tasks.drain()
    task = await app2.tasks.get(created["task_id"])
    assert task["script"]["code"] == WHOLE
    assert "keeps only the first line of the page" in json_calls(llm2, "planner")[2]["messages"][1]["content"]
