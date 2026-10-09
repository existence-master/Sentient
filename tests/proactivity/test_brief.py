"""Daily Brief (#138): a capped, steerable, read-only morning digest that is an ordinary recurring task.

Calendar, Gmail, weather and news tools are replaced by fakes on the registry (no network); the model is
FakeProvider. The fictional persona is Maya Rao.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.proactivity import brief as brief_mod
from sentient.proactivity.brief import BriefError, cap_items, parse_summaries, schedule_for
from sentient.tasks.schedule import calculate_next_run
from sentient.tools.base import Risk
from tests.conftest import FakeProvider

NOW = datetime(2026, 10, 12, 6, 0, tzinfo=UTC)  # a Monday, before the brief's 07:30
DAY = NOW.date().isoformat()

EVENTS = [
    {"id": "ev0", "summary": "Early gym", "start": f"{DAY}T05:00:00+00:00", "end": f"{DAY}T05:45:00+00:00"},
    {"id": "ev1", "summary": "Design review", "start": f"{DAY}T09:30:00+00:00", "end": f"{DAY}T10:00:00+00:00",
     "location": "Room 4", "url": "https://calendar.google.com/event?eid=ev1"},
    {"id": "ev2", "summary": "Lunch with Kavya", "start": f"{DAY}T12:30:00+00:00", "end": f"{DAY}T13:30:00+00:00",
     "url": "https://calendar.google.com/event?eid=ev2"},
    {"id": "ev3", "summary": "Pottery class", "start": f"{DAY}T18:00:00+00:00", "end": f"{DAY}T19:00:00+00:00"},
    {"id": "ev4", "summary": "Call with the landlord", "start": f"{DAY}T19:30:00+00:00", "end": f"{DAY}T20:00:00+00:00"},
]
MESSAGES = [
    {"id": "m1", "thread_id": "t1", "from": "Arjun Mehta <arjun@example.com>", "subject": "Q3 deck",
     "snippet": "Could you review the Q3 deck before Friday?", "url": "https://mail.google.com/mail/u/0/#all/m1"},
    {"id": "m2", "thread_id": "t2", "from": "Kavya Iyer <kavya@example.com>", "subject": "Lunch moved",
     "snippet": "Can we move lunch to 13:00?", "url": "https://mail.google.com/mail/u/0/#all/m2"},
    {"id": "m3", "thread_id": "t3", "from": "billing@example.com", "subject": "Invoice 42",
     "snippet": "Your invoice is attached.", "url": "https://mail.google.com/mail/u/0/#all/m3"},
]
WEATHER = {"location": "Pune, India", "current": {"condition": "Light rain", "temperature_c": 24.4},
           "today": {"max_c": 28.2, "rain_chance_percent": 60}}
NEWS = {"articles": [{"title": "Monsoon ends early", "source": "Example Times", "url": "https://news.example.com/1"}]}


class Fakes:
    def __init__(self):
        self.calls: list[str] = []

    def install(self, app, monkeypatch) -> None:
        replies = {"gcal_list_events": {"events": EVENTS}, "gmail_search": {"messages": MESSAGES},
                   "weather_current": WEATHER, "news_search": NEWS}
        for name, reply in replies.items():
            async def fake(ctx, _name=name, _reply=reply, **kwargs):
                self.calls.append(_name)
                return _reply
            monkeypatch.setattr(app.registry.get(name), "fn", fake)

        async def connected(plugin_id):
            return plugin_id in {"gmail", "gcalendar", "weather", "news"}

        monkeypatch.setattr(app.integrations, "is_connected", connected)


@pytest.fixture
async def app(config, isolated_home, monkeypatch):
    config.assistant.timezone = "UTC"
    config.assistant.user_name = "Maya Rao"
    config.assistant.location = "Pune, India"
    llm = FakeProvider()
    a = SentientApp(config, llm=llm, db_path=isolated_home / "brief.db", enable_background=False)
    await a.start()
    a.fake = llm
    a.fakes = Fakes()
    a.fakes.install(a, monkeypatch)
    yield a
    await a.stop()


async def add_tasks(app) -> None:
    repo = app.tasks.repo
    base = {"description": "", "priority": 1, "assignee": "ai", "enabled": True, "plan": [{"tool": "files", "description": "x"}],
            "task_type": "single", "created_at": NOW.isoformat(), "updated_at": NOW.isoformat()}
    await repo.insert_task({**base, "name": "Book the plumber", "status": "approval_pending"})
    await repo.insert_task({**base, "name": "Pick a flight to Goa", "status": "waiting_for_user"})
    await repo.insert_task({**base, "name": "Send the rent reminder", "status": "pending",
                            "next_execution_at": (NOW + timedelta(hours=9)).isoformat()})
    await repo.insert_task({**base, "name": "Next week's report", "status": "pending",
                            "next_execution_at": (NOW + timedelta(days=3)).isoformat()})


async def add_follow_up(app) -> None:
    suggestion = {"suggestion_type": "follow_up_reply", "description": "Priya is waiting for your reply about the invoice",
                  "action_details": {}, "reasoning": "", "confidence": 0.9,
                  "source_event": {"source": "gmail", "event_type": "follow_up", "item_id": "t9:m9",
                                   "url": "https://mail.google.com/mail/u/0/#all/m9"},
                  "follow_up": {"kind": "waiting_on_you", "days_waiting": 4, "thread_id": "t9", "message_id": "m9"}}
    await app.proactivity._deliver({"suggestion": suggestion, "status": "pending", "task_id": None}, {},
                                   threshold=0.5, now=NOW)


# ---------------------------------------------------------------------------- building
async def test_brief_built_from_calendar_email_tasks_and_weather_and_capped(app):
    await add_tasks(app)
    await add_follow_up(app)
    app.fake.text_replies.append("1. Arjun asks you to review the Q3 deck by Friday\n2. Kavya wants lunch at 13:00\n3. x")
    brief = await app.proactivity.brief.build(NOW)
    items = brief["items"]
    assert len(items) == 7  # 4 + 4 + 3 + 1 found: one line from each section, then the rest in order
    by = {s: [i["text"] for i in items if i["section"] == s] for s in ("calendar", "email", "tasks", "weather")}
    assert by["calendar"] == ["09:30 Design review (Room 4)", "12:30 Lunch with Kavya", "18:00 Pottery class"]
    assert "Early gym" not in " ".join(by["calendar"])
    assert by["email"] == ["Priya is waiting for your reply about the invoice", "Arjun asks you to review the Q3 deck by Friday"]
    assert by["tasks"] == ["Pick a flight to Goa: asked you a question"]  # the most urgent task first
    assert by["weather"] == ["Pune: light rain, 24°C now, high 28°C, 60% chance of rain"]
    assert [s["id"] for s in brief["sections"]] == ["calendar", "email", "tasks", "weather"]
    assert all(i["why"] and i["feedback"] is None for i in items)
    assert items[0]["link"] == "https://calendar.google.com/event?eid=ev1"
    assert next(i for i in items if i["section"] == "tasks")["link"].startswith("/tasks/")
    assert "_mail" not in str(items)


async def test_email_summaries_fall_back_to_sender_and_subject(app):
    app.config.proactivity.brief.sections = ["email"]
    app.fake.text_replies.append("Sure! Here you go:\n2) Kavya asks to move lunch to 13:00\n{\"oops\": 1}")
    brief = await app.proactivity.brief.build(NOW)
    assert [i["text"] for i in brief["items"]] == [
        "Arjun Mehta: Q3 deck", "Kavya asks to move lunch to 13:00", "billing: Invoice 42",
    ]
    assert len(app.fake.text_calls) == 1  # one short call for every email together

    app.config.proactivity.brief.summarize_emails = False
    calls = len(app.fake.text_calls)
    await app.proactivity.brief.build(NOW)
    assert len(app.fake.text_calls) == calls


async def test_sections_turned_off_are_skipped(app):
    app.config.proactivity.brief.sections = ["calendar", "tasks"]
    brief = await app.proactivity.brief.build(NOW)
    assert {i["section"] for i in brief["items"]} == {"calendar"}
    assert app.fakes.calls == ["gcal_list_events"]
    assert app.fake.text_calls == []


async def test_missing_sources_are_explained_not_invented(app, monkeypatch):
    async def nothing_connected(plugin_id):
        return False

    monkeypatch.setattr(app.integrations, "is_connected", nothing_connected)
    app.config.assistant.location = ""
    app.config.proactivity.brief.sections = ["calendar", "email", "tasks", "weather", "news"]
    brief = await app.proactivity.brief.build(NOW)
    assert brief["items"] == []
    assert {s["section"] for s in brief["skipped"]} == {"calendar", "email", "weather", "news"}
    assert app.fakes.calls == []
    assert brief_mod.DailyBrief.as_text(brief) == "Nothing needs you this morning. Enjoy your day."


async def test_read_only_never_calls_anything_but_read_tools(app, monkeypatch):
    acted: list[str] = []
    for t in app.registry.tools(include_hidden=True):
        if t.risk != Risk.read:
            async def trap(ctx, _n=t.name, **kwargs):
                acted.append(_n)
                return {"ok": True}
            monkeypatch.setattr(t, "fn", trap)
    monkeypatch.setattr(app.registry.get("gcal_list_events"), "risk", Risk.send)  # a read tool that changed
    app.config.tools.approvals.rules = {"weather": "never"}
    app.config.proactivity.brief.news_topics = ["climate"]
    app.config.proactivity.brief.sections = ["calendar", "email", "tasks", "weather", "news"]
    brief = await app.proactivity.brief.build(NOW)
    assert acted == []
    assert "gcal_list_events" not in app.fakes.calls and "weather_current" not in app.fakes.calls
    assert set(app.fakes.calls) == {"gmail_search", "news_search"}
    assert {i["section"] for i in brief["items"]} == {"email", "news"}
    assert app.registry.get(brief_mod.READ_TOOL).risk == Risk.read
    assert app.registry.get(brief_mod.BUILD_TOOL).internal  # it only adds a notification


# ---------------------------------------------------------------------------- steering
async def test_feedback_is_recorded_and_changes_later_briefs(app):
    await add_tasks(app)
    b = app.proactivity.brief
    first = await b.deliver(NOW)
    assert [s["id"] for s in first["sections"]] == ["calendar", "email", "tasks", "weather"]
    assert len([i for i in first["items"] if i["section"] == "tasks"]) == 1
    task_item = next(i for i in first["items"] if i["section"] == "tasks")
    weather = next(i for i in first["items"] if i["section"] == "weather")

    await b.feedback(first["id"], "up", section="tasks")
    out = await b.feedback(first["id"], "up", item=task_item["id"])
    await b.feedback(first["id"], "down", item=weather["id"])
    second = await b.deliver(NOW)  # a new brief can be rated again
    await b.feedback(second["id"], "up", section="tasks")
    assert next(s for s in out["sections"] if s["id"] == "tasks")["feedback"] == "up"
    assert next(i for i in out["items"] if i["id"] == task_item["id"])["feedback"] == "up"
    scores = await app.proactivity.preference_scores()
    assert scores["daily_brief_tasks"] == 3 and scores["daily_brief_weather"] == -1

    with pytest.raises(BriefError) as dup:
        await b.feedback(second["id"], "down", section="tasks")
    assert dup.value.status == 409
    with pytest.raises(BriefError) as missing:
        await b.feedback(second["id"], "up", item="nope")
    assert missing.value.status == 404
    with pytest.raises(BriefError):
        await b.feedback(second["id"], "meh", section="email")

    later = await b.build(NOW)
    assert later["sections"][0]["id"] == "tasks"  # liked most, so first
    assert len([i for i in later["items"] if i["section"] == "tasks"]) == 3  # it now gets its lines first


def test_cap_takes_turns_and_respects_section_limits():
    found = {s: [{"section": s, "id": f"{s}{n}"} for n in range(5)] for s in ("calendar", "email", "tasks")}
    items = cap_items(found, ["calendar", "email", "tasks"], {}, 7)
    assert [i["id"] for i in items] == ["calendar0", "calendar1", "calendar2", "email0", "email1", "email2", "tasks0"]
    liked = cap_items(found, ["tasks", "calendar", "email"], {"daily_brief_tasks": 5, "daily_brief_email": -4}, 20)
    assert [i["section"] for i in liked].count("tasks") == 4 and [i["section"] for i in liked].count("email") == 2


def test_parse_summaries_is_tolerant():
    assert parse_summaries("1. One line here\n2) Second line\n2. dup\n9. out of range\n**3:** {x}", 3) == {
        1: "One line here", 2: "Second line",
    }
    assert parse_summaries("", 2) == {}


# ---------------------------------------------------------------------------- the task
async def test_setup_creates_one_editable_recurring_task(app):
    b = app.proactivity.brief
    assert (await b.state())["set_up"] is False
    state = await b.setup({})
    task = await app.tasks.get(state["task_id"])
    assert task["name"] == "Daily Brief" and task["status"] == "active" and task["enabled"]
    assert task["schedule"]["time"] == "07:30" and task["schedule"]["days"] == brief_mod.DEFAULT_DAYS
    assert task["original_context"]["fixed_call"] == {"tool": "daily_brief_build", "arguments": {},
                                                      "done_text": "Your Daily Brief is ready.", "quiet": True}
    assert task["next_execution_at"]
    assert state["days"] == brief_mod.DEFAULT_DAYS and state["time"] == "07:30"

    state = await b.setup({"time": "evening", "days": "daily", "sections": ["calendar", "news"], "news_topics": ["cricket"]})
    assert len(await app.tasks.list()) == 1
    task = await app.tasks.get(state["task_id"])
    assert task["schedule"]["frequency"] == "daily" and task["schedule"]["time"] == "18:00"
    assert state["sections"] == ["calendar", "news"] and state["news_topics"] == ["cricket"]
    assert app.config.proactivity.brief.sections == ["calendar", "news"]

    # it is an ordinary task: the normal task edit changes the schedule, and deleting it turns the brief off
    edited = await app.tasks.update(state["task_id"], {"schedule": schedule_for("08:15", ["Saturday"], "UTC")})
    expected = calculate_next_run(edited["schedule"], app.tasks.now())
    assert edited["schedule"]["time"] == "08:15" and edited["next_execution_at"].startswith(expected.strftime("%Y-%m-%dT08:15"))
    await app.tasks.update(state["task_id"], {"enabled": False})
    assert (await b.state())["enabled"] is False
    await app.tasks.delete(state["task_id"])
    assert (await b.state())["set_up"] is False


async def test_scheduled_run_delivers_the_brief_without_a_task_completed_note(app):
    state = await app.proactivity.brief.setup({"sections": ["calendar", "weather"]})
    task = await app.tasks.get(state["task_id"])
    due = datetime.fromisoformat(task["next_execution_at"])
    app.tasks.clock = lambda: due + timedelta(seconds=1)
    assert await app.tasks.tick()
    await app.tasks.drain()
    notes = await app.notifications.list()
    briefs = [n for n in notes if n["kind"] == "brief"]
    assert len(briefs) == 1 and briefs[0]["payload"]["task_id"] == task["task_id"]
    assert "**Calendar**" in briefs[0]["message"] and "Design review" in briefs[0]["message"]
    assert not [n for n in notes if n["kind"] == "task"]  # quiet: the brief is the notification
    after = await app.tasks.get(task["task_id"])
    assert after["status"] == "active" and after["runs"][-1]["status"] == "completed"
    assert after["next_execution_at"] > task["next_execution_at"]


async def test_brief_expires_at_the_end_of_the_day(app):
    b = app.proactivity.brief
    first = await b.deliver(NOW)
    assert first["expires_at"] == "2026-10-13T00:00:00+00:00"
    assert (await b.today(NOW))["id"] == first["id"]
    assert await b.expire(NOW + timedelta(hours=1)) == 0
    assert await b.expire(NOW + timedelta(days=1)) == 1
    note = await app.notifications.get(first["id"])
    assert note["payload"]["status"] == "expired" and note["read"]
    assert await b.today(NOW + timedelta(days=1)) is None
    with pytest.raises(BriefError) as gone:
        await b.feedback(first["id"], "up", section="calendar")
    assert gone.value.status == 409

    second = await b.deliver(NOW)
    third = await b.deliver(NOW)  # "make one now" replaces the one still showing
    assert (await app.notifications.get(second["id"]))["payload"]["status"] == "expired"
    assert (await b.today(NOW))["id"] == third["id"]


async def test_read_my_brief_tool_reads_today_or_builds_one(app):
    tool = app.registry.get(brief_mod.READ_TOOL)
    ctx = app.agent.tool_context(None, "voice")
    fresh = await tool.call(ctx, {})
    assert fresh["items"] > 0 and "Design review" in fresh["text"] and "**" not in fresh["text"]
    assert not [n for n in await app.notifications.list() if n["kind"] == "brief"]  # reading never delivers
    await app.proactivity.brief.deliver()
    again = await tool.call(ctx, {})
    assert again["title"].startswith("Your Daily Brief for")


# ---------------------------------------------------------------------------- REST
def test_brief_routes(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    config.assistant.timezone = "UTC"
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "routes.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        Fakes().install(core, monkeypatch)
        state = c.get("/api/proactivity/brief").json()
        assert state["set_up"] is False and state["today"] is None and state["available"]["tasks"] is True
        assert c.post("/api/proactivity/brief/run").status_code == 409
        state = c.post("/api/proactivity/brief", json={"time": "07:00", "sections": ["calendar"]}).json()
        assert state["set_up"] and state["time"] == "07:00" and state["sections"] == ["calendar"]
        assert c.post("/api/proactivity/brief", json={"max_items": "lots"}).status_code == 400
        assert c.post("/api/proactivity/brief/feedback", json={"brief_id": "x", "value": "up", "section": "calendar"}).status_code == 404


def test_onboarding_can_set_up_the_brief(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "onb.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        assert c.post("/api/onboarding", json={"user_name": "Maya Rao", "daily_brief": True}).json() == {"ok": True}
        state = c.get("/api/proactivity/brief").json()
        assert state["set_up"] and state["time"] == "07:30"


def test_schedule_words_and_bad_times():
    assert schedule_for("morning")["time"] == "07:30" and schedule_for("Evening", "daily")["frequency"] == "daily"
    assert schedule_for(None, ["Saturday"])["days"] == ["Saturday"]
    with pytest.raises(ValueError):
        schedule_for("25:99")
