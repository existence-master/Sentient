"""Quick-command extras (#359): alarms, "what's next on my calendar", and desktop reminders in the paired chats.

Phrasings that must resolve with no model call, and ones that must still reach the model.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sentient.agent import fastpath
from sentient.agent.fastpath import en
from sentient.llm.events import ToolCallEvent
from tests.conftest import FakeProvider
from tests.test_quick_commands import IST, NOW, _app, _reply, _turn

# NOW is Wednesday 14 October 2026, 3:07 PM in Kolkata
ALARMS = [  # (text, (day offset, hour, minute) or seconds from now, label)
    ("wake me at 7", (1, 7, 0), ""),                        # never 7 PM: the next morning
    ("wake me up at 6:30 am", (1, 6, 30), ""),
    ("Hey Sentient, wake me up at 6:30 tomorrow", (1, 6, 30), ""),
    ("wake me at 4", (0, 16, 0), ""),                       # a nap: the next 4 o'clock
    ("wake me up in 20 minutes", 1200, ""),
    ("set an alarm for 6:30", (0, 18, 30), ""),             # the next 6:30, as on a phone
    ("set an alarm for 6:30 tomorrow", (1, 6, 30), ""),     # tomorrow: the morning one
    ("set an alarm for tomorrow at 7", (1, 7, 0), ""),
    ("set alarm for 7 am", (1, 7, 0), ""),                  # 7 AM has passed today
    ("alarm for 18:30", (0, 18, 30), ""),
    ("set a 6:30 alarm", (0, 18, 30), ""),
    ("set an alarm for 7 called gym", (0, 19, 0), "gym"),
    ("set an alarm for seven thirty", (0, 19, 30), ""),
    ("please set an alarm at 5:45 pm", (0, 17, 45), ""),
    ("set an alarm in 10 minutes", 600, ""),
]
ALARMS_TO_THE_MODEL = [
    "set an alarm",                                # when?
    "set an alarm for every day at 7",             # repeating
    "wake me up every weekday at 6",
    "set an alarm for Friday at 7",                # a day name
    "set an alarm for 2 pm today",                 # already passed today
    "set an alarm for 6:30 tomorrow morning",
    "wake me when the download finishes",
    "get me a coffee at 7",
    "how do alarms work",
]
CALENDAR = [
    "what's next on my calendar",
    "What's next on my calendar?",
    "hey Sentient, what is next on my schedule",
    "what's coming up on my calendar",
    "what's my next meeting",
    "when is my next meeting",
    "what do I have next",
    "show me my next event",
    "next on my calendar",
]
CALENDAR_TO_THE_MODEL = [
    "what's next",                                 # could be anything
    "what's on my calendar tomorrow",
    "add a meeting to my calendar",
    "what's next on my calendar and email it to Ravi",
    "cancel my next meeting",
]


@pytest.mark.parametrize(("text", "when", "label"), ALARMS)
def test_alarm_phrasings(text, when, label):
    cmd = en.parse(text, NOW, ["Sentient"])
    assert cmd is not None and cmd.action == "alarm"
    if isinstance(when, tuple):
        days, hour, minute = when
        assert cmd.seconds is None and cmd.at == datetime(2026, 10, 14 + days, hour, minute, tzinfo=IST)
    else:
        assert cmd.seconds == when and cmd.at == NOW + timedelta(seconds=when)
    assert cmd.label == label


@pytest.mark.parametrize("text", ALARMS_TO_THE_MODEL + CALENDAR_TO_THE_MODEL)
def test_these_still_go_to_the_model(text):
    assert en.parse(text, NOW, ["Sentient"]) is None


@pytest.mark.parametrize("text", CALENDAR)
def test_calendar_phrasings(text):
    cmd = en.parse(text, NOW, ["Sentient"])
    assert cmd is not None and cmd.action == "calendar_next"


@pytest.mark.parametrize(("text", "action", "target"), [
    ("cancel my alarm", "cancel", "alarm"),
    ("turn off the gym alarm", "cancel", "alarm"),
    ("cancel all alarms", "cancel", "alarm"),
    ("what alarms do I have", "list", "alarm"),
    ("when is my alarm", "list", "alarm"),
    ("do I have an alarm set", "list", "alarm"),
])
def test_alarms_are_listed_and_cancelled_like_timers(text, action, target):
    cmd = en.parse(text, NOW, ["Sentient"])
    assert cmd is not None and (cmd.action, cmd.target) == (action, target)


def test_alarm_words():
    at = datetime(2026, 10, 15, 7, 0, tzinfo=IST)
    alarm = {"kind": "alarm", "label": "", "fire_at": at.isoformat(), "fire_at_local": "2026-10-15 07:00:00"}
    assert en.alarm_set(alarm, NOW) == "Alarm set for 7:00 AM tomorrow."
    assert en.alarm_set({**alarm, "label": "gym"}, NOW) == "Gym alarm set for 7:00 AM tomorrow."
    soon = {"kind": "alarm", "label": "", "duration_s": 600, "fire_at": (NOW + timedelta(minutes=10)).isoformat()}
    assert en.alarm_set(soon, NOW) == "Alarm set for 10 minutes from now, at 3:17 PM."
    assert en.alarm_due(alarm, at, False) == ("Alarm", "It's 7:00 AM.")
    assert en.alarm_due({**alarm, "label": "gym"}, at, True) == (
        "Alarm: gym", "Your 7:00 AM alarm went off while Sentient wasn't running.")
    assert en.cancelled([alarm]) == "Cancelled your 7:00 AM alarm."
    assert en.cancelled([alarm, {"kind": "timer", "duration_s": 60, "label": ""}]) == "Cancelled 1 timer and 1 alarm."
    assert en.listing([alarm], NOW) == "You have 1 alarm: 7:00 AM tomorrow."
    assert en.nothing("alarm") == "You don't have any alarms set."


EVENTS = [  # what gcal_list_events returns: soonest first, from now
    {"id": "e0", "summary": "Standup", "start": "2026-10-14T15:00:00+05:30", "end": "2026-10-14T15:15:00+05:30",
     "all_day": False, "status": "confirmed"},
    {"id": "e1", "summary": "Design review with Maya Rao", "start": "2026-10-14T16:30:00+05:30",
     "end": "2026-10-14T17:00:00+05:30", "all_day": False, "location": "Room 4", "status": "confirmed"},
    {"id": "e2", "summary": "Call with the printer", "start": "2026-10-14T18:00:00+05:30",
     "end": "2026-10-14T18:30:00+05:30", "all_day": False, "status": "confirmed"},
    {"id": "e3", "summary": "Maya's birthday", "start": "2026-10-16", "end": "2026-10-17", "all_day": True},
]


def test_calendar_words():
    assert en.calendar_next(EVENTS, NOW, 7) == (
        "Right now: Standup, until 3:15 PM. Next: Design review with Maya Rao, today at 4:30 PM (Room 4). "
        "After that: Call with the printer, today at 6:00 PM."
    )
    assert en.calendar_next(EVENTS[3:], NOW, 7) == "Next on your calendar: Maya's birthday, on Friday, all day."
    cancelled = {**EVENTS[1], "status": "cancelled"}
    assert en.calendar_next([cancelled], NOW, 7) == "Nothing on your calendar in the next 7 days."
    birthday_today = {"id": "b", "summary": "Maya's birthday", "start": "2026-10-14", "end": "2026-10-15", "all_day": True}
    trip = {"id": "t", "summary": "Conference trip", "start": "2026-10-13", "end": "2026-10-16", "all_day": True}
    assert en.calendar_next([birthday_today, trip, EVENTS[1]], NOW, 7) == (
        "All day today: Maya's birthday and Conference trip. "
        "Next: Design review with Maya Rao, today at 4:30 PM (Room 4).")  # the meeting is not hidden
    assert en.calendar_next([birthday_today], NOW, 7) == "All day today: Maya's birthday."
    tomorrow = {**EVENTS[1], "start": "2026-10-15T09:00:00Z"}
    assert en.calendar_next([tomorrow], NOW, 7) == (
        "Next on your calendar: Design review with Maya Rao, tomorrow at 2:30 PM (Room 4).")


# ---------------------------------------------------------------------------- in a chat turn
async def test_an_alarm_is_set_and_rings_without_the_model(config, isolated_home):
    llm = FakeProvider()
    s = await _app(config, isolated_home, llm)
    try:
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "set an alarm in 10 minutes")
        assert llm.calls == [] and llm.text_calls == []
        call = next(e for e in events if isinstance(e, ToolCallEvent))
        assert call.name == "set_alarm" and call.arguments == {"label": "", "in_minutes": 10}
        assert _reply(events).startswith("Alarm set for 10 minutes from now, at ")
        item = (await s.reminders.list())[0]
        assert item["kind"] == "alarm" and item["source"] == "quick"

        plan = await fastpath.plan(s, "wake me at 7", now=NOW)
        assert plan is not None and plan.calls[0].name == "set_alarm"
        assert plan.calls[0].arguments == {"label": "", "at": "2026-10-15T07:00:00+05:30"}

        due = datetime.fromisoformat(item["fire_at"])
        assert await s.reminders.fire_due(due + timedelta(seconds=1)) == 1
        note = (await s.notifications.list())[0]
        assert note["kind"] == "reminder" and note["title"] == "Alarm" and note["message"].startswith("It's ")
        assert note["payload"]["event"] == "alarm" and note["payload"]["late"] is False

        assert _reply(await _turn(s, sid, "what alarms do I have")) == "You don't have any alarms set."
    finally:
        await s.stop()


async def test_an_alarm_can_be_cancelled_by_name(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        sid = await s.store.create_session(channel="desktop")
        assert _reply(await _turn(s, sid, "set an alarm for 23:30 called gym")).startswith("Gym alarm set for 11:30 PM")
        listed = _reply(await _turn(s, sid, "when is my alarm"))
        assert listed.startswith("You have 1 alarm: gym, 11:30 PM")
        assert _reply(await _turn(s, sid, "turn off the gym alarm")) == "Cancelled your gym alarm."
        assert await s.reminders.list() == []
    finally:
        await s.stop()


async def test_turning_off_an_alarm_with_none_set_goes_to_the_model(config, isolated_home):
    """With Home Assistant (#175) "turn off the alarm" may mean the house alarm, which never goes through a quick
    command: with no Sentient alarm set, the model handles it."""
    s = await _app(config, isolated_home)
    try:
        assert await fastpath.plan(s, "turn off the alarm", now=NOW) is None
        assert await fastpath.plan(s, "cancel my reminder to stretch", now=NOW) is not None  # still says "none set"
    finally:
        await s.stop()


async def _calendar_app(config, isolated_home, monkeypatch, *, connected: bool, llm=None):
    s = await _app(config, isolated_home, llm)
    calls: list[dict] = []

    async def fake(ctx, **kwargs):  # one event an hour from the real now
        calls.append(kwargs)
        start = datetime.now(UTC) + timedelta(hours=1)
        event = {**EVENTS[1], "start": start.isoformat(), "end": (start + timedelta(minutes=30)).isoformat()}
        return {"time_min": "now", "time_max": "+7d", "timezone": "Asia/Kolkata", "count": 1, "events": [event]}

    async def is_connected(plugin_id):
        return connected and plugin_id == "gcalendar"

    monkeypatch.setattr(s.registry.get("gcal_list_events"), "fn", fake)
    monkeypatch.setattr(s.integrations, "is_connected", is_connected)
    return s, calls


async def test_whats_next_on_my_calendar_reads_it_with_no_model(config, isolated_home, monkeypatch):
    llm = FakeProvider()
    s, calls = await _calendar_app(config, isolated_home, monkeypatch, connected=True, llm=llm)
    try:
        plan = await fastpath.plan(s, "what's next on my calendar", now=NOW)
        assert plan is not None and plan.calls[0].name == "gcal_list_events"
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "What's next on my calendar?")
        assert llm.calls == [] and llm.text_calls == []
        assert len(calls) == 1 and calls[0]["max_results"] == 10 and not calls[0].get("time_min")
        assert "Design review with Maya Rao" in _reply(events)
    finally:
        await s.stop()


async def test_with_no_calendar_connected_the_model_answers(config, isolated_home, monkeypatch):
    llm = FakeProvider(replies=["Connect Google Calendar in Integrations and I can check."])
    s, calls = await _calendar_app(config, isolated_home, monkeypatch, connected=False, llm=llm)
    try:
        assert await fastpath.plan(s, "what's next on my calendar", now=NOW) is None
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "what's next on my calendar")
        assert len(llm.calls) == 1 and calls == []
        assert _reply(events) == "Connect Google Calendar in Integrations and I can check."
    finally:
        await s.stop()


async def test_a_calendar_behind_a_never_rule_goes_to_the_model(config, isolated_home, monkeypatch):
    llm = FakeProvider(replies=["I can't read your calendar."])
    s, calls = await _calendar_app(config, isolated_home, monkeypatch, connected=True, llm=llm)
    try:
        s.config.tools.approvals.rules = {"gcal_list_events": "never"}
        sid = await s.store.create_session(channel="desktop")
        await _turn(s, sid, "what's next on my calendar")
        assert calls == [] and len(llm.calls) == 1
    finally:
        await s.stop()


async def test_the_calendar_needs_no_reminders_service(config, isolated_home, monkeypatch):
    s, _ = await _calendar_app(config, isolated_home, monkeypatch, connected=True)
    reminders, s.reminders = s.reminders, None
    try:
        assert await fastpath.plan(s, "set a timer for 1 minute", now=NOW) is None
        assert await fastpath.plan(s, "what's my next meeting", now=NOW) is not None
    finally:
        s.reminders = reminders
        await s.stop()


async def test_alarm_over_rest(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        item = await s.reminders.add("alarm", datetime.now(UTC) + timedelta(hours=1), label="gym", source="api")
        assert item["kind"] == "alarm" and item["label"] == "gym"
    finally:
        await s.stop()
