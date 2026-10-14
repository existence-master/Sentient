"""Quick commands (#150): timers and reminders done without the model, and everything else left to it.

The phrasing tables are the regression suite: each phrasing that must resolve with no model call, and each one that
must reach the model unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from sentient.agent import fastpath
from sentient.agent.fastpath import en
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.events import Done, TextDelta, ToolCallEvent, ToolResultEvent
from tests.conftest import FakeProvider

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 14, 15, 7, tzinfo=IST)  # a Wednesday afternoon


# ---------------------------------------------------------------------------- the phrasings
TIMERS = [
    ("set a timer for 10 minutes", 600, ""),
    ("Set a timer for 1 minute.", 60, ""),
    ("Hey Sentient, set a timer for five minutes.", 300, ""),
    ("can you set a timer for 20 seconds please", 20, ""),
    ("start a 10 minute timer", 600, ""),
    ("set a ten-minute timer", 600, ""),
    ("timer for 90 seconds", 90, ""),
    ("10 min timer", 600, ""),
    ("set a pasta timer for 8 minutes", 480, "pasta"),
    ("set a timer for 8 minutes for the pasta", 480, "pasta"),
    ("set a timer for 12 minutes called eggs", 720, "eggs"),
    ("set a timer for an hour and a half", 5400, ""),
    ("set a timer for half an hour", 1800, ""),
    ("set a timer for twenty five minutes", 1500, ""),
    ("set a timer for 1 hour 30 minutes", 5400, ""),
    ("set a timer for two and a half minutes", 150, ""),
]
REMINDERS = [  # (text, (day offset, hour, minute) or seconds from now, message, connector)
    ("remind me at 6 pm to call Ravi", (0, 18, 0), "call Ravi", "to"),
    ("remind me to call Ravi at 6 pm", (0, 18, 0), "call Ravi", "to"),
    ("Remind me at 6:30 p.m. to call Ravi.", (0, 18, 30), "call Ravi", "to"),
    ("remind me at 18:00 to leave", (0, 18, 0), "leave", "to"),
    ("remind me at 9 am to pay rent", (1, 9, 0), "pay rent", "to"),  # 9 AM has passed today: tomorrow
    ("remind me tomorrow at 9 am to pay rent", (1, 9, 0), "pay rent", "to"),
    ("remind me to pay rent at 9 am tomorrow", (1, 9, 0), "pay rent", "to"),
    ("remind me tonight at 8 to call mom", (0, 20, 0), "call mom", "to"),
    ("remind me at noon tomorrow to eat", (1, 12, 0), "eat", "to"),
    ("remind me at seven in the evening to eat", (0, 19, 0), "eat", "to"),
    ("set a reminder for 6 pm to call Ravi", (0, 18, 0), "call Ravi", "to"),
    ("remind me in 20 minutes", 1200, "", ""),
    ("remind me in 2 minutes to stretch", 120, "stretch", "to"),
    ("remind me to stretch in 2 minutes", 120, "stretch", "to"),
    ("remind me to look at the oven in 10 minutes", 600, "look at the oven", "to"),
    ("Remind me about the meeting in 15 min", 900, "the meeting", "about"),
    ("remind me in an hour that the laundry is done", 3600, "the laundry is done", "that"),
    ("could you remind me in 3 days to renew the passport", 3 * 86400, "renew the passport", "to"),
]
CANCELS = [
    ("cancel the timer", "timer", "", False),
    ("stop the pasta timer", "timer", "pasta", False),
    ("cancel the 10 minute timer", "timer", "10 minute", False),
    ("cancel all timers", "timer", "", True),
    ("cancel all of my reminders", "reminder", "", True),
    ("cancel my reminder to stretch", "reminder", "stretch", False),
    ("turn off the reminder about the meeting", "reminder", "the meeting", False),
]
LISTS = [
    ("what timers do I have", "timer"),
    ("how much time is left", "timer"),
    ("how much time is left on my timer", "timer"),
    ("what are my reminders", "reminder"),
    ("do I have any timers running", "timer"),
]
TO_THE_MODEL = [
    "set a timer",                                   # how long?
    "set a timer for a bit",
    "set a timer for 25 hours",                      # too long for a timer
    "set a timer for 10 minutes and email Bob",      # more than a timer
    "remind me at 6 to call Ravi",                   # 6 AM or 6 PM?
    "remind me at 6:30 to call Ravi",
    "remind me at 9 am today to call Ravi",          # already passed today
    "remind me every day at 6 pm to stretch",        # repeating: a task
    "remind me each Monday to water the plants",
    "remind me on Friday to call Ravi",              # a day name
    "remind me tonight to call Ravi",                # no time
    "remind me to call Ravi",
    "remind Ravi at 6 pm to call me",                # a message to someone else
    "send Ravi a reminder at 6 pm",
    "cancel timers",                                 # plural without "all"
    "turn off the calendar reminders",               # may mean a setting
    "delete the timer",                              # never a quick delete
    "buy a kitchen timer",
    "what's the weather",
    "how does a timer work",
    "turn off the living room lights",               # no smart-home integration yet
]


@pytest.mark.parametrize(("text", "seconds", "label"), TIMERS)
def test_timer_phrasings(text, seconds, label):
    cmd = en.parse(text, NOW, ["Sentient"])
    assert cmd is not None and cmd.action == "timer"
    assert cmd.seconds == seconds and cmd.label == label


@pytest.mark.parametrize(("text", "when", "message", "connector"), REMINDERS)
def test_reminder_phrasings(text, when, message, connector):
    cmd = en.parse(text, NOW, ["Sentient"])
    assert cmd is not None and cmd.action == "reminder"
    if isinstance(when, tuple):
        days, hour, minute = when
        assert cmd.seconds is None
        assert cmd.at == datetime(2026, 10, 14 + days, hour, minute, tzinfo=IST)
    else:
        assert cmd.seconds == when and cmd.at == NOW + timedelta(seconds=when)
    assert cmd.message == message and cmd.connector == connector


@pytest.mark.parametrize(("text", "target", "which", "everything"), CANCELS)
def test_cancel_phrasings(text, target, which, everything):
    cmd = en.parse(text, NOW, ["Sentient"])
    assert cmd is not None and cmd.action == "cancel"
    assert (cmd.target, cmd.which, cmd.all) == (target, which, everything)


@pytest.mark.parametrize(("text", "target"), LISTS)
def test_list_phrasings(text, target):
    cmd = en.parse(text, NOW, ["Sentient"])
    assert cmd is not None and cmd.action == "list" and cmd.target == target


@pytest.mark.parametrize("text", TO_THE_MODEL)
def test_everything_else_goes_to_the_model(text):
    assert en.parse(text, NOW, ["Sentient"]) is None


def test_the_assistants_own_name_is_understood():
    assert en.parse("Nova, set a timer for 5 minutes", NOW, ["Nova"]) is not None
    assert fastpath.parse("set a timer for 5 minutes", NOW, lang="de-DE") is not None  # English is always tried


def test_said_back_in_plain_words():
    assert en.timer_set(600, "") == "Timer set for 10 minutes."
    assert en.timer_set(5400, "bread") == "Bread timer set for 1 hour and 30 minutes."
    at = datetime(2026, 10, 14, 18, 0, tzinfo=IST)
    r = {"kind": "reminder", "message": "call my mom", "connector": "to", "fire_at": at.isoformat()}
    assert en.reminder_set(r, NOW) == "OK, I'll remind you to call your mom at 6:00 PM."
    r2 = {**r, "fire_at": (at + timedelta(days=1)).isoformat()}
    assert en.reminder_set(r2, NOW) == "OK, I'll remind you to call your mom tomorrow at 6:00 PM."
    assert en.reminder_set({"kind": "reminder", "message": "", "duration_s": 1200}, NOW) == "OK, I'll remind you in 20 minutes."
    assert en.cancelled([{"kind": "timer", "duration_s": 600, "label": ""}]) == "Cancelled your 10 minute timer."
    assert en.timer_done({"kind": "timer", "duration_s": 60, "label": ""}) == ("Timer done", "Your 1 minute timer is done.")
    assert en.reminder_due({"message": "stretch"}, None) == ("Reminder", "Stretch")


# ---------------------------------------------------------------------------- in a chat turn
async def _app(config, isolated_home, llm=None, **cfg) -> SentientApp:
    config.chat.tool_selection = "all"
    config.tools.approvals.mode = "ask"
    config.assistant.timezone = "Asia/Kolkata"
    for key, value in cfg.items():
        section, _, field = key.partition("__")
        setattr(getattr(config, section), field, value)
    return await SentientApp(config, llm=llm or FakeProvider(), db_path=isolated_home / "q.db",
                             enable_background=False).start()


async def _turn(s: SentientApp, sid: str, text: str, channel: str = "desktop") -> list:
    return [ev async for ev in s.agent.run_turn(sid, text, channel=channel)]


def _reply(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


async def test_timer_is_set_without_the_model(config, isolated_home):
    llm = FakeProvider()
    s = await _app(config, isolated_home, llm)
    try:
        sid = await s.store.create_session(channel="desktop")
        before = datetime.now(UTC)
        events = await _turn(s, sid, "Set a timer for 10 minutes.")
        assert llm.calls == [] and llm.text_calls == []  # no model at all
        assert [type(e).__name__ for e in events] == ["ToolCallEvent", "ToolResultEvent", "TextDelta", "Done"]
        call = next(e for e in events if isinstance(e, ToolCallEvent))
        assert call.name == "set_timer" and call.arguments["seconds"] == 600
        assert _reply(events) == "Timer set for 10 minutes." and events[-1].content == "Timer set for 10 minutes."
        timers = await s.reminders.list()
        assert len(timers) == 1 and timers[0]["kind"] == "timer" and timers[0]["source"] == "quick"
        due = datetime.fromisoformat(timers[0]["fire_at"])
        assert timedelta(seconds=599) < due - before < timedelta(seconds=602)
        # the chat keeps the turn like any other, so the model sees it later
        msgs = await s.store.recent_messages(sid, 10)
        assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
        assert timers[0]["session_id"] == sid and timers[0]["channel"] == "desktop"
    finally:
        await s.stop()


async def test_reminder_uses_the_users_timezone(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        plan = await fastpath.plan(s, "remind me at 6 pm to call Ravi", now=NOW)
        assert plan is not None and plan.calls[0].name == "set_reminder"
        assert plan.calls[0].arguments["at"] == "2026-10-14T18:00:00+05:30"
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "remind me in 2 minutes to stretch")
        assert _reply(events) == "OK, I'll remind you to stretch in 2 minutes."
        item = (await s.reminders.list())[0]
        assert item["kind"] == "reminder" and item["message"] == "stretch" and item["duration_s"] == 120
    finally:
        await s.stop()


async def test_unmatched_text_reaches_the_model_unchanged(config, isolated_home):
    llm = FakeProvider(replies=["How long should it run?"])
    s = await _app(config, isolated_home, llm)
    try:
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "set a timer")
        assert _reply(events) == "How long should it run?"
        assert len(llm.calls) == 1 and llm.calls[0]["messages"][-1]["content"].endswith("set a timer")
        assert await s.reminders.list() == []
    finally:
        await s.stop()


async def test_setting_off_and_approvals_send_it_to_the_model(config, isolated_home):
    llm = FakeProvider(replies=["model 1", "model 2", "model 3"])
    s = await _app(config, isolated_home, llm, chat__quick_commands=False)
    try:
        sid = await s.store.create_session(channel="desktop")
        assert _reply(await _turn(s, sid, "set a timer for 1 minute")) == "model 1"
        s.config.chat.quick_commands = True
        s.config.tools.approvals.mode = "always"  # the user wants to OK every tool call
        assert _reply(await _turn(s, sid, "set a timer for 1 minute")) == "model 2"
        s.config.tools.approvals.mode = "ask"
        s.config.tools.approvals.rules = {"set_timer": "never"}
        assert _reply(await _turn(s, sid, "set a timer for 1 minute")) == "model 3"
        assert await s.reminders.list() == []
    finally:
        await s.stop()


async def test_timer_fires_a_notification_on_time(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        sid = await s.store.create_session(channel="voice")
        await _turn(s, sid, "set a timer for 1 minute", channel="voice")
        item = (await s.reminders.list())[0]
        due = datetime.fromisoformat(item["fire_at"])
        assert await s.reminders.fire_due(due - timedelta(seconds=1)) == 0
        assert await s.reminders.fire_due(due + timedelta(seconds=1)) == 1
        note = (await s.notifications.list())[0]
        assert note["kind"] == "reminder" and note["title"] == "Timer done"
        assert note["message"] == "Your 1 minute timer is done."
        assert note["payload"]["session_id"] == sid and note["payload"]["late"] is False
        fired = await s.reminders.get(item["id"])
        assert fired["status"] == "fired" and fired["notification_id"] == note["id"]
        assert await s.reminders.fire_due(due + timedelta(seconds=5)) == 0  # once only
    finally:
        await s.stop()


async def test_a_reminder_missed_while_off_says_when_it_was_due(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        at = datetime(2026, 10, 14, 18, 0, tzinfo=IST)
        item = await s.reminders.add("reminder", datetime.now(UTC) + timedelta(seconds=30), message="call Ravi",
                                     connector="to")
        await s.store.execute("UPDATE reminders SET fire_at = ? WHERE id = ?", (at.astimezone(UTC).isoformat(), item["id"]))
        assert await s.reminders.fire_due(at + timedelta(hours=2)) == 1
        note = (await s.notifications.list())[0]
        assert note["title"] == "Reminder" and note["message"].startswith("Call Ravi (It was due at 6:00 PM")
    finally:
        await s.stop()


async def test_cancel_by_chat(config, isolated_home):
    llm = FakeProvider(replies=["Which one?"])
    s = await _app(config, isolated_home, llm)
    try:
        sid = await s.store.create_session(channel="desktop")
        await _turn(s, sid, "set a timer for 10 minutes")
        assert _reply(await _turn(s, sid, "cancel the timer")) == "Cancelled your 10 minute timer."
        assert await s.reminders.list() == []
        assert (await s.reminders.list("cancelled"))[0]["kind"] == "timer"
        assert _reply(await _turn(s, sid, "cancel the timer")) == "You don't have any timers running."
        # two timers: which one? the model asks
        await _turn(s, sid, "set a pasta timer for 8 minutes")
        await _turn(s, sid, "set a timer for 3 minutes")
        assert _reply(await _turn(s, sid, "cancel the timer")) == "Which one?"
        assert _reply(await _turn(s, sid, "stop the pasta timer")) == "Cancelled your pasta timer."
        assert _reply(await _turn(s, sid, "how much time is left")).startswith("You have 1 timer: 3 minutes, ")
        assert _reply(await _turn(s, sid, "cancel the 3 minute timer")) == "Cancelled your 3 minute timer."
        assert len(llm.calls) == 1
    finally:
        await s.stop()


async def test_the_model_can_use_the_same_tools(config, isolated_home):
    llm = FakeProvider(replies=[
        [{"id": "c1", "name": "set_reminder", "arguments": {"message": "call the bank", "at": "2026-12-01 10:00"}}],
        "Done, I'll remind you.",
    ])
    s = await _app(config, isolated_home, llm)
    try:
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "remind me on the first of December at 10 to call the bank")
        assert any(isinstance(e, ToolResultEvent) and not e.is_error for e in events)
        item = (await s.reminders.list())[0]
        assert item["source"] == "chat" and item["fire_at"] == "2026-12-01T04:30:00+00:00"  # 10:00 in Kolkata
    finally:
        await s.stop()


async def test_activity_lists_and_cancels_them(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        sid = await s.store.create_session(channel="desktop")
        await _turn(s, sid, "set a timer for 5 minutes")
        await _turn(s, sid, "remind me in 2 minutes to stretch")
        upcoming = (await s.timeline.upcoming())["items"]
        mine = [i for i in upcoming if i["kind"] == "reminders"]
        assert [i["title"] for i in mine] == ["Will remind you: stretch", "Your 5 minute timer ends"]
        assert all(i["undo"] == {"action": "cancel", "label": "Cancel"} and i["status"] == "scheduled" for i in mine)
        assert (await s.timeline.cancel(mine[1]["id"]))["ok"] is True
        assert (await s.timeline.cancel(mine[1]["id"]))["ok"] is False  # already cancelled
        await s.reminders.fire_due(datetime.now(UTC) + timedelta(minutes=3))
        past = (await s.timeline.past(kinds=["reminders"]))["items"]
        assert [i["title"] for i in past] == ["Reminded you: stretch", "Cancelled your 5 minute timer"]
        assert past[0]["why"] == "You set it in a chat" and past[0]["ref"] == {"type": "chat", "id": sid}
        # the fired notification is not listed a second time
        everything = (await s.timeline.past())["items"]
        assert sum(1 for i in everything if "stretch" in (i["title"] or "")) == 1
    finally:
        await s.stop()


def test_rest_routes(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    config.assistant.timezone = "UTC"
    app = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        made = c.post("/api/reminders", json={"kind": "timer", "minutes": 2, "label": "tea"}).json()
        assert made["kind"] == "timer" and made["label"] == "tea" and made["source"] == "api"
        assert [r["id"] for r in c.get("/api/reminders").json()["reminders"]] == [made["id"]]
        assert c.post("/api/reminders", json={"kind": "timer"}).status_code == 400
        assert c.post(f"/api/reminders/{made['id']}/cancel").json()["status"] == "cancelled"
        assert c.post(f"/api/reminders/{made['id']}/cancel").status_code == 409
        assert c.post("/api/reminders/nope/cancel").status_code == 404
        assert c.get("/api/reminders?status=cancelled").json()["reminders"][0]["id"] == made["id"]
        again = c.post("/api/reminders", json={"kind": "reminder", "minutes": 5, "message": "tea"}).json()
        assert c.post("/api/timeline/cancel", json={"id": f"reminder:{again['id']}"}).json()["ok"] is True
        assert c.post("/api/timeline/cancel", json={"id": "chat:x"}).status_code == 409


async def test_done_is_the_last_event_and_carries_the_message(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        sid = await s.store.create_session(channel="desktop")
        events = await _turn(s, sid, "what timers do I have")
        assert isinstance(events[-1], Done) and events[-1].message_id
        assert events[-1].content == "You don't have any timers running."
    finally:
        await s.stop()


async def test_a_failed_quick_turn_still_ends_with_error_and_done(config, isolated_home, monkeypatch):
    s = await _app(config, isolated_home)
    try:
        sid = await s.store.create_session(channel="desktop")

        async def broken(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(s.agent, "_tool_content", broken)
        events = await _turn(s, sid, "set a timer for 1 minute")
        assert [type(e).__name__ for e in events[-2:]] == ["Error", "Done"]
        assert events[-1].error == "disk full" and events[-2].recoverable is False
    finally:
        await s.stop()


async def test_a_reminder_whose_notification_failed_is_tried_again(config, isolated_home, monkeypatch):
    s = await _app(config, isolated_home)
    try:
        first = await s.reminders.add("reminder", datetime.now(UTC) + timedelta(seconds=30), message="stretch")
        second = await s.reminders.add("timer", datetime.now(UTC) + timedelta(seconds=30), duration_s=30)
        real = s.notify
        calls = {"n": 0}

        async def flaky(kind, message, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("database is locked")
            return await real(kind, message, **kwargs)

        monkeypatch.setattr(s, "notify", flaky)
        later = datetime.now(UTC) + timedelta(minutes=1)
        assert await s.reminders.fire_due(later) == 1  # the other one still went off
        statuses = {r["id"]: r["status"] for r in await s.reminders.list(None)}
        assert sorted(statuses.values()) == ["fired", "scheduled"]
        assert await s.reminders.fire_due(later) == 1  # and the failed one on the next try
        assert {(await s.reminders.get(i))["status"] for i in (first["id"], second["id"])} == {"fired"}
        assert len(await s.notifications.list()) == 2
    finally:
        await s.stop()
