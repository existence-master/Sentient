"""Stopping and snoozing a ringing alarm by saying so (#390): a quick command only while a Sentient alarm rings.

With nothing ringing, "stop the alarm" may mean a house alarm (#175), so it reaches the model as before.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from sentient.agent import fastpath
from sentient.agent.fastpath import en
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.events import ToolCallEvent
from sentient.reminders.service import RING_S
from tests.conftest import FakeProvider
from tests.test_quick_commands import _app, _reply, _turn

STOPS = [
    ("stop", ""), ("Stop!", ""), ("stop the alarm", ""), ("stop it", ""), ("turn it off", ""), ("turn that off", ""),
    ("turn off the alarm", ""), ("switch the alarm off", ""), ("Hey Sentient, stop it please", ""), ("ok stop", ""),
    ("stop stop stop", ""), ("I'm up", ""), ("silence the alarm", ""), ("dismiss", ""), ("alarm off", ""),
    ("turn off the gym alarm", "gym"), ("stop my gym alarm", "gym"),
]
SNOOZES = [
    ("snooze", None), ("Snooze.", None), ("snooze it", None), ("snooze the alarm", None),
    ("snooze for 5 minutes", 300), ("snooze 10", 600), ("snooze it for ten min", 600),
    ("snooze the alarm for 15 minutes", 900), ("5 more minutes", 300), ("give me 10 more minutes", 600),
    ("please snooze for another 5 minutes", 300),
]
NOT_WHILE_RINGING = [  # even while an alarm rings, these are not about it
    "stop the music", "turn off the lights", "snooze for a bit", "snooze notifications for an hour",
    "snooze for 4 hours", "stop reminding me about the gym", "what time is it", "stop the timer",
]


@pytest.mark.parametrize(("text", "which"), STOPS)
def test_stop_phrasings(text, which):
    cmd = en.ringing(text, ["Sentient"])
    assert cmd is not None and cmd.action == "alarm_stop" and cmd.which == which


@pytest.mark.parametrize(("text", "seconds"), SNOOZES)
def test_snooze_phrasings(text, seconds):
    cmd = en.ringing(text, ["Sentient"])
    assert cmd is not None and cmd.action == "alarm_snooze" and cmd.seconds == seconds


@pytest.mark.parametrize("text", NOT_WHILE_RINGING)
def test_other_words_are_not_about_the_ringing_alarm(text):
    assert en.ringing(text, ["Sentient"]) is None


def test_ring_words():
    assert en.alarm_stopped([{}]) == "Alarm stopped."
    assert en.alarm_stopped([{}, {}]) == "Alarms stopped."
    now = datetime(2026, 10, 15, 7, 30, tzinfo=UTC)
    assert en.alarm_snoozed({"fire_at": (now + timedelta(minutes=9)).isoformat()}, now) == "Snoozed until 7:39 AM."


async def _ringing(s: SentientApp, label: str = "", channel: str = "desktop") -> dict:
    """An alarm that has just gone off (fired with the real clock, so it counts as ringing)."""
    item = await s.reminders.add("alarm", datetime.now(UTC) - timedelta(seconds=1), label=label, channel=channel,
                                 source="quick")
    assert await s.reminders.fire_due() >= 1
    return item


async def test_snooze_while_ringing_needs_no_model(config, isolated_home):
    llm = FakeProvider()
    s = await _app(config, isolated_home, llm)
    try:
        sid = await s.store.create_session(channel="desktop")
        rang = await _ringing(s, "gym")
        assert [r["id"] for r in await s.reminders.ringing()] == [rang["id"]]
        events_seen: list[dict] = []
        async with s.bus.subscribe() as q:
            before = datetime.now(UTC)
            events = await _turn(s, sid, "snooze")
            while not q.empty():
                ev = q.get_nowait()
                if ev["type"] == "reminder.updated":
                    events_seen.append(ev["data"])
        assert llm.calls == [] and llm.text_calls == []
        call = next(e for e in events if isinstance(e, ToolCallEvent))
        assert call.name == "snooze_alarm" and call.arguments == {"minutes": 9}
        assert _reply(events).startswith("Snoozed until ")
        assert await s.reminders.ringing() == []
        new = (await s.reminders.list(kind="alarm"))[0]
        assert new["snoozed_from"] == rang["id"] and new["label"] == "gym" and new["status"] == "scheduled"
        left = (datetime.fromisoformat(new["fire_at"]) - before).total_seconds()
        assert 9 * 60 - 1 <= left <= 9 * 60 + 5
        assert (await s.reminders.get(rang["id"]))["stopped_at"]
        # the window learns of the new alarm first, then that the ringing one stopped
        assert [e["id"] for e in events_seen] == [new["id"], rang["id"]]
        assert events_seen[1]["stopped_at"] and events_seen[0]["snoozed_from"] == rang["id"]

        # a snooze is a real alarm: it rings again, and "stop" silences it
        assert await s.reminders.fire_due(datetime.fromisoformat(new["fire_at"])) == 1
        await s.store.execute("UPDATE reminders SET fired_at = ? WHERE id = ?", (datetime.now(UTC).isoformat(), new["id"]))
        events = await _turn(s, sid, "stop")
        assert llm.calls == [] and _reply(events) == "Alarm stopped."
        assert await s.reminders.ringing() == []
    finally:
        await s.stop()


async def test_snooze_for_the_minutes_said_or_the_setting(config, isolated_home):
    s = await _app(config, isolated_home, chat__alarm_snooze_minutes=15)
    try:
        sid = await s.store.create_session(channel="desktop")
        await _ringing(s)
        events = await _turn(s, sid, "snooze for 5 minutes")
        assert next(e for e in events if isinstance(e, ToolCallEvent)).arguments == {"minutes": 5}
        await s.reminders.cancel((await s.reminders.list())[0]["id"])
        await _ringing(s)
        events = await _turn(s, sid, "snooze it")
        assert next(e for e in events if isinstance(e, ToolCallEvent)).arguments == {"minutes": 15}
    finally:
        await s.stop()


async def test_with_nothing_ringing_the_words_go_to_the_model(config, isolated_home):
    """#175: "stop the alarm" may mean the house alarm. Nothing ringing: the model answers, as before."""
    llm = FakeProvider(replies=["Which alarm do you mean?"] * 4)
    s = await _app(config, isolated_home, llm)
    try:
        sid = await s.store.create_session(channel="desktop")
        for text in ("stop the alarm", "stop", "snooze", "turn it off"):
            assert await fastpath.plan(s, text) is None
        assert _reply(await _turn(s, sid, "stop the alarm")) == "Which alarm do you mean?"
        assert len(llm.calls) == 1

        # an alarm that went off longer ago than the window rings has stopped by itself
        old = await _ringing(s)
        await s.store.execute("UPDATE reminders SET fired_at = ? WHERE id = ?",
                              ((datetime.now(UTC) - timedelta(seconds=RING_S + 5)).isoformat(), old["id"]))
        assert await s.reminders.ringing() == []
        assert await fastpath.plan(s, "stop") is None
    finally:
        await s.stop()


async def test_naming_another_alarm_is_not_the_ringing_one(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        await _ringing(s, "gym")
        plan = await fastpath.plan(s, "turn off the gym alarm")
        assert plan is not None and plan.calls[0].name == "stop_alarm"
        assert await fastpath.plan(s, "stop the pasta alarm") is None
    finally:
        await s.stop()


async def test_a_never_rule_sends_it_to_the_model(config, isolated_home):
    llm = FakeProvider(replies=["I can't do that."])
    s = await _app(config, isolated_home, llm)
    try:
        s.config.tools.approvals.rules = {"reminders": "never"}
        sid = await s.store.create_session(channel="desktop")
        await _ringing(s)
        await _turn(s, sid, "stop")
        assert len(llm.calls) == 1 and len(await s.reminders.ringing()) == 1
    finally:
        await s.stop()


async def test_the_model_can_stop_and_snooze_too(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        ctx = s.agent.tool_context("x", "desktop")
        stop, snooze = s.registry.get("stop_alarm"), s.registry.get("snooze_alarm")
        assert (await stop.fn(ctx)) == {"error": "No Sentient alarm is ringing right now."}
        await _ringing(s)
        assert (await snooze.fn(ctx, minutes=500))["error"].startswith("A snooze can be")
        done = await snooze.fn(ctx, minutes=2)
        assert done["status"] == "success" and done["reminder"]["source"] == "chat"
        assert (await snooze.fn(ctx)) == {"error": "No Sentient alarm is ringing right now."}
    finally:
        await s.stop()


async def test_a_snooze_that_cannot_set_its_alarm_leaves_it_ringing(config, isolated_home, monkeypatch):
    s = await _app(config, isolated_home)
    try:
        rang = await _ringing(s)

        async def broken(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(s.reminders, "add", broken)
        with pytest.raises(RuntimeError):
            await s.reminders.snooze(9)
        assert [r["id"] for r in await s.reminders.ringing()] == [rang["id"]]
    finally:
        await s.stop()


async def test_stop_silences_every_ringing_alarm(config, isolated_home):
    s = await _app(config, isolated_home)
    try:
        await _ringing(s)
        await _ringing(s, "gym")
        sid = await s.store.create_session(channel="desktop")
        assert _reply(await _turn(s, sid, "stop")) == "Alarms stopped."
        assert await s.reminders.ringing() == []
    finally:
        await s.stop()


def test_rest_stop_and_snooze(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False)
    app = create_app(core)
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        made = c.post("/api/reminders", json={"kind": "alarm", "minutes": 30, "label": "gym"}).json()
        assert c.post(f"/api/reminders/{made['id']}/stop").status_code == 409  # not ringing yet
        assert c.post("/api/reminders/nope/snooze").status_code == 404
        c.portal.call(core.store.execute, "UPDATE reminders SET fire_at = ? WHERE id = ?",
                      ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), made["id"]))
        assert c.portal.call(core.reminders.fire_due) == 1
        snoozed = c.post(f"/api/reminders/{made['id']}/snooze", json={"minutes": 3}).json()
        assert snoozed["stopped"]["id"] == made["id"] and snoozed["stopped"]["stopped_at"]
        assert snoozed["reminder"]["snoozed_from"] == made["id"] and snoozed["reminder"]["source"] == "api"
        assert c.post(f"/api/reminders/{made['id']}/stop").status_code == 409  # already snoozed
        assert c.post(f"/api/reminders/{made['id']}/snooze", json={"minutes": 999}).status_code == 400
