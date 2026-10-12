"""Pause routines while the user is away (#163).

After ``tasks.away_pause_days`` without any sign of the user (or ``tasks.away_pause_after_results`` results since),
Sentient asks once, then pauses recurring and triggered tasks if ``tasks.away_ask_hours`` pass with no answer. Any
activity resets the timer. Tasks marked ``keep_running`` never pause. Resume turns them back on.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.tasks.away import ACTIVE_KEY
from sentient.tasks.schedule import iso
from tests.conftest import FakeProvider

T0 = datetime(2026, 9, 15, 2, 0, tzinfo=UTC)
DAILY = {"type": "recurring", "frequency": "daily", "time": "09:00", "timezone": "Asia/Kolkata"}
TRIGGERED = {"type": "triggered", "source": "gmail", "event": "new_email", "filter": {}}


async def add(app, name: str, schedule: dict, *, status: str = "active", keep_running: bool = False) -> str:
    now = app.tasks.now_iso()
    return await app.tasks.repo.insert_task({
        "name": name, "description": name, "status": status, "schedule": schedule,
        "plan": [{"tool": "time", "description": "Check today's date"}],
        "next_execution_at": iso(T0 + timedelta(days=1)) if schedule["type"] == "recurring" else None,
        "keep_running": keep_running, "created_at": now, "updated_at": now,
    })


async def events(app, event: str) -> list[dict]:
    return [n for n in await app.notifications.list() if n["payload"].get("event") == event]


async def routines(app) -> dict[str, str]:
    return {
        "brief": await add(app, "Daily Brief", DAILY),
        "mail": await add(app, "New invoice emails", TRIGGERED),
        "gym": await add(app, "Gym reminder", DAILY, keep_running=True),
        "once": await add(app, "Book the dentist", {"type": "once", "run_at": iso(T0 + timedelta(days=30))}, status="pending"),
    }


async def test_away_for_days_asks_then_pauses_then_resumes(make_app, clock):
    clock.dt = T0
    app = await make_app(clock=clock)
    await app.user_seen("app")
    ids = await routines(app)

    assert (await app.tasks.away.check(T0 + timedelta(days=6, hours=23)))["state"] == "active"
    asked = await app.tasks.away.check(T0 + timedelta(days=7))
    assert asked["state"] == "asked" and asked["reason"] == "days"
    [question] = await events(app, "away_check")
    assert question["title"] == "Keep your routines running?"
    assert "haven't used Sentient for 7 days" in question["message"] and "pause 2 routines" in question["message"]
    assert "Gym reminder" not in question["message"]  # kept running by the user

    assert (await app.tasks.away.check(T0 + timedelta(days=7, hours=23)))["state"] == "asked"  # still waiting
    paused = await app.tasks.away.check(T0 + timedelta(days=8, minutes=1))
    assert paused["state"] == "paused" and set(paused["task_ids"]) == {ids["brief"], ids["mail"]}
    tasks = {t["task_id"]: t for t in await app.tasks.list()}
    assert not tasks[ids["brief"]]["enabled"] and not tasks[ids["mail"]]["enabled"]
    assert tasks[ids["gym"]]["enabled"] and tasks[ids["once"]]["enabled"]
    [note] = await events(app, "away_paused")
    assert note["title"] == "Routines paused while you were away" and "paused 2 routines" in note["message"]
    assert (await app.notifications.get(question["id"]))["payload"]["status"] == "paused"
    # coming back doesn't switch them on behind the user's back; Resume does
    clock.dt = T0 + timedelta(days=9)
    await app.user_seen("app")
    assert (await app.tasks.away.status())["state"] == "paused"

    done = await app.tasks.away.resume()
    assert done["resumed"] == 2 and done["status"]["state"] == "active"
    brief = await app.tasks.get(ids["brief"])
    assert brief["enabled"] and brief["next_execution_at"] > iso(clock.dt)
    assert (await app.notifications.get(note["id"]))["payload"]["status"] == "resumed"
    assert (await app.tasks.away.check(clock.dt + timedelta(days=1)))["state"] == "active"


async def test_any_activity_cancels_the_question(make_app, clock):
    clock.dt = T0
    app = await make_app(clock=clock)
    await app.user_seen("app")
    ids = await routines(app)
    assert (await app.tasks.away.check(T0 + timedelta(days=7)))["state"] == "asked"
    [question] = await events(app, "away_check")

    clock.dt = T0 + timedelta(days=7, hours=3)
    await app.user_seen("telegram")  # a message in a paired chat
    assert (await app.tasks.away.status())["state"] == "active"
    assert (await app.notifications.get(question["id"]))["payload"]["status"] == "kept"
    assert (await app.tasks.away.check(T0 + timedelta(days=8, hours=6)))["state"] == "active"
    assert (await app.tasks.get(ids["brief"]))["enabled"]


async def test_a_chat_turn_counts_as_activity(make_app, clock):
    clock.dt = T0
    app = await make_app(FakeProvider(replies=["Hi Maya."]), clock=clock)
    assert await app.store.get_meta(ACTIVE_KEY) is None
    sid = await app.store.create_session()
    async for _ in app.agent.run_turn(sid, "hello", channel="desktop"):
        pass
    assert await app.store.get_meta(ACTIVE_KEY) == T0.isoformat()


async def test_results_piling_up_pause_without_asking(make_app, clock):
    app = await make_app(clock=clock)
    app.config.tasks.away_pause_days = 0
    app.config.tasks.away_pause_after_results = 3
    app.config.tasks.away_ask_hours = 0
    clock.dt = datetime.now(UTC)
    ids = await routines(app)
    await app.store.set_meta(ACTIVE_KEY, (clock.dt - timedelta(hours=2)).isoformat())
    for i in range(2):
        await app.notify("task", f"Run {i} finished", payload={"event": "run_completed", "task_id": ids["brief"]})
    await app.notify("info", "Not a result")
    assert (await app.tasks.away.check(clock.dt))["state"] == "active"
    await app.notify("brief", "Your Daily Brief", payload={"task_id": ids["brief"]})
    paused = await app.tasks.away.check(clock.dt)
    assert paused["state"] == "paused" and paused["reason"] == "results"
    assert not await events(app, "away_check")  # no question with away_ask_hours 0
    assert "paused 2 routines" in (await events(app, "away_paused"))[0]["message"]


async def test_off_when_both_limits_are_zero_and_while_stopped(make_app, clock):
    clock.dt = T0
    app = await make_app(clock=clock)
    await app.user_seen("app")
    await routines(app)
    app.config.tasks.away_pause_days = 0
    app.config.tasks.away_pause_after_results = 0
    assert (await app.tasks.away.check(T0 + timedelta(days=60)))["state"] == "active"
    app.config.tasks.away_pause_days = 7
    await app.stop_all("desktop")
    assert (await app.tasks.away.check(T0 + timedelta(days=60)))["state"] == "active"


async def test_first_check_without_a_record_starts_counting(make_app, clock):
    clock.dt = T0
    app = await make_app(clock=clock)
    await routines(app)
    assert (await app.tasks.away.check(T0 + timedelta(days=30)))["state"] == "active"
    assert await app.store.get_meta(ACTIVE_KEY) == (T0 + timedelta(days=30)).isoformat()


async def test_keep_running_is_a_task_field(make_app, clock):
    app = await make_app(clock=clock)
    task_id = await add(app, "Daily Brief", DAILY)
    assert (await app.tasks.get(task_id))["keep_running"] is False
    assert (await app.tasks.update(task_id, {"keep_running": True}))["keep_running"] is True
    assert await app.tasks.away.routines() == []


async def test_channel_message_has_the_answer_as_a_button(make_app, clock):
    clock.dt = T0
    app = await make_app(clock=clock)
    await app.user_seen("app")
    await routines(app)
    await app.tasks.away.check(T0 + timedelta(days=7))
    [question] = await events(app, "away_check")
    md, buttons = await app.channels.format_notification(question)
    assert md.startswith("**Keep your routines running?**") and buttons[0][0].data == "aw:k:0"
    await app.tasks.away.check(T0 + timedelta(days=8, minutes=1))
    [paused] = await events(app, "away_paused")
    _, buttons = await app.channels.format_notification(paused)
    assert buttons[0][0].label == "Resume routines" and buttons[0][0].data == "aw:r:0"
    settled = await app.notifications.get(question["id"])
    assert await app.channels.format_notification(settled) is None  # answered: not sent again


def test_routes(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "away-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "away.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers["Authorization"] = "Bearer away-token"
        seen = c.post("/api/activity").json()
        assert seen["ok"] and seen["last_active_at"]
        status = c.get("/api/tasks/away").json()
        assert status["state"] == "active" and status["away_pause_days"] == 7 and status["away_ask_hours"] == 24

        async def away() -> None:
            core.tasks.clock = lambda: T0
            await core.store.set_meta(ACTIVE_KEY, T0.isoformat())
            await add(core, "Daily Brief", DAILY)
            await core.tasks.away.check(T0 + timedelta(days=7))

        c.portal.call(away)
        assert c.get("/api/tasks/away").json()["state"] == "asked"
        kept = c.post("/api/tasks/away/keep").json()
        assert kept["state"] == "active"
        assert c.post("/api/tasks/away/resume").json() == {"resumed": 0, "status": c.get("/api/tasks/away").json()}
