"""Catching up on scheduled runs missed while the computer was off or asleep (issue #135).

A missed run happens once now if it is less than ``tasks.catch_up_window_hours`` late (or the task says ``run``),
otherwise it is skipped; never a backlog. One "Caught up" notification says what ran and what was skipped. Nothing is
caught up while Sentient is stopped; it happens on the first tick after Resume.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sentient.tasks.schedule import iso, normalize_schedule
from tests.conftest import FakeProvider
from tests.tasks.conftest import RESULT

DAILY = {"type": "recurring", "frequency": "daily", "time": "09:00", "timezone": "Asia/Kolkata"}  # 03:30 UTC


def llm(runs: int = 3) -> FakeProvider:
    return FakeProvider(replies=["Today is Tuesday."] * runs, json_replies=[dict(RESULT) for _ in range(runs)])


async def scheduled(app, name: str, due: datetime, schedule: dict | None = None, status: str = "active") -> str:
    now = app.tasks.now_iso()
    return await app.tasks.repo.insert_task({
        "name": name, "description": name, "status": status, "schedule": schedule or DAILY,
        "plan": [{"tool": "time", "description": "Check today's date"}], "next_execution_at": iso(due),
        "created_at": now, "updated_at": now,
    })


async def caught_up(app) -> list[dict]:
    return [n for n in await app.notifications.list() if n["payload"].get("event") == "caught_up"]


async def tick(app) -> list[str]:
    run_ids = await app.tasks.tick()
    await app.tasks.drain()
    return run_ids


async def test_a_daily_task_missed_a_little_runs_once_on_startup(make_app, clock):
    app = await make_app(llm(), clock=clock)
    clock.dt = datetime(2026, 9, 15, 5, 30, tzinfo=UTC)  # two hours after 09:00 in Pune
    task_id = await scheduled(app, "Morning digest", datetime(2026, 9, 15, 3, 30, tzinfo=UTC))

    assert len(await tick(app)) == 1
    task = await app.tasks.get(task_id)
    assert [r["status"] for r in task["runs"]] == ["completed"]
    assert task["next_execution_at"] == "2026-09-16T03:30:00+00:00"
    [note] = await caught_up(app)
    assert note["title"] == "Caught up after Sentient was off: ran 1, skipped 0"
    assert note["message"] == "Ran once now: 'Morning digest'."
    assert note["task_id"] == task_id and note["payload"]["reason"] == "start"

    assert await tick(app) == [] and len(await caught_up(app)) == 1  # said once


async def test_a_task_missed_for_days_runs_once_not_once_per_day(make_app, clock):
    app = await make_app(llm(), clock=clock)
    task_id = await scheduled(app, "Morning digest", clock.dt - timedelta(days=4), {**DAILY, "catch_up": "run"})

    assert len(await tick(app)) == 1
    clock.dt += timedelta(minutes=1)
    assert await tick(app) == []
    task = await app.tasks.get(task_id)
    assert len(task["runs"]) == 1 and task["next_execution_at"] == "2026-09-15T03:30:00+00:00"


async def test_misses_older_than_the_window_are_skipped_with_a_notice(make_app, clock):
    app = await make_app(llm(), clock=clock)
    daily = await scheduled(app, "Morning digest", clock.dt - timedelta(days=2))
    once = await scheduled(
        app, "Call the bank", clock.dt - timedelta(hours=13), {"type": "once", "run_at": "2026-09-14T18:30"}, "pending"
    )

    assert await tick(app) == []
    task = await app.tasks.get(daily)
    assert task["status"] == "active" and task["runs"] == []
    assert task["next_execution_at"] == "2026-09-15T03:30:00+00:00"  # the next 09:00, not a replay
    one_off = await app.tasks.get(once)
    assert one_off["status"] == "error" and one_off["runs"] == [] and one_off["next_execution_at"] is None
    assert one_off["error"] == (
        "Skipped: it was due Sep 14, 18:30, while the computer was off or asleep. Choose Run now if you still want it."
    )
    [note] = await caught_up(app)
    assert note["title"] == "Caught up after Sentient was off: ran 0, skipped 2"
    assert note["message"].startswith("Skipped: ") and note["message"].endswith(". Open one and choose Run now if you still want it.")
    assert "'Call the bank'" in note["message"] and "'Morning digest'" in note["message"]
    assert {i["task_id"] for i in note["payload"]["skipped"]} == {daily, once}


async def test_a_jump_of_the_wall_clock_is_a_wake_from_sleep(make_app, clock):
    app = await make_app(llm(), clock=clock)
    assert await tick(app) == []  # Sentient is running; nothing was missed
    task_id = await scheduled(app, "Morning digest", datetime(2026, 9, 15, 3, 30, tzinfo=UTC))
    later = await scheduled(app, "Water the plants", clock.dt + timedelta(hours=1))

    clock.dt = datetime(2026, 9, 15, 9, 0, tzinfo=UTC)  # the laptop slept for 7 hours
    assert len(await tick(app)) == 2
    [note] = await caught_up(app)
    assert note["title"] == "Caught up after sleep: ran 2, skipped 0" and note["payload"]["reason"] == "sleep"
    assert {i["task_id"] for i in note["payload"]["ran"]} == {task_id, later}


async def test_a_run_a_moment_late_is_not_a_catch_up(make_app, clock):
    app = await make_app(llm(), clock=clock)
    await scheduled(app, "Morning digest", clock.dt - timedelta(seconds=40))
    interval = {"type": "recurring", "frequency": "interval", "interval_minutes": 60, "timezone": "UTC"}
    await scheduled(app, "Check the price", clock.dt - timedelta(days=2), interval)
    assert len(await tick(app)) == 2  # both run, and an every-hour check is caught up without a notice
    assert await caught_up(app) == []


async def test_an_every_few_minutes_check_that_says_skip_is_skipped(make_app, clock):
    app = await make_app(llm(), clock=clock)
    interval = {"type": "recurring", "frequency": "interval", "interval_minutes": 60, "timezone": "UTC", "catch_up": "skip"}
    task_id = await scheduled(app, "Check the price", clock.dt - timedelta(hours=2), normalize_schedule(interval, "UTC"))
    assert await tick(app) == []
    task = await app.tasks.get(task_id)
    assert task["runs"] == [] and task["next_execution_at"] == "2026-09-15T03:00:00+00:00"
    [note] = await caught_up(app)
    assert note["title"] == "Caught up after Sentient was off: ran 0, skipped 1"


async def test_nothing_is_caught_up_while_stopped_and_it_happens_on_resume(make_app, clock):
    app = await make_app(llm(), clock=clock)
    assert await tick(app) == []
    await app.stop_all()
    task_id = await scheduled(app, "Morning digest", clock.dt + timedelta(minutes=10))
    clock.dt += timedelta(hours=3)
    assert await tick(app) == []
    assert (await app.tasks.get(task_id))["runs"] == [] and await caught_up(app) == []

    await app.resume()
    assert len(await tick(app)) == 1
    [note] = await caught_up(app)
    assert note["title"] == "Caught up after resuming: ran 1, skipped 0"


async def test_a_task_can_say_skip_when_missed(make_app, clock):
    app = await make_app(llm(), clock=clock)
    task_id = await scheduled(app, "Morning digest", clock.dt - timedelta(hours=1))
    await app.tasks.update(task_id, {"schedule": {**DAILY, "catch_up": "skip"}, "status": "active"})
    # PATCHing the schedule recomputes the next time; put the missed one back as if the computer had slept
    await app.tasks.repo.update_task(task_id, {"next_execution_at": iso(clock.dt - timedelta(hours=1))})
    assert await tick(app) == []
    [note] = await caught_up(app)
    assert note["title"].endswith("ran 0, skipped 1")


def test_catch_up_policy_is_kept_only_when_it_means_something():
    assert normalize_schedule({**DAILY, "catch_up": "Run"}, "UTC")["catch_up"] == "run"
    assert "catch_up" not in normalize_schedule({**DAILY, "catch_up": "auto"}, "UTC")
    assert "catch_up" not in normalize_schedule({**DAILY, "catch_up": "all"}, "UTC")  # never a backlog
    trig = {"type": "triggered", "source": "gmail", "event": "new_email", "catch_up": "run"}
    assert "catch_up" not in normalize_schedule(trig, "UTC")
