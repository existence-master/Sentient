"""Fixtures for the tasks package: a started SentientApp, a frozen clock, a clock to skip, a gated fake LLM."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from sentient.app import SentientApp
from tests.conftest import FakeProvider

REFINE_ONCE = {
    "name": "Write haiku",
    "description": "Write a haiku and save it to haiku.txt",
    "priority": 1,
    "schedule": {"type": "once", "run_at": None},
}
REFINE_DAILY = {
    "name": "Morning digest",
    "description": "Every day at 9am tell me the date",
    "priority": 1,
    "schedule": {"type": "recurring", "frequency": "daily", "time": "09:00"},
}
PLAN_FILES = {
    "name": "Haiku file",
    "description": "Write a haiku and save it",
    "plan": [{"tool": "files", "description": "Save the haiku to haiku.txt"}],
    "clarifying_questions": [],
}
PLAN_TIME = {
    "name": "Date check",
    "description": "Tell the user the date",
    "plan": [{"tool": "time", "description": "Get the current date"}],
    "clarifying_questions": [],
}
RESULT = {
    "summary": "Saved a haiku.",
    "links_created": [],
    "links_found": [],
    "files_created": [],
    "tools_used": ["files"],
}


class Clock:
    def __init__(self, dt: datetime):
        self.dt = dt

    def __call__(self) -> datetime:
        return self.dt


class GatedProvider(FakeProvider):
    """FakeProvider whose Nth ``stream`` call blocks until ``gate`` is set."""

    def __init__(self, *args, block_on_call: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_on_call = block_on_call
        self.stream_calls = 0
        self.gate = asyncio.Event()
        self.blocked = asyncio.Event()

    async def stream(self, role, messages, tools=None, *, model=None):
        self.stream_calls += 1
        if self.block_on_call is not None and self.stream_calls == self.block_on_call:
            self.blocked.set()
            await self.gate.wait()
        async for chunk in super().stream(role, messages, tools, model=model):
            yield chunk


class SkipClock:
    """``time.monotonic``, and with it the event loop's clock, made to jump forward on demand: ``asyncio`` timeouts
    and sleeps, run time limits and the stuck check see the time pass at once. Real time still flows underneath,
    so a busy machine only adds to it."""

    def __init__(self, real: Callable[[], float]):
        self._real = real
        self.skipped = 0.0

    def monotonic(self) -> float:
        return self._real() + self.skipped

    async def sleep(self, seconds: float) -> None:
        """Let ``seconds`` pass without waiting for them, then give the loop a turn."""
        self.skipped += seconds
        await asyncio.sleep(0)


@pytest.fixture
def skip_clock(monkeypatch) -> SkipClock:
    """Timing tests skip time instead of sleeping, so they don't depend on how busy the machine is."""
    clock = SkipClock(time.monotonic)
    monkeypatch.setattr(time, "monotonic", clock.monotonic)
    return clock


@pytest.fixture
def clock():
    return Clock(datetime(2026, 9, 15, 2, 0, tzinfo=UTC))


@pytest.fixture
async def make_app(config, isolated_home):
    config.assistant.timezone = "Asia/Kolkata"
    config.assistant.location = "Pune, India"
    apps: list[SentientApp] = []

    async def factory(llm=None, *, db_name: str = "tasks.db", clock=None) -> SentientApp:
        app = SentientApp(config, llm=llm or FakeProvider(), db_path=isolated_home / db_name, enable_background=False)
        if clock is not None:
            app.tasks.clock = clock
        await app.start()
        apps.append(app)
        return app

    yield factory
    for app in apps:
        if app._started:
            await app.stop()


def stream_calls(llm, role: str = "executor") -> list[dict]:
    return [c for c in llm.calls if c.get("role") == role and not c.get("json")]


def json_calls(llm, role: str) -> list[dict]:
    return [c for c in llm.calls if c.get("role") == role and c.get("json")]
