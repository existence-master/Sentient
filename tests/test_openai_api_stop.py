"""Stop owns API turns before initialization and rejects requests queued before it."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from sentient.app import SentientApp
from sentient.gateway.routes import openai_api
from sentient.llm.events import Done
from tests.conftest import FakeProvider
from tests.test_openai_api import WIN, _ask, _sessions
from tests.test_openai_api import client as client
from tests.test_openai_api import keychain as keychain


@pytest.fixture
async def api_app(config, isolated_home):
    llm = FakeProvider(replies=["Fresh reply."])
    core = await SentientApp(
        config, llm=llm, db_path=isolated_home / "api-stop.db", enable_background=False,
    ).start()
    try:
        yield core, llm
    finally:
        tasks = list(getattr(core, "_openai_api_turns", {}).values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await core.stop()


async def _drain(out):
    events = []
    while (event := await asyncio.wait_for(out.get(), timeout=5)) is not None:
        events.append(event)
    return events


def _assert_idle(core, sid):
    assert sid not in core._openai_api_turns
    assert not core.agent.is_busy(sid)
    assert not core.agent.is_running(sid)
    assert sid not in core.agent._session_tasks
    assert not core.agent._turn_tasks


async def _assert_retry_works(core, sid):
    _assert_idle(core, sid)
    accepted = openai_api._start(core, sid, "Fresh explicit request")
    assert accepted is not None
    events = await _drain(accepted[0])
    assert any(isinstance(event, Done) and event.content == "Fresh reply." for event in events)
    _assert_idle(core, sid)


class ObservedMarkLock(asyncio.Lock):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()

    async def acquire(self):
        self.entered.set()
        return await super().acquire()


async def test_stop_cancels_an_api_turn_before_its_first_step(api_app, monkeypatch):
    core, llm = api_app
    sid = await core.store.create_session(channel="api")
    await core.stop_all()
    entered = []
    run_turn = core.agent.run_turn

    async def observe(*args, **kwargs):
        entered.append(True)
        async for event in run_turn(*args, **kwargs):
            yield event

    monkeypatch.setattr(core.agent, "run_turn", observe)
    accepted = openai_api._start(core, sid, "Accepted before the second Stop")
    assert accepted is not None
    stopped = await core.stop_all()
    assert stopped["cancelled"] >= 1
    events = await _drain(accepted[0])
    assert len(events) == 1 and isinstance(events[0], asyncio.CancelledError)
    assert entered == [] and llm.calls == []
    assert await core.store.recent_messages(sid, 10) == []
    await _assert_retry_works(core, sid)


async def test_stop_cancels_an_api_turn_without_releasing_its_mark_lock(api_app):
    core, llm = api_app
    sid = await core.store.create_session(channel="api")
    lock = ObservedMarkLock()
    await lock.acquire()
    lock.entered.clear()
    core.agent._mark_locks[sid] = lock
    try:
        accepted = openai_api._start(core, sid, "Read this note from another app")
        assert accepted is not None
        await asyncio.wait_for(lock.entered.wait(), timeout=5)
        stopped = await asyncio.wait_for(core.stop_all(), timeout=5)
        assert stopped["cancelled"] >= 1 and lock.locked()
        events = await _drain(accepted[0])
        assert len(events) == 1 and isinstance(events[0], asyncio.CancelledError)
        assert llm.calls == []
        assert await core.store.recent_messages(sid, 10) == []
        _assert_idle(core, sid)
    finally:
        lock.release()
    await _assert_retry_works(core, sid)


async def test_initialization_failure_releases_api_turn_ownership(api_app, monkeypatch):
    core, llm = api_app
    sid = await core.store.create_session(channel="api")
    mark_chat = core.agent._mark_chat
    first = True

    async def fail_once(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise RuntimeError("The scripted initialization failed.")
        return await mark_chat(*args, **kwargs)

    monkeypatch.setattr(core.agent, "_mark_chat", fail_once)
    accepted = openai_api._start(core, sid, "First request")
    assert accepted is not None
    events = await _drain(accepted[0])
    assert len(events) == 1 and isinstance(events[0], RuntimeError)
    assert llm.calls == []
    await _assert_retry_works(core, sid)


async def test_closing_desktop_turn_at_done_releases_ownership(api_app):
    core, _ = api_app
    sid = await core.store.create_session(channel="desktop")
    turn = core.agent.run_turn(sid, "Hello", channel="desktop")
    try:
        async for event in turn:
            if isinstance(event, Done):
                break
    finally:
        await turn.aclose()
    assert not core.agent.is_busy(sid)
    assert sid not in core.agent._session_tasks
    assert not core.agent._turn_tasks


async def test_stop_cancels_desktop_initialization_before_quick_planning_finishes(api_app, monkeypatch):
    core, llm = api_app
    sid = await core.store.create_session(channel="desktop")
    entered, release = asyncio.Event(), asyncio.Event()
    quick_plan = core.agent._quick_plan

    async def blocked_plan(*args, **kwargs):
        entered.set()
        await release.wait()
        return await quick_plan(*args, **kwargs)

    async def consume():
        return [event async for event in core.agent.run_turn(sid, "set a timer for 5 minutes")]

    monkeypatch.setattr(core.agent, "_quick_plan", blocked_plan)
    turn = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        stopped = await asyncio.wait_for(core.stop_all(), timeout=5)
        assert stopped["cancelled"] >= 1 and turn.cancelled()
        assert not core.agent.is_busy(sid)
        assert sid not in core.agent._session_tasks
        assert not core.agent._turn_tasks
        assert llm.calls == []
    finally:
        turn.cancel()
        release.set()
        await asyncio.gather(turn, return_exceptions=True)
    monkeypatch.setattr(core.agent, "_quick_plan", quick_plan)
    events = await consume()
    assert isinstance(events[-1], Done) and not events[-1].error
    assert llm.calls == []
    assert not core.agent.is_busy(sid) and not core.agent._turn_tasks


def test_stop_rejects_api_input_waiting_for_a_named_session(client, monkeypatch):
    entered, release = Event(), Event()
    session = openai_api._session

    async def blocked_session(core, name):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return await session(core, name)

    monkeypatch.setattr(openai_api, "_session", blocked_session)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(_ask, client, user="Notes app")
        try:
            assert entered.wait(timeout=10)
            assert client.post("/api/stop-all", headers=WIN).status_code == 200
        finally:
            release.set()
        result = pending.result(timeout=20)
    assert result.status_code == 409 and result.json()["error"]["code"] == "stopped"
    assert client.llm.calls == []
    [chat] = _sessions(client)
    assert client.get(f"/api/sessions/{chat['id']}/messages", headers=WIN).json() == []
    assert _ask(client, user="Notes app").status_code == 200
