"""Overlapping API requests are refused instead of mixing or queuing chat turns."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from sentient.gateway.routes import openai_api
from sentient.llm.events import Done
from sentient.llm.provider import ProviderFailed
from tests.test_openai_api import WIN, _ask, _sessions
from tests.test_openai_api import client as client
from tests.test_openai_api import keychain as keychain


def _gate_first_stream(client, monkeypatch, *, fail=False):
    entered, release = Event(), Event()
    original = client.llm.stream
    first = True

    async def gated(role, messages, tools=None, *, model=None):
        nonlocal first
        if first:
            first = False
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            if fail:
                raise ProviderFailed("The scripted first reply failed.")
        async for chunk in original(role, messages, tools, model=model):
            yield chunk

    monkeypatch.setattr(client.llm, "stream", gated)
    return entered, release


def _assert_busy(response):
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "chat_busy"
    assert response.headers["x-should-retry"] == "false"


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("outcome", ["success", "error", "stop"])
def test_overlap_is_rejected_and_chat_reopens_after_turn_ends(client, monkeypatch, stream, outcome):
    entered, release = _gate_first_stream(client, monkeypatch, fail=outcome == "error")
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(_ask, client, user="Notes app", messages=[{"role": "user", "content": "First request"}])
        try:
            assert entered.wait(timeout=10)
            second = pool.submit(_ask, client, user="Notes app", stream=stream,
                                 messages=[{"role": "user", "content": "Do not queue this request"}])
            _assert_busy(second.result(timeout=3))
        finally:
            if outcome == "stop":
                assert client.post("/api/stop-all", headers=WIN).status_code == 200
            release.set()
        response = first.result(timeout=20)
    assert response.status_code == {"success": 200, "error": 500, "stop": 409}[outcome]
    [session] = _sessions(client)
    rows = client.get(f"/api/sessions/{session['id']}/messages", headers=WIN).json()
    assert [row["content"] for row in rows if row["role"] == "user"] == ["First request"]

    # A fresh explicit request works after completion; a rejected request never runs later, even after Stop.
    assert _ask(client, user="Notes app", messages=[{"role": "user", "content": "Fresh request"}]).status_code == 200
    rows = client.get(f"/api/sessions/{session['id']}/messages", headers=WIN).json()
    assert [row["content"] for row in rows if row["role"] == "user"] == ["First request", "Fresh request"]


def test_different_named_chats_can_be_admitted_together(client, monkeypatch):
    entered, release = _gate_first_stream(client, monkeypatch)
    second_started = Event()
    start = openai_api._start

    def observe_start(core, session_id, text):
        result = start(core, session_id, text)
        if text == "Second chat":
            second_started.set()
        return result

    monkeypatch.setattr(openai_api, "_start", observe_start)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(_ask, client, user="Notes app", messages=[{"role": "user", "content": "First chat"}])
        try:
            assert entered.wait(timeout=10)
            second = pool.submit(_ask, client, user="Editor", messages=[{"role": "user", "content": "Second chat"}])
            assert second_started.wait(timeout=10)
        finally:
            release.set()
        assert first.result(timeout=20).status_code == 200
        assert second.result(timeout=20).status_code == 200
    assert {session["title"] for session in _sessions(client)} == {"Notes app", "Editor"}


def test_api_request_is_rejected_while_desktop_uses_its_chat(client, monkeypatch):
    assert _ask(client, user="Notes app").status_code == 200
    [session] = _sessions(client)
    entered, release = _gate_first_stream(client, monkeypatch)
    with ThreadPoolExecutor(2) as pool:
        desktop = pool.submit(client.post, "/api/chat", headers=WIN,
                              json={"session_id": session["id"], "text": "Desktop request"})
        try:
            assert entered.wait(timeout=10)
            api = pool.submit(_ask, client, user="Notes app")
            _assert_busy(api.result(timeout=3))
        finally:
            release.set()
        assert desktop.result(timeout=20).status_code == 200
    assert _ask(client, user="Notes app").status_code == 200


def test_chat_reopens_after_an_unexpected_turn_exception(client, monkeypatch):
    agent = client.app.state.sentient.agent
    run_turn = agent.run_turn
    first = True

    async def failing_once(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise RuntimeError("The scripted turn broke before producing events.")
        async for event in run_turn(*args, **kwargs):
            yield event

    monkeypatch.setattr(agent, "run_turn", failing_once)
    failed = _ask(client, user="Notes app")
    assert failed.status_code == 500 and failed.json()["error"]["code"] == "turn_failed"
    assert failed.headers["x-should-retry"] == "false"
    assert _ask(client, user="Notes app").status_code == 200


async def test_admission_is_immediate_per_app_and_prestart_cancellation_releases_it(monkeypatch):
    class Agent:
        def create_turn_task(self, coro):
            return asyncio.create_task(coro)

        def is_busy(self, session_id):
            return False

        async def run_turn(self, session_id, text, *, channel):
            yield Done(content=text)

    first_core = SimpleNamespace(agent=Agent())
    second_core = SimpleNamespace(agent=Agent())
    tasks = []
    create_task = asyncio.create_task

    def capture_task(coro):
        task = create_task(coro)
        tasks.append(task)
        return task

    try:
        # No await: the first task has not entered Agent.run_turn when the second request arrives.
        with monkeypatch.context() as patch:
            patch.setattr(openai_api.asyncio, "create_task", capture_task)
            accepted = openai_api._start(first_core, "same-id", "First")
            assert accepted is not None
            assert openai_api._start(first_core, "same-id", "Overlapping") is None
            assert openai_api._start(second_core, "same-id", "Independent") is not None
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    out, _ = accepted
    assert isinstance(await asyncio.wait_for(out.get(), timeout=1), asyncio.CancelledError)
    assert await asyncio.wait_for(out.get(), timeout=1) is None

    reopened = openai_api._start(first_core, "same-id", "After cancellation")
    assert reopened is not None
    out, _ = reopened
    assert isinstance(await asyncio.wait_for(out.get(), timeout=1), Done)
    assert await asyncio.wait_for(out.get(), timeout=1) is None
