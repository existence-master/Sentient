"""Side questions use a chat's context independently of its running reply and stored history."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import tool_call


@pytest.fixture
async def side_chat(config, isolated_home, fake_provider):
    core = SentientApp(config, llm=fake_provider, db_path=isolated_home / "side.db", enable_background=False)
    gateway = create_app(core)
    async with gateway.router.lifespan_context(gateway), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway),
        base_url="http://test",
        headers={"Authorization": f"Bearer {gateway.state.token}"},
    ) as client:
        yield core, client


@pytest.mark.parametrize("incognito", [False, True])
async def test_side_question_uses_that_chats_context_and_model(side_chat, fake_provider, incognito):
    core, client = side_chat
    sid = await core.store.create_session(channel="desktop", model="lm_studio/picked", incognito=incognito)
    other = await core.store.create_session(channel="desktop")
    await core.store.add_message(other, "user", "Separate conversation")
    await core.store.add_message(sid, "user", "Check when the shop opens")
    call = tool_call("web_search", query="shop opening time")
    await core.store.add_message(sid, "assistant", None, tool_calls=[call])
    await core.store.add_message(sid, "tool", "Opens at ten", name="web_search", tool_call_id=call["id"])
    before = await core.store.recent_messages(sid, 20)
    session = await core.store.get_session(sid)
    fake_provider.replies.append("The shop opens at ten.")

    response = await client.post(f"/api/sessions/{sid}/side-question", json={"text": "  When can I go?  "})

    assert response.status_code == 200
    assert response.json() == {"answer": "The shop opens at ten."}
    model_call, = fake_provider.calls
    assert model_call["role"] == "primary" and model_call["model"] == "lm_studio/picked"
    assert model_call["tools"] is None
    snapshot = json.loads(model_call["messages"][-1]["content"])
    assert snapshot["question"] == "When can I go?"
    user, tool_result = snapshot["conversation"]
    assert user == {"speaker": "user", "text": "Check when the shop opens"}
    assert tool_result["speaker"] == "web_search" and "Opens at ten" in tool_result["text"]
    assert await core.store.recent_messages(sid, 20) == before
    assert await core.store.get_session(sid) == session


async def test_side_question_does_not_steer_an_active_reply(side_chat, fake_provider, monkeypatch):
    core, client = side_chat
    sid = await core.store.create_session(channel="desktop")
    fake_provider.replies.extend(["Side answer", "Main answer"])
    started, release = asyncio.Event(), asyncio.Event()
    stream = fake_provider.stream

    async def hold_first_call(*args, **kwargs):
        if not started.is_set():
            started.set()
            await release.wait()
        async for chunk in stream(*args, **kwargs):
            yield chunk

    monkeypatch.setattr(fake_provider, "stream", hold_first_call)

    async def main_reply():
        return [event async for event in core.agent.run_turn(sid, "Main question")]

    turn = asyncio.create_task(main_reply())
    try:
        await asyncio.wait_for(started.wait(), 5)
        response = await asyncio.wait_for(
            client.post(f"/api/sessions/{sid}/side-question", json={"text": "Side question"}), 5,
        )
        assert response.json() == {"answer": "Side answer"}
        snapshot = json.loads(fake_provider.calls[0]["messages"][-1]["content"])
        assert snapshot["reply_in_progress"] is True
        assert snapshot["conversation"] == [{"speaker": "user", "text": "Main question"}]
        assert not turn.done()
    finally:
        release.set()
        events = await asyncio.wait_for(turn, 5)

    assert events[-1].content == "Main answer"
    assert [row["content"] for row in await core.store.recent_messages(sid, 10)] == ["Main question", "Main answer"]


async def test_side_question_validates_the_chat_and_text(side_chat, fake_provider):
    core, client = side_chat
    sid = await core.store.create_session(channel="desktop")
    path = f"/api/sessions/{sid}/side-question"
    assert (await client.post(path, json={"text": "Hello"}, headers={"Authorization": "Bearer invalid"})).status_code == 401
    assert (await client.post(path, json={"text": " "})).status_code == 422
    assert (await client.post("/api/sessions/missing/side-question", json={"text": "Hello"})).status_code == 404
    assert not fake_provider.calls


async def test_side_question_does_not_act_on_a_model_tool_request(side_chat, fake_provider):
    core, client = side_chat
    sid = await core.store.create_session(channel="desktop")
    fake_provider.replies.append([tool_call("current_datetime")])

    response = await client.post(f"/api/sessions/{sid}/side-question", json={"text": "What time is it?"})

    assert response.status_code == 502
    assert len(fake_provider.calls) == 1
    assert await core.store.recent_messages(sid, 10) == []


async def test_stop_everything_cancels_a_side_question(side_chat, fake_provider, monkeypatch):
    core, client = side_chat
    sid = await core.store.create_session(channel="desktop")
    started = asyncio.Event()

    async def hanging_stream(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()
        yield  # pragma: no cover

    monkeypatch.setattr(fake_provider, "stream", hanging_stream)
    task = asyncio.create_task(client.post(f"/api/sessions/{sid}/side-question", json={"text": "What are you doing?"}))
    try:
        await asyncio.wait_for(started.wait(), 5)
        stop = await client.post("/api/stop-all")
        assert stop.json()["cancelled"] == 1
        response = await asyncio.wait_for(task, 5)
        assert response.status_code == 409
        assert await core.store.recent_messages(sid, 10) == []
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
