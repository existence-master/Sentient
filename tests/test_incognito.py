"""Incognito chats (#154): nothing is learned from them, nothing finds them, and they are deleted when closed."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.events import Done, ToolResultEvent
from tests.conftest import FakeProvider, tool_call
from tests.memory.helpers import add_fact, drain


@pytest.fixture
async def app(config, isolated_home):
    llm = FakeProvider()
    a = SentientApp(config, llm=llm, db_path=isolated_home / "inc.db", enable_background=False)
    await a.start()
    a.fake = llm
    yield a
    await a.stop()


def _offered(call: dict) -> set[str]:
    return {t["function"]["name"] for t in call["tools"] or []}


async def _turn(app, sid: str, text: str) -> list:
    events = [ev async for ev in app.agent.run_turn(sid, text, channel="desktop")]
    await app.agent.drain(timeout=5)
    return events


async def test_an_incognito_turn_learns_nothing(app, monkeypatch):
    app.config.memory.extract_after_turn = True
    extracted: list[str] = []

    async def fake_extract(text, user_name, **kw):
        extracted.append(text)
        return []

    monkeypatch.setattr(app.memory, "extract_and_store", fake_extract)
    sid = await app.store.create_session(channel="desktop", incognito=True)
    async with app.bus.subscribe() as q:
        app.fake.replies.append("Got it, good luck with the interview.")
        events = await _turn(app, sid, "I have a job interview at Northwind next Tuesday morning")
        seen = [e["type"] for e in drain(q)]

    assert any(isinstance(e, Done) for e in events)
    assert extracted == []  # no fact extraction
    assert "chat.turn_completed" not in seen  # so skills, the user model and dreaming never hear of it
    call = app.fake.calls[-1]
    assert "memory_remember" not in _offered(call) and "skill_save" not in _offered(call)
    assert "memory_recall" in _offered(call)  # reading what Sentient knows stays on by default
    assert "This is an incognito chat" in call["messages"][0]["content"]

    # the same words in a normal chat are extracted and announced: the exclusion is the flag, not the text
    normal = await app.store.create_session(channel="desktop")
    async with app.bus.subscribe() as q:
        await _turn(app, normal, "I have a job interview at Northwind next Tuesday morning")
        assert "chat.turn_completed" in [e["type"] for e in drain(q)]
    assert extracted == ["I have a job interview at Northwind next Tuesday morning"]
    assert "memory_remember" in _offered(app.fake.calls[-1])


async def test_memory_and_skill_writes_refuse_in_an_incognito_chat(app):
    sid = await app.store.create_session(channel="desktop", incognito=True)
    ctx = app.agent.tool_context(sid, "subagent")  # work the chat starts carries only its id: still refused
    remember = app.registry.get("memory_remember")
    out = await remember.call(ctx, {"fact": "Maya has an interview at Northwind"})
    assert "incognito" in out["error"]
    rows = await app.store.fetchall("SELECT id FROM facts")
    assert rows == []  # nothing stored, not even held for review
    save = app.registry.get("skill_save")
    out = await save.call(ctx, {"name": "interview-prep", "description": "d", "body": "b"})
    assert "incognito" in out["error"] and app.skills.get("interview-prep") is None
    # a normal chat still saves
    normal = app.agent.tool_context(await app.store.create_session(), "desktop")
    assert (await remember.call(normal, {"fact": "Maya likes green tea"})).get("error") is None


async def test_a_model_call_to_memory_remember_is_refused_in_the_turn(app):
    """A model that calls the tool anyway (it wasn't offered) gets an error and nothing is stored."""
    sid = await app.store.create_session(channel="desktop", incognito=True)
    app.fake.replies.extend([[tool_call("memory_remember", fact="Maya has an interview")], "Okay."])
    events = await _turn(app, sid, "I have an interview tomorrow at nine")
    [res] = [e for e in events if isinstance(e, ToolResultEvent)]
    assert res.is_error and "error" in json.dumps(res.result)
    assert await app.store.fetchall("SELECT id FROM facts") == []


async def test_incognito_without_memory_reads_nothing(app):
    app.config.chat.incognito_uses_memory = False
    app.config.memory.min_similarity = 0.0
    await add_fact(app.memory, "Maya lives in Pune", source="manual")
    sid = await app.store.create_session(channel="desktop", incognito=True)
    app.fake.replies.append("I don't know.")
    events = await _turn(app, sid, "where do I live")
    call = app.fake.calls[-1]
    system = call["messages"][0]["content"]
    assert "Maya lives in Pune" not in system and "answer from this chat alone" in system
    assert not {"memory_recall", "history_time_search", "user_model_ask"} & _offered(call)
    [done] = [e for e in events if isinstance(e, Done)]
    assert done.memory_sources == []
    out = await app.registry.get("memory_recall").call(app.agent.tool_context(sid, "desktop"), {"query": "Pune"})
    assert out == []


async def test_incognito_chats_are_never_searched_summarized_or_reviewed(app):
    inc = await app.store.create_session(channel="desktop", incognito=True)
    normal = await app.store.create_session(channel="desktop")
    for sid in (inc, normal):
        for i in range(4):
            await app.store.add_message(sid, "user" if i % 2 == 0 else "assistant", f"the purple umbrella {i}")
            await app.store.add_message(sid, "tool", "{}", tool_call_id=f"c{i}", name="memory_recall")
    old = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    await app.store.execute("UPDATE messages SET created_at = ?", (old,))
    await app.store.execute("UPDATE sessions SET updated_at = ?", (old,))

    hits = await app.store.search_messages('"purple"', limit=50)
    assert hits and {h["session_id"] for h in hits} == {normal}
    chunks = await app.memory.episodic.pending_chunks()
    assert chunks and {sid for sid, _ in chunks} == {normal}
    rows = await app.memory.episodic.messages_between("2000-01-01", "2100-01-01")
    assert rows and {r["session_id"] for r in rows} == {normal}
    _evidence, sections, _outside = await app.user_model._gather("")
    assert sections["messages"].count("purple umbrella") == 2  # the normal chat's two user messages only
    app.config.evolution.min_tool_calls_for_review = 1
    due = await app.evolution.sessions_due(datetime.now(UTC), idle_minutes=0)
    assert normal in due and inc not in due
    assert (await app.evolution.review_session(inc, force=True))["decision"] == "skipped"


async def test_sweeper_deletes_closed_and_idle_incognito_chats(app):
    keep = await app.store.create_session(channel="desktop")
    fresh = await app.store.create_session(channel="desktop", incognito=True)
    idle = await app.store.create_session(channel="desktop", incognito=True)
    await app.store.add_message(idle, "user", "a secret")
    old = (datetime.now(UTC) - timedelta(minutes=app.config.chat.incognito_delete_after_minutes + 1)).isoformat()
    await app.store.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (old, idle))
    async with app.bus.subscribe() as q:
        assert await app.incognito.sweep() == [idle]
        assert [e["data"] for e in drain(q) if e["type"] == "session.deleted"] == [
            {"session_id": idle, "reason": "incognito"}
        ]
    assert await app.store.get_session(idle) is None
    assert await app.store.fetchall("SELECT id FROM messages WHERE session_id = ?", (idle,)) == []
    # shutting down (or the next start) deletes every incognito chat; normal chats stay
    assert await app.incognito.purge() == [fresh]
    assert await app.store.get_session(keep) is not None


async def test_leftover_incognito_chats_are_deleted_at_start(config, isolated_home):
    db = isolated_home / "left.db"
    first = SentientApp(config, llm=FakeProvider(), db_path=db, enable_background=False)
    await first.start()
    sid = await first.store.create_session(channel="desktop", incognito=True)
    await first.store.close()  # the engine went away without a clean stop
    again = SentientApp(config, llm=FakeProvider(), db_path=db, enable_background=False)
    await again.start()
    try:
        assert await again.store.get_session(sid) is None
    finally:
        await again.stop()


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider(replies=["hello", "hi again"])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "g.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def test_gateway_creates_lists_and_deletes_incognito_chats(client):
    with client.websocket_connect("/ws?token=test-token") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "chat.send", "text": "keep this between us", "incognito": True, "client_id": "c1"})
        sid = None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "session":
                sid = msg["session_id"]
            if msg["type"] == "done":
                break
    listed = {s["id"]: s for s in client.get("/api/sessions").json()}
    assert listed[sid]["incognito"] is True
    assert client.get("/api/sessions/search", params={"q": "between us"}).json() == []
    rest = client.post("/api/sessions", params={"incognito": "true"}).json()["session_id"]
    normal = client.post("/api/sessions").json()["session_id"]
    listed = {s["id"]: s for s in client.get("/api/sessions").json()}
    assert listed[rest]["incognito"] is True and listed[normal]["incognito"] is False
    assert client.delete(f"/api/sessions/{sid}").json() == {"ok": True}
    assert sid not in {s["id"] for s in client.get("/api/sessions").json()}
    assert client.get(f"/api/sessions/{sid}/messages").status_code == 404  # the window shows it no longer exists


async def test_without_memory_the_prompt_has_no_personal_details(app):
    app.config.chat.incognito_uses_memory = False
    app.config.assistant.location = "Koramangala, Bengaluru"
    sid = await app.store.create_session(channel="desktop", incognito=True)
    app.fake.replies.append("Hello.")
    await _turn(app, sid, "hi there")
    system = app.fake.calls[-1]["messages"][0]["content"]
    # the configured name and place stay out (the persona the user wrote is kept as it is)
    assert "Koramangala" not in system and "- User: Sarthak" not in system


async def test_a_reply_still_writing_is_not_swept_and_deleting_cancels_helpers(app, monkeypatch):
    sid = await app.store.create_session(channel="desktop", incognito=True)
    app.agent._busy[sid] = 1  # a turn between its last model call and its final write
    assert await app.incognito.purge() == []
    app.agent._turn_done(sid)
    assert not app.agent.is_busy(sid)
    cancelled: list[str] = []

    async def fake_cancel(session_id):
        cancelled.append(session_id)
        return 0

    monkeypatch.setattr(app.subagents, "cancel_for_session", fake_cancel)
    assert await app.incognito.purge() == [sid]
    assert cancelled == [sid] and await app.store.get_session(sid) is None


async def test_turn_busy_count_covers_the_final_write(app):
    sid = await app.store.create_session(channel="desktop", incognito=True)
    seen: list[bool] = []
    real_add = app.store.add_message

    async def spy(session_id, role, content, **kw):
        if role == "assistant":
            seen.append(app.agent.is_busy(session_id))
        return await real_add(session_id, role, content, **kw)

    app.store.add_message = spy
    app.fake.replies.append("Done.")
    await _turn(app, sid, "hello")
    assert seen == [True] and not app.agent.is_busy(sid)


async def test_closing_the_stream_at_done_still_ends_the_busy_count(app):
    sid = await app.store.create_session(channel="desktop", incognito=True)
    app.fake.replies.append("Hi.")
    gen = app.agent.run_turn(sid, "hello", channel="desktop")
    async for ev in gen:
        if isinstance(ev, Done):
            break
    await gen.aclose()  # the window went away right at the final event
    assert not app.agent.is_busy(sid)
    assert await app.incognito.purge() == [sid]


async def test_deleting_a_chat_stops_its_running_reply_first(app):
    import asyncio

    sid = await app.store.create_session(channel="desktop", incognito=True)
    started, release = asyncio.Event(), asyncio.Event()
    real_stream = app.fake.stream

    async def slow_stream(*a, **k):
        started.set()
        await release.wait()
        async for chunk in real_stream(*a, **k):
            yield chunk

    app.fake.stream = slow_stream

    async def consume():
        return [ev async for ev in app.agent.run_turn(sid, "tell me something", channel="desktop")]

    turn = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 5)
    await app.delete_chat(sid, reason="incognito")
    assert turn.done() and turn.cancelled()
    assert await app.store.get_session(sid) is None
    assert await app.store.fetchall("SELECT id FROM messages WHERE session_id = ?", (sid,)) == []
    assert not app.agent.is_busy(sid)


async def test_helpers_of_a_private_incognito_chat_do_not_get_the_name(app):
    app.config.chat.incognito_uses_memory = False
    assert "Sarthak" in app.subagents._system_prompt()
    assert "Sarthak" not in app.subagents._system_prompt(private=True)
