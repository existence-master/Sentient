"""Memory sources (issue #136): a reply records which memories it had in mind."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.events import Done
from sentient.memory.sources import MemorySources
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call
from tests.memory.helpers import add_fact


async def _turn(app, sid: str, text: str) -> list:
    return [ev async for ev in app.agent.run_turn(sid, text, channel="cli")]


def _done(events: list) -> Done:
    [done] = [e for e in events if isinstance(e, Done)]
    return done


async def _assistant_rows(app, sid: str) -> list[dict]:
    return [r for r in await app.store.recent_messages(sid, 50) if r["role"] == "assistant"]


async def test_prompt_facts_and_insights_are_recorded_on_the_reply(app):
    app.config.memory.min_similarity = 0.0
    fid = await add_fact(app.memory, "Maya lives in Pune", source="manual")
    ins = await app.user_model.add_insight("Maya prefers morning meetings", "preferences")
    app.fake.replies.append("Morning works, in Pune.")
    sid = await app.store.create_session(channel="cli")

    done = _done(await _turn(app, sid, "where do I live and when should we meet"))

    system = app.fake.calls[-1]["messages"][0]["content"]
    assert "Maya lives in Pune" in system and "morning meetings" in system
    expected = [
        {"kind": "fact", "id": fid, "text": "Maya lives in Pune", "source": "manual", "via": "prompt"},
        {"kind": "insight", "id": ins["id"], "text": "Maya prefers morning meetings", "source": "user", "via": "prompt"},
    ]
    assert done.memory_sources == expected
    [row] = await _assistant_rows(app, sid)
    assert row["id"] == done.message_id and row["memory_sources"] == expected


async def test_memory_recall_results_are_recorded_with_their_source(app):
    app.config.memory.facts_top_k = 0  # nothing pushed into the prompt: only the tool finds it
    app.config.user_model.enabled = False
    fid = await add_fact(app.memory, "Maya's sister Aditi lives in Mumbai", source="file:family.md")
    app.fake.replies.extend([[tool_call("memory_recall", query="sister Aditi Mumbai")], "Aditi lives in Mumbai."])
    sid = await app.store.create_session(channel="cli")

    done = _done(await _turn(app, sid, "where does my sister live"))

    assert done.memory_sources == [
        {"kind": "fact", "id": fid, "text": "Maya's sister Aditi lives in Mumbai", "source": "file:family.md",
         "via": "tool"},
    ]
    rows = await _assistant_rows(app, sid)
    assert rows[0]["tool_calls"] and rows[0]["memory_sources"] == []  # the tool-call step carries none
    assert rows[-1]["memory_sources"] == done.memory_sources


async def test_facts_cut_from_a_long_result_are_not_attributed(app):
    """A long tool result is cut before the model reads it; facts from the cut part are not listed."""
    app.config.memory.facts_top_k = 0
    app.config.user_model.enabled = False
    await add_fact(app.memory, "Maya's sister Aditi lives in Mumbai", source="manual")
    for i in range(30):
        await add_fact(app.memory, f"Maya noted detail number {i} about the Mumbai trip plans", source="manual")
    app.config.chat.tool_result_max_chars = 400
    app.fake.replies.extend([[tool_call("memory_recall", query="Mumbai", limit=20)], "Noted."])
    sid = await app.store.create_session(channel="cli")

    done = _done(await _turn(app, sid, "what do you know about Mumbai"))

    ids = [s["id"] for s in done.memory_sources]
    assert 0 < len(ids) < 20  # only the rows inside the 400 characters the model read
    tool_msg = next(m for m in app.fake.calls[-1]["messages"] if m["role"] == "tool")
    for s in done.memory_sources:
        assert s["text"] in tool_msg["content"]


async def test_no_memories_means_no_sources(app):
    app.config.user_model.enabled = False
    app.fake.replies.append("Hello!")
    sid = await app.store.create_session(channel="cli")

    done = _done(await _turn(app, sid, "hi there"))

    assert done.memory_sources == []
    [row] = await _assistant_rows(app, sid)
    assert row["memory_sources"] == []
    raw = await app.store.fetchone("SELECT memory_sources FROM messages WHERE id = ?", (row["id"],))
    assert raw["memory_sources"] is None


async def test_collector_dedups_and_ignores_other_tools():
    s = MemorySources()
    s.add_facts([{"id": 1, "content": "Maya likes tea", "source": "conversation"}])
    s.add_tool_result("memory_recall", [{"id": 1, "fact": "Maya likes tea"}, {"id": 2, "fact": "Maya runs"}, "junk"])
    s.add_tool_result("memory_search_history", [{"id": 3, "fact": "not a fact list"}])
    s.add_tool_result("memory_recall", {"error": "memory unavailable"})
    s.add_insights([{"id": "abc", "statement": "  Maya is an early riser ", "source": "inferred"}, {"id": "x"}])
    out = await s.resolve(None)
    assert [(x["kind"], x["id"], x["via"]) for x in out] == [
        ("fact", 1, "prompt"), ("fact", 2, "tool"), ("insight", "abc", "prompt")
    ]
    assert out[2]["text"] == "Maya is an early riser"


async def test_context_with_sources_matches_the_block(app):
    um = app.user_model
    assert await um.context_with_sources("anything") == ("", [])
    mine = await um.add_insight("Maya is vegetarian", "preferences")
    block, used = await um.context_with_sources("dinner ideas")
    assert "vegetarian" in block and [i["id"] for i in used] == [mine["id"]]
    assert block == await um.context_for("dinner ideas")


def test_messages_api_and_stream_carry_sources(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider(replies=["Noted, mornings it is."])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "src.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        ins = c.post("/api/user-model/insights", json={"statement": "Maya prefers morning meetings",
                                                        "dimension": "preferences"}).json()
        lines = [json.loads(x) for x in c.post("/api/chat", json={"text": "book a call with Aditi"}).text.splitlines()]
        sid = lines[0]["session_id"]
        [done] = [x for x in lines if x.get("type") == "done"]
        source = {"kind": "insight", "id": ins["id"], "text": "Maya prefers morning meetings", "source": "user",
                  "via": "prompt"}
        assert done["memory_sources"] == [source]
        rows = c.get(f"/api/sessions/{sid}/messages").json()
        assert [r["memory_sources"] for r in rows] == [[], [source]]


def test_a_stopped_reply_carries_its_sources_on_the_live_done(config, isolated_home, monkeypatch):
    """Stop while a tool runs: the live ``done`` has the kept reply's id and memory sources, no reload needed."""
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    started = threading.Event()

    @tool("slow_lookup", risk=Risk.read)
    async def slow_lookup(ctx: ToolContext) -> dict:
        """Slow lookup."""
        started.set()
        await asyncio.Event().wait()
        return {"ok": True}

    class Slow(ToolPlugin):
        id = "slow"
        display_name = "Slow"
        tools = [slow_lookup]

    llm = FakeProvider(replies=[[tool_call("slow_lookup")]])
    core = SentientApp(config, llm=llm, db_path=isolated_home / "stop.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        core.registry.register(Slow())
        c.headers.update({"Authorization": "Bearer test-token"})
        ins = c.post("/api/user-model/insights", json={"statement": "Maya prefers morning meetings",
                                                        "dimension": "preferences"}).json()
        with c.websocket_connect("/ws?token=test-token") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "chat.send", "text": "book a call with Aditi"})
            sid = _until(ws, "session")["session_id"]
            assert started.wait(5)
            ws.send_json({"type": "chat.cancel", "session_id": sid})
            done = _until(ws, "done")
        source = {"kind": "insight", "id": ins["id"], "text": "Maya prefers morning meetings", "source": "user",
                  "via": "prompt"}
        assert done["cancelled"] is True and done["memory_sources"] == [source]
        kept = c.get(f"/api/sessions/{sid}/messages").json()[-1]
        assert kept["id"] == done["message_id"] and kept["memory_sources"] == [source]
        assert kept["content"].endswith("_(stopped)_")



async def test_a_second_stop_while_saving_still_reports_the_kept_reply(app, monkeypatch):
    """Stop pressed twice: the kept reply is still saved and handed to ``on_stopped``."""
    ins = await app.user_model.add_insight("Maya prefers morning meetings", "preferences")
    started, saving, gate = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @tool("slow_lookup", risk=Risk.read)
    async def slow_lookup(ctx: ToolContext) -> dict:
        """Slow lookup."""
        started.set()
        await asyncio.Event().wait()
        return {"ok": True}

    class Slow(ToolPlugin):
        id = "slow"
        display_name = "Slow"
        tools = [slow_lookup]

    app.registry.register(Slow())
    original = app.agent._persist_stopped

    async def slow_persist(*args):
        saving.set()
        await gate.wait()
        return await original(*args)

    monkeypatch.setattr(app.agent, "_persist_stopped", slow_persist)
    app.fake.replies.append([tool_call("slow_lookup")])
    sid = await app.store.create_session(channel="cli")
    stopped: dict = {}

    async def consume() -> None:
        async for _ in app.agent.run_turn(sid, "book a call with Aditi", channel="cli", on_stopped=stopped.update):
            pass

    turn = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 5)
    turn.cancel()
    await asyncio.wait_for(saving.wait(), 5)
    turn.cancel()  # pressed again while the kept reply is being saved
    await asyncio.sleep(0)
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert [s["id"] for s in stopped["memory_sources"]] == [ins["id"]]
    kept = (await _assistant_rows(app, sid))[-1]
    assert kept["id"] == stopped["message_id"] and kept["content"].endswith("_(stopped)_")

def _until(ws, kind: str) -> dict:
    while True:
        msg = ws.receive_json()
        if msg.get("type") == kind:
            return msg
