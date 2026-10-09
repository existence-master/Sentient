"""Memory sources (issue #136): a reply records which memories it had in mind."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.events import Done
from sentient.memory.sources import MemorySources
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
    fid = await add_fact(app.memory, "Sarthak lives in Pune", source="manual")
    ins = await app.user_model.add_insight("Sarthak prefers morning meetings", "preferences")
    app.fake.replies.append("Morning works, in Pune.")
    sid = await app.store.create_session(channel="cli")

    done = _done(await _turn(app, sid, "where do I live and when should we meet"))

    system = app.fake.calls[-1]["messages"][0]["content"]
    assert "Sarthak lives in Pune" in system and "morning meetings" in system
    expected = [
        {"kind": "fact", "id": fid, "text": "Sarthak lives in Pune", "source": "manual", "via": "prompt"},
        {"kind": "insight", "id": ins["id"], "text": "Sarthak prefers morning meetings", "source": "user", "via": "prompt"},
    ]
    assert done.memory_sources == expected
    [row] = await _assistant_rows(app, sid)
    assert row["id"] == done.message_id and row["memory_sources"] == expected


async def test_memory_recall_results_are_recorded_with_their_source(app):
    app.config.memory.facts_top_k = 0  # nothing pushed into the prompt: only the tool finds it
    app.config.user_model.enabled = False
    fid = await add_fact(app.memory, "Sarthak's sister Aditi lives in Mumbai", source="file:family.md")
    app.fake.replies.extend([[tool_call("memory_recall", query="sister Aditi Mumbai")], "Aditi lives in Mumbai."])
    sid = await app.store.create_session(channel="cli")

    done = _done(await _turn(app, sid, "where does my sister live"))

    assert done.memory_sources == [
        {"kind": "fact", "id": fid, "text": "Sarthak's sister Aditi lives in Mumbai", "source": "file:family.md",
         "via": "tool"},
    ]
    rows = await _assistant_rows(app, sid)
    assert rows[0]["tool_calls"] and rows[0]["memory_sources"] == []  # the tool-call step carries none
    assert rows[-1]["memory_sources"] == done.memory_sources


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
    mine = await um.add_insight("Sarthak is vegetarian", "preferences")
    block, used = await um.context_with_sources("dinner ideas")
    assert "vegetarian" in block and [i["id"] for i in used] == [mine["id"]]
    assert block == await um.context_for("dinner ideas")


def test_messages_api_and_stream_carry_sources(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider(replies=["Noted, mornings it is."])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "src.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        ins = c.post("/api/user-model/insights", json={"statement": "Sarthak prefers morning meetings",
                                                        "dimension": "preferences"}).json()
        lines = [json.loads(x) for x in c.post("/api/chat", json={"text": "book a call with Aditi"}).text.splitlines()]
        sid = lines[0]["session_id"]
        [done] = [x for x in lines if x.get("type") == "done"]
        source = {"kind": "insight", "id": ins["id"], "text": "Sarthak prefers morning meetings", "source": "user",
                  "via": "prompt"}
        assert done["memory_sources"] == [source]
        rows = c.get(f"/api/sessions/{sid}/messages").json()
        assert [r["memory_sources"] for r in rows] == [[], [source]]
