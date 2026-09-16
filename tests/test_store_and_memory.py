import pytest

from sentient.memory.facts import FactMemory, parse_duration
from sentient.store.db import Store


@pytest.fixture
async def store(isolated_home):
    s = await Store(isolated_home / "t.db").open()
    yield s
    await s.close()


async def test_schema_and_messages(store):
    sid = await store.create_session(channel="cli")
    await store.add_message(sid, "user", "hello there")
    await store.add_message(sid, "assistant", "hi", tool_calls=[{"id": "1"}])
    msgs = await store.recent_messages(sid, 10)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["tool_calls"] == [{"id": "1"}]
    hits = await store.search_messages("hello")
    assert hits and hits[0]["content"] == "hello there"


async def test_vec_loaded(store):
    assert store.vec_available, "sqlite-vec must load on this platform"


def test_parse_duration():
    assert parse_duration("3 days").days == 3
    assert parse_duration("2 weeks").days == 14
    assert parse_duration(None) is None


async def test_remember_recall_dedup_update(store, fake_provider, config):
    mem = FactMemory(store, fake_provider, config)
    r1 = await mem.remember("Sarthak's favorite snack is almonds", use_llm=False)
    assert r1["action"] == "ADD"
    r2 = await mem.remember("Sarthak's favorite snack is almonds", use_llm=False)
    assert r2["action"] == "SKIP" and r2["id"] == r1["id"]
    hits = await mem.recall("favorite snack", top_k=3, min_similarity=0.0)
    assert hits and hits[0]["id"] == r1["id"]

    fake_provider.json_replies.append(
        {"action": "UPDATE", "fact_id": r1["id"], "content": "Sarthak's favorite snack is cashews",
         "memory_type": "long-term", "duration": None, "topics": ["preferences"]}
    )
    r3 = await mem.remember("Sarthak's favorite snack is cashews now", use_llm=True)
    assert r3["action"] == "UPDATE" and r3["id"] == r1["id"]
    facts = await mem.list_facts()
    assert len(facts) == 1 and facts[0]["content"] == "Sarthak's favorite snack is cashews"

    assert await mem.forget(r1["id"])
    assert await mem.count() == 0


async def test_short_term_expiry(store, fake_provider, config):
    mem = FactMemory(store, fake_provider, config)
    fake_provider.json_replies.append(
        {"action": "ADD", "fact_id": None, "content": "Sarthak flies to Delhi on Friday",
         "memory_type": "short-term", "duration": "0 hours", "topics": ["travel"]}
    )
    r = await mem.remember("I fly to Delhi on Friday")
    assert r["action"] == "ADD"
    await store.execute("UPDATE facts SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (r["id"],))
    assert await mem.recall("Delhi flight", min_similarity=0.0) == []
    assert await mem.purge_expired() == 1


async def test_extract_and_store(store, fake_provider, config):
    mem = FactMemory(store, fake_provider, config)
    fake_provider.json_replies.append({"facts": ["Sarthak lives in Pune", "Sarthak's manager is Jane"]})
    fake_provider.json_replies.append({"action": "ADD", "content": "Sarthak lives in Pune", "memory_type": "long-term", "topics": ["identity"], "fact_id": None, "duration": None})
    fake_provider.json_replies.append({"action": "ADD", "content": "Sarthak's manager is Jane", "memory_type": "long-term", "topics": ["work"], "fact_id": None, "duration": None})
    results = await mem.extract_and_store("User: I live in Pune and my manager is Jane", "Sarthak")
    assert [r["action"] for r in results] == ["ADD", "ADD"]
    assert await mem.count() == 2
