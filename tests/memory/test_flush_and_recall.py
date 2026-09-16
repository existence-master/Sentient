from __future__ import annotations

from sentient.memory.facts import FactMemory, keyword_tokens, word_overlap
from tests.memory.helpers import add_fact, drain

TRANSCRIPT = """user: I just moved to Mumbai last week for the new office.
assistant: Congratulations on the move! How is it going?
tool: {"weather": "sunny and 31C in Chennai"}
continued tool output that should be dropped
user: Pretty good. My sister Riya is joining me in December.
system: internal note"""


def _add(content: str) -> dict:
    return {
        "action": "ADD", "fact_id": None, "content": content,
        "analysis": {"topics": ["Personal Identity"], "memory_type": "long-term", "duration": None},
    }


def test_flush_transcript_drops_tools_and_bounds_size():
    text = FactMemory.flush_transcript(TRANSCRIPT, 5000)
    assert "Mumbai" in text and "Riya" in text
    assert "Chennai" not in text and "dropped" not in text and "internal note" not in text
    long = "\n".join([f"assistant: {'blah ' * 200}", "user: I am allergic to peanuts and shellfish."] * 20)
    bounded = FactMemory.flush_transcript(long, 1000)
    assert len(bounded) <= 1000 and "peanuts" in bounded


async def test_flush_conversation_extracts_and_stores(app):
    mem, llm = app.memory, app.fake
    app.config.memory.duplicate_similarity = 0.999
    llm.json_replies.extend(
        [
            {"facts": ["Sarthak moved to Mumbai for a new office.", "Sarthak asked how the move is going",
                       "Sarthak's sister Riya is joining him in December."]},
            _add("Sarthak moved to Mumbai for a new office."),
            _add("Sarthak's sister Riya is joining him in December."),
        ]
    )
    async with app.bus.subscribe() as q:
        results = await mem.flush_conversation(TRANSCRIPT, "Sarthak")
        events = drain(q)
    assert [r["action"] for r in results] == ["ADD", "ADD"]
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "about to be compressed" in prompt and "Chennai" not in prompt
    assert {f["content"] for f in await mem.list_facts()} == {
        "Sarthak moved to Mumbai for a new office.", "Sarthak's sister Riya is joining him in December."
    }
    assert sum(1 for e in events if e["type"] == "memory.updated" and e["data"]["action"] == "ADD") == 2


async def test_flush_conversation_is_bounded_and_tolerant(app):
    mem, llm = app.memory, app.fake
    app.config.memory.flush_enabled = False
    assert await mem.flush_conversation(TRANSCRIPT, "Sarthak") == []
    assert llm.calls == []
    app.config.memory.flush_enabled = True
    assert await mem.flush_conversation("user: hi", "Sarthak") == []  # too short to be worth a call
    assert llm.calls == []
    llm.json_replies.append("definitely not a list of facts")
    assert await mem.flush_conversation(TRANSCRIPT, "Sarthak") == []
    app.config.memory.flush_max_facts = 1
    llm.json_replies.extend([{"facts": [{"fact": "Sarthak lives in Mumbai."}, "Sarthak has a sister."]},
                             _add("Sarthak lives in Mumbai.")])
    out = await mem.flush_conversation(TRANSCRIPT, "Sarthak")
    assert [r["content"] for r in out] == ["Sarthak lives in Mumbai."]


def test_keyword_helpers():
    assert keyword_tokens("What is Sarthak's dentist called?") == ["sarthak", "dentist", "called"]
    assert word_overlap("Sarthak lives in Pune", "Sarthak lives in Mumbai") == 0.5
    assert word_overlap("", "x") == 0.0


async def test_hybrid_recall_keyword_match_and_counters(app):
    mem = app.memory
    dentist = await add_fact(mem, "Sarthak's dentist is Dr Mehta in Kothrud")
    await add_fact(mem, "Sarthak enjoys playing chess on weekends")
    # a vector threshold nothing passes: the keyword match still surfaces the fact
    hits = await mem.recall("who is my dentist Mehta", min_similarity=0.999)
    assert [h["id"] for h in hits] == [dentist]
    assert "score" in hits[0] and hits[0]["score"] >= hits[0]["similarity"]
    row = await app.store.fetchone("SELECT recall_count, last_recalled_at FROM facts WHERE id = ?", (dentist,))
    assert row["recall_count"] == 1 and row["last_recalled_at"]
    # internal lookups do not count as recalls
    await mem.recall("dentist Mehta", min_similarity=0.0, track=False)
    row = await app.store.fetchone("SELECT recall_count FROM facts WHERE id = ?", (dentist,))
    assert row["recall_count"] == 1
    # the keyword index follows edits and deletes
    await mem.update_content(dentist, "Sarthak's orthodontist is Dr Kulkarni", use_llm=False)
    assert [h["id"] for h in await mem.recall("Kulkarni orthodontist", min_similarity=0.999)] == [dentist]
    assert await mem.recall("Mehta", min_similarity=0.999) == []
    await mem.forget(dentist)
    assert await mem.recall("Kulkarni orthodontist", min_similarity=0.999) == []


async def test_existing_database_gets_keyword_index(config, isolated_home):
    from sentient.store.db import Store
    from tests.conftest import FakeProvider

    path = isolated_home / "old.db"
    store = await Store(path).open()
    await store.execute(
        "INSERT INTO facts(content, source, topics, memory_type, created_at, updated_at) VALUES(?,?,?,?,?,?)",
        ("Sarthak collects vintage fountain pens", "manual", "[]", "long-term", "2026-01-01", "2026-01-01"),
    )
    await store.close()
    store = await Store(path).open()
    try:
        mem = FactMemory(store, FakeProvider(), config)
        await mem.ensure_schema()
        rows = await store.fetchall("SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'fountain'")
        assert len(rows) == 1
    finally:
        await store.close()
