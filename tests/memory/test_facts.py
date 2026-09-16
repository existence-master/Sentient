from __future__ import annotations

from sentient.memory.topics import TOPIC_NAMES, normalize_topics


def test_topics_are_the_eight_v2_topics():
    assert TOPIC_NAMES == [
        "Personal Identity", "Interests & Lifestyle", "Work & Learning", "Health & Wellbeing",
        "Relationships & Social Life", "Financial", "Goals & Challenges", "Miscellaneous",
    ]
    assert normalize_topics(["work", "family"]) == ["Work & Learning", "Relationships & Social Life"]
    assert normalize_topics("nonsense-label") == ["Miscellaneous"]
    assert normalize_topics([]) == ["Miscellaneous"]


async def test_cud_add_update_delete_skip(app):
    mem, llm = app.memory, app.fake
    app.config.memory.duplicate_similarity = 0.999  # keep hashed toy embeddings out of the near-duplicate short-circuit
    llm.json_replies.append(
        {"action": "ADD", "fact_id": None, "content": "Sarthak drinks black coffee",
         "analysis": {"topics": ["Personal Identity"], "memory_type": "long-term", "duration": None}}
    )
    added = await mem.remember("Sarthak drinks black coffee")
    assert added["action"] == "ADD"
    fact = await mem.get_fact(added["id"])
    assert fact["topics"] == ["Personal Identity"] and fact["memory_type"] == "long-term"

    # exact duplicate short-circuits before any LLM call
    calls = len(llm.calls)
    assert (await mem.remember("Sarthak drinks black coffee"))["action"] == "SKIP"
    assert len(llm.calls) == calls

    llm.json_replies.append(
        {"action": "UPDATE", "fact_id": added["id"], "content": "Sarthak drinks black coffee with oat milk",
         "analysis": {"topics": ["Personal Identity"], "memory_type": "long-term", "duration": None}}
    )
    upd = await mem.remember("Sarthak drinks black coffee with oat milk now")
    assert upd == {"action": "UPDATE", "id": added["id"], "content": "Sarthak drinks black coffee with oat milk"}

    llm.json_replies.append({"action": "SKIP", "fact_id": added["id"], "content": None, "analysis": None})
    assert (await mem.remember("Sarthak drinks black coffee with oat milk daily"))["action"] == "SKIP"

    llm.json_replies.append({"action": "DELETE", "fact_id": added["id"], "content": None, "analysis": None})
    assert (await mem.remember("Sarthak drinks black coffee no more"))["action"] == "DELETE"
    assert await mem.count() == 0


async def test_update_by_id_keeps_id_and_reanalyzes(app):
    mem, llm = app.memory, app.fake
    r = await mem.remember("Sarthak works at Existence", use_llm=False)
    llm.json_replies.append({"topics": ["Work & Learning"], "memory_type": "long-term", "duration": None})
    out = await mem.update_content(r["id"], "Sarthak is the founder of Existence")
    assert out["id"] == r["id"] and out["content"] == "Sarthak is the founder of Existence"
    assert out["topics"] == ["Work & Learning"]
    row = await app.store.fetchone("SELECT previous_content FROM facts WHERE id = ?", (r["id"],))
    assert row["previous_content"] == "Sarthak works at Existence"
    hits = await mem.recall("founder of Existence", min_similarity=0.0)
    assert hits and hits[0]["id"] == r["id"]


async def test_short_term_expiry_and_purge(app):
    mem, llm = app.memory, app.fake
    llm.json_replies.append(
        {"action": "ADD", "fact_id": None, "content": "Sarthak has a dentist appointment on Friday",
         "analysis": {"topics": ["Health & Wellbeing"], "memory_type": "short-term", "duration": "3 days"}}
    )
    r = await mem.remember("Sarthak has a dentist appointment on Friday")
    fact = await mem.get_fact(r["id"])
    assert fact["memory_type"] == "short-term" and fact["expires_at"]
    assert (await mem.expiring_soon())== []
    await app.store.execute("UPDATE facts SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (r["id"],))
    assert await mem.recall("dentist", min_similarity=0.0) == []
    assert await mem.list_facts() == []
    assert await mem.purge_expired() == 1
    assert await mem.count() == 0


async def test_list_filters_topics_and_delete_by_source(app):
    mem = app.memory
    a = await mem.remember("Sarthak plays chess on weekends", use_llm=False, source="manual")
    await app.store.execute("UPDATE facts SET topics = ? WHERE id = ?", ('["Interests & Lifestyle"]', a["id"]))
    await mem.remember("Sarthak studied computer engineering", use_llm=False, source="file:cv.txt")
    await mem.remember("Sarthak knows Python very well", use_llm=False, source="file:cv.txt")

    assert [f["id"] for f in await mem.list_facts(topic="Interests & Lifestyle")] == [a["id"]]
    assert len(await mem.list_facts(source="file:cv.txt")) == 2
    hits = await mem.list_facts(q="chess weekends")
    assert hits and hits[0]["id"] == a["id"]
    counts = {t["name"]: t["count"] for t in await mem.topic_counts()}
    assert counts["Interests & Lifestyle"] == 1 and counts["Miscellaneous"] == 2

    by_source = await mem.search_by_source("python", "file:cv.txt")
    assert by_source and all(f["id"] != a["id"] for f in by_source)
    assert await mem.forget_source("file:cv.txt") == 2
    assert await mem.count() == 1


async def test_graph_links_similar_facts(app):
    mem = app.memory
    app.config.memory.graph_link_similarity = 0.8
    app.config.memory.duplicate_similarity = 0.999
    a = await mem.remember("Sarthak loves hiking mountains", use_llm=False)
    b = await mem.remember("Sarthak loves hiking mountains often", use_llm=False)
    await mem.remember("zebra quantum violin", use_llm=False)
    g = await mem.graph()
    assert len(g["nodes"]) == 3
    assert {"id", "title", "content", "topics", "memory_type"} <= set(g["nodes"][0])
    pairs = {frozenset((link["source"], link["target"])) for link in g["links"]}
    assert frozenset((a["id"], b["id"])) in pairs
    assert all(link["value"] >= 0.8 for link in g["links"])


async def test_extraction_personalizes_and_drops_requests(app):
    mem, llm = app.memory, app.fake
    llm.json_replies.append(
        {"facts": ["The user's sister Riya lives in Pune", "Sarthak asked what day it is", "USERNAME likes tea"]}
    )
    facts = await mem.extract_facts("my sister riya lives in pune. what day is it? I like tea", "Sarthak")
    assert facts == ["Sarthak's sister Riya lives in Pune", "Sarthak likes tea"]
    system = llm.calls[-1]["messages"][0]["content"]
    assert "Never write the words" in system and "Sarthak" in system


async def test_bus_events_on_changes(app):
    events = []
    async with app.bus.subscribe() as q:
        r = await app.memory.remember("Sarthak owns a red bicycle", use_llm=False, notify=True)
        await app.memory.forget(r["id"], notify=True)
        while not q.empty():
            events.append(await q.get())
    kinds = [(e["type"], e["data"]["action"]) for e in events]
    assert ("memory.updated", "ADD") in kinds and ("memory.updated", "DELETE") in kinds
