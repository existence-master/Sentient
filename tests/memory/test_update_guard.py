"""Regression from the real qwen3:8b run: the model rewrote "Riya is moving to Berlin next month" as
"Riya is pursuing her masters", losing Berlin. An UPDATE may only drop a name, place or number that the
new fact replaces."""

from sentient.memory.facts import FactMemory
from sentient.store.db import Store
from tests.conftest import FakeProvider

ANALYSIS = {"topics": ["Relationships & Social Life"], "memory_type": "long-term", "duration": None}


async def _memory(config, isolated_home, name):
    store = await Store(isolated_home / name).open()
    config.memory.duplicate_similarity = 0.999
    llm = FakeProvider()
    return store, llm, FactMemory(store, llm, config)


async def test_update_that_drops_a_place_is_kept_as_a_new_fact(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "guard.db")
    try:
        first = await mem.remember("Sarthak's sister Riya is moving to Berlin next month.", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": first["id"], "content": "Sarthak's sister Riya is pursuing her masters.",
             "analysis": None},
            ANALYSIS, ANALYSIS, ANALYSIS,
        ])
        second = await mem.remember("Sarthak's sister Riya is pursuing her masters.", use_llm=True)
        stored = [f["content"] for f in await mem.list_facts()]
        assert second["action"] == "ADD"
        assert any("Berlin" in f for f in stored)
        assert any("masters" in f for f in stored)
    finally:
        await store.close()


async def test_update_that_replaces_a_place_still_updates(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "guard2.db")
    try:
        first = await mem.remember("Sarthak's friend Aman lives in Pune.", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": first["id"], "content": "Sarthak's friend Aman lives in Mumbai.",
             "analysis": None},
            ANALYSIS, ANALYSIS, ANALYSIS,
        ])
        second = await mem.remember("Sarthak's friend Aman now lives in Mumbai.", use_llm=True)
        stored = [f["content"] for f in await mem.list_facts()]
        assert second["action"] == "UPDATE" and second["id"] == first["id"]
        assert stored == ["Sarthak's friend Aman lives in Mumbai."]
    finally:
        await store.close()
