from sentient.memory.facts import FactMemory
from sentient.store.db import Store
from tests.conftest import FakeProvider


async def test_tautological_facts_are_dropped(config, isolated_home):
    store = await Store(isolated_home / "taut.db").open()
    try:
        llm = FakeProvider(
            json_replies=[
                {"facts": ["Maya's name is Maya.", "Maya lives in Pune.", "maya’s name is Maya"]},  # noqa: RUF001 - curly apostrophe is the case under test
                {
                    "action": "ADD",
                    "fact_id": None,
                    "content": "Maya lives in Pune.",
                    "analysis": {"topics": ["Personal Identity"], "memory_type": "long-term", "duration": None},
                },
            ]
        )
        mem = FactMemory(store, llm, config)
        results = await mem.extract_and_store("My name is Maya and I live in Pune.", "Maya", source="onboarding")
        stored = [f["content"] for f in await mem.list_facts()]
        assert stored == ["Maya lives in Pune."]
        assert all("name is" not in r["content"] for r in results)
    finally:
        await store.close()
