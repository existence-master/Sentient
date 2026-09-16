from sentient.config.schema import SentientConfig
from sentient.memory.facts import FactMemory, new_details
from sentient.store.db import Store
from tests.conftest import FakeProvider


def test_new_details_finds_places_names_numbers():
    assert new_details("Sarthak lives in Mumbai.", "Sarthak lives in Pune.") == {"Mumbai"}
    assert "Berlin" in new_details(
        "Sarthak's sister Riya is moving to Berlin next month for her masters.",
        "Sarthak's sister Riya is pursuing her masters.",
    )
    assert "10" in new_details("Sarthak runs 10k every Sunday.", "Sarthak runs every Sunday.")
    assert new_details("Sarthak runs every Sunday.", "Sarthak runs every Sunday.") == set()
    # possessives of known words and the sentence-initial capital are not "new"
    assert new_details("Sarthak's sister lives in Pune.", "Sarthak has a sister who lives in Pune.") == set()


def test_duplicate_threshold_default_is_conservative():
    assert SentientConfig().memory.duplicate_similarity >= 0.98


async def test_model_skip_overruled_when_new_fact_adds_detail(config, isolated_home):
    store = await Store(isolated_home / "dedup.db").open()
    try:
        config.memory.duplicate_similarity = 0.999
        llm = FakeProvider()
        mem = FactMemory(store, llm, config)
        first = await mem.remember("Sarthak's sister Riya is pursuing her masters.", use_llm=False)
        assert first["action"] == "ADD"

        llm.json_replies.extend(
            [
                # the model wrongly says the Berlin fact is already covered
                {"action": "SKIP", "fact_id": first["id"], "content": None, "analysis": None},
                # analysis call for the overruled ADD
                {"topics": ["Relationships & Social Life"], "memory_type": "short-term", "duration": "1 month"},
            ]
        )
        second = await mem.remember(
            "Sarthak's sister Riya is moving to Berlin next month for her masters.", use_llm=True
        )
        stored = [f["content"] for f in await mem.list_facts()]
        assert second["action"] == "ADD"
        assert any("Berlin" in f for f in stored)

        llm.json_replies.append({"action": "SKIP", "fact_id": first["id"], "content": None, "analysis": None})
        third = await mem.remember("Sarthak's sister Riya is pursuing her masters now.", use_llm=True)
        assert third["action"] == "SKIP"
    finally:
        await store.close()
