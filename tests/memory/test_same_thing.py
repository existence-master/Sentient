"""#257, from the real qwen3:8b run: saving "does not want files to be written" rewrote the different preference
"does not want their files deleted", and extraction added "The user's sister lives in a city." next to a fact that
already said where she lives. An UPDATE needs the same subject and action; an ADD that an existing fact already
covers is skipped."""

import pytest

from sentient.memory.facts import FactMemory, covered_by, fact_actions, same_thing
from sentient.store.db import Store
from tests.conftest import FakeProvider

ANALYSIS = {"topics": ["Personal Preferences"], "memory_type": "long-term", "duration": None}


async def _memory(config, isolated_home, name):
    store = await Store(isolated_home / name).open()
    config.memory.duplicate_similarity = 0.999
    llm = FakeProvider()
    return store, llm, FactMemory(store, llm, config)


def test_fact_actions_reads_verb_forms():
    assert fact_actions("Sarthak does not want their files deleted.") == {"delete"}
    assert fact_actions("Sarthak does not want files to be written.") == {"write"}
    assert fact_actions("Sarthak's favourite tea is Darjeeling.") == set()


@pytest.mark.parametrize(
    ("new", "old", "same"),
    [
        ("Sarthak does not want files to be written.", "Sarthak does not want their files deleted.", False),
        # the same action on something else is a second rule, not a change of the first
        ("Sarthak does not want her emails deleted.", "Sarthak does not want her files deleted.", False),
        ("Sarthak does not want calls before 10am.", "Sarthak does not want calls before 9am.", True),
        # Sentient and "the assistant" are the same
        ("Sarthak does not want the assistant to write files for her.",
         "Sarthak doesn't want Sentient to write files for her", True),
        ("Sarthak's brother lives in Pune.", "Sarthak's sister lives in Pune.", False),
        ("Sarthak's favourite tea is black tea.", "Sarthak's favourite tea is green tea.", True),
        ("Sarthak no longer wants emails sent at night.", "Sarthak wants emails sent at night.", True),
        ("The user's sister lives in Porto.", "Sarthak's sister Meera lives in Lisbon.", True),
    ],
)
def test_same_thing(new, old, same):
    assert same_thing(new, old, "Sarthak") is same


@pytest.mark.parametrize(
    ("new", "old", "covered"),
    [
        ("The user's sister lives in a city.", "Sarthak's sister Meera lives in Lisbon and works as a marine biologist.",
         True),
        ("Sarthak's sister lives in Lisbon.", "Sarthak's sister Meera lives in Lisbon.", True),
        # a new detail, a different word, a negation or the past tense is not covered
        ("Sarthak's sister lives in Porto.", "Sarthak's sister Meera lives in Lisbon.", False),
        ("Sarthak's sister lives near the sea.", "Sarthak's sister Meera lives in Lisbon.", False),
        ("Sarthak eats meat.", "Sarthak does not eat meat.", False),
        ("Sarthak lives in Pune.", "Sarthak lived in Pune as a child.", False),
        ("Sarthak's brother lives in a city.", "Sarthak's sister Meera lives in Lisbon.", False),
        # saying a relation exists is covered by any fact about that relation, but not a fact about someone else
        ("Sarthak has a sister.", "Sarthak's sister Meera lives in Lisbon.", True),
        ("Sarthak does not want the assistant to write files for her.",
         "Sarthak doesn't want Sentient to write files for her", True),
        ("Sarthak has a brother.", "Sarthak's sister Meera lives in Lisbon.", False),
        ("Sarthak lives in Lisbon.", "Sarthak's sister Meera lives in Lisbon.", False),
    ],
)
def test_covered_by(new, old, covered):
    assert covered_by(new, old, "Sarthak") is covered


async def test_update_about_a_different_action_keeps_both(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "action.db")
    try:
        first = await mem.remember("Sarthak does not want their files deleted.", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": first["id"], "content": "Sarthak does not want files to be written.",
             "analysis": None},
            ANALYSIS, ANALYSIS,
        ])
        second = await mem.remember("Sarthak does not want files to be written.", use_llm=True)
        stored = sorted(f["content"] for f in await mem.list_facts())
        assert second["action"] == "ADD" and second["id"] != first["id"]
        assert stored == ["Sarthak does not want files to be written.", "Sarthak does not want their files deleted."]
    finally:
        await store.close()


async def test_update_about_the_same_thing_still_updates(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "same.db")
    try:
        first = await mem.remember("Sarthak's favourite tea is green tea.", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": first["id"], "content": "Sarthak's favourite tea is black tea.",
             "analysis": None},
            ANALYSIS, ANALYSIS,
        ])
        second = await mem.remember("Sarthak's favourite tea is now black tea.", use_llm=True)
        assert second["action"] == "UPDATE" and second["id"] == first["id"]
        assert [f["content"] for f in await mem.list_facts()] == ["Sarthak's favourite tea is black tea."]
    finally:
        await store.close()


async def test_a_vaguer_fact_is_skipped(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "vague.db")
    try:
        full = "Sarthak's sister Meera lives in Lisbon and works as a marine biologist."
        first = await mem.remember(full, use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": first["id"], "content": "The user's sister lives in a city.",
             "analysis": None},
            ANALYSIS, ANALYSIS,
        ])
        second = await mem.remember("The user's sister lives in a city.", use_llm=True)
        assert second == {"action": "SKIP", "id": first["id"], "content": full}
        assert [f["content"] for f in await mem.list_facts()] == [full]
    finally:
        await store.close()


@pytest.mark.parametrize(
    "fact", ["Sarthak's sister's city of residence is unknown.", "Sarthak's sister's occupation is not mentioned."]
)
def test_a_gap_is_not_a_fact(fact):
    assert FactMemory.clean_fact(fact, "Sarthak") is None
    assert FactMemory.clean_fact("Sarthak's sister is a marine biologist.", "Sarthak")
