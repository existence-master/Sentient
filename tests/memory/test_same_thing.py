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
    assert fact_actions("Maya does not want their files deleted.") == {"delete"}
    assert fact_actions("Maya does not want files to be written.") == {"write"}
    assert fact_actions("Maya's favourite tea is Darjeeling.") == set()


@pytest.mark.parametrize(
    ("new", "old", "same"),
    [
        ("Maya does not want files to be written.", "Maya does not want their files deleted.", False),
        # the same action on something else is a second rule, not a change of the first
        ("Maya does not want her emails deleted.", "Maya does not want her files deleted.", False),
        ("Maya does not want calls before 10am.", "Maya does not want calls before 9am.", True),
        # a replacement that drops an action loses a preference
        ("Maya does not want files deleted.", "Maya does not want files deleted or sent.", False),
        ("Maya does not want files deleted or sent.", "Maya does not want files deleted.", True),
        # Sentient and "the assistant" are the same
        ("Maya does not want the assistant to write files for her.",
         "Maya doesn't want Sentient to write files for her", True),
        ("Maya's brother lives in Pune.", "Maya's sister lives in Pune.", False),
        ("Maya's favourite tea is black tea.", "Maya's favourite tea is green tea.", True),
        ("Maya no longer wants emails sent at night.", "Maya wants emails sent at night.", True),
        ("The user's sister lives in Porto.", "Maya's sister Meera lives in Lisbon.", True),
    ],
)
def test_same_thing(new, old, same):
    assert same_thing(new, old, "Maya") is same


@pytest.mark.parametrize(
    ("new", "old", "covered"),
    [
        ("The user's sister lives in a city.", "Maya's sister Meera lives in Lisbon and works as a marine biologist.",
         True),
        ("Maya's sister lives in Lisbon.", "Maya's sister Meera lives in Lisbon.", True),
        # a new detail, a different word, a negation or the past tense is not covered
        ("Maya's sister lives in Porto.", "Maya's sister Meera lives in Lisbon.", False),
        ("Maya's sister lives near the sea.", "Maya's sister Meera lives in Lisbon.", False),
        ("Maya eats meat.", "Maya does not eat meat.", False),
        ("Maya lives in Pune.", "Maya lived in Pune as a child.", False),
        ("Maya's brother lives in a city.", "Maya's sister Meera lives in Lisbon.", False),
        # a placeholder needs a fact that names it
        ("Maya has a job.", "Maya has a dog.", False),
        ("Maya has a job.", "Maya has a dog named Rex.", False),
        ("Maya works at a company.", "Maya works at Acme.", True),
        # saying a relation exists is covered by any fact about that relation, but not a fact about someone else
        ("Maya has a sister.", "Maya's sister Meera lives in Lisbon.", True),
        ("Maya does not want the assistant to write files for her.",
         "Maya doesn't want Sentient to write files for her", True),
        ("Maya has a brother.", "Maya's sister Meera lives in Lisbon.", False),
        ("Maya lives in Lisbon.", "Maya's sister Meera lives in Lisbon.", False),
    ],
)
def test_covered_by(new, old, covered):
    assert covered_by(new, old, "Maya") is covered


async def test_update_about_a_different_action_keeps_both(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "action.db")
    try:
        first = await mem.remember("Maya does not want their files deleted.", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": first["id"], "content": "Maya does not want files to be written.",
             "analysis": None},
            ANALYSIS, ANALYSIS,
        ])
        second = await mem.remember("Maya does not want files to be written.", use_llm=True)
        stored = sorted(f["content"] for f in await mem.list_facts())
        assert second["action"] == "ADD" and second["id"] != first["id"]
        assert stored == ["Maya does not want files to be written.", "Maya does not want their files deleted."]
    finally:
        await store.close()


async def test_update_about_the_same_thing_still_updates(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "same.db")
    try:
        first = await mem.remember("Maya's favourite tea is green tea.", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": first["id"], "content": "Maya's favourite tea is black tea.",
             "analysis": None},
            ANALYSIS, ANALYSIS,
        ])
        second = await mem.remember("Maya's favourite tea is now black tea.", use_llm=True)
        assert second["action"] == "UPDATE" and second["id"] == first["id"]
        assert [f["content"] for f in await mem.list_facts()] == ["Maya's favourite tea is black tea."]
    finally:
        await store.close()


async def test_a_vaguer_fact_is_skipped(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "vague.db")
    try:
        full = "Maya's sister Meera lives in Lisbon and works as a marine biologist."
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
    "fact", ["Maya's sister's city of residence is unknown.", "Maya's sister's occupation is not mentioned."]
)
def test_a_gap_is_not_a_fact(fact):
    assert FactMemory.clean_fact(fact, "Maya") is None
    assert FactMemory.clean_fact("Maya's sister is a marine biologist.", "Maya")


async def test_a_rewrite_about_another_action_is_not_stored(config, isolated_home):
    """The new fact matches, but the model's rewrite that would be stored is about writing, not deleting."""
    store, llm, mem = await _memory(config, isolated_home, "rewrite.db")
    try:
        first = await mem.remember("Maya does not want their files deleted.", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": first["id"], "content": "Maya does not want files to be written.",
             "analysis": None},
            ANALYSIS, ANALYSIS,
        ])
        second = await mem.remember("Maya never wants files deleted.", use_llm=True)
        stored = sorted(f["content"] for f in await mem.list_facts())
        assert second["action"] == "ADD" and second["content"] == "Maya never wants files deleted."
        assert "Maya does not want their files deleted." in stored
        assert "Maya does not want files to be written." not in stored
    finally:
        await store.close()
