"""#364, from the real qwen3:8b run for #363: after "Mira starts school in June" was saved, "Mira starts school in
July" was skipped, so the old date stayed. A correction (the same subject and action with a changed date, place,
number or name) always updates the old fact, without asking the model."""

import pytest

from sentient.memory import review as reviews
from sentient.memory.facts import FactMemory, correction_of, says_something_else
from sentient.memory.prompts import CUD_DECISION_SYSTEM
from sentient.store.db import Store
from tests.conftest import FakeProvider

ANALYSIS = {"topics": ["Family"], "memory_type": "long-term", "duration": None}


async def _memory(config, isolated_home, name, duplicate_similarity=0.999):
    store = await Store(isolated_home / name).open()
    config.assistant.user_name = "Maya"
    config.memory.duplicate_similarity = duplicate_similarity
    llm = FakeProvider()
    return store, llm, FactMemory(store, llm, config)


@pytest.mark.parametrize(
    ("new", "old", "stored"),
    [
        # a date: the old wording is kept with the new value
        ("Mira's school now starts in July.", "Maya's daughter Mira starts school in June.",
         "Maya's daughter Mira starts school in July."),
        ("Actually Mira's school starts in July", "Maya's daughter Mira starts school in June.",
         "Maya's daughter Mira starts school in July."),
        ("Mira starts school in July, not June.", "Mira starts school in June.", "Mira starts school in July."),
        # a place
        ("Ravi moved from Pune to Delhi.", "Ravi lives in Pune.", "Ravi moved from Pune to Delhi."),
        ("Maya's friend Ravi now lives in Delhi.", "Maya's friend Ravi lives in Pune.",
         "Maya's friend Ravi lives in Delhi."),
        # a number
        ("Maya now runs 8 km every morning.", "Maya runs 5 km every morning.", "Maya runs 8 km every morning."),
        ("Maya's team has grown to 15 people.", "Maya's team has 12 people.", "Maya's team has grown to 15 people."),
        # a name
        ("Maya's dog is now named Max.", "Maya's dog is named Bruno.", "Maya's dog is named Max."),
    ],
)
def test_a_changed_value_is_a_correction(new, old, stored):
    assert correction_of(new, old, "Maya") == stored


@pytest.mark.parametrize(
    ("new", "old"),
    [
        # another thing about the same person
        ("Mira has a piano recital in July.", "Maya's daughter Mira starts school in June."),
        ("Mira's piano class starts in July.", "Mira starts school in June."),
        # another person
        ("Maya's son Kabir starts school in July.", "Maya's daughter Mira starts school in June."),
        ("Aman lives in Delhi.", "Ravi lives in Pune."),
        # only a new detail, nothing replaced
        ("Mira starts school in June at Green Valley.", "Mira starts school in June."),
        ("Mira starts school in June.", "Mira starts school in June."),
        # a value would be lost with nothing in its place: left to the model and its guard
        ("Riya got a job at Google.", "Maya's sister Riya works at Infosys in Pune."),
        # a negation or the past is not a correction of the present
        ("Ravi lived in Delhi as a child.", "Ravi lives in Pune."),
        ("Maya does not run 8 km every morning.", "Maya runs 5 km every morning."),
    ],
)
def test_other_facts_are_not_corrections(new, old):
    assert correction_of(new, old, "Maya") is None


async def test_a_correction_updates_even_when_the_model_says_skip(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "fix.db")
    try:
        first = await mem.remember("Maya's daughter Mira starts school in June.", use_llm=False)
        # what qwen3:8b answered in the real run, then the analysis of the corrected fact
        llm.json_replies.extend([
            {"action": "SKIP", "fact_id": first["id"], "content": None, "analysis": None}, ANALYSIS, ANALYSIS,
        ])
        second = await mem.remember("Mira's school now starts in July.", use_llm=True)
        assert second["action"] == "UPDATE" and second["id"] == first["id"]
        assert second["previous"] == "Maya's daughter Mira starts school in June."
        assert [f["content"] for f in await mem.list_facts()] == ["Maya's daughter Mira starts school in July."]
        # decided without the CUD model: only the analysis of the new wording was asked for
        assert not any("memory management" in str(c) for c in llm.calls)
    finally:
        await store.close()


async def test_a_near_identical_correction_is_not_skipped_as_a_duplicate(config, isolated_home):
    # embedding models score "... in June" and "... in July" almost the same; the duplicate check must not win
    store, llm, mem = await _memory(config, isolated_home, "dup.db", duplicate_similarity=0.5)
    try:
        first = await mem.remember("Mira starts school in June.", use_llm=False)
        llm.json_replies.extend([ANALYSIS, ANALYSIS])
        second = await mem.remember("Mira starts school in July.", use_llm=True)
        assert second["action"] == "UPDATE" and second["id"] == first["id"]
        assert [f["content"] for f in await mem.list_facts()] == ["Mira starts school in July."]
    finally:
        await store.close()


async def test_unrelated_facts_are_not_merged(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "apart.db")
    try:
        await mem.remember("Maya's daughter Mira starts school in June.", use_llm=False)
        llm.json_replies.extend([
            {"action": "ADD", "fact_id": None, "content": "Mira has a piano recital in July.", "analysis": ANALYSIS},
        ] + [ANALYSIS] * 3)
        second = await mem.remember("Mira has a piano recital in July.", use_llm=True)
        assert second["action"] == "ADD"
        assert sorted(f["content"] for f in await mem.list_facts()) == [
            "Maya's daughter Mira starts school in June.", "Mira has a piano recital in July.",
        ]
    finally:
        await store.close()


async def test_the_model_is_told_which_fact_a_looser_correction_changes(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "hint.db")
    try:
        first = await mem.remember("Mira's first school day is in June.", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": first["id"], "content": "Mira's school year opens in July.",
             "analysis": ANALYSIS},
        ] + [ANALYSIS] * 3)
        second = await mem.remember("Mira's school year opens in July.", use_llm=True)
        assert second["action"] == "UPDATE"
        cud = next(c for c in llm.calls if "memory management" in str(c))
        assert '"changes_value": true' in str(cud)
        assert "A correction is never SKIP" in CUD_DECISION_SYSTEM
    finally:
        await store.close()


async def test_a_held_correction_is_not_skipped_as_a_duplicate(config, isolated_home):
    store, _llm, mem = await _memory(config, isolated_home, "held.db", duplicate_similarity=0.5)
    try:
        await mem.remember("Mira starts school in June.", use_llm=False)
        held = await mem.remember(
            "Mira starts school in July.", use_llm=False, review=reviews.note("Mail")
        )
        assert held["action"] == "ADD" and held["status"] == "pending"
    finally:
        await store.close()


async def test_an_unrelated_fact_the_model_calls_an_update_is_added(config, isolated_home):
    # from the real qwen3:8b run: "Mira has a piano recital in July" replaced "Mira starts school in July"
    store, llm, mem = await _memory(config, isolated_home, "unrelated.db")
    try:
        first = await mem.remember("Maya's daughter Mira starts school in July", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": first["id"], "content": "Mira has a piano recital in July.",
             "analysis": ANALYSIS},
        ] + [ANALYSIS] * 3)
        second = await mem.remember("Mira has a piano recital in July.", use_llm=True)
        assert second["action"] == "ADD"
        assert sorted(f["content"] for f in await mem.list_facts()) == [
            "Maya's daughter Mira starts school in July", "Mira has a piano recital in July.",
        ]
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("new", "old", "unrelated"),
    [
        ("Mira has a piano recital in July.", "Maya's daughter Mira starts school in July", True),
        ("Maya started eating fish.", "Maya is vegetarian.", False),  # the same attribute: diet
        ("Riya got a job at Google.", "Maya's sister Riya works at Infosys.", False),
        ("Maya now runs 8 km every morning.", "Maya runs 5 km every morning.", False),
    ],
)
def test_says_something_else(new, old, unrelated):
    assert says_something_else(new, old, "Maya") is unrelated


async def test_a_correction_updates_without_the_model_too(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "nollm.db", duplicate_similarity=0.5)
    try:
        first = await mem.remember("Mira starts school in June.", use_llm=False)
        second = await mem.remember("Mira starts school in July.", use_llm=False)
        assert second["action"] == "UPDATE" and second["id"] == first["id"]
        assert [f["content"] for f in await mem.list_facts()] == ["Mira starts school in July."]
        assert llm.calls == []
    finally:
        await store.close()
