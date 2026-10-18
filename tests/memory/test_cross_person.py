"""#463: saving "Maya Rao's sister Priya lives in Mumbai." replaced the user's own "Maya Rao moved from Pune to
Bengaluru." The subject reader stopped at "Maya", so the sister's home counted as a correction of Maya's. A fact about
another person (someone's sister, my friend, her colleague) never replaces one about the user, in code or by the
model, and a real correction of the user's own fact still does."""

from __future__ import annotations

import pytest

from sentient.memory.facts import (
    FactMemory,
    correction_of,
    fact_subjects,
    same_person,
    same_subject,
)
from sentient.memory.scripts import devanagari_subjects
from sentient.store.db import Store
from tests.conftest import FakeProvider
from tests.memory.helpers import add_fact

ANALYSIS = {"topics": ["Family"], "memory_type": "long-term", "duration": None}
USER = "Maya Rao"


class SameVectorProvider(FakeProvider):
    """An embedding model that can't tell texts apart, so every saved fact is a close neighbour of the new one."""

    async def embed(self, texts, *, model=None):
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


async def _memory(config, isolated_home, name):
    store = await Store(isolated_home / name).open()
    config.assistant.user_name = USER
    config.memory.duplicate_similarity = 0.999
    llm = SameVectorProvider()
    return store, llm, FactMemory(store, llm, config)


async def _seed(mem, *facts):
    return [await add_fact(mem, f) for f in facts]  # saved as they are, with no merging between them


def _cud(action: str, content: str, fact_id: int | None = None) -> dict:
    return {"action": action, "fact_id": fact_id, "content": content, "analysis": ANALYSIS}


@pytest.mark.parametrize(
    ("text", "subjects"),
    [
        ("Maya Rao's sister Priya lives in Mumbai.", {"maya's sister", "priya"}),
        ("Maya's sister Priya lives in Mumbai.", {"maya's sister", "priya"}),
        ("Maya Rao's younger sister lives in Mumbai.", {"maya's sister"}),
        ("Maya's mom lives in Mumbai.", {"maya's mom"}),
        ("My friend Arjun moved to Delhi.", {"maya rao's friend", "arjun"}),
        ("My best friend moved to Delhi.", {"maya rao's friend"}),
        ("Her colleague Ravi lives in Delhi.", {"someone's colleague", "ravi"}),
        ("Priya, Maya's sister, lives in Mumbai.", {"priya", "maya's sister"}),
        ("Maya Rao moved from Pune to Bengaluru.", {"maya"}),
        ("I moved to Mumbai.", {"maya rao"}),
        ("Last month Maya moved to Mumbai.", {"maya"}),
        ("She moved to Mumbai.", set()),
    ],
)
def test_who_a_fact_is_about(text, subjects):
    assert fact_subjects(text, USER) == subjects


def test_hindi_relations_with_a_surname_or_a_pronoun():
    assert devanagari_subjects("माया राव की बहन प्रिया मुंबई में रहती है।") == {"माया's sister"}
    assert devanagari_subjects("उसकी बहन मुंबई में रहती है।") == {"someone's sister"}
    assert not same_subject("माया राव की बहन प्रिया मुंबई में रहती है।", "Maya lives in Bengaluru.", USER)


@pytest.mark.parametrize(
    "new",
    [
        "Maya Rao's sister Priya lives in Mumbai.",
        "Maya's sister Priya lives in Mumbai.",
        "Maya's mom lives in Mumbai.",
        "My friend Arjun moved to Delhi.",
        "Her colleague Ravi lives in Delhi.",
        "Priya, Maya's sister, lives in Mumbai.",
        "She moved to Mumbai.",  # nobody named: never decided in code
    ],
)
@pytest.mark.parametrize("old", ["Maya Rao moved from Pune to Bengaluru.", "Maya lives in Bengaluru."])
def test_someone_elses_home_is_not_a_correction_of_the_users(new, old):
    assert correction_of(new, old, USER) is None


def test_the_users_own_new_home_is_a_correction():
    assert correction_of("Maya moved to Mumbai.", "Maya lives in Bengaluru.", USER) == "Maya moved to Mumbai."
    assert correction_of("I moved to Mumbai.", "Maya lives in Bengaluru.", USER) == "I moved to Mumbai."
    assert same_person("I moved to Mumbai.", "Maya Rao lives in Bengaluru.", USER)


def test_a_pronoun_relation_matches_the_same_relation_only():
    assert same_subject("Her sister lives in Delhi.", "Maya's sister lives in Pune.", USER)
    assert not same_subject("Her sister lives in Delhi.", "Maya's brother lives in Pune.", USER)
    assert not same_subject("Her sister lives in Delhi.", "Maya lives in Pune.", USER)


@pytest.mark.parametrize(
    "new",
    ["Maya Rao's sister Priya lives in Mumbai.", "Maya's sister Priya lives in Mumbai.", "My friend Arjun moved to Delhi."],
)
async def test_another_persons_home_is_added_and_the_users_is_kept(config, isolated_home, new):
    store, llm, mem = await _memory(config, isolated_home, "home.db")
    try:
        [moved, lives] = await _seed(mem, "Maya Rao moved from Pune to Bengaluru.", "Maya lives in Bengaluru.")
        # a small model that calls it an update of Maya's home anyway
        llm.json_replies.extend([_cud("UPDATE", new, moved)] + [ANALYSIS] * 3)
        out = await mem.remember(new)
        assert out["action"] == "ADD" and out["content"] == new
        assert (await mem.get_fact(moved))["content"] == "Maya Rao moved from Pune to Bengaluru."
        assert (await mem.get_fact(lives))["content"] == "Maya lives in Bengaluru."
        # Maya's own facts are never offered to the model as something this one could update
        cud = next(c for c in llm.calls if "memory management" in str(c))
        assert "Bengaluru" not in cud["messages"][1]["content"]
    finally:
        await store.close()


async def test_the_users_move_still_updates_her_home(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "move.db")
    try:
        [sister, lives] = await _seed(mem, "Maya's sister Priya lives in Pune.", "Maya lives in Bengaluru.")
        llm.json_replies.extend([ANALYSIS] * 3)
        out = await mem.remember("Maya moved to Mumbai.")
        assert out["action"] == "UPDATE" and out["id"] == lives and out["previous"] == "Maya lives in Bengaluru."
        assert (await mem.get_fact(sister))["content"] == "Maya's sister Priya lives in Pune."
    finally:
        await store.close()


async def test_the_users_move_replaces_a_moved_from_home(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "moved.db")
    try:
        [moved] = await _seed(mem, "Maya Rao moved from Pune to Bengaluru.")
        llm.json_replies.extend([_cud("ADD", "Maya moved to Mumbai.")] + [ANALYSIS] * 3)
        out = await mem.remember("Maya moved to Mumbai.")
        assert out["action"] == "UPDATE" and out["id"] == moved
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("own", "other", "correction"),
    [
        ("Maya works at Infosys.", "Maya's husband Dev works at Google.", "Maya works at Google."),
        ("Maya is vegetarian.", "Maya's brother Kabir eats fish every week.", None),
        ("Maya is married to Dev.", "Maya's sister Priya is married to Arjun.", None),
    ],
)
async def test_job_diet_and_relationship_of_someone_else_leave_the_users_alone(
    config, isolated_home, own, other, correction
):
    store, llm, mem = await _memory(config, isolated_home, "attr.db")
    try:
        [mine] = await _seed(mem, own)
        llm.json_replies.extend([_cud("UPDATE", other, mine)] + [ANALYSIS] * 3)
        out = await mem.remember(other)
        assert out["action"] == "ADD"
        assert (await mem.get_fact(mine))["content"] == own
        if correction:
            llm.json_replies.extend([ANALYSIS] * 3)
            fixed = await mem.remember(correction)
            assert fixed["action"] == "UPDATE" and fixed["id"] == mine
    finally:
        await store.close()


async def test_a_hindi_fact_about_the_sister_never_updates_the_users_home(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "hindi.db")
    try:
        [lives] = await _seed(mem, "Maya lives in Bengaluru.")
        new = "माया राव की बहन प्रिया मुंबई में रहती है।"
        llm.json_replies.extend([_cud("UPDATE", new, lives)] + [ANALYSIS] * 3)
        out = await mem.remember(new)
        assert out["action"] == "ADD"
        assert (await mem.get_fact(lives))["content"] == "Maya lives in Bengaluru."
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("new", "old"),
    [
        # "her" may be anyone: a code-decided update needs an owner it can name (CodeRabbit on #471)
        ("Her sister lives in Delhi.", "Maya's sister lives in Pune."),
        ("उसकी बहन दिल्ली में रहती है।", "Maya's sister lives in Pune."),
        # a namesake of the user is someone else
        ("Maya Singh moved to Mumbai.", "Maya Rao lives in Bengaluru."),
        ("Maya Singh moved to Mumbai.", "Maya lives in Bengaluru."),
        ("Maya Singh's sister lives in Delhi.", "Maya's sister lives in Pune."),
    ],
)
def test_no_code_update_without_a_clear_same_person(new, old):
    assert not same_person(new, old, USER)
    assert correction_of(new, old, USER) is None


def test_a_pronoun_relation_still_counts_by_a_shared_name():
    assert same_person("Her sister Priya lives in Delhi.", "Maya's sister Priya lives in Pune.", USER)
    assert fact_subjects("Maya Singh moved to Mumbai.", USER) == {"maya singh"}
    assert fact_subjects("Maya Rao moved to Mumbai.", USER) == {"maya"}


async def test_a_namesake_moving_leaves_the_users_home_alone(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "namesake.db")
    try:
        [home] = await _seed(mem, "Maya Rao lives in Bengaluru.")
        llm.json_replies.extend([_cud("ADD", "Maya Singh moved to Mumbai.")] + [ANALYSIS] * 3)
        out = await mem.remember("Maya Singh moved to Mumbai.")
        assert out["action"] == "ADD"
        assert (await mem.get_fact(home))["content"] == "Maya Rao lives in Bengaluru."
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("content", "action"),
    [
        ("Her sister lives in Delhi.", "ADD"),  # whose sister is unclear: never stored over Maya's sister's home
        ("Maya's sister lives in Delhi.", "UPDATE"),  # the model named her: it is Maya's sister
    ],
)
async def test_the_model_must_name_whose_relation_it_updates(config, isolated_home, content, action):
    # CodeRabbit on #471: "Her sister" may be offered next to "Maya's sister", but stored over it only when named
    store, llm, mem = await _memory(config, isolated_home, "her.db")
    config.memory.duplicate_similarity = 1.01  # this embedder scores everything 1.0: no near-duplicate skip
    try:
        [sister] = await _seed(mem, "Maya's sister lives in Pune.")
        llm.json_replies.extend([_cud("UPDATE", content, sister)] + [ANALYSIS] * 3)
        out = await mem.remember("Her sister lives in Delhi.")
        assert out["action"] == action
        stored = (await mem.get_fact(sister))["content"]
        assert stored == (content if action == "UPDATE" else "Maya's sister lives in Pune.")
    finally:
        await store.close()
