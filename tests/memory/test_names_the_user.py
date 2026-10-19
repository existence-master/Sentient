"""#475: the memory tool saved "Priya, the user's sister, lives in Mumbai." as given. A saved fact says the user's
configured name instead of "the user", whichever way it is saved."""

from __future__ import annotations

import pytest

from sentient.memory.facts import FactMemory, name_the_user
from sentient.store.db import Store
from tests.conftest import FakeProvider
from tests.memory.helpers import add_fact

ANALYSIS = {"topics": ["Family"], "memory_type": "long-term", "duration": None}
USER = "Maya Rao"


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("Priya, the user's sister, lives in Mumbai.", "Priya, Maya Rao's sister, lives in Mumbai."),
        ("Priya, user's sister, lives in Mumbai.", "Priya, Maya Rao's sister, lives in Mumbai."),
        ("Priya, the user\u2019s sister, lives in Mumbai.", "Priya, Maya Rao's sister, lives in Mumbai."),
        ("The user moved to Bengaluru.", "Maya Rao moved to Bengaluru."),
        ("USERNAME's dog is called Biscuit.", "Maya Rao's dog is called Biscuit."),
        ("Maya's sister lives in Mumbai.", "Maya's sister lives in Mumbai."),
        ("Maya built the user-facing screen.", "Maya built the user-facing screen."),
        ("Maya fixed an end-user's bug.", "Maya fixed an end-user's bug."),
        ("User's sister lives in Mumbai.", "Maya Rao's sister lives in Mumbai."),
        ("Maya fixed the end user's laptop.", "Maya fixed the end user's laptop."),
        ("Maya reset another user's password.", "Maya reset another user's password."),
        ("Maya opened a user's account.", "Maya opened a user's account."),
        ("My GitHub username is maya-rao.", "My GitHub username is maya-rao."),
        ("Username: maya-rao.", "Username: maya-rao."),
        ("USERNAME moved to Pune.", "Maya Rao moved to Pune."),
        ("Maya works on the user research team.", "Maya works on the user research team."),
        ("Maya redesigned the user interface.", "Maya redesigned the user interface."),
        ("The user experience got better after Maya's fix.", "The user experience got better after Maya's fix."),
        ("Maya wrote the user stories for checkout.", "Maya wrote the user stories for checkout."),
        ("The user agent is Chrome.", "The user agent is Chrome."),
        ("Maya forgot the user name for the router.", "Maya forgot the user name for the router."),
        ("The user ID for the portal is 4512.", "The user ID for the portal is 4512."),
        ("Maya locked the user account.", "Maya locked the user account."),
        ("The user researched flights to Goa.", "Maya Rao researched flights to Goa."),
        ("The user named her dog Biscuit.", "Maya Rao named her dog Biscuit."),
        ("Priya met the user at work.", "Priya met Maya Rao at work."),
    ],
)
def test_the_user_becomes_the_users_name(text, named):
    assert name_the_user(text, USER) == named


def test_without_a_name_the_words_stay():
    assert name_the_user("Priya, the user's sister, lives in Mumbai.", " ") == "Priya, the user's sister, lives in Mumbai."


async def test_a_saved_fact_says_the_users_name(config, isolated_home):
    store = await Store(isolated_home / "named.db").open()
    config.assistant.user_name = USER
    mem = FactMemory(store, FakeProvider(json_replies=[ANALYSIS] * 3), config)
    try:
        out = await mem.remember("Priya, the user's sister, lives in Mumbai.")
        assert out["action"] == "ADD"
        assert (await mem.get_fact(out["id"]))["content"] == "Priya, Maya Rao's sister, lives in Mumbai."
    finally:
        await store.close()


class SameVectorProvider(FakeProvider):
    """Every text embeds the same, so a saved fact is always offered to the CUD model next to the new one."""

    async def embed(self, texts, *, model=None):
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


async def _memory(config, isolated_home, name):
    store = await Store(isolated_home / name).open()
    config.assistant.user_name = USER
    config.memory.duplicate_similarity = 1.01  # this embedder scores everything 1.0: no near-duplicate skip
    llm = SameVectorProvider()
    return store, llm, FactMemory(store, llm, config)


def _cud(action: str, content: str, fact_id: int | None = None) -> dict:
    return {"action": action, "fact_id": fact_id, "content": content, "analysis": ANALYSIS}


async def test_the_models_added_wording_says_the_users_name(config, isolated_home):
    # the CUD model may hand back its own wording for a new fact, with "the user" in it
    store, llm, mem = await _memory(config, isolated_home, "add.db")
    try:
        llm.json_replies.extend([_cud("ADD", "Priya, the user's sister, lives in Mumbai.")] + [ANALYSIS] * 3)
        out = await mem.remember("Priya lives in Mumbai.")
        assert out["action"] == "ADD"
        assert (await mem.get_fact(out["id"]))["content"] == "Priya, Maya Rao's sister, lives in Mumbai."
    finally:
        await store.close()


async def test_the_models_updated_wording_says_the_users_name(config, isolated_home):
    # the model, not the correction shortcut, decides this update ("Her sister" names no one), and writes "the user"
    store, llm, mem = await _memory(config, isolated_home, "update.db")
    try:
        sister = await add_fact(mem, "Maya's sister lives in Pune.")
        llm.json_replies.extend([_cud("UPDATE", "The user's sister lives in Delhi.", sister)] + [ANALYSIS] * 3)
        out = await mem.remember("Her sister lives in Delhi.")
        assert out["action"] == "UPDATE" and out["id"] == sister
        assert (await mem.get_fact(sister))["content"] == "Maya Rao's sister lives in Delhi."
        assert any("memory management" in str(c) for c in llm.calls)  # the CUD model was asked
    finally:
        await store.close()
