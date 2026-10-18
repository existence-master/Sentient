"""#475: the memory tool saved "Priya, the user's sister, lives in Mumbai." as given. A saved fact says the user's
configured name instead of "the user", whichever way it is saved."""

from __future__ import annotations

import pytest

from sentient.memory.facts import FactMemory, name_the_user
from sentient.store.db import Store
from tests.conftest import FakeProvider

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
