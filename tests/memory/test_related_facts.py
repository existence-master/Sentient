"""The live CUD path shows the model facts about the same subject and attribute, even when their
embeddings are not similar, and a new home replaces the old one instead of being added alongside it."""

from __future__ import annotations

import pytest

from tests.memory.helpers import add_fact


def _reply(action: str, content: str, fact_id: int | None = None) -> dict:
    return {
        "action": action, "fact_id": fact_id, "content": content,
        "analysis": {"topics": ["Personal Identity"], "memory_type": "long-term", "duration": None},
    }


async def test_move_updates_residence_even_if_model_says_add(app):
    mem, llm = app.memory, app.fake
    app.config.memory.duplicate_similarity = 0.999
    pune = await add_fact(mem, "Maya lives in Pune")
    await add_fact(mem, "Maya's sister Riya lives in Mumbai")
    await add_fact(mem, "Maya lived in Nagpur as a child")
    llm.json_replies.append(_reply("ADD", "Maya moved to Bengaluru last month."))
    out = await mem.remember("Maya moved to Bengaluru last month.")
    assert out == {"action": "UPDATE", "id": pune, "content": "Maya moved to Bengaluru last month.",
                   "previous": "Maya lives in Pune"}
    # a new home for the same person is a correction (#364): decided without the model
    assert not any("memory management" in str(c) for c in llm.calls)
    assert (await mem.get_fact(pune))["previous_content"] == "Maya lives in Pune"
    contents = {f["content"] for f in await mem.list_facts()}
    assert contents == {
        "Maya moved to Bengaluru last month.", "Maya's sister Riya lives in Mumbai", "Maya lived in Nagpur as a child",
    }


async def test_past_residence_is_added_not_replacing_home(app):
    mem, llm = app.memory, app.fake
    app.config.memory.duplicate_similarity = 0.999
    await add_fact(mem, "Maya lives in Pune")
    llm.json_replies.append(_reply("ADD", "Maya lived in Delhi in 2015."))
    assert (await mem.remember("Maya lived in Delhi in 2015."))["action"] == "ADD"
    assert await mem.count() == 2


async def test_a_new_job_replaces_the_old_one(app):
    mem = app.memory
    app.config.memory.duplicate_similarity = 0.999
    old = await add_fact(mem, "Maya's sister Riya works at Infosys")
    out = await mem.remember("Riya got a job at Google")
    assert out == {"action": "UPDATE", "id": old, "content": "Riya got a job at Google",
                   "previous": "Maya's sister Riya works at Infosys"}
    assert await mem.count() == 1


async def test_a_new_home_is_offered_to_the_model_when_no_value_changed(app):
    mem, llm = app.memory, app.fake
    app.config.memory.duplicate_similarity = 0.999
    await add_fact(mem, "Maya lives in Pune")
    await add_fact(mem, "Maya's sister Riya lives in Mumbai")
    llm.json_replies.append(_reply("ADD", "Maya moved out of her flat."))
    await mem.remember("Maya moved out of her flat.")
    shown = llm.calls[-1]["messages"][1]["content"]
    assert "Maya lives in Pune" in shown and "same_subject_attribute" in shown
    assert "Maya's sister Riya lives in Mumbai" not in shown


@pytest.mark.parametrize(
    ("old_text", "new_text"),
    [("Maya is vegetarian", "Maya started eating fish")],
)
async def test_model_can_update_related_fact_with_low_similarity(app, old_text, new_text):
    mem, llm = app.memory, app.fake
    app.config.memory.duplicate_similarity = 0.999
    old = await add_fact(mem, old_text)
    llm.json_replies.append(_reply("UPDATE", new_text, old))
    out = await mem.remember(new_text)
    assert out == {"action": "UPDATE", "id": old, "content": new_text, "previous": old_text}
    assert old_text in llm.calls[-1]["messages"][1]["content"]
    assert await mem.count() == 1
