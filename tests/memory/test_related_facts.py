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
    pune = await add_fact(mem, "Sarthak lives in Pune")
    await add_fact(mem, "Sarthak's sister Riya lives in Mumbai")
    await add_fact(mem, "Sarthak lived in Nagpur as a child")
    llm.json_replies.append(_reply("ADD", "Sarthak moved to Bengaluru last month."))
    out = await mem.remember("Sarthak moved to Bengaluru last month.")
    assert out == {"action": "UPDATE", "id": pune, "content": "Sarthak moved to Bengaluru last month."}
    shown = llm.calls[-1]["messages"][1]["content"]
    assert "Sarthak lives in Pune" in shown and "same_subject_attribute" in shown
    assert "Sarthak's sister Riya lives in Mumbai" not in shown
    assert (await mem.get_fact(pune))["previous_content"] == "Sarthak lives in Pune"
    contents = {f["content"] for f in await mem.list_facts()}
    assert contents == {
        "Sarthak moved to Bengaluru last month.", "Sarthak's sister Riya lives in Mumbai", "Sarthak lived in Nagpur as a child",
    }


async def test_past_residence_is_added_not_replacing_home(app):
    mem, llm = app.memory, app.fake
    app.config.memory.duplicate_similarity = 0.999
    await add_fact(mem, "Sarthak lives in Pune")
    llm.json_replies.append(_reply("ADD", "Sarthak lived in Delhi in 2015."))
    assert (await mem.remember("Sarthak lived in Delhi in 2015."))["action"] == "ADD"
    assert await mem.count() == 2


@pytest.mark.parametrize(
    ("old_text", "new_text"),
    [("Sarthak's sister Riya works at Infosys", "Riya got a job at Google"),
     ("Sarthak is vegetarian", "Sarthak started eating fish")],
)
async def test_model_can_update_related_fact_with_low_similarity(app, old_text, new_text):
    mem, llm = app.memory, app.fake
    app.config.memory.duplicate_similarity = 0.999
    old = await add_fact(mem, old_text)
    llm.json_replies.append(_reply("UPDATE", new_text, old))
    out = await mem.remember(new_text)
    assert out == {"action": "UPDATE", "id": old, "content": new_text}
    assert old_text in llm.calls[-1]["messages"][1]["content"]
    assert await mem.count() == 1
