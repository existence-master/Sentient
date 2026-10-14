"""The nightly tidy-up merges the same memory saved in two languages (#353), reversibly. Offline: FakeProvider."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.memory.languages import (
    details_covered,
    language_pairs,
    multilingual,
    numbers,
    same_words,
)
from tests.conftest import FakeProvider
from tests.memory.helpers import add_fact

SAME = {"a_in_english": "Maya is a vegetarian.", "same": True, "extra": "none", "conflict": False}


def _json_calls(llm) -> int:
    return sum(1 for c in llm.calls if c.get("json"))


def judge(llm) -> None:
    """The model says only the vegetarian pair is the same thing; every other pair differs."""

    async def complete_json(role, messages, *, model=None):
        llm.calls.append({"role": role, "messages": messages, "json": True})
        text = messages[-1]["content"]
        if "शाकाहारी" in text and "vegetarian" in text:
            return SAME
        return {"same": False, "extra": "A", "conflict": False}

    llm.complete_json = complete_json


def test_pair_rules():
    assert multilingual("ollama/bge-m3") and not multilingual("ollama/nomic-embed-text")
    assert numbers("माया की उम्र ३२ है") == {"32"} == numbers("Maya is 32")
    assert details_covered("Maya lives in Pune", "माया पुणे में रहती है")
    assert not details_covered("Maya lives in Pune", "माया मुंबई में रहती है")
    facts = [
        {"id": 1, "content": "माया शाकाहारी है"},
        {"id": 2, "content": "Maya is vegetarian"},
        {"id": 3, "content": "Riya is vegetarian"},  # someone else
        {"id": 4, "content": "Maya is 32"},  # a number the Hindi one lacks
        {"id": 5, "content": "likes tea"},  # no clear subject
        {"id": 6, "content": "Maya lives in Pune"},  # a place the Hindi one lacks
    ]
    assert [(i, j) for _s, i, j in language_pairs(facts, "Maya", None)] == [(0, 1)]
    assert same_words("Maya is a vegetarian.", "Maya is vegetarian")
    assert same_words("Maya likes tea", "Maya like tea")
    assert not same_words("Maya does yoga every morning", "Maya does yoga")
    assert not same_words("Maya likes tea", "Maya likes coffee")


async def test_merges_the_same_fact_in_two_languages_and_undo_brings_it_back(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    hindi = await add_fact(mem, "माया शाकाहारी है", source="conversation")
    english = await add_fact(mem, "Maya is vegetarian", source="file:profile.pdf")
    other = await add_fact(mem, "माया पुणे में रहती है")
    work = await add_fact(mem, "Maya works at Lumen Labs")
    runs = await add_fact(mem, "Maya runs on Sundays")
    judge(llm)
    dream = await app.dreaming.run()
    assert dream["stats"]["merged"] == 1 and dream["stats"]["languages_merged"] == 1
    assert "said the same thing in two languages" in dream["journal_md"]
    # most memories are in English, so the English wording stays; the Hindi one waits in the review history
    assert await mem.get_fact(english) is not None
    assert (await mem.get_fact(hindi))["status"] == "merged"
    assert all(f["id"] != hindi for f in await mem.list_facts())
    assert all(r["id"] != hindi for r in await mem.recall("माया शाकाहारी", min_similarity=0.0))
    assert {f["id"] for f in await mem.active_facts()} == {english, other, work, runs}
    asked = [c["messages"][1]["content"] for c in llm.calls if c.get("json")]
    assert 'A: "माया शाकाहारी है"\nB: "Maya is vegetarian"' in asked
    # facts naming a place or day the Hindi facts lack are never even asked about
    assert not any("Lumen" in a or "Sundays" in a for a in asked)
    [m] = await mem.merged_facts()
    assert m["id"] == hindi and m["merged_into"] == {"id": english, "text": "Maya is vegetarian"}
    assert m["source"] == "conversation" and m["by"] == "nightly tidying"

    # undo: it is back as it was, and the next tidy-up leaves the pair alone
    back = await mem.unmerge(hindi)
    assert back["status"] == "active" and back["content"] == "माया शाकाहारी है"
    assert await mem.merged_facts() == [] and await mem.unmerge(hindi) is None
    calls = _json_calls(llm)
    await app.dreaming.run()
    assert (await mem.get_fact(hindi))["status"] == "active" and _json_calls(llm) == calls


async def test_never_merges_when_one_adds_a_detail_or_they_conflict(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    await add_fact(mem, "माया शाकाहारी है")
    await add_fact(mem, "Maya is vegetarian")
    llm.json_replies.append({"same": False, "extra": "B", "conflict": False})
    dream = await app.dreaming.run()
    assert dream["stats"]["merged"] == 0 and len(await mem.active_facts()) == 2
    # a pair the model said differs is not asked again until one of them changes
    calls = _json_calls(llm)
    await app.dreaming.run()
    assert _json_calls(llm) == calls

    # a small model may call "every morning" no extra detail: its own translation gives it away
    every_morning = {"a_in_english": "Maya is vegetarian every morning", "same": True, "extra": "none", "conflict": False}
    for reply in ({**SAME, "extra": "A"}, {**SAME, "conflict": True}, every_morning, {**SAME, "a_in_english": ""},
                  {"same": "maybe"}, "not json"):
        await mem.store.execute("DELETE FROM meta WHERE key = 'dreaming.language_pairs'")
        llm.json_replies.append(reply)
        assert (await app.dreaming.run())["stats"]["merged"] == 0
    assert len(await mem.active_facts()) == 2


async def test_with_a_multilingual_embedding_model_only_close_pairs_are_asked(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    app.config.models.roles.embedding = "ollama/bge-m3"
    llm.model_for = lambda role: "ollama/bge-m3" if role == "embedding" else f"fake/{role}"
    app.config.dreaming.language_merge_similarity = 0.99  # the fake embeddings can't compare languages
    await add_fact(mem, "माया शाकाहारी है")
    await add_fact(mem, "Maya is vegetarian")
    dream = await app.dreaming.run()
    assert dream["stats"]["merged"] == 0 and _json_calls(llm) == 0


async def test_off_switch(app):
    app.config.user_model.enabled = False
    app.config.dreaming.merge_languages = False
    await add_fact(app.memory, "माया शाकाहारी है")
    await add_fact(app.memory, "Maya is vegetarian")
    await app.dreaming.run()
    assert _json_calls(app.fake) == 0


def test_review_history_and_undo_routes(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider()

    async def complete_json(role, messages, *, model=None):
        text = messages[-1]["content"]
        return SAME if "शाकाहारी" in text and "vegetarian" in text else {"facts": []}

    llm.complete_json = complete_json
    core = SentientApp(config, llm=llm, db_path=isolated_home / "routes.db", enable_background=False)
    core.config.user_model.enabled = False
    with TestClient(create_app(core)) as c:
        c.headers["Authorization"] = "Bearer test-token"
        hindi = c.post("/api/memories", json={"content": "माया शाकाहारी है"}).json()
        english = c.post("/api/memories", json={"content": "Maya is vegetarian"}).json()
        assert hindi.get("id") and english.get("id")
        c.post("/api/memories/dreams/run")
        for _ in range(200):
            if c.get("/api/memories/review").json().get("merged"):
                break
            time.sleep(0.02)
        review = c.get("/api/memories/review").json()
        assert review["count"] == 0
        [m] = review["merged"]
        assert m["merged_into"]["text"] in {"Maya is vegetarian", "माया शाकाहारी है"}
        assert c.post(f"/api/memories/review/merged/{m['id']}/undo").json()["ok"] is True
        assert c.get("/api/memories/review").json()["merged"] == []
        assert c.post(f"/api/memories/review/merged/{m['id']}/undo").status_code == 404
