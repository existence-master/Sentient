from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sentient.memory.dreaming import DreamingService, covers, parse_hhmm
from sentient.memory.facts import (
    _FAMILY_PROTOTYPES,  # white-box: the real prototype text per family
    attribute_families,
    embedding_attribute_families,
    fact_subjects,
    profile_fact,
    profile_fact_async,
    subject_aliases,
)
from tests.memory.helpers import add_fact, drain


def _json_calls(llm) -> int:
    return sum(1 for c in llm.calls if c.get("json"))


def test_helpers():
    assert parse_hhmm("03:00") == (3, 0) and parse_hhmm("7:45") == (7, 45) and parse_hhmm("bad") == (3, 0)
    assert covers("Maya plays chess with Riya", ["Maya plays chess"])
    assert not covers("Maya plays chess", ["Maya plays chess with Riya"])
    assert DreamingService.contradiction_candidate("Maya lives in Pune", "Maya lives in Mumbai")
    assert DreamingService.contradiction_candidate("Maya eats meat", "Maya no longer eats meat")
    assert not DreamingService.contradiction_candidate("Maya lives in Pune", "Riya lives in Mumbai")
    assert not DreamingService.contradiction_candidate("Maya's sister lives in Pune", "Maya lives in Mumbai")
    assert DreamingService.contradiction_candidate("I live in Pune", "Maya moved to Bengaluru", "Maya")


def test_subjects_and_families():
    assert fact_subjects("Maya moved to Bengaluru last month") == {"maya"}
    assert fact_subjects("Maya's sister Riya works at Infosys") == {"maya's sister", "riya"}
    assert fact_subjects("Every morning Maya drinks black coffee") == {"maya"}
    assert fact_subjects("My manager is Jane", "Maya") == {"maya's manager"}
    assert fact_subjects("The user lives in Pune", "Maya") == {"maya"}
    assert fact_subjects("likes tea") == set()
    assert attribute_families("Riya got a job at Google") == {"job"}
    assert attribute_families("Maya started eating fish") == {"diet"}
    assert "residence" in attribute_families("Maya moved to Bengaluru last month")


def test_subject_in_non_english_script():
    # fact_subjects only reads capital letters (Devanagari has none), so it finds nothing here - a Devanagari
    # fact's subject comes from scripts.devanagari_subjects instead, via subject_aliases (and profile_fact,
    # which dreaming's contradiction grouping actually uses) (#95).
    assert fact_subjects("माया पुणे में रहती है") == set()  # "Maya lives in Pune"
    assert subject_aliases("माया पुणे में रहती है") == {"माया"}
    assert subject_aliases("माया बेंगलुरु चली गई") == {"माया"}  # "Maya moved to Bengaluru"
    assert profile_fact("माया पुणे में रहती है").subjects == {"माया"}


def test_subject_past_a_long_leading_clause():
    # "March" and "wedding" used to be mistaken for (or block) the subject: "March" is capitalized but is a
    # month, not a name, and the old code gave up entirely on the first lowercase, unrecognised word (#95).
    assert fact_subjects("Last year in March, after the wedding, Riya moved to Mumbai.") == {"riya"}
    # with no leading clause (no comma), an unrecognised word still gives up as before - no guessing
    assert fact_subjects("She moved to Mumbai.") == set()


def test_subject_never_a_leading_number_or_a_place_after_a_preposition():
    # Found in review of #95: scanning past a leading clause must not pick up the clause's own words as the
    # subject. A bare year is never name-like (no capital letter), and a capitalized word straight after a
    # preposition is a place or a time, not a person - both differ only by the actual subject's name, or the
    # old code paired the two facts below as a contradiction about the same "person".
    assert fact_subjects("In 2021 Kabir moved to Delhi.") == {"kabir"}
    assert fact_subjects("In 2021 Karan moved to Goa.") == {"karan"}
    assert fact_subjects("After the trip to Goa, Kabir started working at Infosys.") == {"kabir"}
    assert fact_subjects("After the trip to Goa, Karan started working at Wipro.") == {"karan"}


async def test_embedding_attribute_families_matches_the_closest_prototype():
    # Distinct, non-orthogonal vectors per family (orthogonal ones would make every margin equally wide,
    # which isn't how a real embedding model's scores for related-but-wrong families behave).
    names = list(_FAMILY_PROTOTYPES)
    vectors = {name: [1.0, float(n)] for n, name in enumerate(names)}  # residence=[1,0], job=[1,1], ...
    fact_text = "माया बेंगलुरु चली गई"  # "Maya moved to Bengaluru"

    async def embed_one(text: str) -> list[float]:
        for name, proto in _FAMILY_PROTOTYPES.items():
            if text == proto:
                return vectors[name]
        return [1.0, 0.05]  # the fact text itself: closest to "residence" ([1, 0]), next closest a bit further

    assert await embedding_attribute_families(fact_text, embed_one, "test-model", margin=0.01) == {"residence"}
    # a wider margin than the real gap between the top two scores: neither stands out enough to call it a match
    assert await embedding_attribute_families(fact_text, embed_one, "test-model", margin=0.5) == set()


async def test_profile_fact_async_only_falls_back_when_keywords_find_nothing():
    calls: list[str] = []

    async def embed_one(text: str) -> list[float]:
        calls.append(text)
        for n, (_name, proto) in enumerate(_FAMILY_PROTOTYPES.items()):
            if text == proto:
                return [1.0, float(n)]
        return [1.0, 0.0]  # the fact text itself: lands on "residence" ([1, 0], the first prototype)

    # English text: the keyword family already matches, so the embedder is never called
    prof = await profile_fact_async("Maya moved to Bengaluru", embed_one, "Maya", "test-model")
    assert prof.families == {"residence"} and calls == []

    # non-English text with no keyword match: falls back, calling the embedder for the
    # prototypes (once, then cached) plus the fact text
    prof = await profile_fact_async("माया बेंगलुरु चली गई", embed_one, "Maya", "test-model-2", margin=0.01)
    assert calls  # the fallback ran
    assert prof.families == {"residence"}
    calls.clear()

    # a stored vector of the right dimension is reused instead of embedding the text again
    prof = await profile_fact_async(
        "माया बेंगलुरु चली गई", embed_one, "Maya", "test-model-2", margin=0.01, vector=[1.0, 0.0]
    )
    assert calls == []  # only the (already-cached) prototypes would have needed embedding, and nothing did
    assert prof.families == {"residence"}


async def test_profile_fact_async_falls_back_to_the_keyword_profile_when_embedding_fails():
    async def broken(text: str) -> list[float]:
        raise RuntimeError("embedding provider unavailable")

    prof = await profile_fact_async("माया बेंगलुरु चली गई", broken, "Maya", "test-model-3", margin=0.01)
    assert prof.families == set()  # no exception, and no family guessed


EXAMPLES = [
    ("Maya lives in Pune", "Maya moved to Bengaluru last month"),
    ("Maya's sister Riya works at Infosys", "Riya got a job at Google"),
    ("Maya is vegetarian", "Maya started eating fish"),
]


async def superseded(mem, fid: int, *, by: int) -> bool:
    """A contradiction's loser stays as history (#152): superseded by the winner, and recall never finds it."""
    old, new = await mem.get_fact(fid), await mem.get_fact(by)
    hits = await mem.recall(old["content"], top_k=10, min_similarity=0.0, track=False)
    return (old["state"] == "superseded" and old["superseded_by"] == by and new["state"] == "current"
            and fid not in [h["id"] for h in hits])


@pytest.mark.parametrize(("old_text", "new_text"), EXAMPLES)
async def test_contradictions_found_without_shared_wording(app, old_text, new_text):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    app.config.dreaming.contradiction_similarity = 0.99  # tight families are checked regardless of similarity
    old = await add_fact(mem, old_text, updated_at="2025-01-01T00:00:00+00:00")
    new = await add_fact(mem, new_text)
    await add_fact(mem, "Maya's manager is Jane")
    llm.json_replies.append({"conflict": True, "current": "B", "change": "it changed"})
    dream = await app.dreaming.run()
    assert dream["stats"]["contradictions_resolved"] == 1
    assert await superseded(mem, old, by=new)  # kept as history, out of every prompt (#152)
    assert _json_calls(llm) == 1
    prompt = llm.calls[-1]["messages"][1]["content"]
    assert prompt.index(old_text) < prompt.index(new_text)  # A is the older memory


async def test_older_fact_wins_only_with_explicit_recent_wording(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    older = await add_fact(mem, "Maya moved back to Pune this month", updated_at="2026-09-01T00:00:00+00:00")
    stale = await add_fact(mem, "Maya lives in Bengaluru")  # e.g. re-imported from an old document
    llm.json_replies.append({"conflict": True, "current": "A"})
    await app.dreaming.run()
    assert await superseded(mem, stale, by=older)

    a = await add_fact(mem, "Riya works at Infosys", updated_at="2025-01-01T00:00:00+00:00")
    b = await add_fact(mem, "Riya works at Google now")
    llm.json_replies.append({"conflict": True, "current": "A"})  # no recent wording on A: newer still wins
    await app.dreaming.run()
    assert await superseded(mem, a, by=b)


async def test_pure_duplicates_merge_without_model(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    a = await add_fact(mem, "Maya drinks black coffee every morning", topics=["Personal Identity"])
    b = await add_fact(mem, "Every morning Maya drinks black coffee.", topics=["Interests & Lifestyle"])
    await add_fact(mem, "Maya's manager is Jane")
    dream = await app.dreaming.run()
    assert dream["status"] == "completed" and dream["trigger"] == "manual"
    assert dream["stats"]["merged"] == 1 and dream["stats"]["facts_reviewed"] == 3
    assert _json_calls(llm) == 0
    kept = await mem.get_fact(a)
    assert kept is not None and await mem.get_fact(b) is None
    assert set(kept["topics"]) == {"Personal Identity", "Interests & Lifestyle"}
    assert dream["journal_md"].startswith("Tonight I merged 2 duplicate memories into 1")


async def test_model_merge_keeps_details_and_ids(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    app.config.dreaming.merge_similarity = 0.75
    a = await add_fact(mem, "Maya plays chess every weekend")
    b = await add_fact(mem, "Maya plays chess every weekend with Riya")
    llm.json_replies.append({"keep_id": a, "fact": "Maya plays chess with Riya every weekend."})
    dream = await app.dreaming.run()
    assert dream["stats"]["merged"] == 1
    kept = await mem.get_fact(a)
    assert kept["content"] == "Maya plays chess with Riya every weekend."
    assert kept["previous_content"] == "Maya plays chess every weekend"
    assert await mem.get_fact(b) is None
    assert "Maya plays chess" in llm.calls[-1]["messages"][1]["content"]


async def test_model_merge_that_drops_a_detail_is_rejected(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    app.config.dreaming.merge_similarity = 0.75
    a = await add_fact(mem, "Maya plays chess every weekend")
    b = await add_fact(mem, "Maya plays chess every weekend with Riya")
    llm.json_replies.append({"keep_id": b, "fact": "Maya plays chess every weekend."})  # lost "Riya"
    await app.dreaming.run()
    # fallback: the fact that already carries every detail wins, the oldest id is kept
    kept = await mem.get_fact(a)
    assert kept["content"] == "Maya plays chess every weekend with Riya" and await mem.get_fact(b) is None


async def test_contradiction_latest_wins_after_model_confirms(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    app.config.dreaming.contradiction_similarity = 0.5
    old = await add_fact(mem, "Maya lives in Pune", updated_at="2025-01-01T00:00:00+00:00")
    new = await add_fact(mem, "Maya lives in Mumbai")
    llm.json_replies.append({"conflict": "true", "change": "moved from Pune to Mumbai"})
    async with app.bus.subscribe() as q:
        dream = await app.dreaming.run()
        events = drain(q)
    assert dream["stats"]["contradictions_resolved"] == 1
    assert await superseded(mem, old, by=new)
    assert (await mem.get_fact(new))["previous_content"] == "Maya lives in Pune"
    assert 'I used to remember "Maya lives in Pune", but now "Maya lives in Mumbai"' in dream["journal_md"]
    assert "moved from Pune to Mumbai" in dream["journal_md"]
    assert [e["data"]["status"] for e in events if e["type"] == "dream.updated"] == ["running", "completed"]
    assert any(e["type"] == "memory.updated" and e["data"].get("reason") == "contradicted" for e in events)
    notes = await app.store.fetchall("SELECT kind, payload FROM notifications")
    assert [n["kind"] for n in notes] == ["info"] and dream["id"] in notes[0]["payload"]


async def test_contradiction_not_confirmed_or_malformed_keeps_both(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    app.config.dreaming.contradiction_similarity = 0.5
    a = await add_fact(mem, "Maya lives in Pune", updated_at="2025-01-01T00:00:00+00:00")
    b = await add_fact(mem, "Maya lives in Mumbai")
    llm.json_replies.append("garbage")
    await app.dreaming.run()
    assert _json_calls(llm) == 1
    llm.json_replies.append({"conflict": False})  # malformed answers are asked again
    dream = await app.dreaming.run()
    assert dream["stats"]["contradictions_resolved"] == 0 and _json_calls(llm) == 2
    assert await mem.get_fact(a) and await mem.get_fact(b)
    # nothing meaningful happened: no notification
    assert await app.store.fetchall("SELECT id FROM notifications") == []
    # a pair the model cleared is not asked about again while neither fact changes
    await app.dreaming.run()
    assert _json_calls(llm) == 2
    await mem.update_content(b, "Maya lives in Mumbai near Bandra", use_llm=False)
    app.config.dreaming.max_model_calls = 0  # and the cap holds
    await app.dreaming.run()
    assert _json_calls(llm) == 2


async def test_promotion_purge_user_model_and_journal(app):
    mem, llm, store = app.memory, app.fake, app.store
    app.config.dreaming.promote_min_recalls = 2
    hot = await add_fact(mem, "Maya is preparing for a marathon in December", memory_type="short-term")
    await store.execute(
        "UPDATE facts SET expires_at = ?, recall_count = 3 WHERE id = ?",
        ((datetime.now(UTC) + timedelta(days=2)).isoformat(), hot),
    )
    stale = await add_fact(mem, "Maya has a dentist appointment on Friday", memory_type="short-term")
    await store.execute("UPDATE facts SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (stale,))
    llm.json_replies.append(
        {"operations": [{"op": "add", "dimension": "goals", "statement": "Maya trains for endurance events.",
                         "evidence": [f"f{hot}"]}], "summary": "Maya is training hard."}
    )
    dream = await app.dreaming.run()
    s = dream["stats"]
    assert (s["promoted"], s["expired"], s["insights_updated"]) == (1, 1, 1)
    fact = await mem.get_fact(hot)
    assert fact["memory_type"] == "long-term" and fact["expires_at"] is None
    assert await mem.get_fact(stale) is None
    journal = dream["journal_md"]
    assert journal.startswith("Tonight I kept 1 short-term memory for the long run")
    assert "let go of 1 expired memory" in journal and "keeps coming up" in journal
    assert "My picture of Maya: 1 new." in journal
    assert (await app.user_model.get_state())["insights"][0]["statement"] == "Maya trains for endurance events."
    listed = await app.dreaming.list()
    assert listed[0]["id"] == dream["id"] and (await app.dreaming.get(dream["id"]))["journal_md"] == journal


async def test_recall_counts_feed_promotion(app):
    mem = app.memory
    fid = await add_fact(mem, "Maya is reading Dune this month", memory_type="short-term")
    for _ in range(3):
        await mem.recall("reading Dune", min_similarity=0.0)
    promoted = await mem.promote_recalled(3)
    assert [p["id"] for p in promoted] == [fid]


async def test_schedule_time_idle_and_once_per_day(app):
    dreaming, cfg = app.dreaming, app.config.dreaming
    app.config.assistant.timezone = "UTC"
    cfg.time, cfg.require_idle_minutes = "03:00", 30
    day = datetime(2026, 9, 15, tzinfo=UTC)
    dreaming.clock = lambda: day + timedelta(hours=2)
    assert await dreaming.is_due() == (False, "not yet time")
    assert await dreaming.tick() is None

    now = day + timedelta(hours=3, minutes=10)
    dreaming.note_activity(now - timedelta(minutes=5))
    assert await dreaming.is_due(now) == (False, "user is active")
    dreaming.note_activity(now - timedelta(hours=2))
    assert await dreaming.is_due(now) == (True, "due")
    dream = await dreaming.tick(now)
    assert dream is not None and dream["trigger"] == "schedule" and dream["status"] == "completed"
    assert dream["journal_md"].startswith("Tonight I looked over 0 memories")
    assert await dreaming.is_due(now + timedelta(hours=5)) == (False, "already dreamed today")
    assert await dreaming.tick(now + timedelta(hours=5)) is None

    # the next night, a recent user message counts as activity too
    tomorrow = now + timedelta(days=1)
    sid = await app.store.create_session()
    await app.store.add_message(sid, "user", "still up?")
    await app.store.execute("UPDATE messages SET created_at = ?", ((tomorrow - timedelta(minutes=10)).isoformat(),))
    assert await dreaming.is_due(tomorrow) == (False, "user is active")
    assert await dreaming.is_due(tomorrow + timedelta(hours=1)) == (True, "due")
    cfg.enabled = False
    assert await dreaming.is_due(tomorrow + timedelta(hours=1)) == (False, "disabled")


async def test_local_timezone_is_used(app):
    app.config.assistant.timezone = "Asia/Kolkata"  # UTC+5:30
    app.config.dreaming.time = "03:00"
    app.config.dreaming.require_idle_minutes = 0
    utc_2130 = datetime(2026, 9, 14, 21, 30, tzinfo=UTC)  # 03:00 in Kolkata on the 15th
    assert await app.dreaming.is_due(utc_2130 - timedelta(minutes=1)) == (False, "not yet time")
    assert await app.dreaming.is_due(utc_2130 + timedelta(minutes=1)) == (True, "due")
