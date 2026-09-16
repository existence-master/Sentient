from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sentient.memory.dreaming import DreamingService, covers, parse_hhmm
from sentient.memory.facts import attribute_families, fact_subjects
from tests.memory.helpers import add_fact, drain


def _json_calls(llm) -> int:
    return sum(1 for c in llm.calls if c.get("json"))


def test_helpers():
    assert parse_hhmm("03:00") == (3, 0) and parse_hhmm("7:45") == (7, 45) and parse_hhmm("bad") == (3, 0)
    assert covers("Sarthak plays chess with Riya", ["Sarthak plays chess"])
    assert not covers("Sarthak plays chess", ["Sarthak plays chess with Riya"])
    assert DreamingService.contradiction_candidate("Sarthak lives in Pune", "Sarthak lives in Mumbai")
    assert DreamingService.contradiction_candidate("Sarthak eats meat", "Sarthak no longer eats meat")
    assert not DreamingService.contradiction_candidate("Sarthak lives in Pune", "Riya lives in Mumbai")
    assert not DreamingService.contradiction_candidate("Sarthak's sister lives in Pune", "Sarthak lives in Mumbai")
    assert DreamingService.contradiction_candidate("I live in Pune", "Sarthak moved to Bengaluru", "Sarthak")


def test_subjects_and_families():
    assert fact_subjects("Sarthak moved to Bengaluru last month") == {"sarthak"}
    assert fact_subjects("Sarthak's sister Riya works at Infosys") == {"sarthak's sister", "riya"}
    assert fact_subjects("Every morning Sarthak drinks black coffee") == {"sarthak"}
    assert fact_subjects("My manager is Jane", "Sarthak") == {"sarthak's manager"}
    assert fact_subjects("The user lives in Pune", "Sarthak") == {"sarthak"}
    assert fact_subjects("likes tea") == set()
    assert attribute_families("Riya got a job at Google") == {"job"}
    assert attribute_families("Sarthak started eating fish") == {"diet"}
    assert "residence" in attribute_families("Sarthak moved to Bengaluru last month")


EXAMPLES = [
    ("Sarthak lives in Pune", "Sarthak moved to Bengaluru last month"),
    ("Sarthak's sister Riya works at Infosys", "Riya got a job at Google"),
    ("Sarthak is vegetarian", "Sarthak started eating fish"),
]


@pytest.mark.parametrize(("old_text", "new_text"), EXAMPLES)
async def test_contradictions_found_without_shared_wording(app, old_text, new_text):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    app.config.dreaming.contradiction_similarity = 0.99  # tight families are checked regardless of similarity
    old = await add_fact(mem, old_text, updated_at="2025-01-01T00:00:00+00:00")
    new = await add_fact(mem, new_text)
    await add_fact(mem, "Sarthak's manager is Jane")
    llm.json_replies.append({"conflict": True, "current": "B", "change": "it changed"})
    dream = await app.dreaming.run()
    assert dream["stats"]["contradictions_resolved"] == 1
    assert await mem.get_fact(old) is None and await mem.get_fact(new) is not None
    assert _json_calls(llm) == 1
    prompt = llm.calls[-1]["messages"][1]["content"]
    assert prompt.index(old_text) < prompt.index(new_text)  # A is the older memory


async def test_older_fact_wins_only_with_explicit_recent_wording(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    older = await add_fact(mem, "Sarthak moved back to Pune this month", updated_at="2026-09-01T00:00:00+00:00")
    stale = await add_fact(mem, "Sarthak lives in Bengaluru")  # e.g. re-imported from an old document
    llm.json_replies.append({"conflict": True, "current": "A"})
    await app.dreaming.run()
    assert await mem.get_fact(stale) is None and await mem.get_fact(older) is not None

    a = await add_fact(mem, "Riya works at Infosys", updated_at="2025-01-01T00:00:00+00:00")
    b = await add_fact(mem, "Riya works at Google now")
    llm.json_replies.append({"conflict": True, "current": "A"})  # no recent wording on A: newer still wins
    await app.dreaming.run()
    assert await mem.get_fact(a) is None and await mem.get_fact(b) is not None


async def test_pure_duplicates_merge_without_model(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    a = await add_fact(mem, "Sarthak drinks black coffee every morning", topics=["Personal Identity"])
    b = await add_fact(mem, "Every morning Sarthak drinks black coffee.", topics=["Interests & Lifestyle"])
    await add_fact(mem, "Sarthak's manager is Jane")
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
    a = await add_fact(mem, "Sarthak plays chess every weekend")
    b = await add_fact(mem, "Sarthak plays chess every weekend with Riya")
    llm.json_replies.append({"keep_id": a, "fact": "Sarthak plays chess with Riya every weekend."})
    dream = await app.dreaming.run()
    assert dream["stats"]["merged"] == 1
    kept = await mem.get_fact(a)
    assert kept["content"] == "Sarthak plays chess with Riya every weekend."
    assert kept["previous_content"] == "Sarthak plays chess every weekend"
    assert await mem.get_fact(b) is None
    assert "Sarthak plays chess" in llm.calls[-1]["messages"][1]["content"]


async def test_model_merge_that_drops_a_detail_is_rejected(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    app.config.dreaming.merge_similarity = 0.75
    a = await add_fact(mem, "Sarthak plays chess every weekend")
    b = await add_fact(mem, "Sarthak plays chess every weekend with Riya")
    llm.json_replies.append({"keep_id": b, "fact": "Sarthak plays chess every weekend."})  # lost "Riya"
    await app.dreaming.run()
    # fallback: the fact that already carries every detail wins, the oldest id is kept
    kept = await mem.get_fact(a)
    assert kept["content"] == "Sarthak plays chess every weekend with Riya" and await mem.get_fact(b) is None


async def test_contradiction_latest_wins_after_model_confirms(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    app.config.dreaming.contradiction_similarity = 0.5
    old = await add_fact(mem, "Sarthak lives in Pune", updated_at="2025-01-01T00:00:00+00:00")
    new = await add_fact(mem, "Sarthak lives in Mumbai")
    llm.json_replies.append({"conflict": "true", "change": "moved from Pune to Mumbai"})
    async with app.bus.subscribe() as q:
        dream = await app.dreaming.run()
        events = drain(q)
    assert dream["stats"]["contradictions_resolved"] == 1
    assert await mem.get_fact(old) is None
    assert (await mem.get_fact(new))["previous_content"] == "Sarthak lives in Pune"
    assert 'I used to remember "Sarthak lives in Pune", but now "Sarthak lives in Mumbai"' in dream["journal_md"]
    assert "moved from Pune to Mumbai" in dream["journal_md"]
    assert [e["data"]["status"] for e in events if e["type"] == "dream.updated"] == ["running", "completed"]
    assert any(e["type"] == "memory.updated" and e["data"].get("reason") == "contradicted" for e in events)
    notes = await app.store.fetchall("SELECT kind, payload FROM notifications")
    assert [n["kind"] for n in notes] == ["info"] and dream["id"] in notes[0]["payload"]


async def test_contradiction_not_confirmed_or_malformed_keeps_both(app):
    mem, llm = app.memory, app.fake
    app.config.user_model.enabled = False
    app.config.dreaming.contradiction_similarity = 0.5
    a = await add_fact(mem, "Sarthak lives in Pune", updated_at="2025-01-01T00:00:00+00:00")
    b = await add_fact(mem, "Sarthak lives in Mumbai")
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
    await mem.update_content(b, "Sarthak lives in Mumbai near Bandra", use_llm=False)
    app.config.dreaming.max_model_calls = 0  # and the cap holds
    await app.dreaming.run()
    assert _json_calls(llm) == 2


async def test_promotion_purge_user_model_and_journal(app):
    mem, llm, store = app.memory, app.fake, app.store
    app.config.dreaming.promote_min_recalls = 2
    hot = await add_fact(mem, "Sarthak is preparing for a marathon in December", memory_type="short-term")
    await store.execute(
        "UPDATE facts SET expires_at = ?, recall_count = 3 WHERE id = ?",
        ((datetime.now(UTC) + timedelta(days=2)).isoformat(), hot),
    )
    stale = await add_fact(mem, "Sarthak has a dentist appointment on Friday", memory_type="short-term")
    await store.execute("UPDATE facts SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (stale,))
    llm.json_replies.append(
        {"operations": [{"op": "add", "dimension": "goals", "statement": "Sarthak trains for endurance events.",
                         "evidence": [f"f{hot}"]}], "summary": "Sarthak is training hard."}
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
    assert "My picture of Sarthak: 1 new." in journal
    assert (await app.user_model.get_state())["insights"][0]["statement"] == "Sarthak trains for endurance events."
    listed = await app.dreaming.list()
    assert listed[0]["id"] == dream["id"] and (await app.dreaming.get(dream["id"]))["journal_md"] == journal


async def test_recall_counts_feed_promotion(app):
    mem = app.memory
    fid = await add_fact(mem, "Sarthak is reading Dune this month", memory_type="short-term")
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
