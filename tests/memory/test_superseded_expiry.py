"""Superseded memories are history that expires (#441): ``memory.keep_superseded_days``, run by the nightly purge."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sentient.store.db import now_iso
from tests.memory.helpers import add_fact


def _ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


async def _superseded_pair(app, old_text: str, new_text: str, *, age_days: int) -> tuple[int, int]:
    mem = app.memory
    old, new = await add_fact(mem, old_text), await add_fact(mem, new_text)
    assert await mem.supersede(new, old)
    await app.store.execute("UPDATE facts SET updated_at = ? WHERE id = ?", (_ago(age_days), old))
    return old, new


async def test_old_superseded_memories_expire_and_current_ones_stay(app):
    mem = app.memory
    assert app.config.memory.keep_superseded_days == 180
    old, new = await _superseded_pair(app, "Maya lives in Pune", "Maya lives in Bengaluru", age_days=181)
    recent, newer = await _superseded_pair(app, "Maya works at Acme", "Maya works at Existence", age_days=10)
    disputed = await add_fact(mem, "Maya is left-handed")
    await mem.set_state(disputed, "disputed")
    await app.store.execute("UPDATE facts SET updated_at = ? WHERE id IN (?, ?, ?)", (_ago(400), new, newer, disputed))

    assert await mem.purge_expired() == 1

    assert await mem.get_fact(old) is None
    for fid in (new, recent, newer, disputed):
        assert await mem.get_fact(fid) is not None
    assert (await mem.get_fact(new))["state"] == "current"


async def test_zero_keeps_superseded_memories_forever(app):
    mem = app.memory
    app.config.memory.keep_superseded_days = 0
    old, _ = await _superseded_pair(app, "Maya lives in Pune", "Maya lives in Bengaluru", age_days=5000)
    assert await mem.purge_expired() == 0
    assert (await mem.get_fact(old))["state"] == "superseded"


async def test_explicit_expiry_still_purges_alongside(app):
    mem = app.memory
    short = await add_fact(mem, "Maya has a dentist appointment on Friday", memory_type="short-term")
    await app.store.execute("UPDATE facts SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (short,))
    old, _ = await _superseded_pair(app, "Maya lives in Pune", "Maya lives in Bengaluru", age_days=200)
    assert await mem.purge_expired() == 2
    assert await mem.get_fact(short) is None and await mem.get_fact(old) is None


async def test_expired_superseded_fact_uses_the_cascade_before_its_retention_limit(app):
    mem, store = app.memory, app.store
    old, new = await _superseded_pair(app, "Maya lives in Pune", "Maya lives in Bengaluru", age_days=10)
    await store.execute("UPDATE facts SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (old,))
    await store.execute(
        "INSERT INTO dreams(id, started_at, status, trigger, stats, journal_md) VALUES('d1', ?, 'completed', 'manual',"
        " '{}', ?)",
        (now_iso(), "- I used to remember Maya lives in Pune"),
    )
    insight = await app.user_model._insert(
        "Maya is from Pune", "personal", confidence=0.6, status="active", source="inferred",
        evidence=[{"kind": "fact", "ref": str(old), "quote": "Maya lives in Pune"}],
    )

    assert await mem.purge_expired() == 1

    assert await mem.get_fact(old) is None and await mem.get_fact(new) is not None
    assert await store.fetchall("SELECT id FROM user_insights WHERE id = ?", (insight["id"],)) == []
    journal = (await store.fetchone("SELECT journal_md FROM dreams WHERE id = 'd1'"))["journal_md"]
    assert "Pune" not in journal and "(a memory you deleted)" in journal


async def test_expiring_a_superseded_memory_cascades(app):
    mem, store = app.memory, app.store
    old, new = await _superseded_pair(app, "Maya lives in Pune", "Maya lives in Bengaluru", age_days=300)
    await mem.link_sources(old, ["file:cv.txt"])
    assert await store.fetchall("SELECT fact_id FROM fact_sources WHERE fact_id = ?", (old,))
    await store.execute(
        "INSERT INTO dreams(id, started_at, status, trigger, stats, journal_md) VALUES('d1', ?, 'completed', 'manual',"
        " '{}', ?)",
        (now_iso(), "Tonight I tidied up.\n- I used to remember \"Maya lives in Pune\"\n- Maya lives in Bengaluru"),
    )
    kept = await app.user_model._insert(
        "Maya has moved around", "personal", confidence=0.6, status="active", source="inferred",
        evidence=[{"kind": "fact", "ref": str(old), "quote": "x"}, {"kind": "fact", "ref": str(new), "quote": "y"}],
    )
    gone = await app.user_model._insert(
        "Maya is from Pune", "personal", confidence=0.6, status="active", source="inferred",
        evidence=[{"kind": "fact", "ref": str(old), "quote": "x"}],
    )

    assert await mem.purge_superseded() == 1

    assert await mem.get_fact(old) is None
    assert await store.fetchall("SELECT rowid FROM facts_vec WHERE rowid = ?", (old,)) == []
    assert await store.fetchall("SELECT fact_id FROM fact_sources WHERE fact_id = ?", (old,)) == []
    journal = (await store.fetchone("SELECT journal_md FROM dreams WHERE id = 'd1'"))["journal_md"]
    assert "Pune" not in journal and "Bengaluru" in journal and "(a memory you deleted)" in journal
    insights = {r["id"]: json.loads(r["evidence"]) for r in await store.fetchall("SELECT id, evidence FROM user_insights")}
    assert gone["id"] not in insights and [e["ref"] for e in insights[kept["id"]]] == [str(new)]
    assert await mem.get_fact(new) is not None
