"""Superseded memories are history that expires (#441): ``memory.keep_superseded_days``, run by the nightly purge."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

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


@pytest.mark.parametrize("expiry", ["retention", "explicit"])
@pytest.mark.parametrize("preserve_unrelated", [False, True])
async def test_expiry_clears_only_the_matching_previous_content(app, expiry, preserve_unrelated):
    """Expired history must not remain on its successor or erase an unrelated prior version."""
    mem, store = app.memory, app.store
    old_text = "Maya lives in Pune"
    old, new = await _superseded_pair(
        app, old_text, "Maya lives in Bengaluru", age_days=181 if expiry == "retention" else 10
    )
    if preserve_unrelated:
        await store.execute("UPDATE facts SET previous_content = 'another earlier version' WHERE id = ?", (new,))
    if expiry == "explicit":
        await store.execute("UPDATE facts SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (old,))

    assert await mem.purge_expired() == 1

    assert await mem.get_fact(old) is None
    current = await mem.get_fact(new)
    assert current["content"] == "Maya lives in Bengaluru" and current["state"] == "current"
    assert current["previous_content"] == ("another earlier version" if preserve_unrelated else None)


async def test_moving_superseded_memory_preserves_retention_age(app):
    """Moving overdue history to another scope must not restart its retention period."""
    mem = app.memory
    old, _ = await _superseded_pair(app, "Maya prefers tea", "Maya prefers coffee", age_days=181)
    superseded_at = (await mem.get_fact(old))["updated_at"]

    moved = await mem.set_scope(old, "work")

    assert moved["scope"] == "work" and moved["updated_at"] == superseded_at
    assert await mem.purge_expired() == 1
    assert await mem.get_fact(old) is None


@pytest.mark.parametrize("state", ["current", "disputed"])
async def test_moving_other_memory_states_still_updates_timestamp(app, state):
    """The timestamp preservation exception must be limited to superseded history."""
    mem = app.memory
    fid = await add_fact(mem, "Maya uses a standing desk")
    await mem.set_state(fid, state)
    before = _ago(181)
    await app.store.execute("UPDATE facts SET updated_at = ? WHERE id = ?", (before, fid))

    moved = await mem.set_scope(fid, "work")

    assert moved["updated_at"] != before and moved["state"] == state
