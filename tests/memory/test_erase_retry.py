"""A cleanup step that fails during an erase is retried until it succeeds (#462).

`erase()` deletes the memory itself first and then removes everything derived
from it. One failing step (a locked file, a transient database error) used to
drop the rest: the derived data stayed forever and nothing ever tried again.
The unfinished cleanup is now recorded in `pending_erase` and replayed by
memory upkeep (`drain_pending_erases`) until it succeeds.
"""

from __future__ import annotations

import json

import pytest

from sentient.store.db import now_iso

HONDA = "Maya drives a red Honda City"


async def test_a_cleanup_step_that_fails_once_is_retried_by_the_next_upkeep(app, monkeypatch):
    mem, store = app.memory, app.store
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "I drive a red Honda City to work")
    first = await mem.remember(HONDA, use_llm=False, session_id=tg)
    honda = int(first["id"])
    summary = await mem.episodic.add_summary(
        tg, "Maya drives a red Honda City to the office.", await store.recent_messages(tg, 10)
    )

    real_cascade = mem._cascade
    calls = []

    async def fails_once(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return await real_cascade(*args, **kwargs)

    monkeypatch.setattr(mem, "_cascade", fails_once)

    # The delete itself still succeeds: the memory row goes, the failing cleanup is recorded.
    assert await mem.erase(honda) is True
    assert len(calls) == 1
    assert await mem.get_fact(honda) is None
    assert await store.fetchone("SELECT id FROM summaries WHERE id = ?", (summary["id"],))
    pending = await store.fetchall("SELECT id, fact_id, payload FROM pending_erase")
    assert len(pending) == 1 and int(pending[0]["fact_id"]) == honda
    payload = json.loads(pending[0]["payload"])
    assert honda in payload["ids"] and HONDA in payload["texts"]

    # The next upkeep run drains the queue: the retry replays the same cleanup,
    # the derived data is gone and the wait is cleared.
    # We simulate the hourly upkeep path (purge_expired + drain_pending_erases).
    await app.memory.purge_expired()
    assert await app.memory.drain_pending_erases() == 1
    assert len(calls) == 2
    assert await store.fetchone("SELECT id FROM summaries WHERE id = ?", (summary["id"],)) is None
    assert not await store.fetchall("SELECT id FROM pending_erase")


async def test_a_cleanup_that_keeps_failing_is_kept_and_retried_next_time(app, monkeypatch):
    mem, store = app.memory, app.store
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "I drive a red Honda City to work")
    first = await mem.remember(HONDA, use_llm=False, session_id=tg)
    honda = int(first["id"])

    async def always_fails(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mem, "_cascade", always_fails)
    assert await mem.erase(honda) is True
    assert await mem.drain_pending_erases() == 0
    assert len(await store.fetchall("SELECT id FROM pending_erase")) == 1  # kept for the next run

    monkeypatch.undo()
    assert await mem.drain_pending_erases() == 1
    assert not await store.fetchall("SELECT id FROM pending_erase")


async def test_a_clean_erase_records_nothing(app):
    mem, store = app.memory, app.store
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "I drive a red Honda City to work")
    first = await mem.remember(HONDA, use_llm=False, session_id=tg)
    honda = int(first["id"])
    assert await mem.erase(honda) is True
    assert not await store.fetchall("SELECT id FROM pending_erase")


async def test_upkeep_does_not_strip_a_memory_still_being_deleted(app, monkeypatch):
    """A drain that lands while the delete is still running must do nothing (#462)."""
    mem, store = app.memory, app.store
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "I drive a red Honda City to work")
    honda = int((await mem.remember(HONDA, use_llm=False, session_id=tg))["id"])
    summary = await mem.episodic.add_summary(
        tg, "Maya drives a red Honda City to the office.", await store.recent_messages(tg, 10)
    )

    real_forget = mem.forget

    async def forget_slow(fid, *, notify=False):
        # A concurrent upkeep tick lands while the delete is suspended.
        assert await mem.drain_pending_erases() == 0
        # The summary is still there: the wait is not complete yet.
        assert await store.fetchone("SELECT id FROM summaries WHERE id = ?", (summary["id"],))
        return await real_forget(fid, notify=notify)

    monkeypatch.setattr(mem, "forget", forget_slow)
    assert await mem.erase(honda) is True
    # Only the finished erase cleaned up: the summary is gone now.
    assert await store.fetchone("SELECT id FROM summaries WHERE id = ?", (summary["id"],)) is None
    assert not await store.fetchall("SELECT id FROM pending_erase")


async def test_a_summary_vector_whose_delete_failed_goes_on_the_retry(app, monkeypatch):
    """A vector whose delete failed once still goes on the retry (#462): its row ID is
    recorded in the pending payload before the summary row can disappear."""
    mem, store = app.memory, app.store
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "I drive a red Honda City to work")
    honda = int((await mem.remember(HONDA, use_llm=False, session_id=tg))["id"])
    summary = await mem.episodic.add_summary(
        tg, "Maya drives a red Honda City to the office.", await store.recent_messages(tg, 10)
    )

    real_delete = mem.episodic.vec.delete
    calls = []

    async def delete_once(rowid):
        calls.append(rowid)
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return await real_delete(rowid)

    monkeypatch.setattr(mem.episodic.vec, "delete", delete_once)
    assert await mem.erase(honda) is True  # the cleanup fails at the first vector delete
    # The vector went first, so the summary row survived the failed step.
    row = await store.fetchone("SELECT rowid FROM summaries WHERE id = ?", (summary["id"],))
    assert row is not None
    pending = await store.fetchone("SELECT payload FROM pending_erase")
    assert int(json.loads(pending["payload"])["summary_vec_ids"][0]) == int(row["rowid"])

    monkeypatch.undo()
    assert await mem.drain_pending_erases() == 1
    assert await store.fetchone("SELECT rowid FROM summaries WHERE id = ?", (summary["id"],)) is None
    assert not await store.fetchall("SELECT rowid FROM summaries_vec WHERE rowid = ?", (int(row["rowid"]),))


async def test_a_retry_deletes_vectors_whose_summary_row_is_already_gone(app):
    """Retries use the vector IDs stored in the payload, so a vector still goes even
    when its summary row disappeared by another route (#462)."""
    mem, store = app.memory, app.store
    await mem.episodic.vec.upsert(4242, [1.0, 0.0])
    await store.execute(
        "INSERT INTO pending_erase(fact_id, payload, at, complete) VALUES(?,?,?,1)",
        (1, json.dumps({"ids": [], "texts": [], "refs": [], "summary_vec_ids": [4242]}), now_iso()),
    )

    assert await mem.drain_pending_erases() == 1
    assert not await store.fetchall("SELECT rowid FROM summaries_vec WHERE rowid = 4242")
    assert not await store.fetchall("SELECT id FROM pending_erase")


async def test_hourly_upkeep_retries_cleanups_even_when_the_purge_fails(app, monkeypatch):
    """A purge failure must not block the pending-erase retries (#462)."""
    mem, store = app.memory, app.store
    app.config.evolution.review_enabled = False
    app.config.memory.summaries_enabled = False
    app.config.evolution.curator_enabled = False
    app.config.evolution.user_profile_updates = False
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "I drive a red Honda City to work")
    honda = int((await mem.remember(HONDA, use_llm=False, session_id=tg))["id"])
    summary = await mem.episodic.add_summary(
        tg, "Maya drives a red Honda City to the office.", await store.recent_messages(tg, 10)
    )

    async def always_fails(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mem, "_cascade", always_fails)
    assert await mem.erase(honda) is True  # the wait is recorded for the next upkeep
    monkeypatch.undo()

    async def purge_raises():
        raise RuntimeError("database is locked")

    monkeypatch.setattr(app.memory, "purge_expired", purge_raises)
    with pytest.raises(RuntimeError):
        await app.evolution._tick()
    # The purge failed, but the retry still ran and cleared the wait.
    assert not await store.fetchall("SELECT id FROM pending_erase")
    assert await store.fetchone("SELECT id FROM summaries WHERE id = ?", (summary["id"],)) is None
