"""A cleanup step that fails during an erase is retried until it succeeds (#462).

`erase()` deletes the memory itself first and then removes everything derived
from it. One failing step (a locked file, a transient database error) used to
drop the rest: the derived data stayed forever and nothing ever tried again.
The unfinished cleanup is now recorded in `pending_erase` and replayed by
memory upkeep (`drain_pending_erases`) until it succeeds.
"""

from __future__ import annotations

import json

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

    async def fails_once(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return await real_cascade(*args)

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
    assert await mem.drain_pending_erases() == 1
    assert len(calls) == 2
    assert await store.fetchone("SELECT id FROM summaries WHERE id = ?", (summary["id"],)) is None
    assert not await store.fetchall("SELECT id FROM pending_erase")


async def test_a_cleanup_that_keeps_failing_is_kept_and_retried_next_time(app, monkeypatch):
    mem, store = app.memory, app.store
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "I drive a red Honda City to work")
    first = await mem.remember(HONDA, use_llm=False, session_id=tg)
    honda = int(first["id"])

    async def always_fails(*args):
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
