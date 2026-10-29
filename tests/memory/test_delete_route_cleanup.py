"""`DELETE /api/memories/{id}` when part of the erase fails (#534).

A failure after the memory row is gone used to answer a bare 500, although the
memory was deleted and its queued cleanup (#462) finishes on the next upkeep.
It is a successful delete now, flagged `cleanup_pending`. A failure before the
row is deleted still answers 503 "try again", and a missing id 404.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider

HONDA = "Maya drives a red Honda City"


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "del.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        c.core = core
        yield c


def _seed(client):
    mem, store = client.core.memory, client.core.store

    async def setup():
        tg = await store.create_session(channel="telegram")
        await store.add_message(tg, "user", "I drive a red Honda City to work")
        honda = int((await mem.remember(HONDA, use_llm=False, session_id=tg))["id"])
        summary = await mem.episodic.add_summary(
            tg, "Maya drives a red Honda City to the office.", await store.recent_messages(tg, 10)
        )
        return honda, summary["id"]

    return client.portal.call(setup)


def _rows(client, sql, params=()):
    return client.portal.call(client.core.store.fetchall, sql, params)


def test_a_forget_that_raises_after_the_row_is_gone_answers_success(client, monkeypatch):
    mem = client.core.memory
    honda, summary = _seed(client)
    real_forget = mem.forget

    async def forget_crashes(fid, *, notify=False):
        assert await real_forget(fid, notify=notify)
        raise RuntimeError("vector delete failed after the row went")

    monkeypatch.setattr(mem, "forget", forget_crashes)
    r = client.delete(f"/api/memories/{honda}")
    assert r.status_code == 200
    assert r.json() == {"deleted": True, "cleanup_pending": True}
    monkeypatch.undo()

    # The memory is no longer listed, and its cleanup waits in the queue.
    assert honda not in [m["id"] for m in client.get("/api/memories", params={"state": "all"}).json()]
    assert client.get(f"/api/memories/{honda}").status_code == 404
    queued = _rows(client, "SELECT id, fact_id, complete FROM pending_erase")
    assert [int(q["fact_id"]) for q in queued] == [honda]
    assert _rows(client, "SELECT id FROM summaries WHERE id = ?", (summary,))

    # The hourly upkeep replays it once the wait has settled, and the derived summary goes.
    client.portal.call(
        client.core.store.execute,
        "UPDATE pending_erase SET at = ?",
        ((datetime.now(UTC) - timedelta(hours=2)).isoformat(),),
    )
    assert client.portal.call(mem.drain_pending_erases) == 1
    assert not _rows(client, "SELECT id FROM summaries WHERE id = ?", (summary,))
    assert not _rows(client, "SELECT id FROM pending_erase")
    # A second delete of the same id is a plain 404.
    assert client.delete(f"/api/memories/{honda}").status_code == 404


def test_a_forget_that_raises_before_the_row_goes_answers_503_and_keeps_the_memory(client, monkeypatch):
    mem = client.core.memory
    honda, _ = _seed(client)

    async def forget_fails_first(fid, *, notify=False):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mem, "forget", forget_fails_first)
    r = client.delete(f"/api/memories/{honda}")
    assert r.status_code == 503 and "try again" in r.json()["detail"]
    monkeypatch.undo()
    assert client.get(f"/api/memories/{honda}").status_code == 200
    # A retry works.
    assert client.delete(f"/api/memories/{honda}").json() == {"deleted": True}


def test_a_failed_cleanup_step_is_flagged_and_a_clean_delete_is_not(client, monkeypatch):
    mem = client.core.memory
    honda, _ = _seed(client)
    other = int(client.portal.call(lambda: mem.remember("Maya likes masala chai", source="manual", use_llm=False))["id"])

    async def cascade_fails(*args, **kwargs):
        raise RuntimeError("MEMORY.md is locked by another program")

    monkeypatch.setattr(mem, "_cascade", cascade_fails)
    assert client.delete(f"/api/memories/{honda}").json() == {"deleted": True, "cleanup_pending": True}
    monkeypatch.undo()
    assert client.delete(f"/api/memories/{other}").json() == {"deleted": True}
    assert client.delete("/api/memories/4242").status_code == 404


def test_a_missing_id_is_404_even_when_the_erase_would_fail(client, monkeypatch):
    mem = client.core.memory

    async def erase_fails(fid, *, notify=False):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mem, "erase", erase_fails)
    r = client.delete("/api/memories/4242")
    assert r.status_code == 404


def test_an_unreadable_queue_counts_as_cleanup_pending(client, monkeypatch):
    mem, store = client.core.memory, client.core.store
    real_fetchone = store.fetchone

    async def fetchone_failing_on_queue(sql, *args, **kwargs):
        if "FROM pending_erase" in sql:
            raise RuntimeError("database is locked")
        return await real_fetchone(sql, *args, **kwargs)

    monkeypatch.setattr(store, "fetchone", fetchone_failing_on_queue)
    assert client.portal.call(mem.cleanup_pending, 1) is True
