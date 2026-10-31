"""Concurrent API requests keep one persistent chat per name."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from sentient.gateway.routes.openai_api import CHANNEL, SESSIONS_META, _session
from sentient.store.db import Store


@pytest.fixture
async def store(isolated_home):
    db = await Store(isolated_home / "api-sessions.db").open()
    try:
        yield db
    finally:
        await db.close()


@pytest.mark.parametrize("names", [["Notes app"] * 8, [f"App {i}" for i in range(8)]])
async def test_concurrent_first_requests_keep_one_chat_per_name(store, names):
    core = SimpleNamespace(store=store)

    ids = await asyncio.gather(*(_session(core, name) for name in names))

    known = json.loads(await store.get_meta(SESSIONS_META))
    assert set(known) == set(names)
    assert ids == [known[name] for name in names]
    rows = await store.fetchall("SELECT id, title, channel FROM sessions")
    assert len(rows) == len(set(names))
    assert {row["title"]: row["id"] for row in rows} == known
    assert all(row["channel"] == CHANNEL for row in rows)
    assert await asyncio.gather(*(_session(core, name) for name in names)) == ids


async def test_concurrent_requests_recreate_a_deleted_chat_once(store):
    core = SimpleNamespace(store=store)
    old = await _session(core, "Notes app")
    other = await _session(core, "Editor")
    await store.delete_session(old)

    ids = await asyncio.gather(*(_session(core, "Notes app") for _ in range(8)))

    assert len(set(ids)) == 1 and ids[0] != old
    assert json.loads(await store.get_meta(SESSIONS_META)) == {"Notes app": ids[0], "Editor": other}
    assert len(await store.fetchall("SELECT id FROM sessions")) == 2


async def test_session_creation_does_not_block_another_app(store, isolated_home, monkeypatch):
    other = await Store(isolated_home / "other-api-sessions.db").open()
    entered, release = asyncio.Event(), asyncio.Event()
    create_session = store.create_session

    async def waiting_create_session(**kwargs):
        entered.set()
        await release.wait()
        return await create_session(**kwargs)

    monkeypatch.setattr(store, "create_session", waiting_create_session)
    pending = asyncio.create_task(_session(SimpleNamespace(store=store), "Notes app"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        sid = await asyncio.wait_for(_session(SimpleNamespace(store=other), "Notes app"), timeout=5)
        assert await other.get_session(sid) is not None
        assert not pending.done()
    finally:
        release.set()
        await pending
        await other.close()
