"""Rows saved in the same clock tick keep their order and are never skipped (issue #236).

Windows' clock moves about every 15.6 ms, so these tests freeze the clock to make every row share one timestamp."""

from __future__ import annotations

import json
from datetime import datetime

from sentient.app import SentientApp
from sentient.store import db as store_db
from sentient.store.db import after_watermark, next_watermark
from tests.conftest import FakeProvider

TICK = "2026-10-11T06:00:00.000000+00:00"


def freeze(monkeypatch, *modules) -> None:
    for m in modules:
        monkeypatch.setattr(m, "now_iso", lambda: TICK)


def test_watermark_keeps_the_ids_read_at_its_timestamp():
    assert after_watermark(None, "t", "id") == ("t > ?", [""])
    assert after_watermark(TICK, "t", "id") == ("t > ?", [TICK])  # an older build's plain timestamp
    mark = next_watermark(TICK, [{"t": "2026-10-11T05:00", "id": 1}, {"t": TICK, "id": 2}, {"t": TICK, "id": 3}], "t", "id")
    assert json.loads(mark) == {"at": TICK, "ids": [2, 3]}
    where, params = after_watermark(mark, "t", "id")
    assert where == "(t > ? OR (t = ? AND id NOT IN (?,?)))" and params == [TICK, TICK, 2, 3]
    # more rows at the same timestamp add to the ids; a newer timestamp starts over
    assert json.loads(next_watermark(mark, [{"t": TICK, "id": 4}], "t", "id"))["ids"] == [2, 3, 4]
    assert json.loads(next_watermark(mark, [{"t": "2026-10-12", "id": 5}], "t", "id")) == {"at": "2026-10-12", "ids": [5]}
    assert next_watermark(mark, [], "t", "id") == mark


async def test_same_tick_messages_keep_their_order(config, isolated_home, monkeypatch):
    freeze(monkeypatch, store_db)
    s = await SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "order.db", enable_background=False).start()
    try:
        sid = await s.store.create_session(channel="cli")
        for i in range(30):
            await s.store.add_message(sid, "user" if i % 2 == 0 else "assistant", f"message {i}")
        recent = await s.store.recent_messages(sid, 30)
        assert [m["content"] for m in recent] == [f"message {i}" for i in range(30)]
        between = await s.memory.episodic.messages_between("2026-10-11", "2026-10-11") if s.memory else []
        assert [m["content"] for m in between] == [f"message {i}" for i in range(30)]
    finally:
        await s.stop()


async def test_compression_does_not_skip_messages_saved_in_the_same_tick(config, isolated_home, monkeypatch):
    freeze(monkeypatch, store_db)
    seen: list[str] = []

    class Recorder(FakeProvider):
        async def complete_text(self, role, messages, *, model=None):
            seen.append(messages[-1]["content"])
            return "Running summary."

    config.chat.history_window = 2
    config.chat.compress_after_messages = 10
    s = await SentientApp(config, llm=Recorder(), db_path=isolated_home / "compress.db", enable_background=False).start()
    try:
        sid = await s.store.create_session(channel="cli")
        for i in range(20):
            await s.store.add_message(sid, "user", f"early {i}")
        await s.agent._maybe_compress(sid)
        assert "early 17" in seen[0] and "early 18" not in seen[0]  # the last two stay in the window
        for i in range(12):
            await s.store.add_message(sid, "user", f"later {i}")
        await s.agent._maybe_compress(sid)  # every message shares the first run's timestamp
        assert len(seen) == 2
        assert "early 17" not in seen[1] and "early 18" in seen[1] and "later 9" in seen[1]
    finally:
        await s.stop()


async def test_profile_upkeep_learns_a_fact_saved_in_the_same_tick(config, isolated_home, monkeypatch):
    from sentient.memory import facts as facts_mod

    freeze(monkeypatch, store_db, facts_mod)
    config.evolution.user_profile_updates = True
    llm = FakeProvider()
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "upkeep.db", enable_background=False).start()
    try:
        mem, now = s.memory, datetime.fromisoformat(TICK)
        assert mem is not None
        a = await mem.remember("Maya is training for a marathon", use_llm=False)
        await s.store.execute("UPDATE facts SET topics = ? WHERE id = ?", ('["Health & Wellbeing"]', a["id"]))
        llm.text_replies.append("# Long-term memory\n\n## Goals\n- Maya is training for a marathon.\n")
        assert (await s.evolution.update_profile(now, force=True))["learned_appended"] == 1
        # saved after that run's read, in the same clock tick
        b = await mem.remember("Maya mentors two junior developers", use_llm=False)
        await s.store.execute("UPDATE facts SET topics = ? WHERE id = ?", ('["Work & Learning"]', b["id"]))
        llm.text_replies.append("# Long-term memory\n\n## Work\n- Maya mentors two junior developers.\n")
        out = await s.evolution.update_profile(now, force=True)
        assert out["updated"] and out["learned_appended"] == 1
        user_md = s.workspace.user.read_text(encoding="utf-8")
        assert user_md.count("marathon") == 1 and "mentors two junior" in user_md
        assert (await s.evolution.update_profile(now, force=True))["updated"] is False  # nothing new is read twice
    finally:
        await s.stop()
