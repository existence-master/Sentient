"""Rows saved in the same clock tick keep their order and are never skipped (issue #236).

Windows' clock moves about every 15.6 ms, so these tests freeze the clock to make every row share one timestamp."""

from __future__ import annotations

import json
from datetime import datetime

from sentient.app import SentientApp
from sentient.store import db as store_db
from sentient.store.db import after_watermark, next_watermark
from tests.conftest import FakeProvider

TICK = "2026-10-11T06:00:00.123456+00:00"


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


async def test_profile_upkeep_after_an_empty_read_still_learns_a_same_tick_fact(config, isolated_home, monkeypatch):
    from sentient.memory import facts as facts_mod

    freeze(monkeypatch, store_db, facts_mod)
    llm = FakeProvider()
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "empty.db", enable_background=False).start()
    try:
        now = datetime.fromisoformat(TICK)
        assert (await s.evolution.update_profile(now, force=True))["updated"] is False  # nothing to read yet
        assert s.memory is not None
        a = await s.memory.remember("Maya is training for a marathon", use_llm=False)  # same tick as that run
        await s.store.execute("UPDATE facts SET topics = ? WHERE id = ?", ('["Health & Wellbeing"]', a["id"]))
        llm.text_replies.append("# Long-term memory\n\n## Goals\n- Maya is training for a marathon.\n")
        assert (await s.evolution.update_profile(now, force=True))["learned_appended"] == 1
    finally:
        await s.stop()


async def test_user_model_refresh_reads_rows_saved_in_the_same_tick(config, isolated_home, monkeypatch):
    """#297: the refresh starts after what it read last time, not at its own time."""
    from sentient.memory import facts as facts_mod

    freeze(monkeypatch, store_db, facts_mod)
    llm = FakeProvider()
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "um.db", enable_background=False).start()
    try:
        assert s.memory is not None
        sid = await s.store.create_session(channel="cli")
        await s.store.add_message(sid, "user", "keep answers short please")
        await s.memory.remember("Maya runs every morning", use_llm=False)
        llm.json_replies.append({"operations": [], "summary": "Maya likes short answers."})
        await s.user_model.refresh()
        first = llm.calls[-1]["messages"][1]["content"]
        assert "keep answers short" in first and "runs every morning" in first
        # saved after that read, in the same clock tick
        await s.store.add_message(sid, "user", "I'm moving to Lisbon in spring")
        await s.memory.remember("Maya mentors two junior developers", use_llm=False)
        llm.json_replies.append({"operations": [], "summary": "Maya likes short answers."})
        await s.user_model.refresh()
        second = llm.calls[-1]["messages"][1]["content"]
        assert "Lisbon" in second and "mentors two junior" in second
        assert "keep answers short" not in second and "runs every morning" not in second  # never read twice
        calls = len(llm.calls)
        await s.user_model.refresh()
        assert len(llm.calls) == calls  # nothing new: no model call
        marks = json.loads(await s.store.get_meta("user_model.messages_mark") or "{}")
        assert marks["at"] == TICK and len(marks["ids"]) == 2
    finally:
        await s.stop()


async def test_a_kind_with_nothing_read_yet_is_not_skipped_later(config, isolated_home, monkeypatch):
    """The first refresh reads a message but no fact: a fact saved later in the same tick is still read."""
    from sentient.memory import facts as facts_mod

    freeze(monkeypatch, store_db, facts_mod)
    llm = FakeProvider()
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "um-first.db", enable_background=False).start()
    try:
        assert s.memory is not None
        sid = await s.store.create_session(channel="cli")
        await s.store.add_message(sid, "user", "keep answers short please")
        llm.json_replies.append({"operations": [], "summary": "Maya likes short answers."})
        await s.user_model.refresh(now=datetime.fromisoformat(TICK))
        assert await s.store.get_meta("user_model.facts_mark") == ""  # nothing read yet, saved as such
        await s.memory.remember("Maya mentors two junior developers", use_llm=False)
        llm.json_replies.append({"operations": [], "summary": "Maya likes short answers."})
        await s.user_model.refresh(now=datetime.fromisoformat(TICK))
        assert "mentors two junior" in llm.calls[-1]["messages"][1]["content"]
    finally:
        await s.stop()

async def test_session_review_reads_messages_saved_in_the_same_tick(config, isolated_home, monkeypatch):
    """#297: the skill review of a chat starts after the messages it reviewed, not after their timestamp."""
    from tests.evolution.conftest import add_tool_chat

    freeze(monkeypatch, store_db)
    llm = FakeProvider()
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "evo.db", enable_background=False).start()
    try:
        config.evolution.review_idle_minutes = 0
        sid = await add_tool_chat(s, tool_calls=3)
        llm.json_replies.append({"decision": "none", "reason": "nothing reusable"})
        assert (await s.evolution.review_session(sid))["decision"] == "none"
        # three more tool calls in the same clock tick as the last reviewed message
        for i in range(3):
            call = {"id": f"d{i}", "type": "function", "function": {"name": "gmail_read", "arguments": "{}"}}
            await s.store.add_message(sid, "assistant", None, tool_calls=[call])
            await s.store.add_message(sid, "tool", '{"thread": "ok"}', tool_call_id=f"d{i}", name="gmail_read")
        now = datetime.fromisoformat(TICK)
        assert await s.evolution.sessions_due(now, idle_minutes=0) == [sid]
        llm.json_replies.append({"decision": "none", "reason": "nothing reusable"})
        assert (await s.evolution.review_session(sid))["reviewed"] is True
        transcript = llm.calls[-1]["messages"][1]["content"]
        assert "gmail_read" in transcript and "gmail_search" not in transcript  # only the new part
        assert await s.evolution.sessions_due(now, idle_minutes=0) == []  # all reviewed now
    finally:
        await s.stop()
