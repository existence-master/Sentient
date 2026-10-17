"""Disconnect and forget (#153): facts know every source they came from, and forgetting a source deletes what only it
backs, with its chats, summaries, vectors and cached content. Without the box nothing is deleted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentient import secrets
from sentient.activity.timeline import Timeline, forgot_detail
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.memory import forget as memory_forget
from sentient.memory import origins
from sentient.memory import review as reviews
from sentient.store.db import now_iso
from tests.conftest import FakeProvider


@pytest.fixture(autouse=True)
def keychain(monkeypatch) -> dict[str, str]:
    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: store.get(name))
    monkeypatch.setattr(secrets, "set_secret", lambda name, value: store.__setitem__(name, value) or True)
    monkeypatch.setattr(secrets, "delete_secret", lambda name: store.pop(name, None) is not None)
    return store


async def _folder(app, tmp_path: Path, fid: str = "fold1", name: str = "Notes") -> Path:
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    await app.store.execute(
        "INSERT INTO folders(id, path, name, added_at) VALUES(?,?,?,?)", (fid, str(root), name, now_iso())
    )
    return root


NOTE = "My dentist is Dr. Mehta at Smile Care in Pune. Next checkup in March."


async def _read_note(app, sid: str, path: Path, snippet: str = NOTE) -> None:
    """A reply in chat ``sid`` that used a note from the folder (its answer sources, #174)."""
    await app.store.add_message(sid, "assistant", "From your notes: ...", sources=[
        {"kind": "file", "title": path.stem, "path": str(path), "folder": "Notes", "app": "folders",
         "tool": "folder_search", "snippet": snippet},
    ])


async def _vec_ids(app, table: str) -> set[int]:
    return {int(r["rowid"]) for r in await app.store.fetchall(f"SELECT rowid FROM {table}")}


async def test_chat_facts_take_the_chats_source(app):
    mem, store = app.memory, app.store
    tg = await store.create_session(channel="telegram")
    dev = await store.create_session(channel="glasses", node_id="node-1")
    desk = await store.create_session(channel="desktop")
    a = await mem.remember("Maya is allergic to peanuts", use_llm=False, session_id=tg)
    b = await mem.remember("Maya runs every Sunday morning", use_llm=False, session_id=dev)
    c = await mem.remember("Maya drinks green tea daily", use_llm=False, session_id=desk)
    d = await mem.remember("Maya's sister is called Riya", use_llm=False)
    assert [(await mem.get_fact(r["id"]))["source"] for r in (a, b, c, d)] == [
        "channel:telegram", "device:node-1", "conversation", "conversation",
    ]


async def test_a_chat_links_the_apps_and_folders_it_read(app, tmp_path):
    mem, store = app.memory, app.store
    root = await _folder(app, tmp_path)
    tg = await store.create_session(channel="telegram")
    await _read_note(app, tg, root / "health.md")
    await store.add_message(tg, "assistant", "You have mail", sources=[
        {"kind": "email", "title": "Flight update", "from": "a@example.com", "app": "gmail", "tool": "gmail_search",
         "snippet": "Your flight to Lisbon lands at 9:40"},
        {"kind": "web", "title": "x", "url": "https://example.com", "app": "web", "tool": "web_fetch"},
    ])
    told = await mem.remember("Maya is allergic to peanuts", use_llm=False, session_id=tg)
    assert await mem.sources_of(told["id"]) == ["channel:telegram"]  # the notes it read don't say it
    noted = await mem.remember("Maya's dentist is Dr. Mehta in Pune", use_llm=False, session_id=tg)
    assert await mem.sources_of(noted["id"]) == ["channel:telegram", "folder:fold1"]  # her notes say it too
    held = await mem.remember(
        "Maya's flight to Lisbon lands at 9:40", use_llm=False, session_id=tg,
        review=reviews.note("Gmail", session_id=tg),
    )
    assert await mem.sources_of(held["id"]) == ["app:gmail"]  # from what the chat read, not from Maya


async def test_held_facts_belong_to_what_the_chat_read(app):
    mem, store = app.memory, app.store
    desk = await store.create_session(channel="desktop")
    await store.add_message(desk, "assistant", "Your inbox", sources=[
        {"kind": "email", "title": "Flight", "from": "air@example.com", "app": "gmail", "tool": "gmail_search"},
    ])
    held = await mem.remember(
        "Maya flies to Lisbon on Friday", use_llm=False, session_id=desk, review=reviews.note("Gmail", session_id=desk)
    )
    assert held["status"] == "pending"
    assert await mem.sources_of(held["id"]) == ["app:gmail"]


async def test_purge_unlinks_shared_facts_and_deletes_the_rest(app, tmp_path):
    mem, store = app.memory, app.store
    root = await _folder(app, tmp_path)
    tg = await store.create_session(channel="telegram")
    only = await mem.remember("Maya is allergic to peanuts", use_llm=False, session_id=tg)
    shared = await mem.remember("Maya's dentist is Dr. Mehta in Pune", use_llm=False, session_id=tg)
    desk = await store.create_session(channel="desktop")
    await _read_note(app, desk, root / "health.md")
    again = await mem.remember("Maya's dentist is Dr. Mehta in Pune", use_llm=False, session_id=desk)
    assert again["action"] == "SKIP" and again["id"] == shared["id"]
    assert await mem.sources_of(shared["id"]) == ["channel:telegram", "conversation", "folder:fold1"]

    assert await mem.source_counts("channel:telegram") == {"memories": 1, "shared": 1}
    assert await mem.purge_source("channel:telegram") == {"deleted": 1, "unlinked": 1}
    assert await mem.get_fact(only["id"]) is None
    assert only["id"] not in await _vec_ids(app, "facts_vec")
    kept = await mem.get_fact(shared["id"])
    assert kept["source"] == "conversation"
    assert await mem.sources_of(shared["id"]) == ["conversation", "folder:fold1"]
    assert shared["id"] in await _vec_ids(app, "facts_vec")
    rows = await store.fetchall("SELECT source FROM fact_sources WHERE source = 'channel:telegram'")
    assert rows == []

    assert await mem.purge_source("folder:fold1") == {"deleted": 0, "unlinked": 1}
    assert await mem.purge_source("conversation") == {"deleted": 1, "unlinked": 0}
    assert await mem.get_fact(shared["id"]) is None
    assert await store.fetchall("SELECT * FROM fact_sources") == []


async def test_merged_copies_count_as_sources(app):
    mem, store = app.memory, app.store
    tg = await store.create_session(channel="telegram")
    keep = await mem.remember("Maya lives in Pune", use_llm=False, session_id=tg)
    copy = await mem.remember("माया पुणे में रहती है", use_llm=False)  # learned in a desktop chat
    assert await mem.merge_translation(keep["id"], copy["id"])
    assert await mem.sources_of(keep["id"]) == ["channel:telegram", "conversation"]
    assert await mem.purge_source("channel:telegram") == {"deleted": 0, "unlinked": 1}
    assert (await mem.get_fact(keep["id"]))["source"] == "conversation"


async def test_forget_a_messaging_app(app, tmp_path):
    mem, store = app.memory, app.store
    root = await _folder(app, tmp_path)
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "I'm allergic to peanuts and my dentist is Dr. Mehta")
    await _read_note(app, tg, root / "health.md")
    only = await mem.remember("Maya is allergic to peanuts", use_llm=False, session_id=tg)
    shared = await mem.remember("Maya's dentist is Dr. Mehta in Pune", use_llm=False, session_id=tg)
    summary = await mem.episodic.add_summary(tg, "Maya talked about her allergy.", await store.recent_messages(tg, 100))
    desk = await store.create_session(channel="desktop")
    await store.add_message(desk, "user", "What's the weather?")
    other = await mem.episodic.add_summary(desk, "Maya asked about the weather.", await store.recent_messages(desk, 100))

    before = await memory_forget.preview(app, "channel:telegram")
    assert before == {"source": "channel:telegram", "memories": 1, "shared": 1, "chats": 1, "messages": 2,
                      "summaries": 1, "results": 0, "items": 0, "item_noun": "items"}
    done = await memory_forget.forget(app, "channel:telegram", "Telegram")
    assert {k: done[k] for k in ("memories", "shared", "chats", "messages", "summaries")} == {
        "memories": 1, "shared": 1, "chats": 1, "messages": 2, "summaries": 1,
    }
    assert await mem.get_fact(only["id"]) is None
    assert await mem.sources_of(shared["id"]) == ["folder:fold1"]
    assert await store.get_session(tg) is None
    assert await store.fetchall("SELECT id FROM messages WHERE session_id = ?", (tg,)) == []
    summaries = {r["id"]: r["rowid"] for r in await store.fetchall("SELECT rowid, id FROM summaries")}
    assert summary["id"] not in summaries and other["id"] in summaries
    assert await _vec_ids(app, "summaries_vec") == {summaries[other["id"]]}
    assert await store.get_session(desk) is not None
    # logged for Activity
    rows = await store.fetchall("SELECT source, label, counts FROM memory_forgets")
    assert [(r["source"], r["label"]) for r in rows] == [("channel:telegram", "Telegram")]
    page = await Timeline(app).past(kinds=["memory"])
    item = next(i for i in page["items"] if i["id"].startswith("forget:"))
    assert item["title"] == "Forgot what Sentient learned from Telegram"
    assert item["detail"] == "1 memory, 1 chat (2 messages) and 1 chat summary. 1 memory you also shared elsewhere stayed"


async def test_forget_an_app_scrubs_its_results_and_cache(app):
    mem, store = app.memory, app.store
    desk = await store.create_session(channel="desktop")
    await store.add_message(desk, "user", "Any news from the airline?")
    await store.add_message(desk, "assistant", None, tool_calls=[{"id": "c1", "name": "gmail_search", "arguments": {}}])
    await store.add_message(desk, "tool", json.dumps({"messages": [{"subject": "Flight", "from": "air@example.com",
                                                                    "snippet": "Your seat 14C to Lisbon"}]}),
                            tool_call_id="c1", name="gmail_search")
    await store.add_message(desk, "assistant", "Your flight to Lisbon is on Friday, seat 14C.", sources=[
        {"kind": "email", "title": "Flight", "from": "air@example.com", "app": "gmail", "tool": "gmail_search"},
    ])
    held = await mem.remember(
        "Maya flies to Lisbon on Friday", use_llm=False, session_id=desk, review=reviews.note("Gmail", session_id=desk)
    )
    own = await mem.remember("Maya prefers aisle seats", use_llm=False, session_id=desk)
    await mem.episodic.add_summary(desk, "Maya checked her flight to Lisbon.", await store.recent_messages(desk, 100))
    for i in range(3):
        await store.execute(
            "INSERT INTO proactive_seen(source, item_id, event_type, item, seen_at) VALUES('gmail', ?, 'new_email', ?, ?)",
            (f"m{i}", json.dumps({"subject": "Flight", "body": "seat 14C"}), now_iso()),
        )
    await store.execute(
        "INSERT INTO integration_seen(source, item_key, origin, seen_at) VALUES('gmail', 'm1', 'poll', ?)", (now_iso(),)
    )

    before = await memory_forget.preview(app, "app:gmail")
    assert before == {"source": "app:gmail", "memories": 1, "shared": 0, "chats": 0, "messages": 0, "summaries": 1,
                      "results": 1, "items": 3, "item_noun": "emails"}
    done = await memory_forget.forget(app, "app:gmail", "Gmail")
    assert done["memories"] == 1 and done["results"] == 1 and done["items"] == 3
    assert await mem.get_fact(held["id"]) is None
    assert (await mem.get_fact(own["id"]))["source"] == "conversation"  # what Maya said herself stays
    # the chat stays, without the email
    msgs = await store.recent_messages(desk, 100)
    tool = next(m for m in msgs if m["role"] == "tool")
    assert "14C" not in tool["content"] and "forgotten" in tool["content"]
    hits = await store.fetchall("SELECT rowid FROM messages_fts WHERE messages_fts MATCH '\"14C\"'")
    assert len(hits) == 1  # only the assistant's own reply still says it
    assert all(not m.get("sources") for m in msgs if m["role"] == "assistant")
    for table in ("proactive_seen", "integration_seen", "summaries"):
        assert await store.fetchall(f"SELECT * FROM {table}") == [], table
    assert forgot_detail(done).startswith("1 memory, 3 saved emails, 1 result in your chats and 1 chat summary")


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "forget.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        c.core = core
        yield c


def test_disconnect_without_the_box_deletes_nothing_and_with_it_forgets(client):
    app = client.core
    mem, store = app.memory, app.store

    async def setup():
        tg = await store.create_session(channel="telegram")
        await store.add_message(tg, "user", "I'm allergic to peanuts")
        return tg, await mem.remember("Maya is allergic to peanuts", use_llm=False, session_id=tg)

    tg, fact = client.portal.call(setup)
    preview = client.get("/api/memories/forget-preview", params={"source": "channel:telegram"}).json()
    assert preview["memories"] == 1 and preview["chats"] == 1 and preview["messages"] == 1
    assert client.get("/api/memories/forget-preview", params={"source": "telegram"}).status_code == 400

    plain = client.post("/api/channels/telegram/disconnect").json()
    assert "forgotten" not in plain
    assert client.portal.call(mem.get_fact, fact["id"]) is not None
    assert client.portal.call(store.get_session, tg) is not None
    assert client.portal.call(store.fetchall, "SELECT * FROM memory_forgets", ()) == []

    out = client.post("/api/channels/telegram/disconnect", params={"forget": "true"}).json()
    assert out["status"] == "disconnected"
    assert out["forgotten"]["memories"] == 1 and out["forgotten"]["chats"] == 1
    assert client.portal.call(mem.get_fact, fact["id"]) is None
    assert client.portal.call(store.get_session, tg) is None
    activity = client.get("/api/timeline", params={"kinds": "memory"}).json()
    assert any(i["title"] == "Forgot what Sentient learned from Telegram" for i in activity["items"])


def test_removing_a_device_or_folder_can_forget(client, tmp_path):
    app = client.core
    mem, store = app.memory, app.store

    async def setup():
        await store.execute(
            "INSERT INTO nodes(id, name, kind, created_at) VALUES('node-1', 'Maya''s glasses', 'glasses', ?)",
            (now_iso(),),
        )
        dev = await store.create_session(channel="glasses", node_id="node-1")
        await store.add_message(dev, "user", "Remind me I park on level 3")
        fact = await mem.remember("Maya parks on level 3 at work", use_llm=False, session_id=dev)
        root = await _folder(app, tmp_path)
        desk = await store.create_session(channel="desktop")
        await _read_note(app, desk, root / "trips.md", "Next trip: Goa in December, booked with Rohan")
        held = await mem.remember(
            "Maya's next trip is to Goa", use_llm=False, session_id=desk, review=reviews.note("Notes", session_id=desk)
        )
        return dev, fact, held

    dev, fact, held = client.portal.call(setup)
    out = client.delete("/api/nodes/node-1", params={"forget": "true"}).json()
    assert out["ok"] and out["forgotten"]["memories"] == 1 and out["forgotten"]["chats"] == 1
    assert client.portal.call(mem.get_fact, fact["id"]) is None
    assert client.portal.call(store.get_session, dev) is None

    out = client.delete("/api/folders/fold1", params={"forget": "true"}).json()
    assert out["removed"] and out["forgotten"]["memories"] == 1
    assert client.portal.call(mem.get_fact, held["id"]) is None


def test_an_excerpt_backs_a_fact_by_its_words():
    assert origins.backs("Maya's dentist is Dr. Mehta", NOTE, "Maya")
    assert origins.backs("Maya's checkup is on 14 March", "Checkup: 14 March", "Maya")
    assert not origins.backs("Maya is allergic to peanuts", NOTE, "Maya")
    assert not origins.backs("Maya likes Pune", "Maya Rao lives in Pune", "Maya Rao")


def test_source_keys():
    assert origins.key("channel", "telegram") == "channel:telegram"
    assert origins.split("app:gmail") == ("app", "gmail")
    assert origins.split("file:cv.pdf") == ("", "file:cv.pdf")
    assert origins.is_chat("device:abc") and not origins.is_chat("app:gmail")
    with pytest.raises(ValueError):
        origins.key("file", "x")
