"""Source links, confidence and state, and deletes that cascade (#152).

Each fact points at exactly where it was said (a chat message, an email, a note, a file), knows how sure Sentient is of
it, and is current, disputed or superseded. Only current facts go into prompts. Deleting a fact takes everything derived
from it with it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentient import paths
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.memory import review as reviews
from sentient.store.db import SCHEMA_VERSION, now_iso
from tests.conftest import FakeProvider

NOTE = "My dentist is Dr. Mehta at Smile Care in Pune. Next checkup in March."


async def _folder(app, tmp_path: Path) -> Path:
    root = tmp_path / "Notes"
    root.mkdir(exist_ok=True)
    await app.store.execute(
        "INSERT INTO folders(id, path, name, added_at) VALUES('fold1', ?, 'Notes', ?)", (str(root), now_iso())
    )
    return root


async def _ids(app, sql: str, params: tuple = ()) -> list:
    return [r[0] for r in await app.store.fetchall(sql, params)]


# ---------------------------------------------------------------- spans
async def test_a_chat_fact_points_at_the_message_that_said_it(app, tmp_path):
    mem, store = app.memory, app.store
    root = await _folder(app, tmp_path)
    tg = await store.create_session(channel="telegram")
    said = await store.add_message(tg, "user", "Hi! I'm allergic to peanuts, by the way.")
    await store.add_message(tg, "assistant", "Noted.")
    await store.add_message(tg, "user", "Also, what does my health note say about my dentist?")
    await store.add_message(tg, "assistant", "Dr. Mehta at Smile Care.", sources=[
        {"kind": "file", "title": "health", "path": str(root / "health.md"), "folder": "Notes", "app": "folders",
         "tool": "folder_search", "snippet": NOTE},
    ])
    await store.add_message(tg, "user", "Right, my dentist is Dr. Mehta in Pune.")
    allergy = await mem.remember("Maya is allergic to peanuts", use_llm=False, session_id=tg)
    fact = await mem.get_fact(allergy["id"])
    assert fact["ref"] == {"kind": "message", "session_id": tg, "message_id": said,
                           "quote": "I'm allergic to peanuts, by the way."}
    dentist = await mem.get_fact((await mem.remember("Maya's dentist is Dr. Mehta in Pune", use_llm=False,
                                                     session_id=tg))["id"])
    assert [s["source"] for s in dentist["sources"]] == ["channel:telegram", "folder:fold1"]
    note = dentist["sources"][1]["ref"]
    assert note["kind"] == "note" and note["folder_id"] == "fold1" and note["path"].endswith("health.md")
    assert note["quote"] == "My dentist is Dr. Mehta at Smile Care in Pune."


async def test_a_held_fact_points_at_the_email(app):
    mem, store = app.memory, app.store
    desk = await store.create_session(channel="desktop")
    await store.add_message(desk, "assistant", "Your inbox", sources=[
        {"kind": "email", "title": "Flight update", "from": "air@example.com", "id": "m-42", "app": "gmail",
         "tool": "gmail_search", "url": "https://mail.example.com/m-42",
         "snippet": "Your flight to Lisbon lands at 9:40. Seat 14C."},
    ])
    held = await mem.remember("Maya's flight to Lisbon lands at 9:40", use_llm=False, session_id=desk,
                              review=reviews.note("Gmail", session_id=desk))
    fact = await mem.get_fact(held["id"])
    assert fact["source"] == "app:gmail" and fact["confidence"] == 0.6
    assert fact["ref"]["kind"] == "email" and fact["ref"]["id"] == "m-42"
    assert fact["ref"]["url"] == "https://mail.example.com/m-42" and fact["ref"]["quote"] == "Your flight to Lisbon lands at 9:40."


async def test_an_imported_file_fact_points_at_the_file(app, tmp_path):
    mem, llm = app.memory, app.fake
    path = tmp_path / "cv.txt"
    path.write_text("I studied design at NID. I like hiking.", encoding="utf-8")
    llm.json_replies.append({"facts": ["Maya studied design at NID"]})
    await mem.import_document(path, username="Maya")
    row = await app.store.fetchone("SELECT source, source_ref FROM facts")
    assert row["source"] == "file:cv.txt"
    assert json.loads(row["source_ref"]) == {"kind": "file", "file": "cv.txt", "quote": "I studied design at NID."}


# ---------------------------------------------------------------- confidence
async def test_confidence_comes_from_the_source_and_grows_when_learned_again(app):
    mem, store = app.memory, app.store
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "I run 5 km every Sunday")
    told = await mem.remember("Maya runs 5 km every Sunday", use_llm=False, session_id=tg)
    assert (await mem.get_fact(told["id"]))["confidence"] == 0.9
    manual = await mem.remember("Maya's cat is called Biscuit", source="manual", use_llm=False)
    assert (await mem.get_fact(manual["id"]))["confidence"] == 1.0
    # said again in another chat: one more source, a little surer
    desk = await store.create_session(channel="desktop")
    await store.add_message(desk, "user", "I run 5 km every Sunday")
    again = await mem.remember("Maya runs 5 km every Sunday", use_llm=False, session_id=desk)
    assert again["action"] == "SKIP"
    fact = await mem.get_fact(told["id"])
    assert fact["confidence"] == 1.0 and [s["source"] for s in fact["sources"]] == ["channel:telegram", "conversation"]
    assert fact["sources"][1]["ref"]["session_id"] == desk
    # the same source again adds nothing
    await mem.remember("Maya runs 5 km every Sunday", use_llm=False, session_id=desk)
    assert len((await mem.get_fact(told["id"]))["sources"]) == 2


# ---------------------------------------------------------------- state
async def test_disputed_and_superseded_memories_stay_out_of_prompts(app):
    mem = app.memory
    pune = await mem.remember("Maya lives in Pune", source="manual", use_llm=False)
    chai = await mem.remember("Maya drinks masala chai every morning", source="manual", use_llm=False)
    assert pune["id"] in [h["id"] for h in await mem.recall("Where does Maya live?", min_similarity=0.0)]
    disputed = await mem.set_state(pune["id"], "disputed")
    assert disputed["state"] == "disputed"
    assert pune["id"] not in [h["id"] for h in await mem.recall("Where does Maya live?", min_similarity=0.0)]
    tool = app.registry.get("memory_recall")
    ctx = app.agent.tool_context(None, "desktop")
    assert pune["id"] not in [r["id"] for r in await tool.call(ctx, {"query": "Where does Maya live?"})]
    # still on the Memory screen
    assert pune["id"] in [m["id"] for m in await mem.list_facts()]
    # the user says it's right after all
    back = await mem.set_state(pune["id"], "current")
    assert back["state"] == "current" and back["confidence"] == 1.0
    # superseded by a newer one: history, never recalled, not listed unless asked for
    assert await mem.supersede(chai["id"], pune["id"])
    assert pune["id"] not in [h["id"] for h in await mem.recall("Where does Maya live?", min_similarity=0.0)]
    assert pune["id"] not in [m["id"] for m in await mem.list_facts()]
    assert pune["id"] in [m["id"] for m in await mem.list_facts(state="superseded")]
    with pytest.raises(ValueError):
        await mem.set_state(chai["id"], "superseded")


async def test_saying_a_disputed_memory_again_makes_it_current(app):
    mem, store = app.memory, app.store
    fid = (await mem.remember("Maya lives in Pune", source="manual", use_llm=False))["id"]
    await mem.set_state(fid, "disputed")
    desk = await store.create_session(channel="desktop")
    await store.add_message(desk, "user", "I live in Pune")
    again = await mem.remember("Maya lives in Pune", use_llm=False, session_id=desk)
    assert again["id"] == fid and (await mem.get_fact(fid))["state"] == "current"


async def test_a_memory_out_of_prompts_also_leaves_the_workspace_files(app):
    mem, ws = app.memory, app.workspace
    pune = (await mem.remember("Maya lives in Pune", source="manual", use_llm=False))["id"]
    tea = (await mem.remember("Maya drinks masala chai every morning", source="manual", use_llm=False))["id"]
    ws.append_learned(["Maya lives in Pune", "Maya drinks masala chai every morning"])
    await mem.set_state(pune, "disputed")
    assert "Pune" not in ws.user.read_text(encoding="utf-8") and "chai" in ws.user.read_text(encoding="utf-8")
    await mem.supersede(pune, tea)
    assert "chai" not in ws.user.read_text(encoding="utf-8")
    # history is found by its words
    hits = await mem.list_facts(q="chai", state="all")
    assert [h["id"] for h in hits] == [tea] and hits[0]["state"] == "superseded"


async def test_no_message_saying_it_points_at_the_chat_only(app):
    mem, store = app.memory, app.store
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "ok see you tomorrow")
    fid = (await mem.remember("Maya's sister Riya lives in Lisbon", use_llm=False, session_id=tg))["id"]
    assert (await mem.get_fact(fid))["ref"] == {"kind": "message", "session_id": tg}


# ---------------------------------------------------------------- correct, then delete
async def test_correct_then_delete_leaves_nothing_derived(app):
    mem, store, ws = app.memory, app.store, app.workspace
    tg = await store.create_session(channel="telegram")
    said = await store.add_message(tg, "user", "Btw my sister Riya lives in Lisbon now")
    reply = await store.add_message(tg, "assistant", "Lovely!", memory_sources=[])
    other = await store.add_message(tg, "user", "Also I'm learning the veena on weekends")
    fid = (await mem.remember("Maya's sister Riya lives in Lisbon", use_llm=False, session_id=tg))["id"]
    keep = (await mem.remember("Maya is learning the veena on weekends", use_llm=False, session_id=tg))["id"]
    assert (await mem.get_fact(fid))["ref"]["message_id"] == said
    # the user corrects it
    await mem.update_content(fid, "Maya's sister Riya lives in Porto", use_llm=False)
    fact = await mem.get_fact(fid)
    assert fact["previous_content"] == "Maya's sister Riya lives in Lisbon" and fact["confidence"] == 1.0
    # what was derived from it meanwhile
    await store.execute("UPDATE messages SET memory_sources = ? WHERE id = ?", (json.dumps([
        {"kind": "fact", "id": fid, "text": "Maya's sister Riya lives in Porto", "source": "channel:telegram", "via": "prompt"},
        {"kind": "fact", "id": keep, "text": "Maya is learning the veena on weekends", "source": "channel:telegram", "via": "prompt"},
    ]), reply))
    msgs = await store.recent_messages(tg, 10)
    gone_summary = await mem.episodic.add_summary(tg, "Maya said her sister Riya lives in Lisbon now.", msgs[:2])
    kept_summary = await mem.episodic.add_summary(tg, "Maya is learning the veena.", [m for m in msgs if m["id"] == other])
    users = app.user_model
    only = await users._insert("Maya's family lives abroad", "relationships", confidence=0.6, status="active",
                               source="inferred", evidence=[{"kind": "fact", "ref": str(fid), "quote": "x"}])
    mixed = await users._insert("Maya keeps in touch with family", "relationships", confidence=0.6, status="active",
                                source="inferred", evidence=[{"kind": "fact", "ref": str(fid), "quote": "x"},
                                                             {"kind": "fact", "ref": str(keep), "quote": "y"}])
    await store.execute(
        "INSERT INTO dreams(id, started_at, status, trigger, stats, journal_md) VALUES('d1', ?, 'completed', 'manual',"
        " '{}', ?)",
        (now_iso(), "Tonight I tidied up.\n- I used to remember \"Maya's sister Riya lives in Lisbon\", but now "
                    "\"Maya's sister Riya lives in Porto\"\n- Maya is learning the veena"),
    )
    ws.append_learned(["Maya's sister Riya lives in Porto", "Maya is learning the veena on weekends"])
    ws.memory.write_text("# Memory\n- Riya, Maya's sister, lives in Porto\n- Maya plays the veena\n", encoding="utf-8")

    # the user deletes it
    assert await mem.erase(fid, notify=True)

    assert await mem.get_fact(fid) is None
    assert fid not in await _ids(app, "SELECT rowid FROM facts_vec")
    assert await _ids(app, "SELECT fact_id FROM fact_sources WHERE fact_id = ?", (fid,)) == []
    for word in ("Porto", "Lisbon"):
        assert await _ids(app, "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ?", (word,)) == []
    summaries = await _ids(app, "SELECT id FROM summaries")
    assert gone_summary["id"] not in summaries and kept_summary["id"] in summaries
    assert len(await _ids(app, "SELECT rowid FROM summaries_vec")) == 1
    journal = (await store.fetchone("SELECT journal_md FROM dreams WHERE id = 'd1'"))["journal_md"]
    assert "Riya" not in journal and "veena" in journal and "(a memory you deleted)" in journal
    insights = {r["id"]: json.loads(r["evidence"]) for r in await store.fetchall("SELECT id, evidence FROM user_insights")}
    assert only["id"] not in insights and [e["ref"] for e in insights[mixed["id"]]] == [str(keep)]
    chips = json.loads((await store.fetchone("SELECT memory_sources FROM messages WHERE id = ?", (reply,)))["memory_sources"])
    assert [c["id"] for c in chips] == [keep]
    assert "Porto" not in ws.user.read_text(encoding="utf-8") and "veena" in ws.user.read_text(encoding="utf-8")
    assert "Porto" not in ws.memory.read_text(encoding="utf-8") and "veena" in ws.memory.read_text(encoding="utf-8")
    # the chat itself is the user's own record and stays
    assert said in await _ids(app, "SELECT id FROM messages")


# ---------------------------------------------------------------- the one-off backfill (#153 left them as chats)
async def test_older_messaging_app_and_device_facts_get_their_source(config, isolated_home):
    app = SentientApp(config, llm=FakeProvider(), enable_background=False)
    await app.start()
    store = app.store
    tg = await store.create_session(channel="telegram")
    await store.add_message(tg, "user", "I'm allergic to peanuts")
    dev = await store.create_session(channel="glasses", node_id="node-1")
    await store.execute(
        "INSERT INTO messages(id, session_id, role, content, name, created_at) VALUES('t1', ?, 'tool', ?,"
        " 'memory_remember', ?)",
        (dev, json.dumps({"action": "ADD", "id": 2, "content": "Maya parks on level 3"}), now_iso()),
    )
    desk = await store.create_session(channel="desktop")
    await store.add_message(desk, "user", "I drink masala chai")
    await app.stop()
    con = sqlite3.connect(paths.db_file())
    for col in ("source_ref", "confidence", "state", "superseded_by"):  # what a #153 build had
        con.execute(f"ALTER TABLE facts DROP COLUMN {col}")
    soon = now_iso()
    for content in ("Maya is allergic to peanuts", "Maya parks on level 3", "Maya drinks masala chai"):
        con.execute(
            "INSERT INTO facts(content, source, topics, memory_type, created_at, updated_at) VALUES"
            " (?, 'conversation', '[]', 'long-term', ?, ?)",
            (content, soon, soon),
        )
    con.execute("PRAGMA user_version = 14")
    con.commit()
    con.close()
    app = SentientApp(config, llm=FakeProvider(), enable_background=False)
    await app.start()
    try:
        rows = {r["content"]: dict(r) for r in await app.store.fetchall(
            "SELECT content, source, source_ref, confidence, state FROM facts")}
        assert rows["Maya is allergic to peanuts"]["source"] == "channel:telegram"
        assert json.loads(rows["Maya is allergic to peanuts"]["source_ref"])["session_id"] == tg
        assert rows["Maya parks on level 3"]["source"] == "device:node-1"
        assert rows["Maya drinks masala chai"]["source"] == "conversation"  # a desktop chat stays one
        assert {r["confidence"] for r in rows.values()} == {0.9} and {r["state"] for r in rows.values()} == {"current"}
        assert await app.memory.forget_source("channel:telegram") == 1  # so forgetting Telegram covers it now
    finally:
        await app.stop()
    con = sqlite3.connect(paths.db_file())
    assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    con.close()


# ---------------------------------------------------------------- REST
@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "links.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        c.core = core
        yield c


def test_memory_routes_show_sources_and_change_state(client):
    mem, store = client.core.memory, client.core.store

    async def setup():
        tg = await store.create_session(channel="telegram")
        await store.add_message(tg, "user", "I'm allergic to peanuts")
        a = await mem.remember("Maya is allergic to peanuts", use_llm=False, session_id=tg)
        b = await mem.remember("Maya likes cashews", source="manual", use_llm=False)
        await mem.supersede(b["id"], a["id"])
        return tg, a["id"], b["id"]

    tg, old, new = client.portal.call(setup)
    one = client.get(f"/api/memories/{old}").json()
    assert one["state"] == "superseded" and one["superseded_by_content"] == "Maya likes cashews"
    assert one["sources"][0]["source"] == "channel:telegram" and one["sources"][0]["ref"]["session_id"] == tg
    assert client.get("/api/memories/9999").status_code == 404
    listed = client.get("/api/memories").json()
    assert [m["id"] for m in listed] == [new] and listed[0]["sources"][0]["source"] == "manual"
    assert [m["id"] for m in client.get("/api/memories", params={"state": "all"}).json()] == [new, old]
    assert client.get("/api/memories", params={"state": "nope"}).status_code == 400
    assert client.patch(f"/api/memories/{new}", json={"state": "disputed"}).json()["state"] == "disputed"
    assert client.patch(f"/api/memories/{new}", json={"state": "maybe"}).status_code == 400
    assert client.patch("/api/memories/9999", json={"state": "disputed"}).status_code == 404
    assert client.delete(f"/api/memories/{new}").json() == {"deleted": True}
    # a new name in Settings reaches memory at once (what it scrubs on a delete depends on it)
    assert client.patch("/api/config", json={"assistant": {"user_name": "Riya"}}).status_code == 200
    assert client.core.memory.config.assistant.user_name == "Riya"
