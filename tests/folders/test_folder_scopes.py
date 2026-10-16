"""Folder notes follow memory scopes (#392): a folder kept for Work only comes up in work chats, Always (the default,
also for folders added before scopes) everywhere. Offline: FakeProvider makes the embeddings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.memory import scopes
from sentient.store.db import SCHEMA_VERSION
from tests.conftest import FakeProvider, tool_call
from tests.folders.test_prompt_notes import topic_embed

KESTREL = "Kestrel launch checklist: the bank partner demo is on Friday the 23rd. Arjun signs off the launch.\n"
LEASE = "Flat lease for 304, Lavender Court. Rent 38,000 rupees a month. Lease ends 31 May 2027.\n"


@pytest.fixture
def work(tmp_path) -> Path:
    d = tmp_path / "Work notes"
    d.mkdir()
    (d / "kestrel.txt").write_text(KESTREL, encoding="utf-8")
    return d


@pytest.fixture
def home_docs(tmp_path) -> Path:
    d = tmp_path / "Home docs"
    d.mkdir()
    (d / "lease.txt").write_text(LEASE, encoding="utf-8")
    return d


@pytest.fixture
async def app(config, isolated_home):
    llm = FakeProvider()
    llm.embed = topic_embed
    a = SentientApp(config, llm=llm, db_path=isolated_home / "scoped.db", enable_background=False)
    await a.start()
    a.fake = llm
    yield a
    await a.stop()


async def _add(app, path: Path, scope: str | None = None) -> dict:
    folder = await app.folders.add(str(path), scope=scope)
    await app.folders.idle()
    return folder


async def test_a_new_folder_is_used_everywhere_until_given_a_scope(app, work):
    folder = await _add(app, work)
    assert folder["scope"] == "always"
    for scope in (None, "personal", "work", "Atlas"):
        hits = (await app.folders.search("Kestrel bank demo", scope=scope))["results"]
        assert [h["path"] for h in hits] == ["kestrel.txt"], scope
    changed = await app.folders.update(folder["id"], None, "Work")
    assert changed["scope"] == "work"
    assert {s["name"] for s in await scopes.list_scopes(app)} >= {"always", "personal", "work"}
    await app.folders.update(folder["id"], None, "Kestrel launch")
    assert "Kestrel launch" in [s["name"] for s in await scopes.list_scopes(app)]  # a project a folder uses is listed
    await app.folders.update(folder["id"], None, "")
    assert (await app.folders.get(folder["id"]))["scope"] == "always"


async def test_a_work_folder_stays_out_of_personal_chats(app, work, home_docs):
    await _add(app, work, "work")
    await _add(app, home_docs, "personal")

    assert [h["path"] for h in (await app.folders.search("Kestrel bank demo", scope="work"))["results"]] == ["kestrel.txt"]
    personal = await app.folders.search("Kestrel bank demo", scope="personal")
    assert all(h["path"] != "kestrel.txt" for h in personal["results"])
    assert all(h["path"] != "lease.txt" for h in (await app.folders.search("lease rent flat", scope="work"))["results"])

    # asked directly: everywhere, and the result says where it is kept
    wide = await app.folders.search("Kestrel bank demo", scope="personal", everywhere=True)
    assert wide["results"][0]["path"] == "kestrel.txt" and wide["results"][0]["kept_for"] == "Work"
    assert "kept_for" not in (await app.folders.search("Kestrel bank demo", scope="work"))["results"][0]
    # work not tied to a chat reads every folder
    assert {h["path"] for h in (await app.folders.search("Kestrel lease", scope=None))["results"]} >= {"kestrel.txt"}

    # note pieces for the prompt follow the chat's scope
    assert await app.folders.prompt_pieces("when does my lease end?", "work") == []
    assert [p["path"] for p in await app.folders.prompt_pieces("when does my lease end?", "personal")] == ["lease.txt"]
    # a project chat reads only always folders and its own
    assert await app.folders.prompt_pieces("when does my lease end?", "Atlas") == []


async def test_only_folders_of_another_scope_say_so(app, work):
    await _add(app, work, "work")
    out = await app.folders.search("Kestrel", scope="personal")
    assert out["results"] == [] and "kept for other chats (Work)" in out["note"] and "everywhere=true" in out["note"]


async def test_chats_and_tools_use_the_chats_scope(app, work, home_docs):
    await _add(app, work, "work")
    await _add(app, home_docs, "personal")
    work_chat = await app.store.create_session(channel="desktop", memory_scope="work")
    home_chat = await app.store.create_session(channel="desktop")

    _, _, notes = await app.agent._system_prompt_with_sources(
        "when does my lease end?", "desktop", await app.store.get_session(home_chat))
    assert [n["path"] for n in notes] == ["lease.txt"]
    prompt, _, notes = await app.agent._system_prompt_with_sources(
        "when does my lease end?", "desktop", await app.store.get_session(work_chat))
    assert notes == [] and "Lavender Court" not in prompt

    search, read = app.registry.get("folder_search"), app.registry.get("folder_read")
    ctx = app.agent.tool_context(work_chat, "desktop")
    found = await search.call(ctx, {"query": "Kestrel bank demo"})
    assert [r["path"] for r in found["results"]] == ["kestrel.txt"]
    assert all(r["path"] != "lease.txt" for r in (await search.call(ctx, {"query": "lease rent flat"}))["results"])
    lease_file = str(home_docs / "lease.txt")
    refused = await read.call(ctx, {"path": lease_file})
    assert "kept for Personal chats" in refused["error"]
    assert "Lavender Court" in (await read.call(ctx, {"path": lease_file, "everywhere": True}))["content"]
    assert "Kestrel" in (await read.call(ctx, {"path": str(work / "kestrel.txt")}))["content"]

    # a real turn in the work chat that searches: only the work folder answers
    app.fake.replies.extend([[tool_call("folder_search", query="lease rent")], "Nothing in your work notes."])
    _ = [ev async for ev in app.agent.run_turn(work_chat, "what do my notes say about rent?", channel="desktop")]
    tool_msg = next(m for m in app.fake.calls[-1]["messages"] if m.get("role") == "tool")
    assert "Lavender Court" not in tool_msg["content"]


def test_folder_scope_over_rest(config, isolated_home, monkeypatch, work):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        added = c.post("/api/folders", json={"path": str(work), "scope": "work"}).json()
        assert added["scope"] == "work"
        fid = added["id"]
        assert c.patch(f"/api/folders/{fid}", json={"scope": "Kestrel"}).json()["scope"] == "Kestrel"
        assert c.get("/api/folders").json()["folders"][0]["scope"] == "Kestrel"
        assert "Kestrel" in [s["name"] for s in c.get("/api/memories/scopes").json()]
        assert c.patch(f"/api/folders/{fid}", json={"scope": "always"}).json()["scope"] == "always"
        assert c.patch("/api/folders/nope", json={"scope": "work"}).status_code == 404


async def test_folders_added_before_scopes_are_used_everywhere(config, isolated_home, work):
    isolated_home.mkdir(parents=True, exist_ok=True)
    db = isolated_home / "old.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE folders (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, name TEXT NOT NULL,"
        " kind TEXT NOT NULL DEFAULT 'folder', ignore TEXT NOT NULL DEFAULT '[]', added_at TEXT NOT NULL,"
        " last_scan_at TEXT, last_scan_ms INTEGER, skipped INTEGER NOT NULL DEFAULT 0, error TEXT);"
    )
    con.execute("INSERT INTO folders(id, path, name, added_at) VALUES('f1', ?, 'Work notes', '2026-10-01T00:00:00')",
                (str(work),))
    con.execute("PRAGMA user_version = 11")
    con.commit()
    con.close()
    a = SentientApp(config, llm=FakeProvider(), db_path=db, enable_background=False)
    await a.start()
    try:
        [folder] = (await a.folders.overview())["folders"]
        assert folder["scope"] == "always"
    finally:
        await a.stop()
    con = sqlite3.connect(db)
    assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    con.close()


async def test_a_closer_note_in_another_scope_never_hides_this_ones(app, home_docs, tmp_path):
    office = tmp_path / "Office"
    office.mkdir()
    (office / "office lease.txt").write_text("Office lease: lease rent, lease renewal, lease rent lease.\n", encoding="utf-8")
    for i in range(70):  # more matching pieces than one search looks at
        (office / f"desk {i}.txt").write_text(f"Desk {i} lease and rent for the office lease.\n", encoding="utf-8")
    await _add(app, office, "work")
    await _add(app, home_docs, "personal")
    wide = await app.folders.search("lease rent", scope=None)
    assert "lease.txt" not in [r["path"] for r in wide["results"]]  # the office notes fill a search of every folder
    personal = await app.folders.search("lease rent", scope="personal")
    assert [r["path"] for r in personal["results"]] == ["lease.txt"]
    assert [p["path"] for p in await app.folders.prompt_pieces("when does my lease rent end?", "personal")] == ["lease.txt"]

    # nothing here matches: the note names the folders left out, never what is in them
    out = await app.folders.search("orchard apple trees", scope="personal")
    assert out["results"] == [] and "Folders kept for Work chats weren't searched" in out["note"]
    out = await app.folders.search("Office lease", scope="personal", folder="Office")
    assert all(r["path"] != "office lease.txt" for r in out["results"])
