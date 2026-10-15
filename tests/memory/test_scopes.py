"""Memory scopes (#155): work, personal and project memories stay apart, always ones are everywhere."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentient import paths
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.memory import export as memory_export
from sentient.memory import review, scopes
from sentient.store.db import SCHEMA_VERSION
from sentient.tasks import executor
from sentient.tasks.service import TaskConflict
from tests.conftest import FakeProvider
from tests.memory.helpers import drain

SISTER = "Maya's sister Meera lives in Lisbon"
HEALTH = "Maya has asthma and uses an inhaler"
KESTREL = "Maya leads Project Kestrel, a payments launch"
ARJUN = "Arjun is Maya's engineering manager on Kestrel"
NAME = "Maya prefers short, direct answers"
ATLAS = "Project Atlas ships the new onboarding flow"


async def _put(mem, content: str, scope: str) -> int:
    out = await mem.remember(content, use_llm=False, scope=scope)
    assert out["action"] == "ADD", out
    return out["id"]


async def _seed(mem) -> dict[str, int]:
    return {
        "sister": await _put(mem, SISTER, "personal"),
        "health": await _put(mem, HEALTH, "personal"),
        "kestrel": await _put(mem, KESTREL, "work"),
        "arjun": await _put(mem, ARJUN, "work"),
        "name": await _put(mem, NAME, "always"),
        "atlas": await _put(mem, ATLAS, "Atlas"),
    }


def _texts(facts: list[dict]) -> set[str]:
    return {f["content"] for f in facts}


# ---------------------------------------------------------------- helpers
def test_scope_names_and_recall_sets():
    assert scopes.normalize(" Work ") == "work" and scopes.normalize("ALWAYS") == "always"
    assert scopes.normalize("  Kestrel   launch ") == "Kestrel launch"
    assert scopes.normalize("", "personal") == "personal" and scopes.normalize(None) is None
    assert scopes.recall_scopes("work") == ["work", "always"]
    assert scopes.recall_scopes("always") == ["always"] and scopes.recall_scopes(None) is None
    assert scopes.label("work") == "Work" and scopes.label("Kestrel") == "Kestrel" and scopes.label(None) == "Personal"
    with pytest.raises(ValueError):
        scopes.chat_scope("always")
    assert scopes.of_session(None, {"memory_scope": None, "channel": "desktop"}) == "personal"
    user = "# About Maya\n\nMy words.\n\n## Learned\n\n- Maya has asthma\n\n## Later\n- kept\n"
    assert scopes.without_learned(user) == "# About Maya\n\nMy words.\n\n## Later\n- kept"


# ---------------------------------------------------------------- recall and saving
async def test_recall_reads_the_scope_plus_always(app):
    mem = app.memory
    await _seed(mem)
    query = "what do you know about Maya sister health Kestrel Arjun answers Atlas"
    work = _texts(await mem.recall(query, top_k=20, min_similarity=0.0, scopes=scopes.recall_scopes("work")))
    assert work == {KESTREL, ARJUN, NAME}
    personal = _texts(await mem.recall(query, top_k=20, min_similarity=0.0, scopes=scopes.recall_scopes("personal")))
    assert personal == {SISTER, HEALTH, NAME}
    atlas = _texts(await mem.recall(query, top_k=20, min_similarity=0.0, scopes=scopes.recall_scopes("Atlas")))
    assert atlas == {ATLAS, NAME}  # a project's memory is never recalled in another project, and vice versa
    everywhere = _texts(await mem.recall(query, top_k=20, min_similarity=0.0))
    assert everywhere == {SISTER, HEALTH, KESTREL, ARJUN, NAME, ATLAS}
    assert _texts(await mem.list_facts(scope="work")) == {KESTREL, ARJUN}
    assert _texts(await mem.list_facts(scope="Work", q="Kestrel")) == {KESTREL, ARJUN}


async def test_saving_lands_in_its_scope_and_never_changes_another(app):
    mem = app.memory
    await _put(mem, "Maya lives in Pune", "personal")
    # the same words in work are a work memory of their own, not a duplicate of the personal one
    again = await mem.remember("Maya lives in Pune", use_llm=False, scope="work")
    assert again["action"] == "ADD" and again["scope"] == "work"
    # a correction in a work chat updates the work memory, never the personal one
    moved = await mem.remember("Maya lives in Mumbai", use_llm=True, scope="work")
    assert moved["action"] == "UPDATE" and moved["id"] == again["id"]
    rows = {f["scope"]: f["content"] for f in await mem.list_facts()}
    assert rows == {"personal": "Maya lives in Pune", "work": "Maya lives in Mumbai"}
    # an always memory is shared, so a save anywhere still finds it
    name = await _put(mem, NAME, "always")
    assert (await mem.remember(NAME, use_llm=False, scope="work")) == {"action": "SKIP", "id": name, "content": NAME}


async def test_move_and_reset_a_scope(app):
    mem = app.memory
    ids = await _seed(mem)
    async with app.bus.subscribe() as q:
        moved = await mem.set_scope(ids["arjun"], "Kestrel")
        events = drain(q)
    assert moved["scope"] == "Kestrel"
    assert events[-1]["data"] | {"id": None} == {
        "action": "UPDATE", "id": None, "content": ARJUN, "scope": "Kestrel", "previous_scope": "work", "reason": "moved"
    }
    with pytest.raises(KeyError):
        await mem.set_scope(99999, "work")
    # resetting work deletes only work (also what waits for review there)
    held = await mem.remember("Maya's standup moved to 9:30", scope="work", use_llm=False, review=review.note("Gmail"))
    assert held["status"] == "pending"
    assert await mem.forget_scope("work") == 2
    assert _texts(await mem.list_facts()) == {SISTER, HEALTH, NAME, ATLAS, ARJUN}
    assert await mem.pending_facts() == []
    assert await mem.forget_scope("nothing here") == 0


async def test_a_held_memory_is_approved_into_its_scope(app):
    mem = app.memory
    held = await mem.remember(KESTREL, scope="work", use_llm=False, review=review.note("Gmail"))
    [item] = (await review.inbox(app))["items"]
    assert item["scope"] == "work"
    await review.approve(app, "fact", str(held["id"]))
    assert (await mem.get_fact(held["id"]))["scope"] == "work"


# ---------------------------------------------------------------- chats
async def _turn(app, sid: str, text: str) -> list:
    events = [ev async for ev in app.agent.run_turn(sid, text, channel="desktop")]
    await app.agent.drain(timeout=5)
    return events


async def test_a_work_chat_uses_work_memories_and_saves_into_work(app):
    mem, llm = app.memory, app.fake
    await _seed(mem)
    app.workspace.write("memory", "# Long-term memory\n\n- Maya's sister Meera visits in May\n")
    app.workspace.write("user", "# About Maya\n\nMaya Rao, product lead.\n\n## Learned\n\n- Maya has asthma\n")
    app.config.memory.min_similarity = 0.0
    app.config.memory.extract_after_turn = True
    await app.user_model.add_insight("Maya is very close to her sister Meera", "relationships")
    await app.user_model.add_insight("Maya likes answers as short bullet points", "communication")
    work = await app.store.create_session(channel="desktop", memory_scope="work")
    llm.replies.append("You lead Kestrel and Arjun is your manager.")
    llm.json_replies.extend([
        {"facts": ["Maya's Kestrel launch review is on Thursday"]},
        {"action": "ADD", "fact_id": None, "content": "Maya's Kestrel launch review is on Thursday",
         "analysis": {"topics": ["Work & Learning"], "memory_type": "long-term", "duration": None}},
    ])
    await _turn(app, work, "What do you know about me? My Kestrel launch review is on Thursday.")
    prompt = llm.calls[0]["messages"][0]["content"]
    assert KESTREL in prompt and ARJUN in prompt and NAME in prompt
    for personal in (SISTER, HEALTH, "Meera visits in May", "Maya has asthma", "close to her sister"):
        assert personal not in prompt
    assert "short bullet points" in prompt  # how she likes things still holds at work
    assert "Maya Rao, product lead." in prompt  # the user's own words in USER.md stay
    assert "This chat uses the user's Work memories" in prompt
    saved = next(f for f in await mem.list_facts() if "review is on Thursday" in f["content"])
    assert saved["scope"] == "work"

    # a personal chat sees personal memories and the profile files, and nothing from work
    llm.calls.clear()
    llm.replies.append("Your sister Meera lives in Lisbon.")
    personal = await app.store.create_session(channel="desktop")
    app.config.memory.extract_after_turn = False
    await _turn(app, personal, "What do you know about me and my sister and Kestrel?")
    prompt = llm.calls[0]["messages"][0]["content"]
    assert SISTER in prompt and NAME in prompt and "Meera visits in May" in prompt and "close to her sister" in prompt
    assert KESTREL not in prompt and ARJUN not in prompt and "Memory scope" not in prompt


async def test_tools_follow_the_chats_scope_unless_asked_everywhere(app):
    mem = app.memory
    await _seed(mem)
    app.config.memory.min_similarity = 0.0
    work = await app.store.create_session(channel="desktop", memory_scope="work")
    ctx = app.agent.tool_context(work, "subagent")  # a helper the chat started carries only its id
    recall = app.registry.get("memory_recall")
    found = await recall.call(ctx, {"query": "Maya sister asthma Kestrel Arjun answers", "limit": 20})
    assert {f["fact"] for f in found} == {KESTREL, ARJUN, NAME}
    found = await recall.call(ctx, {"query": "Maya sister asthma", "limit": 20, "everywhere": True})
    health = next(f for f in found if f["fact"] == HEALTH)
    assert health["kept_for"] == "Personal"
    out = await app.registry.get("memory_remember").call(ctx, {"fact": "Maya's Kestrel demo is on Friday"})
    assert (await mem.get_fact(out["id"]))["scope"] == "work"
    # work not tied to a chat or task reads every scope
    proactive = app.agent.tool_context(None, "proactive", origin="proactive")
    found = await recall.call(proactive, {"query": "Maya sister asthma Kestrel", "limit": 20})
    assert {SISTER, KESTREL} <= {f["fact"] for f in found}


async def test_a_messaging_app_can_be_set_to_a_scope(app):
    app.config.channels.telegram.memory_scope = "Work"
    sid = await app.store.create_session(channel="telegram")
    assert scopes.of_session(app.config, await app.store.get_session(sid)) == "work"
    await app.store.set_session_scope(sid, "Atlas")  # the chat's own pick wins
    assert scopes.of_session(app.config, await app.store.get_session(sid)) == "Atlas"
    other = await app.store.create_session(channel="discord")
    assert scopes.of_session(app.config, await app.store.get_session(other)) == "personal"


async def test_an_incognito_work_chat_reads_work_and_saves_nothing(app):
    mem, llm = app.memory, app.fake
    await _seed(mem)
    app.config.memory.min_similarity = 0.0
    sid = await app.store.create_session(channel="desktop", incognito=True, memory_scope="work")
    llm.replies.append("ok")
    await _turn(app, sid, "Tell me about Kestrel and my sister's health")
    prompt = llm.calls[0]["messages"][0]["content"]
    assert KESTREL in prompt and SISTER not in prompt and "incognito chat" in prompt
    assert len(await mem.list_facts()) == 6


# ---------------------------------------------------------------- tasks
async def test_tasks_keep_the_scope_of_the_chat_that_made_them(app, monkeypatch):
    mem = app.memory
    await _seed(mem)
    app.config.memory.min_similarity = 0.0
    monkeypatch.setattr(app.tasks, "_spawn", lambda coro, name: coro.close())  # no planning in the background
    work = await app.store.create_session(channel="desktop", memory_scope="work")
    out = await app.registry.get("create_task_from_prompt").call(
        app.agent.tool_context(work, "desktop"), {"prompt": "Every Friday summarise Kestrel and Arjun's notes"}
    )
    assert (await app.tasks.get(out["task_id"]))["memory_scope"] == "work"
    task = await app.tasks.repo.get_task(out["task_id"])
    _, facts, _ = await executor._executor_system_prompt(app.tasks, task, {"id": "run1"}, [], {})
    assert _texts(facts) <= {KESTREL, ARJUN, NAME} and KESTREL in _texts(facts)
    plain = await app.tasks.create_task("Summarise my week, my sister and Kestrel")
    assert plain["memory_scope"] == "personal"
    with pytest.raises(ValueError):
        await app.tasks.update(plain["task_id"], {"memory_scope": "always"})
    updated = await app.tasks.update(plain["task_id"], {"memory_scope": "Atlas"})
    assert updated["memory_scope"] == "Atlas"
    await app.tasks.repo.update_task(plain["task_id"], {"status": "processing"})
    with pytest.raises(TaskConflict):  # a run keeps the memories it started with
        await app.tasks.update(plain["task_id"], {"memory_scope": "work"})


# ---------------------------------------------------------------- the rest of memory
async def test_dreaming_never_merges_across_scopes(app):
    mem = app.memory
    app.config.user_model.enabled = False
    a = await _put(mem, "Maya drinks black coffee every morning", "personal")
    b = await mem.remember("Every morning Maya drinks black coffee.", use_llm=False, scope="work")
    dream = await app.dreaming.run()
    assert dream["stats"]["merged"] == 0
    assert await mem.get_fact(a) is not None and await mem.get_fact(b["id"]) is not None


async def test_profile_files_are_built_from_personal_and_always_only(app):
    mem = app.memory
    for content, scope in ((HEALTH, "personal"), (KESTREL, "work"), (NAME, "always")):
        fid = await _put(mem, content, scope)
        await app.store.execute("UPDATE facts SET topics = ? WHERE id = ?", ('["Personal Identity"]', fid))
    app.fake.text_replies.append("# Long-term memory\n\n- notes\n")
    await app.evolution.update_profile(force=True)
    sent = app.fake.text_calls[-1]["messages"][1]["content"]
    assert HEALTH in sent and NAME in sent and KESTREL not in sent
    learned = app.workspace.user.read_text(encoding="utf-8")
    assert HEALTH in learned and KESTREL not in learned


async def test_export_says_where_each_memory_is_kept(app):
    await _seed(app.memory)
    md = await memory_export.export(app, "markdown")
    text = Path(md["path"]).read_text(encoding="utf-8")
    assert f"- {KESTREL} _(Work, " in text and f"- {SISTER} _(a chat" in text
    db = await memory_export.export(app, "sqlite")
    con = sqlite3.connect(db["path"])
    try:
        rows = dict(con.execute("SELECT content, scope FROM memories").fetchall())
    finally:
        con.close()
    assert rows[KESTREL] == "work" and rows[SISTER] == "personal" and rows[ATLAS] == "Atlas"


async def test_older_memories_become_personal(config, isolated_home):
    """An older database (and a backup made by an older build) gets every memory in personal."""
    app = SentientApp(config, llm=FakeProvider(), enable_background=False)
    await app.start()
    await app.stop()
    con = sqlite3.connect(paths.db_file())
    con.execute("DROP INDEX idx_facts_scope")
    con.execute("ALTER TABLE facts DROP COLUMN scope")
    con.execute("ALTER TABLE sessions DROP COLUMN memory_scope")
    con.execute(
        "INSERT INTO facts(content, source, topics, memory_type, created_at, updated_at) VALUES"
        " ('Maya has asthma', 'conversation', '[]', 'long-term', '2026-01-01', '2026-01-01')"
    )
    con.execute("PRAGMA user_version = 10")
    con.commit()
    con.close()
    app = SentientApp(config, llm=FakeProvider(), enable_background=False)
    await app.start()
    try:
        row = await app.store.fetchone("SELECT scope FROM facts WHERE content = 'Maya has asthma'")
        assert row["scope"] == "personal"
        sid = await app.store.create_session(channel="desktop")
        assert (await app.store.get_session(sid))["memory_scope"] is None
    finally:
        await app.stop()
    con = sqlite3.connect(paths.db_file())
    assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    con.close()


# ---------------------------------------------------------------- REST
@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider(replies=["one", "two", "three"])
    core = SentientApp(config, llm=llm, db_path=isolated_home / "scopes.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        c.core = core
        yield c


def test_scopes_over_rest(client):
    personal = client.post("/api/memories", json={"content": SISTER, "source": "manual"}).json()
    work = client.post("/api/memories", json={"content": KESTREL, "scope": "Work"}).json()
    atlas = client.post("/api/memories", json={"content": ATLAS, "scope": "Atlas"}).json()
    assert [m["content"] for m in client.get("/api/memories", params={"scope": "work"}).json()] == [KESTREL]
    listed = {s["name"]: s for s in client.get("/api/memories/scopes").json()}
    assert list(listed)[:3] == ["always", "personal", "work"]
    assert listed["work"]["count"] == 1 and listed["Atlas"] == {
        "name": "Atlas", "label": "Atlas", "count": 1, "built_in": False
    }
    moved = client.patch(f"/api/memories/{personal['id']}", json={"scope": "always"}).json()
    assert moved["scope"] == "always" and moved["content"] == SISTER
    assert client.patch(f"/api/memories/{personal['id']}", json={"scope": ""}).status_code == 400
    assert client.patch("/api/memories/99999", json={"scope": "work"}).status_code == 404
    assert client.delete("/api/memories/scope/work").json() == {"deleted": 1}
    assert client.get("/api/memories/scopes").json()[2] | {"count": 0} == {
        "name": "work", "label": "Work", "count": 0, "built_in": True
    }
    assert {m["id"] for m in client.get("/api/memories").json()} == {personal["id"], atlas["id"]}
    assert work["id"] not in {m["id"] for m in client.get("/api/memories").json()}


def test_a_chats_scope_over_rest(client):
    sid = client.post("/api/sessions", params={"memory_scope": "Work"}).json()["session_id"]
    row = next(s for s in client.get("/api/sessions").json() if s["id"] == sid)
    assert row["memory_scope"] == "work"
    with client.websocket_connect("/ws?token=test-token") as ws:
        assert ws.receive_json()["type"] == "hello"
        assert client.patch(f"/api/sessions/{sid}", json={"memory_scope": "Kestrel"}).json() == {
            "ok": True, "memory_scope": "Kestrel"
        }
        ev = ws.receive_json()
        while ev.get("type") != "session.updated":
            ev = ws.receive_json()
        assert ev["data"] == {"session_id": sid, "memory_scope": "Kestrel"}
    assert client.patch(f"/api/sessions/{sid}", json={"memory_scope": None}).json()["memory_scope"] == "personal"
    assert client.patch(f"/api/sessions/{sid}", json={"memory_scope": "always"}).status_code == 422
    assert client.post("/api/sessions", params={"memory_scope": "always"}).status_code == 422
    plain = client.post("/api/sessions").json()["session_id"]
    assert next(s for s in client.get("/api/sessions").json() if s["id"] == plain)["memory_scope"] == "personal"
    assert json.dumps(client.get("/api/memories/review").json()["items"]) == "[]"


async def test_profile_upkeep_leaves_out_work_chat_summaries(app):
    app.config.channels.telegram.memory_scope = "work"
    personal = await app.store.create_session(channel="desktop")
    work = await app.store.create_session(channel="desktop", memory_scope="Kestrel")
    telegram = await app.store.create_session(channel="telegram")  # work through its app's scope
    for sid, text in ((personal, "We planned Meera's visit"), (work, "We reviewed the Kestrel launch plan"),
                      (telegram, "We triaged Arjun's payment bugs")):
        await app.store.execute(
            "INSERT INTO summaries(id, session_id, content, start_at, end_at, message_ids, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (f"s-{sid}", sid, text, "2026-10-01", "2026-10-01", "[]", "2026-10-01T00:00:00+00:00"),
        )
    await _put(app.memory, HEALTH, "personal")
    app.fake.text_replies.append("# Long-term memory\n\n- notes\n")
    await app.evolution.update_profile(force=True)
    sent = app.fake.text_calls[-1]["messages"][1]["content"]
    assert "Meera's visit" in sent and "Kestrel launch plan" not in sent and "payment bugs" not in sent


async def test_moving_a_memory_out_of_personal_takes_it_out_of_user_md(client):
    core = client.core
    added = client.post("/api/memories", json={"content": HEALTH}).json()
    core.workspace.write("user", f"# About Maya\n\n- {HEALTH} (my words)\n\n## Learned\n\n- {HEALTH}\n- {SISTER}\n")
    client.patch(f"/api/memories/{added['id']}", json={"scope": "work"})
    text = core.workspace.user.read_text(encoding="utf-8")
    assert f"- {SISTER}" in text and f"- {HEALTH} (my words)" in text and f"- {HEALTH}\n" not in text


def test_a_bad_scope_changes_nothing_else(client):
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.patch(f"/api/sessions/{sid}", json={"title": "Renamed", "memory_scope": "always"})
    assert r.status_code == 422
    assert next(s for s in client.get("/api/sessions").json() if s["id"] == sid)["title"] != "Renamed"
