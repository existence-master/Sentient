from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from sentient.gateway.app import create_app
from tests.evolution.conftest import SKILL_BODY, add_tool_chat

PROPOSAL = {
    "decision": "create", "name": "weekly-inbox-digest", "description": "Build a digest of important unread email.",
    "tags": ["email"], "requires_tools": ["gmail", "not-a-plugin"], "reason": "repeatable", "body": SKILL_BODY,
}


async def test_reviewer_creates_pending_skill(app):
    sid = await add_tool_chat(app, tool_calls=3)
    app.fake.json_replies.append(PROPOSAL)
    res = await app.evolution.review_session(sid)
    assert res == {"reviewed": True, "decision": "create", "skill": "weekly-inbox-digest"}
    pending = app.skills.get_pending("weekly-inbox-digest")
    assert pending and pending.created_by_review and pending.author == "assistant"
    assert "not-a-plugin" not in pending.requires_tools  # unknown plugins are dropped so the skill stays visible
    for sec in ("When to use", "Procedure", "Pitfalls", "Verification"):
        assert f"## {sec}" in pending.body
    transcript = app.fake.calls[-1]["messages"][1]["content"]
    assert "TOOL CALL gmail_search" in transcript and "digest" in transcript
    notes = await app.notifications.list()
    assert notes[0]["kind"] == "skill" and notes[0]["payload"]["skill"] == "weekly-inbox-digest"
    log = await app.store.fetchall("SELECT kind, detail FROM evolution_log")
    assert log[0]["kind"] == "skill_created" and json.loads(log[0]["detail"])["session_id"] == sid
    # already reviewed up to the last message: no second review
    calls = len(app.fake.calls)
    assert (await app.evolution.review_session(sid))["decision"] == "skipped"
    assert len(app.fake.calls) == calls


async def test_reviewer_ignores_trivial_chats(app):
    sid = await app.store.create_session()
    await app.store.add_message(sid, "user", "hi")
    await app.store.add_message(sid, "assistant", "hello!")
    one_tool = await add_tool_chat(app, tool_calls=1)
    assert (await app.evolution.review_session(sid))["reviewed"] is False
    assert (await app.evolution.review_session(one_tool, force=True))["reviewed"] is False
    assert app.fake.calls == [] and app.skills.list_pending() == []


async def test_reviewer_rejects_incomplete_and_patches_existing(app):
    sid = await add_tool_chat(app)
    app.fake.json_replies.append({"decision": "create", "name": "x-thing", "description": "d", "body": "too short"})
    assert (await app.evolution.review_session(sid))["decision"] == "none"
    app.skills.write("weekly-inbox-digest", "Old digest", "## Procedure\n1. old", author="user")
    app.skills.reload()
    sid2 = await add_tool_chat(app)
    app.fake.json_replies.append({**PROPOSAL, "decision": "create", "body": SKILL_BODY + "\nExtra pitfall."})
    res = await app.evolution.review_session(sid2)
    assert res["decision"] == "patch"
    diff = app.skills.diff("weekly-inbox-digest")
    assert "old" in diff["current"] and "Extra pitfall" in diff["proposed"] and "version: '2'" in diff["proposed"]
    assert app.skills.get_pending("weekly-inbox-digest").author == "user"


async def test_sessions_due_uses_idle_time_and_tool_count(app):
    sid = await add_tool_chat(app)
    now = datetime.now(UTC)
    assert await app.evolution.sessions_due(now) == []  # not idle yet
    assert await app.evolution.sessions_due(now + timedelta(minutes=11)) == [sid]
    app.fake.json_replies.append({"decision": "none", "reason": "one-off"})
    out = await app.evolution.review_now()
    assert out == {"reviewed": 1, "proposed": []}


async def test_bus_hooks_record_turns_and_task_runs(app):
    app.evolution.on_event({"type": "chat.turn_completed", "data": {"session_id": "s1", "tool_calls": 2}})
    app.evolution.on_event({"type": "chat.turn_completed", "data": {"session_id": "s1", "tool_calls": 2}})
    assert app.evolution._dirty_sessions["s1"]["tool_calls"] == 4
    app.evolution.on_event({"type": "task.updated", "data": {"task_id": "t1", "status": "completed",
                                                              "runs": [{"run_id": "r1", "status": "completed"}]}})
    assert ("t1", "r1") in app.evolution._task_runs


async def test_review_task_run_from_tasks_service(app, monkeypatch):
    run = {"run_id": "r1", "status": "completed", "finished_at": "2026-09-15T10:00:00+00:00",
           "progress_updates": [
               {"message": {"type": "tool_call", "tool_name": "gmail_search", "parameters": {"q": "x"}}},
               {"message": {"type": "tool_result", "tool_name": "gmail_search", "result": "4 threads"}},
           ] * 3,
           "result": {"summary": "Digest sent"}}

    class Tasks:
        name = "fake-tasks"

        async def stop(self):
            return None

        async def get(self, task_id):
            return {"task_id": task_id, "name": "Digest", "description": "weekly", "runs": [run]}

    monkeypatch.setattr(app, "tasks", Tasks())
    app.fake.json_replies.append(PROPOSAL)
    res = await app.evolution.review_task_run("t1", "r1")
    assert res["skill"] == "weekly-inbox-digest"
    assert (await app.evolution.review_task_run("t1", "r1"))["reviewed"] is False


async def test_skill_lifecycle_approve_reject_edit_archive_restore(app):
    lib = app.skills
    lib.write("weekly-inbox-digest", "Digest", SKILL_BODY, pending=True, created_by_review=True)
    lib.approve_pending("weekly-inbox-digest")
    lib.reload()
    assert lib.get("weekly-inbox-digest").version == "1"
    edited = lib.edit("weekly-inbox-digest", description="Weekly digest of unread mail")
    assert edited.version == "2" and edited.description == "Weekly digest of unread mail"
    lib.write("other-skill", "Other", "## Procedure\n1. y", pending=True)
    lib.reject_pending("other-skill")
    assert lib.get_any("other-skill") is None
    lib.archive("weekly-inbox-digest")
    lib.reload()
    assert lib.get("weekly-inbox-digest") is None and lib.get_archived("weekly-inbox-digest")
    lib.restore("weekly-inbox-digest")
    lib.reload()
    assert lib.get("weekly-inbox-digest") is not None
    assert lib.delete("weekly-inbox-digest") and lib.get_any("weekly-inbox-digest") is None


async def test_skill_tools_count_use_and_stage_patches(app):
    ctx = app.agent.tool_context("sess", "desktop")
    app.skills.write("weekly-inbox-digest", "Digest", SKILL_BODY, author="user")
    app.skills.reload()
    view = await app.registry.get("skill_view").call(ctx, {"name": "weekly-inbox-digest"})
    assert "Procedure" in view["body"]
    stats = (await app.skills.stats())["weekly-inbox-digest"]
    assert stats["use_count"] == 1 and stats["view_count"] == 1 and stats["last_used_at"]
    app.config.skills.write_approval = True
    out = await app.registry.get("skill_save").call(
        ctx, {"name": "weekly-inbox-digest", "description": "Digest v2", "body": SKILL_BODY + "\n- more"}
    )
    assert out["action"] == "patched" and out["pending_review"]
    assert app.skills.get("weekly-inbox-digest").description == "Digest"  # active untouched until approved
    assert (await app.skills.stats())["weekly-inbox-digest"]["state"] == "active"
    app.config.skills.write_approval = False
    out = await app.registry.get("skill_save").call(ctx, {"name": "New Skill!", "description": "d", "body": "## Procedure\n1. a"})
    assert out["name"] == "new-skill" and not out["pending_review"] and app.skills.get("new-skill")


async def test_curator_lifecycle_with_injected_clock(app):
    lib, evo = app.skills, app.evolution
    app.config.evolution.curator_merge_suggestions = False
    lib.write("old-assistant-skill", "Old", "## Procedure\n1. a", author="assistant")
    lib.write("user-skill", "Mine", "## Procedure\n1. b", author="user")
    lib.write("busy-skill", "Busy", "## Procedure\n1. c", author="assistant")
    lib.reload()
    await lib.sync_stats()
    t0 = datetime.now(UTC)
    await lib.mark_used("busy-skill", when=t0 + timedelta(days=14))
    await lib.mark_used("user-skill", when=t0 + timedelta(days=1))

    r1 = await evo.run_curator(t0 + timedelta(days=15))
    assert set(r1["staled"]) == {"old-assistant-skill", "user-skill"} and r1["archived"] == []
    r2 = await evo.run_curator(t0 + timedelta(days=31))
    assert r2["archived"] == ["old-assistant-skill"]  # user skill used on day 1 is protected
    assert lib.get_archived("old-assistant-skill") is not None
    # using a stale skill makes it active again
    await lib.mark_used("user-skill", when=t0 + timedelta(days=32))
    assert (await lib.stats())["user-skill"]["state"] == "active"
    kinds = [r["kind"] for r in await app.store.fetchall("SELECT kind FROM evolution_log ORDER BY id")]
    assert kinds.count("curator_run") == 2 and "skill_archived" in kinds


async def test_curator_merge_proposal(app):
    lib = app.skills
    lib.write("email-digest", "weekly digest of unread email", "## Procedure\n1. a", author="assistant")
    lib.write("email-digests", "weekly digest of unread email", "## Procedure\n1. b", author="assistant")
    lib.reload()
    app.fake.json_replies.append({"name": "email-digest", "description": "Weekly digest of unread email", "body": SKILL_BODY})
    out = await app.evolution.run_curator()
    assert out["merge_proposals"] == ["email-digest"]
    assert app.skills.get_pending("email-digest") is not None


async def test_profile_upkeep_appends_without_clobbering(app):
    ws, mem = app.workspace, app.memory
    ws.write("user", "# About Sarthak\n\nMy own words: I like quiet mornings.\n")
    a = await mem.remember("Sarthak is training for a marathon", use_llm=False)
    await app.store.execute("UPDATE facts SET topics = ? WHERE id = ?", ('["Health & Wellbeing"]', a["id"]))
    await mem.remember("Sarthak random misc fact", use_llm=False)
    app.fake.text_replies.append("# Long-term memory\n\n## Goals\n- Sarthak is training for a marathon.\n")
    out = await app.evolution.update_profile(force=True)
    assert out["updated"] and out["learned_appended"] == 1
    user_md = ws.user.read_text(encoding="utf-8")
    assert "My own words: I like quiet mornings." in user_md
    assert "## Learned\n\n- Sarthak is training for a marathon" in user_md
    assert "random misc" not in user_md
    assert ws.memory.read_text(encoding="utf-8").startswith("# Long-term memory")
    # nothing new: no rewrite, no duplicate bullets
    assert (await app.evolution.update_profile(force=True))["updated"] is False
    b = await mem.remember("Sarthak mentors two junior developers", use_llm=False)
    await app.store.execute("UPDATE facts SET topics = ? WHERE id = ?", ('["Work & Learning"]', b["id"]))
    app.fake.text_replies.append("no heading but long enough content for the memory file here")
    await app.evolution.update_profile(force=True)
    user_md = ws.user.read_text(encoding="utf-8")
    assert user_md.count("## Learned") == 1 and user_md.count("marathon") == 1 and "mentors two junior" in user_md
    assert ws.memory.read_text(encoding="utf-8").startswith("# Long-term memory")


def test_skills_routes(config, isolated_home, monkeypatch):
    from sentient.app import SentientApp
    from tests.conftest import FakeProvider

    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider()
    core = SentientApp(config, llm=llm, db_path=isolated_home / "sk.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        created = c.post("/api/skills", json={"name": "weekly-review", "description": "Review week", "body": SKILL_BODY}).json()
        assert created["state"] == "active" and created["author"] == "user" and "body" in created
        assert c.post("/api/skills", json={"name": "weekly-review", "description": "x", "body": "y"}).status_code == 409
        assert c.post("/api/skills", json={"name": "Bad Name", "description": "x", "body": "y"}).status_code == 400
        upd = c.put("/api/skills/weekly-review", json={"description": "Review my week"}).json()
        assert upd["version"] == "2" and upd["patch_count"] == 1

        core.skills.write("weekly-review", "Proposed", SKILL_BODY + "\n- new", pending=True, created_by_review=True)
        core.skills.write("fresh-skill", "Fresh", SKILL_BODY, pending=True, created_by_review=True)
        listing = c.get("/api/skills").json()
        assert [s["name"] for s in listing["active"]] == ["weekly-review"]
        assert {s["name"] for s in listing["pending"]} == {"weekly-review", "fresh-skill"}
        diff = c.get("/api/skills/weekly-review/diff").json()
        assert "Review my week" in diff["current"] and "- new" in diff["proposed"]
        assert c.post("/api/skills/fresh-skill/reject").json() == {"ok": True}
        assert c.post("/api/skills/weekly-review/approve").json()["description"] == "Proposed"
        assert c.get("/api/skills/weekly-review/diff").status_code == 404
        assert c.post("/api/skills/weekly-review/archive").json()["state"] == "archived"
        assert c.get("/api/skills").json()["archived"][0]["name"] == "weekly-review"
        assert c.post("/api/skills/weekly-review/restore").json()["state"] == "active"
        assert c.get("/api/skills/weekly-review").json()["body"]
        log = c.get("/api/skills/evolution-log").json()
        assert {e["kind"] for e in log} >= {"skill_created", "skill_patched", "skill_archived"}
        llm.json_replies.append({"decision": "none"})
        assert c.post("/api/skills/review-now", json={}).json() == {"reviewed": 0, "proposed": []}
        assert c.delete("/api/skills/weekly-review").json() == {"ok": True}
        assert c.get("/api/skills/weekly-review").status_code == 404
