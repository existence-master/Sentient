"""Skills that repair themselves (docs/API.md section 15) and skill usage outcomes."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from sentient.evolution.service import looks_like_correction
from sentient.gateway.app import create_app
from tests.evolution.conftest import SKILL_BODY, add_tool_chat

NAME = "weekly-inbox-digest"
REPAIR = {
    "decision": "patch",
    "reason": "Search needs the inbox filter.",
    "description": "Build a digest of important unread email.",
    "body": SKILL_BODY.replace("`is:unread newer_than:7d`", "`in:inbox is:unread newer_than:7d`")
    + "- Gmail search without `in:inbox` returns archived mail.\n",
}


def turn(sid: str, turn_id: str, **extra) -> dict:
    data = {"session_id": sid, "turn_id": turn_id, "tool_calls": 3, "tool_errors": 0, "skills_viewed": [NAME],
            "user_text": "Make me a digest of my unread email", "reply": "Here is your digest."}
    return {"type": "chat.turn_completed", "data": {**data, **extra}}


def add_skill(app) -> None:
    app.skills.write(NAME, "Build a digest of important unread email.", SKILL_BODY, author="user")
    app.skills.reload()


async def stats(app) -> dict:
    return (await app.skills.stats())[NAME]


async def test_tool_errors_propose_a_pending_repair(app):
    add_skill(app)
    sid = await add_tool_chat(app)
    app.fake.json_replies.append(REPAIR)
    errors = [{"name": "gmail_search", "error": "HTTP 400: invalid query"}]
    out = await app.evolution.handle_repair_event(turn(sid, "t1", tool_errors=errors))
    assert out == [NAME]

    # pending with a diff against the active skill, which is untouched
    assert app.skills.get_active_file(NAME).body == SKILL_BODY.strip()
    diff = app.skills.diff(NAME)
    assert "`is:unread newer_than:7d`" in diff["current"] and "in:inbox" in diff["proposed"]
    assert app.skills.get_pending(NAME).author == "user"

    prompt = app.fake.calls[-1]["messages"][1]["content"]
    assert "HTTP 400: invalid query" in prompt and "TOOL CALL gmail_search" in prompt and "## Procedure" in prompt

    note = (await app.notifications.list())[0]
    assert note["kind"] == "skill" and note["title"] == "Skill fix to review"
    assert note["payload"]["origin"] == "repair" and note["payload"]["skill"] == NAME
    assert "1 tool call failed" in note["message"]

    row = await app.store.fetchone("SELECT kind, detail FROM evolution_log WHERE kind = 'skill_repair_proposed'")
    detail = json.loads(row["detail"])
    assert detail["origin"] == "repair" and detail["session_id"] == sid and detail["failure"] == "tool_errors"
    assert "1 tool call failed" in detail["reason"] and "inbox filter" in detail["reason"]
    st = await stats(app)
    assert st["failure_count"] == 1 and st["success_count"] == 0 and st["state"] == "active"


async def test_user_correction_on_next_message(app):
    add_skill(app)
    sid = await add_tool_chat(app)
    evo = app.evolution
    assert await evo.handle_repair_event(turn(sid, "t1")) == []
    assert app.fake.calls == [] and (await stats(app))["success_count"] == 1

    # an ordinary follow-up: no model call, success stands
    assert await evo.handle_repair_event(turn(sid, "t2", skills_viewed=[], user_text="Great, now add Slack too")) == []
    assert app.fake.calls == []

    await evo.handle_repair_event(turn(sid, "t3"))
    await app.store.add_message(sid, "user", "No, that's wrong, you forgot the archived threads")
    app.fake.json_replies += [{"correction": True, "what_went_wrong": "missed archived threads"}, REPAIR]
    out = await evo.handle_repair_event(
        turn(sid, "t4", skills_viewed=[], user_text="No, that's wrong, you forgot the archived threads")
    )
    assert out == [NAME]
    assert "No, that's wrong" in app.fake.calls[0]["messages"][1]["content"]
    st = await stats(app)
    assert st["success_count"] == 1 and st["failure_count"] == 1  # t3 success turned into a failure
    detail = json.loads((await app.store.fetchone(
        "SELECT detail FROM evolution_log WHERE kind = 'skill_repair_proposed'"))["detail"])
    assert detail["failure"] == "user_correction" and detail["turn_id"] == "t3"
    assert detail["reason"].startswith("You corrected the result: missed archived threads")


async def test_correction_not_confirmed_by_model(app):
    add_skill(app)
    sid = await add_tool_chat(app)
    await app.evolution.handle_repair_event(turn(sid, "t1"))
    app.fake.json_replies.append({"correction": False, "what_went_wrong": ""})
    assert await app.evolution.handle_repair_event(turn(sid, "t2", skills_viewed=[], user_text="No worries, try again tomorrow")) == []
    assert len(app.fake.calls) == 1 and app.skills.get_pending(NAME) is None
    assert (await stats(app))["failure_count"] == 0


async def test_failed_task_run_proposes_repair_and_success_is_counted(app, monkeypatch):
    add_skill(app)
    run = {"run_id": "r1", "status": "error", "finished_at": "2026-09-15T10:00:00+00:00",
           "progress_updates": [
               {"message": {"type": "tool_call", "tool_name": "gmail_search", "parameters": {"q": "is:unread"}}},
               {"message": {"type": "tool_result", "tool_name": "gmail_search", "result": "error: label missing"}},
           ]}

    class Tasks:
        name = "fake-tasks"

        async def stop(self):
            return None

        async def get(self, task_id):
            return {"task_id": task_id, "name": "Digest", "description": "weekly", "runs": [run]}

    monkeypatch.setattr(app, "tasks", Tasks())
    evo = app.evolution
    ok = {"type": "task.run_finished", "data": {"task_id": "t1", "run_id": "r0", "status": "completed",
                                                 "tool_errors": 0, "skills_viewed": [NAME]}}
    assert await evo.handle_repair_event(ok) == []
    assert app.fake.calls == [] and (await stats(app))["success_count"] == 1
    evo.on_event(ok)  # the bus hook: completed runs still queue a normal review
    assert ("t1", "r0") in evo._task_runs

    app.fake.json_replies.append(REPAIR)
    failed = {"type": "task.run_finished", "data": {"task_id": "t1", "run_id": "r1", "status": "error",
                                                     "tool_errors": 1, "skills_viewed": [NAME], "error": "label missing"}}
    assert await evo.handle_repair_event(failed) == [NAME]
    prompt = app.fake.calls[-1]["messages"][1]["content"]
    assert "Task run status: error" in prompt and "TOOL RESULT gmail_search" in prompt
    detail = json.loads((await app.store.fetchone(
        "SELECT detail FROM evolution_log WHERE kind = 'skill_repair_proposed'"))["detail"])
    assert detail == {**detail, "task_id": "t1", "run_id": "r1", "failure": "run_failed", "origin": "repair"}
    assert detail["reason"].startswith("the task run failed")
    assert (await stats(app))["failure_count"] == 1


async def test_one_open_repair_per_skill_and_cooldown(app):
    add_skill(app)
    sid = await add_tool_chat(app)
    evo = app.evolution
    app.fake.json_replies.append(REPAIR)
    assert await evo.handle_repair_event(turn(sid, "t1", tool_errors=2)) == [NAME]
    calls = len(app.fake.calls)
    assert await evo.handle_repair_event(turn(sid, "t2", tool_errors=1, user_text="again")) == []
    assert len(app.fake.calls) == calls  # proposal still open: no model call
    app.skills.reject_pending(NAME)
    assert await evo.handle_repair_event(turn(sid, "t3", tool_errors=1, user_text="again")) == []
    assert len(app.fake.calls) == calls  # rejected recently: cooling down
    app.config.evolution.repair_cooldown_hours = 0
    app.fake.json_replies.append(REPAIR)
    assert await evo.handle_repair_event(turn(sid, "t4", tool_errors=1, user_text="again")) == [NAME]
    assert (await stats(app))["failure_count"] == 4  # every failed use is counted, proposal or not


async def test_never_auto_activates_and_rejects_bad_proposals(app):
    add_skill(app)
    sid = await add_tool_chat(app)
    evo = app.evolution
    app.config.skills.write_approval = False
    app.config.evolution.repair_cooldown_hours = 0
    app.fake.json_replies += [{"decision": "none", "reason": "network was down"}]
    assert await evo.handle_repair_event(turn(sid, "t1", tool_errors=1)) == []
    app.fake.json_replies += [{"decision": "patch", "body": "too short"}]
    assert await evo.handle_repair_event(turn(sid, "t2", tool_errors=1)) == []
    app.fake.json_replies += [{"decision": "patch", "body": SKILL_BODY}]  # unchanged body
    assert await evo.handle_repair_event(turn(sid, "t3", tool_errors=1)) == []
    assert app.skills.get_pending(NAME) is None and await app.notifications.list() == []
    app.fake.json_replies.append(REPAIR)
    assert await evo.handle_repair_event(turn(sid, "t4", tool_errors=1)) == [NAME]
    app.skills.reload()
    assert app.skills.get(NAME).body == SKILL_BODY.strip()  # still pending, even with write_approval off
    app.config.evolution.skill_repair = False
    app.skills.reject_pending(NAME)
    calls = len(app.fake.calls)
    assert await evo.handle_repair_event(turn(sid, "t5", tool_errors=1)) == []
    assert len(app.fake.calls) == calls


async def test_defensive_event_fields(app):
    add_skill(app)
    evo = app.evolution
    assert await evo.handle_repair_event({"type": "chat.turn_completed", "data": None}) == []
    assert await evo.handle_repair_event({"type": "chat.turn_completed", "data": {"session_id": "s1", "tool_calls": 2}}) == []
    assert await evo.handle_repair_event(turn("s1", "t1", skills_viewed="unknown-skill", tool_errors="3")) == []
    assert await evo.handle_repair_event({"type": "task.run_finished", "data": {"task_id": "t", "status": "error"}}) == []
    evo.on_event({"type": "chat.turn_completed", "data": {"session_id": "s1", "tool_calls": [1, 2]}})
    assert evo._dirty_sessions["s1"]["tool_calls"] == 2
    assert app.fake.calls == []


def test_looks_like_correction():
    for text in ["No, that's wrong", "that didn't work", "That" + chr(0x2019) + "s not what I asked", "you forgot the attachment",
                 "Not what I wanted at all", "wrong file", "still broken", "nope"]:
        assert looks_like_correction(text), text
    for text in ["Thanks, that's perfect", "Now do the same for Slack", "Can you also add Bob?", "Noted, great work"]:
        assert not looks_like_correction(text), text


async def test_curator_merge_keeps_the_more_reliable_skill(app):
    lib = app.skills
    lib.write("email-digest", "weekly digest of unread email", "## Procedure\n1. a", author="assistant")
    lib.write("email-digests", "weekly digest of unread email", "## Procedure\n1. b", author="assistant")
    lib.reload()
    for _ in range(3):
        await lib.mark_used("email-digest")
        await lib.record_outcome("email-digest", False)
    await lib.mark_used("email-digests")
    await lib.record_outcome("email-digests", True)
    app.fake.json_replies.append({"name": "email-digest", "description": "Weekly digest of unread email", "body": SKILL_BODY})
    out = await app.evolution.run_curator()
    assert out["merge_proposals"] == ["email-digests"]
    catalog = await lib.catalog()
    by_name = {s["name"]: s for s in catalog["active"]}
    assert by_name["email-digest"]["failure_count"] == 3 and by_name["email-digests"]["success_count"] == 1


def test_pending_repair_listing_carries_reason_and_origin(config, isolated_home, monkeypatch):
    from sentient.app import SentientApp
    from tests.conftest import FakeProvider

    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "repair-token")
    llm = FakeProvider()
    core = SentientApp(config, llm=llm, db_path=isolated_home / "rp.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers["Authorization"] = "Bearer repair-token"
        assert c.post("/api/skills", json={"name": NAME, "description": "Digest", "body": SKILL_BODY}).status_code == 200
        llm.json_replies.append(REPAIR)

        async def fail():
            return await core.evolution.handle_repair_event(turn("sess-1", "t1", tool_errors=1))

        assert c.portal.call(fail) == [NAME]
        pending = next(p for p in c.get("/api/skills").json()["pending"] if p["name"] == NAME)
        assert pending["origin"] == {"session_id": "sess-1", "repair": True}
        assert "tool call failed" in pending["reason"] and pending["proposed_at"]
        assert "in:inbox" in c.get(f"/api/skills/{NAME}/diff").json()["proposed"]
        log = c.get("/api/skills/evolution-log").json()
        assert log[0]["kind"] == "skill_repair_proposed"
