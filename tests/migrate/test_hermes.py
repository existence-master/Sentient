"""Importing a Hermes home folder: preview, then skills, memory, persona, scheduled jobs and MCP servers."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from sentient import paths
from sentient.gateway.app import create_app
from sentient.migrate import hermes
from sentient.migrate.hermes import _has_secret, cron_to_schedule
from sentient.tasks.service import TaskConflict
from tests.migrate.conftest import TRAP

ALL = ["skills", "memory", "persona", "jobs", "mcp"]


def by_key(items: list[dict]) -> dict[str, dict]:
    return {i["key"]: i for i in items}


async def test_preview_counts_and_what_will_happen(app, hermes_home, opened_secrets):
    plan = await hermes.preview(app, str(hermes_home))
    assert plan["counts"] == {"skills": 2, "memory": 5, "persona": 1, "jobs": 3, "mcp": 3}

    skills = {i["folder"]: i for i in plan["skills"]}
    assert skills["productivity/weekly-review"]["action"] == "import"
    assert skills["software-development/plan"]["action"] == "import"
    assert skills["software-development/plan"]["changed_builtin"] is True
    assert skills["research/arxiv"]["action"] == "skip"
    assert "Built into Hermes" in skills["research/arxiv"]["note"]
    assert skills["broken"]["action"] == "skip"
    assert not any(".archive" in f or ".hub" in f for f in skills)

    texts = [(i["kind"], i["text"], i["action"]) for i in plan["memory"]]
    assert ("fact", "This machine runs Ubuntu 22.04 with Docker installed", "import") in texts
    assert ("fact", "The project lives at ~/code/myapi and uses Axum + SQLx. Tests run with cargo nextest.", "import") in texts
    # a bare § inside an entry is text, not a separator
    assert any("§ sign marks sections" in t for _, t, _ in texts)
    assert ("insight", "Sarthak prefers concise answers without bullet points", "import") in texts
    assert ("insight", "Sarthak works mostly in the evenings", "import") in texts
    assert [i["note"] for i in plan["memory"] if i["action"] == "skip"] == ["It's in the file twice."]

    assert plan["persona"]["action"] == "import"
    assert "You are Hermes" in plan["persona"]["proposed"]
    assert "Hermes by name" in plan["persona"]["note"]
    assert plan["suggestions"] == {"wake_word": "hey jarvis", "tts_provider": "edge", "tts_voice": "en-GB-SoniaNeural"}
    assert plan["never_read"] == [".env", "auth.json"]
    assert opened_secrets == []


async def test_bad_folder_is_a_plain_error(app, tmp_path):
    with pytest.raises(hermes.HermesImportError, match="There's no Hermes folder"):
        await hermes.preview(app, str(tmp_path / "nope"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(hermes.HermesImportError, match="doesn't look like a Hermes folder"):
        await hermes.preview(app, str(tmp_path / "empty"))


async def test_skills_go_to_pending_and_unchanged_builtins_are_skipped(app, hermes_home, opened_secrets):
    result = await hermes.apply(app, str(hermes_home), ["skills"])
    assert sorted(result["skills"]["imported"]) == ["plan", "weekly-review"]
    assert {s["name"] for s in result["skills"]["skipped"]} == {"arxiv", "broken"}

    pending = app.skills.pending_dir
    assert (pending / "weekly-review" / "references" / "notes.md").is_file()
    assert not (pending / "weekly-review" / ".env").exists()
    assert not (pending / "arxiv").exists()
    assert app.skills.get("weekly-review") is None  # never active
    skill = app.skills.get_pending("weekly-review")
    assert skill is not None and skill.state == "pending_review"
    assert skill.tags == ["productivity", "review"]
    assert "Read this week's calendar" in skill.body
    catalog = await app.skills.catalog(app.store)
    assert {s["name"] for s in catalog["pending"]} == {"plan", "weekly-review"}
    assert catalog["active"] == []
    assert opened_secrets == []


async def test_skill_names_stay_unique(app, hermes_home):
    app.skills.write("weekly-review", "Mine", "My own weekly review.", author="user")
    first = await hermes.apply(app, str(hermes_home), ["skills"])
    assert "weekly-review-hermes" in first["skills"]["imported"]
    assert app.skills.get_active_file("weekly-review").body == "My own weekly review."
    assert app.skills.get_pending("weekly-review-hermes").name == "weekly-review-hermes"
    # importing again does not make copies of skills Sentient already has
    again = await hermes.preview(app, str(hermes_home))
    assert again["counts"]["skills"] == 0


async def test_memory_entries_become_facts_and_insights_with_source(app, hermes_home):
    result = await hermes.apply(app, str(hermes_home), ["memory"])
    assert result["memory"]["facts"] == 3
    assert result["memory"]["insights"] == 2
    facts = await app.memory.list_facts(100, 0, source=hermes.SOURCE)
    assert {f["content"] for f in facts} >= {"This machine runs Ubuntu 22.04 with Docker installed"}
    state = await app.user_model.get_state()
    imported = [i for i in state["insights"] if i["source"] == hermes.SOURCE]
    assert {i["statement"] for i in imported} == {
        "Sarthak prefers concise answers without bullet points", "Sarthak works mostly in the evenings",
    }
    assert all(i["status"] == "active" for i in imported)
    assert app.fake.calls == []  # no model calls, only embeddings

    again = await hermes.preview(app, str(hermes_home))
    assert again["counts"]["memory"] == 0

    removed = await hermes.remove_memories(app)
    assert removed == {"facts": 3, "insights": 2}
    assert await app.memory.list_facts(100, 0, source=hermes.SOURCE) == []


async def test_soul_changes_only_when_persona_is_picked(app, hermes_home):
    before = app.workspace.read_full()["soul"]
    await hermes.apply(app, str(hermes_home), ["memory"])
    assert app.workspace.read_full()["soul"] == before

    plan = await hermes.preview(app, str(hermes_home))
    assert plan["persona"]["current"] == before
    result = await hermes.apply(app, str(hermes_home), ["persona"])
    assert result["persona"] == {"updated": True}
    assert app.workspace.read_full()["soul"].startswith("# Soul\n\nYou are Hermes")
    again = await hermes.preview(app, str(hermes_home))
    assert again["persona"]["action"] == "skip"


async def test_jobs_become_paused_tasks_with_schedules(app, hermes_home):
    plan = await hermes.preview(app, str(hermes_home))
    jobs = by_key(plan["jobs"])
    assert jobs["job:a1b2c3d4e5f6"]["delivery"] == "desktop"
    assert "WhatsApp isn't set up in Sentient yet" in jobs["job:a1b2c3d4e5f6"]["note"]
    assert jobs["job:c3d4e5f6a1b2"]["script"]["path"] == "scripts/check_prices.py"
    assert "print(json.dumps" in jobs["job:c3d4e5f6a1b2"]["script"]["code"]
    skipped = {k: i["note"] for k, i in jobs.items() if i["action"] == "skip"}
    assert skipped == {
        "job:d4e5f6a1b2c3": "Its script scripts/missing.py isn't there.",
        "job:e5f6a1b2c3d4": "Its script scripts/backup.sh isn't Python; Sentient's check scripts are Python only.",
        "job:f6a1b2c3d4e5": "\"0 9 1 * *\" runs on certain days of the month, which Sentient can't do yet.",
        "job:a6b1c2d3e4f5": "It runs every 2 minutes; Sentient runs things at most every 5 minutes.",
    }

    result = await hermes.apply(app, str(hermes_home), ["jobs"])
    assert [c["name"] for c in result["jobs"]["created"]] == ["Morning brief", "Weekday standup notes", "Watch prices"]
    assert len(result["jobs"]["skipped"]) == 4
    tasks = {t["name"]: t for t in await app.tasks.list()}
    brief, standup, watch = tasks["Morning brief"], tasks["Weekday standup notes"], tasks["Watch prices"]
    for t in (brief, standup, watch):
        assert t["enabled"] is False
        assert t["status"] == "active"
        assert t["plan"] == []
        assert t["next_execution_at"] is None
        assert t["original_context"]["imported_from"] == "hermes"
    assert brief["schedule"]["frequency"] == "daily" and brief["schedule"]["time"] == "08:00"
    assert standup["schedule"]["frequency"] == "weekly"
    assert standup["schedule"]["days"] == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    assert standup["schedule"]["time"] == "09:30"
    assert "weekly-review" in standup["description"]
    assert watch["task_type"] == "script"
    assert watch["schedule"] == {**watch["schedule"], "frequency": "interval", "interval_minutes": 30}
    assert watch["script"]["then"] == "notify" and watch["script"]["condition"] == "changed"
    # nothing runs on its own
    assert await app.tasks.tick() == []
    # importing again does not add the same jobs twice
    again = await hermes.preview(app, str(hermes_home))
    assert again["counts"]["jobs"] == 0
    assert by_key(again["jobs"])["job:a1b2c3d4e5f6"]["note"] == "Already brought over."


async def test_whatsapp_delivery_when_paired(app, hermes_home):
    await app.channels.store.add_chat("whatsapp", "self", "Me", deliver=True, session_id=None)
    plan = await hermes.preview(app, str(hermes_home))
    brief = by_key(plan["jobs"])["job:a1b2c3d4e5f6"]
    assert brief["delivery"] == "whatsapp"
    assert "paired WhatsApp chat" in brief["note"]


async def test_resuming_an_imported_task_plans_it_for_approval(app, hermes_home):
    await hermes.apply(app, str(hermes_home), ["jobs"])
    tasks = {t["name"]: t for t in await app.tasks.list()}
    brief = tasks["Morning brief"]
    with pytest.raises(TaskConflict, match="Resume this task first"):
        await app.tasks.run_now(brief["task_id"])

    app.fake.json_replies.append({
        "name": "Morning brief", "description": "Calendar and weather for today",
        "plan": [{"tool": "time", "description": "Get today's date"}], "clarifying_questions": [],
    })
    out = await app.tasks.update(brief["task_id"], {"enabled": True, "plan": [{"tool": "files", "description": "x"}]})
    assert out["status"] == "planning"  # a plan sent with the resume can't skip planning and approval
    for _ in range(100):
        task = await app.tasks.get(brief["task_id"])
        if task["status"] != "planning":
            break
        await asyncio.sleep(0.02)
    assert task["status"] == "approval_pending"
    assert task["plan"] == [{"tool": "time", "description": "Get today's date"}]
    assert task["schedule"]["time"] == "08:00"  # the schedule is kept

    approved = await app.tasks.approve(brief["task_id"])
    assert approved["status"] == "active" and approved["enabled"] is True
    assert approved["next_execution_at"]

    watch = tasks["Watch prices"]
    calls = len(app.fake.calls)
    out = await app.tasks.update(watch["task_id"], {"enabled": True})
    assert out["status"] == "approval_pending"  # the script itself is what gets approved, no planner
    assert out["plan"][0]["description"].startswith("Runs a small check script")
    assert len(app.fake.calls) == calls


async def test_mcp_servers_are_added_turned_off_without_secrets(app, hermes_home, keychain):
    result = await hermes.apply(app, str(hermes_home), ["mcp"])
    assert sorted(result["mcp"]["added"]) == ["github", "linear", "notes"]
    assert [s["name"] for s in result["mcp"]["skipped"]] == ["empty", "inline"]
    assert "looks like a key" in result["mcp"]["skipped"][1]["note"]
    assert "inline" not in app.config.integrations.mcp_servers
    servers = app.config.integrations.mcp_servers
    assert servers["github"] == {
        "transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"], "url": None,
        "env_keys": ["GITHUB_PERSONAL_ACCESS_TOKEN"], "auth": "none", "header_keys": [], "enabled": False,
    }
    assert servers["linear"]["auth"] == "oauth" and servers["linear"]["enabled"] is False
    assert servers["notes"]["header_keys"] == ["Authorization"] and servers["notes"]["auth"] == "headers"
    assert keychain == {}
    saved = paths.config_file().read_text(encoding="utf-8")
    assert "fixture-header-value" not in saved and "GITHUB_TOKEN" not in saved and "fixture-not-a-key" not in saved
    listed = {s["name"]: s for s in app.integrations.mcp.list()}
    assert listed["notes"]["status"] == "disabled"

    again = await hermes.preview(app, str(hermes_home))
    assert again["counts"]["mcp"] == 0


async def test_an_imported_server_can_be_turned_on_and_off(app):
    await app.integrations.mcp.import_server("local", {"transport": "http", "url": "http://127.0.0.1:9/mcp"})
    on = await app.integrations.mcp.set_enabled("local", True)
    assert on["enabled"] is True and on["status"] != "disabled"
    off = await app.integrations.mcp.set_enabled("local", False)
    assert off["status"] == "disabled"
    assert app.config.integrations.mcp_servers["local"]["enabled"] is False
    with pytest.raises(KeyError):
        await app.integrations.mcp.set_enabled("missing", True)


async def test_secrets_files_are_never_opened(app, hermes_home, opened_secrets, keychain):
    await hermes.preview(app, str(hermes_home))
    await hermes.apply(app, str(hermes_home), ALL)
    assert opened_secrets == []
    assert keychain == {}
    for f in paths.home().rglob("*"):
        if f.is_file() and f.suffix in {".md", ".yaml", ".json", ".py", ".txt"}:
            assert TRAP not in f.read_text(encoding="utf-8", errors="replace"), f


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("0 8 * * *", {"type": "recurring", "frequency": "daily", "time": "08:00"}),
        ("30 9 * * 1-5", {"type": "recurring", "frequency": "weekly", "time": "09:30",
                          "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}),
        ("0 18 * * SAT,SUN", {"type": "recurring", "frequency": "weekly", "time": "18:00", "days": ["Saturday", "Sunday"]}),
        ("15 7 * * 0", {"type": "recurring", "frequency": "weekly", "time": "07:15", "days": ["Sunday"]}),
        ("*/15 * * * *", {"type": "recurring", "frequency": "interval", "interval_minutes": 15}),
        ("5 * * * *", {"type": "recurring", "frequency": "interval", "interval_minutes": 60}),
        ("0 */3 * * *", {"type": "recurring", "frequency": "interval", "interval_minutes": 180}),
        ("0 9 1 * *", None),
        ("0 9,17 * * *", None),
        ("* * * * *", None),
        ("*/2 * * * *", None),
        ("0 9 * * */2", None),
        ("0 0 9 * * *", None),
    ],
)
def test_cron_to_schedule(expr, expected):
    schedule, reason = cron_to_schedule(expr)
    assert schedule == expected
    assert bool(reason) == (expected is None)


def test_routes(app, hermes_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    with TestClient(create_app(app)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        assert c.get("/api/import/hermes").json()["path"].endswith(".hermes")
        bad = c.post("/api/import/hermes/preview", json={"path": str(hermes_home / "missing")})
        assert bad.status_code == 400 and "There's no Hermes folder" in bad.json()["detail"]
        plan = c.post("/api/import/hermes/preview", json={"path": str(hermes_home)}).json()
        assert plan["counts"]["jobs"] == 3
        assert c.post("/api/import/hermes/apply", json={"path": str(hermes_home), "parts": []}).status_code == 400
        out = c.post(
            "/api/import/hermes/apply",
            json={"path": str(hermes_home), "parts": ["jobs", "mcp"], "skip": ["job:a1b2c3d4e5f6", "mcp:linear"]},
        ).json()
        assert [j["name"] for j in out["jobs"]["created"]] == ["Weekday standup notes", "Watch prices"]
        assert {"key": "job:a1b2c3d4e5f6", "name": "Morning brief", "note": "You left it out."} in out["jobs"]["skipped"]
        assert sorted(out["mcp"]["added"]) == ["github", "notes"]
        assert c.post("/api/integrations/mcp/notes/enabled", json={"enabled": "false"}).status_code == 422
        on = c.post("/api/integrations/mcp/notes/enabled", json={"enabled": False})
        assert on.status_code == 200 and on.json()["status"] == "disabled"
        assert c.delete("/api/import/hermes/memories").json() == {"facts": 0, "insights": 0}


def test_the_trap_catches_a_read(hermes_home, opened_secrets):
    with pytest.raises(AssertionError, match="must never be read"):
        (hermes_home / "auth.json").read_text(encoding="utf-8")
    with pytest.raises(AssertionError, match="must never be read"):
        open(hermes_home / ".env", encoding="utf-8")  # noqa: SIM115
    assert len(opened_secrets) == 2
    opened_secrets.clear()


@pytest.mark.parametrize(
    ("parts", "secret"),
    [
        (["npx", "-y", "@modelcontextprotocol/server-github"], False),
        (["npx", "--api-key", "abc123"], True),
        (["npx", "--api-key", "${KEY}"], False),
        (["x", "--access-token=abcdef"], True),
        (["x", "--token=${T}"], False),
        (["https://example.com/mcp?api_key=abc"], True),
        (["https://example.com/mcp"], False),
        (["run", "ghp_abcdefghijklmnop"], True),
        (["--keyboard-layout", "us"], False),
    ],
)
def test_keys_on_a_command_line_are_spotted(parts, secret):
    assert _has_secret(parts) is secret


async def test_origin_delivery_uses_the_chat_the_job_came_from(app, hermes_home):
    import json

    jobs = hermes_home / "cron" / "jobs.json"
    data = json.loads(jobs.read_text(encoding="utf-8"))
    data["jobs"][0]["deliver"] = "origin"
    data["jobs"][0]["origin"] = {"platform": "telegram", "chat_id": "fixture-chat"}
    jobs.write_text(json.dumps(data), encoding="utf-8")
    await app.channels.store.add_chat("telegram", "1", "Me", deliver=True, session_id=None)
    plan = await hermes.preview(app, str(hermes_home))
    assert by_key(plan["jobs"])["job:a1b2c3d4e5f6"]["delivery"] == "telegram"
