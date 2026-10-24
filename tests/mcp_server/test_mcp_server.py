"""The MCP server's tool set and its safety gates (issue #443, ADR 0034).

A fake engine client stands in for the REST API, and the tools are called through a real MCP
client connected to the server in-process, so what is tested is the protocol surface a client
like Claude Code actually sees.
"""

from __future__ import annotations

import json

from mcp import Client
from typer.testing import CliRunner

from sentient import cli
from sentient.mcp_server.client import EngineError
from sentient.mcp_server.server import build_server, main

EXPECTED_TOOLS = {
    "memory_search", "memory_add", "memory_get",
    "task_create", "task_list", "task_status",
    "notify_me", "ask_sentient",
}


class FakeEngine:
    """Just the EngineClient surface the tools use, with the answers scripted."""

    def __init__(self, *, enabled: bool = True, skip_review: bool = False):
        self.base_url = "http://127.0.0.1:7777"
        self.section = {
            "enabled": enabled,
            "memories_skip_review": skip_review,
            "token_issued": True,
        }
        self.memories = [
            {"id": 1, "content": "Maya Rao's daughter Mira starts school in June",
             "topics": ["Personal Identity"], "source": "conversation", "scope": "personal",
             "created_at": "2026-10-01T09:00:00Z"},
            {"id": 2, "content": "Maya Rao drinks tea, not coffee",
             "topics": ["Interests & Lifestyle"], "source": "conversation", "scope": "personal",
             "created_at": "2026-10-02T09:00:00Z"},
        ]
        self.added: list[dict] = []
        self.tasks = [
            {"id": "t1", "name": "Book the dentist", "status": "waiting_approval",
             "description": "Book the dentist", "plan": [{"step": "find one"}], "updated_at": "2026-10-20T10:00:00Z"},
        ]
        self.notes: list[dict] = []
        self.fail_search = False
        self.fail_config = False
        self.prove_ok = True
        self.status_calls = 0
        self.memory_held = False

    def refresh_address(self):
        pass  # the fake never moves ports

    def prove_identity(self):
        return self.prove_ok

    def status(self):
        self.status_calls += 1
        if self.fail_config:
            raise EngineError("The engine refused this server's token (401).")
        return dict(self.section)

    def search_memories(self, query, limit=10):
        if self.fail_search:
            raise EngineError("The engine said no (500).")
        q = query.lower()
        return [m for m in self.memories if q in m["content"].lower()][:limit]

    def get_memory(self, memory_id):
        for m in self.memories:
            if m["id"] == memory_id:
                return {**m, "memory_type": "long-term", "state": "current"}
        raise EngineError("The engine said no (404): memory 999 not found.")

    def add_memory(self, content, *, hold_for_review, scope=None, app=None):
        self.added.append({"content": content, "hold_for_review": hold_for_review, "scope": scope, "app": app})
        out = {"action": "ADD", "id": 99, "content": content}
        if hold_for_review:
            out["status"] = "pending"
        return out

    def create_task(self, prompt, *, app=None):
        task = {"id": "t2", "name": prompt[:40], "status": "planning", "description": prompt,
                "plan": [], "updated_at": "2026-10-21T10:00:00Z"}
        self.tasks.append(task)
        self.task_apps = [*getattr(self, "task_apps", []), app]
        return task

    def list_tasks(self):
        return list(self.tasks)

    def get_task(self, task_id):
        for t in self.tasks:
            if t["id"] == task_id:
                return t
        raise EngineError("The engine said no (404).")

    def notify(self, message, title=None, *, app=None):
        self.notes.append({"message": message, "title": title, "app": app})
        return {"id": "n1"}

    def chat_turn(self, text, *, app=None):
        out = {"content": f"Sentient replies to: {text}", "session_id": "s1"}
        if self.memory_held:
            out["memory_held"] = True
        return out


def _result_json(result) -> dict:
    assert result.content, "the tool returned no content"
    return json.loads(result.content[0].text)


async def _call(server, name: str, args: dict | None = None) -> dict:
    async with Client(server) as client:
        return _result_json(await client.call_tool(name, args or {}))


async def test_the_tool_list_is_exactly_the_safe_whitelist():
    async with Client(build_server(FakeEngine())) as client:
        tools = await client.list_tools()
    names = {t.name for t in tools.tools}
    assert names == EXPECTED_TOOLS
    # the red line: nothing here can send, delete or run anything
    for forbidden in ("send", "delete", "exec", "run", "shell", "remove"):
        assert not any(forbidden in name for name in names), names


async def test_every_tool_refuses_while_turned_off():
    server = build_server(FakeEngine(enabled=False))
    for name, args in [
        ("memory_search", {"query": "Maya"}),
        ("memory_add", {"content": "Maya likes tea"}),
        ("memory_get", {"memory_id": 1}),
        ("task_create", {"prompt": "do a thing"}),
        ("task_list", {}),
        ("task_status", {"task_id": "t1"}),
        ("notify_me", {"message": "hi"}),
        ("ask_sentient", {"message": "hi"}),
    ]:
        out = await _call(server, name, args)
        assert "turned off" in out["error"], (name, out)


async def test_memory_search_and_get():
    server = build_server(FakeEngine())
    found = await _call(server, "memory_search", {"query": "Maya"})
    assert [m["id"] for m in found["memories"]] == [1, 2]
    one = await _call(server, "memory_get", {"memory_id": 2})
    assert one["content"] == "Maya Rao drinks tea, not coffee"
    missing = await _call(server, "memory_get", {"memory_id": 999})
    assert "error" in missing and "404" in missing["error"]


async def test_memory_add_waits_for_review_by_default():
    engine = FakeEngine()
    out = await _call(build_server(engine), "memory_add", {"content": "Maya's office is in Lisbon"})
    assert engine.added[0]["content"] == "Maya's office is in Lisbon"
    assert engine.added[0]["hold_for_review"] is True and engine.added[0]["scope"] is None
    assert out["status"] == "pending" and "review" in out["note"].lower()


async def test_memory_add_saves_straight_away_only_when_the_user_allowed_it():
    engine = FakeEngine(skip_review=True)
    out = await _call(build_server(engine), "memory_add", {"content": "Maya's office is in Lisbon"})
    assert engine.added[0]["hold_for_review"] is False
    assert out["status"] == "active"


async def test_task_create_notes_that_the_plan_waits_for_approval():
    engine = FakeEngine()
    out = await _call(build_server(engine), "task_create", {"prompt": "Book the dentist for next week"})
    assert out["id"] == "t2" and out["status"] == "planning"
    assert "approval" in out["note"]
    listed = await _call(build_server(engine), "task_list")
    assert {t["id"] for t in listed["tasks"]} == {"t1", "t2"}
    status = await _call(build_server(engine), "task_status", {"task_id": "t1"})
    assert status["status"] == "waiting_approval" and status["plan_steps"] == 1


async def test_notify_me_and_ask_sentient():
    engine = FakeEngine()
    out = await _call(build_server(engine), "notify_me", {"message": "Claude Code finished the report"})
    assert out == {"notified": True, "id": "n1"}
    assert engine.notes[0]["message"] == "Claude Code finished the report" and engine.notes[0]["title"] is None
    reply = await _call(build_server(engine), "ask_sentient", {"message": "what is on today?"})
    assert reply["reply"] == "Sentient replies to: what is on today?"


async def test_engine_errors_come_back_as_error_json_not_a_crash():
    engine = FakeEngine()
    engine.fail_search = True
    out = await _call(build_server(engine), "memory_search", {"query": "Maya"})
    assert out == {"error": "The engine said no (500)."}


def test_the_mcp_command_exists_and_refuses_without_an_engine(isolated_home):
    runner = CliRunner()
    help_out = runner.invoke(cli.app, ["mcp", "--help"])
    assert help_out.exit_code == 0, help_out.output
    # no discovery file in the isolated home: no engine is running, whatever this computer has
    refused = runner.invoke(cli.app, ["mcp"])
    assert refused.exit_code == 2, refused.output
    assert "Start Sentient first" in refused.output
    # and starting (or refusing) never creates the gateway token file: that token is the engine's
    # own, and this process must not mint one as a side effect (review item 3)
    assert not (isolated_home / "gateway.token").exists()


def test_main_refuses_until_the_user_turns_the_feature_on(isolated_home, monkeypatch):
    from sentient import paths, secrets

    found = paths.discovery_file()
    found.parent.mkdir(parents=True, exist_ok=True)
    found.write_text('{"port": 9}', encoding="utf-8")
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: None)
    assert main() == 2  # no MCP token issued yet: turning the feature on creates it


def test_main_never_writes_the_gateway_token(isolated_home, monkeypatch):
    from sentient import paths, secrets

    found = paths.discovery_file()
    found.parent.mkdir(parents=True, exist_ok=True)
    found.write_text('{"port": 9}', encoding="utf-8")  # port 9: nothing answers there
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: "issued-token")
    assert main() == 2  # the engine is not answering, so the server refuses to start
    assert not (isolated_home / "gateway.token").exists()


def test_client_name_comes_from_the_handshake():
    # the SDK's InitializeRequestParams exposes the client info as client_info (the
    # camelCase spelling is only the wire alias); a wrong attribute silently yields None
    from types import SimpleNamespace

    from sentient.mcp_server.server import _client_name

    ctx = SimpleNamespace(request_context=SimpleNamespace(session=SimpleNamespace(
        client_params=SimpleNamespace(client_info=SimpleNamespace(name="Claude Code")))))
    assert _client_name(ctx) == "Claude Code"
    assert _client_name(SimpleNamespace()) is None

    def ctx_named(name):
        return SimpleNamespace(request_context=SimpleNamespace(session=SimpleNamespace(
            client_params=SimpleNamespace(client_info=SimpleNamespace(name=name)))))

    assert _client_name(ctx_named(None)) is None  # an explicit null is not the app "None"
    assert _client_name(ctx_named("   ")) is None
    assert _client_name(ctx_named("  Cursor ")) == "Cursor"


def test_check_ready_refuses_whatever_is_not_sentient():
    # the discovery file holds only a port, and ports get recycled: the readiness probe runs
    # without the token, and only an engine that proves it holds the token may receive it
    from sentient.mcp_server.server import check_ready

    class Stranger:
        base_url = "http://127.0.0.1:7777"

        def health(self):
            return {"ok": True, "version": "9.9"}  # answers, but never says it is Sentient

        def prove_identity(self):
            raise AssertionError("no proof is asked of something that is not even Sentient")

        def status(self):
            raise AssertionError("the token must not travel before the engine is identified")

    problem = check_ready(Stranger())
    assert problem is not None and "not Sentient" in problem

    class Impostor:
        # the regression that matters: the health marker is public, so a listener can return
        # the exact expected answer; without the token it still cannot answer the challenge
        base_url = "http://127.0.0.1:7777"

        def health(self):
            return {"ok": True, "name": "sentient", "version": "9.9"}

        def prove_identity(self):
            return False

        def status(self):
            raise AssertionError("the token must not travel to an unproven engine")

    problem = check_ready(Impostor())
    assert problem is not None and "cannot prove" in problem

    class Engine:
        base_url = "http://127.0.0.1:7777"

        def health(self):
            return {"ok": True, "name": "sentient"}

        def prove_identity(self):
            return True

        def status(self):
            return {"enabled": True}

    assert check_ready(Engine()) is None


async def test_an_engine_error_at_the_gate_is_reported_not_masked_as_turned_off():
    engine = FakeEngine()
    engine.fail_config = True
    out = await _call(build_server(engine), "memory_search", {"query": "Maya"})
    assert out == {"error": "The engine refused this server's token (401)."}



async def test_the_gate_sends_the_token_nowhere_until_the_engine_proves_itself():
    # the proof at startup is not enough: the engine restarts on a fresh port, and whatever
    # took over the old one must not be sent the token on the next call (review 5609721491)
    engine = FakeEngine()
    engine.prove_ok = False
    out = await _call(build_server(engine), "memory_search", {"query": "Maya"})
    assert "could not prove it holds the MCP token" in out["error"]
    # and the wording is honest: the gate retries on the next call and recovers by itself,
    # only a replaced token needs a restart (review 5624575781, the nit)
    assert "clears by itself" in out["error"] and "to reconnect" not in out["error"]
    assert engine.status_calls == 0  # the status call is the first token carrier: it never happened


async def test_ask_sentient_says_when_a_memory_from_the_turn_waits_in_review():
    # the reply itself may say "noted" while the fact is only waiting in Review; the tool's
    # answer carries the correction as a note (review 5609721491, the nit)
    engine = FakeEngine()
    engine.memory_held = True
    out = await _call(build_server(engine), "ask_sentient", {"message": "Remember this about me: I drink tea"})
    assert "Review list" in out["note"] and "only saved if they approve" in out["note"]
