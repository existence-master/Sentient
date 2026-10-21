"""The MCP server's tool set and its safety gates (issue #443, ADR 0034).

A fake engine client stands in for the REST API, and the tools are called through a real MCP
client connected to the server in-process, so what is tested is the protocol surface a client
like Claude Code actually sees.
"""

from __future__ import annotations

import json

import httpx
import respx
from mcp import Client
from typer.testing import CliRunner

from sentient import cli
from sentient.mcp_server.client import EngineError
from sentient.mcp_server.server import build_server

EXPECTED_TOOLS = {
    "memory_search", "memory_add", "memory_get",
    "task_create", "task_list", "task_status",
    "notify_me", "ask_sentient",
}


class FakeEngine:
    """Just the EngineClient surface the tools use, with the answers scripted."""

    def __init__(self, *, enabled: bool = True, skip_review: bool = False):
        self.section = {
            "mcp_server_enabled": enabled,
            "mcp_server_memories_skip_review": skip_review,
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

    def integrations_config(self):
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

    def add_memory(self, content, *, hold_for_review, scope=None):
        self.added.append({"content": content, "hold_for_review": hold_for_review, "scope": scope})
        out = {"action": "ADD", "id": 99, "content": content}
        if hold_for_review:
            out["status"] = "pending"
        return out

    def create_task(self, prompt):
        task = {"id": "t2", "name": prompt[:40], "status": "planning", "description": prompt,
                "plan": [], "updated_at": "2026-10-21T10:00:00Z"}
        self.tasks.append(task)
        return task

    def list_tasks(self):
        return list(self.tasks)

    def get_task(self, task_id):
        for t in self.tasks:
            if t["id"] == task_id:
                return t
        raise EngineError("The engine said no (404).")

    def notify(self, message, title=None):
        self.notes.append({"message": message, "title": title})
        return {"id": "n1"}

    def chat_turn(self, text):
        return {"content": f"Sentient replies to: {text}", "session_id": "s1"}


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
    assert engine.added == [{"content": "Maya's office is in Lisbon", "hold_for_review": True, "scope": None}]
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
    assert engine.notes == [{"message": "Claude Code finished the report", "title": None}]
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
    # the health check is mocked to fail the way a stopped engine fails, so the test never
    # depends on whether a real Sentient happens to be running on this computer
    with respx.mock() as router:
        router.get(url__regex=r".*/api/health").mock(side_effect=httpx.ConnectError("down"))
        refused = runner.invoke(cli.app, ["mcp"])
    assert refused.exit_code == 2, refused.output
    assert "Start Sentient first" in refused.output


async def test_an_engine_error_at_the_gate_is_reported_not_masked_as_turned_off():
    engine = FakeEngine()
    engine.fail_config = True
    out = await _call(build_server(engine), "memory_search", {"query": "Maya"})
    assert out == {"error": "The engine refused this server's token (401)."}


def test_the_http_transport_refuses_calls_without_the_engine_token():
    from starlette.testclient import TestClient

    engine = FakeEngine()
    server = build_server(engine, http_token="test-token", http_port=8765)
    init = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "0"}},
    }
    headers = {"Accept": "application/json, text/event-stream"}
    with TestClient(server.streamable_http_app(), base_url="http://127.0.0.1:8765") as http:
        refused = http.post("/mcp", json=init, headers=headers)
        assert refused.status_code == 401, refused.text
        wrong = http.post("/mcp", json=init, headers={**headers, "Authorization": "Bearer nope"})
        assert wrong.status_code == 401, wrong.text
        allowed = http.post("/mcp", json=init, headers={**headers, "Authorization": "Bearer test-token"})
        assert allowed.status_code == 200, allowed.text
        assert "sentient" in allowed.text
