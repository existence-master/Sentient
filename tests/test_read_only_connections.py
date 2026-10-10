"""Read only or Read and write per connection (#141): write tools are hidden and refused, switching needs no restart."""

from __future__ import annotations

import functools
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentient import secrets
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.integrations.base import JsonSchemaTool
from sentient.integrations.mcp import _runs_slugs, _slug_call_risk, slug_risk
from sentient.llm.events import ApprovalRequest, ToolResultEvent
from sentient.sandbox.policy import BridgePolicy, Refused
from sentient.tasks.executor import select_tools
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call

ECHO_SERVER = Path(__file__).parent / "integrations" / "mcp_echo_server.py"

MULTI_EXECUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"tool_slug": {"type": "string"}, "arguments": {"type": "object"}},
            },
        },
    },
}


def _pantry(log: list[str]) -> ToolPlugin:
    @tool("pantry_list_items", risk=Risk.read)
    async def pantry_list_items(ctx: ToolContext) -> dict:
        """List what is in the pantry."""
        log.append("list")
        return {"items": ["rice"]}

    @tool("pantry_add_item", risk=Risk.write)
    async def pantry_add_item(ctx: ToolContext, item: str) -> dict:
        """Add an item to the pantry list."""
        log.append(f"add:{item}")
        return {"added": item}

    async def run(ctx: ToolContext, **arguments) -> dict:
        log.append("run:" + ",".join(t["tool_slug"] for t in arguments.get("tools", [])))
        return {"ok": True}

    multi = JsonSchemaTool(name="pantry_multi_execute", description="Run pantry tools by slug.", fn=run,
                           input_schema=MULTI_EXECUTE_SCHEMA, risk=Risk.write, plugin="pantry")
    multi.risk_fn = functools.partial(_slug_call_risk, Risk.write)

    class Pantry(ToolPlugin):
        id = "pantry"
        display_name = "Pantry"
        tools = [pantry_list_items, pantry_add_item, multi]

    return Pantry()


async def _start(config, isolated_home, llm, name: str, *plugins: ToolPlugin) -> SentientApp:
    s = await SentientApp(config, llm=llm, db_path=isolated_home / f"{name}.db", enable_background=False).start()
    for p in plugins:
        s.registry.register(p)
    return s


async def _turn(s: SentientApp, sid: str, text: str, decision: str = "allow") -> tuple[list[ApprovalRequest], list]:
    asked: list[ApprovalRequest] = []
    events = []
    async for ev in s.agent.run_turn(sid, text, channel="cli"):
        events.append(ev)
        if isinstance(ev, ApprovalRequest):
            asked.append(ev)
            s.approvals.resolve(ev.approval_id, decision)
    return asked, events


def _offered(llm: FakeProvider, call: int) -> set[str]:
    return {t["function"]["name"] for t in (llm.calls[call]["tools"] or [])}


# ---------------------------------------------------------------------------- slugs
def test_slug_risk_reads_the_verb():
    assert slug_risk("GMAIL_FETCH_EMAILS") == Risk.read
    assert slug_risk("GITHUB_LIST_REPOSITORY_ISSUES") == Risk.read
    assert slug_risk("GITHUB_GET_A_WORKFLOW_RUN") == Risk.read  # "RUN" here is a noun
    assert slug_risk("GMAIL_SEND_EMAIL") == Risk.send
    assert slug_risk("GMAIL_CREATE_EMAIL_DRAFT") == Risk.write
    assert slug_risk("GMAIL_LIST_AND_DELETE_THREADS") == Risk.send  # a later verb that changes things raises it
    assert slug_risk("NOTION_DO_SOMETHING") == Risk.write  # no known verb: not a look-up
    assert slug_risk("") == Risk.write


def test_slug_call_risk_only_lowers_for_plain_look_ups():
    one = {"tools": [{"tool_slug": "GMAIL_FETCH_EMAILS", "arguments": {}}]}
    two = {"tools": [{"tool_slug": "GMAIL_FETCH_EMAILS"}, {"tool_slug": "GMAIL_SEND_EMAIL"}]}
    assert _slug_call_risk(Risk.write, one, None) == Risk.read
    assert _slug_call_risk(Risk.write, two, None) == Risk.send
    assert _slug_call_risk(Risk.send, {"tools": [{"tool_slug": "GMAIL_CREATE_EMAIL_DRAFT"}]}, None) == Risk.send
    assert _slug_call_risk(Risk.write, {"tools": []}, None) is None  # nothing named: the tool's own risk
    assert _runs_slugs(MULTI_EXECUTE_SCHEMA)
    assert not _runs_slugs({"type": "object", "properties": {"tool_slugs": {"type": "array"}}})


# ---------------------------------------------------------------------------- hiding and refusing
async def test_read_only_hides_write_tools_and_switching_needs_no_restart(config, isolated_home):
    log: list[str] = []
    config.integrations.read_only = ["pantry"]
    s = await _start(config, isolated_home, FakeProvider(), "hide", _pantry(log))
    try:
        offered = {t["function"]["name"] for t in s.registry.openai_schemas()}
        assert {"pantry_list_items", "pantry_multi_execute"} <= offered  # look-ups and per-call tools stay
        assert "pantry_add_item" not in offered
        assert "pantry_add_item" not in select_tools(s.registry, ["pantry"])[0]  # planners don't list it
        listed = next(p for p in s.registry.catalog() if p["id"] == "pantry")
        assert [t["name"] for t in listed["tools"]] == ["pantry_list_items", "pantry_multi_execute"]
        assert "Pantry" in await s.agent.system_prompt("hi", "cli")

        s.config.integrations.read_only = []  # Read and write, no restart
        assert "pantry_add_item" in {t["function"]["name"] for t in s.registry.openai_schemas()}
        assert "Read only apps" not in await s.agent.system_prompt("hi", "cli")
    finally:
        await s.stop()


async def test_a_forced_write_call_is_refused_then_asks_after_switching(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"pantry": "allow"}  # even an Allow rule can't open a Read only connection
    config.integrations.read_only = ["pantry"]
    llm = FakeProvider(replies=[
        [tool_call("pantry_add_item", item="oats")], "It is read only.",
        [tool_call("pantry_add_item", item="oats")], "Added.",
    ])
    s = await _start(config, isolated_home, llm, "refuse", _pantry(log))
    try:
        sid = await s.store.create_session(channel="cli")
        asked, events = await _turn(s, sid, "add oats to my pantry")
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert asked == [] and log == []
        assert result.is_error and "Pantry is set to Read only" in str(result.result)
        assert "pantry_add_item" not in _offered(llm, 0)

        s.config.tools.approvals.rules = {}
        await s.integrations.set_access("weather", "read_write")  # unrelated app: nothing else changes
        s.config.integrations.read_only = []
        asked, _ = await _turn(s, sid, "add oats to my pantry")
        assert "pantry_add_item" in _offered(llm, 2)
        assert [a.name for a in asked] == ["pantry_add_item"] and log == ["add:oats"]
    finally:
        await s.stop()


async def test_a_tool_that_runs_slugs_may_look_up_but_not_change(config, isolated_home):
    log: list[str] = []
    config.integrations.read_only = ["pantry"]
    llm = FakeProvider(replies=[
        [tool_call("pantry_multi_execute", tools=[{"tool_slug": "PANTRY_LIST_ITEMS", "arguments": {}}])],
        [tool_call("pantry_multi_execute", tools=[{"tool_slug": "PANTRY_LIST_ITEMS"}, {"tool_slug": "PANTRY_ADD_ITEM"}])],
        "done",
    ])
    s = await _start(config, isolated_home, llm, "slugs", _pantry(log))
    try:
        sid = await s.store.create_session(channel="cli")
        _, events = await _turn(s, sid, "check the pantry, then add oats")
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert not results[0].is_error and results[1].is_error
        assert "Read only" in str(results[1].result)
        assert log == ["run:PANTRY_LIST_ITEMS"]
    finally:
        await s.stop()


async def test_switching_to_read_only_while_a_call_waits_stops_it(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[[tool_call("pantry_add_item", item="oats")], "ok"])
    s = await _start(config, isolated_home, llm, "waiting", _pantry(log))
    try:
        sid = await s.store.create_session(channel="cli")
        events = []
        async for ev in s.agent.run_turn(sid, "add oats", channel="cli"):
            events.append(ev)
            if isinstance(ev, ApprovalRequest):
                s.config.integrations.read_only = ["pantry"]  # switched while the question is open
                s.approvals.resolve(ev.approval_id, "allow")
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result.is_error and "Read only" in str(result.result) and log == []
    finally:
        await s.stop()


async def test_scripts_cannot_reach_write_tools_of_a_read_only_connection():
    log: list[str] = []
    plugin = _pantry(log)
    for t in plugin.tools:
        t.plugin = "pantry"
    add, multi = plugin.tools[1], plugin.tools[2]
    apps = ["pantry"]
    policy = BridgePolicy(approvals_mode="off", read_only_apps_source=lambda: apps)
    ctx = ToolContext(store=None, config=None, llm=None)
    assert not policy.is_available(add) and policy.is_available(multi)
    with pytest.raises(Refused, match="Read only"):
        await policy.check(add, add.name, {"item": "oats"}, ctx, 0)
    with pytest.raises(Refused, match="Read only"):
        await policy.check(multi, multi.name, {"tools": [{"tool_slug": "PANTRY_ADD_ITEM"}]}, ctx, 0)
    assert await policy.check(multi, multi.name, {"tools": [{"tool_slug": "PANTRY_LIST_ITEMS"}]}, ctx, 0) == Risk.read
    apps.clear()  # read on every call: the switch applies to a running script
    assert await policy.check(add, add.name, {"item": "oats"}, ctx, 0) == Risk.write


# ---------------------------------------------------------------------------- API
@pytest.fixture
def keychain(monkeypatch) -> dict[str, str]:
    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: store.get(name))
    monkeypatch.setattr(secrets, "set_secret", lambda name, value: store.__setitem__(name, value) or True)
    monkeypatch.setattr(secrets, "delete_secret", lambda name: store.pop(name, None) is not None)
    return store


@pytest.fixture
def client(config, isolated_home, monkeypatch, keychain):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "routes.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        c.core = core  # type: ignore[attr-defined]
        yield c


def test_access_routes_for_integrations(client):
    core = client.core  # type: ignore[attr-defined]
    assert client.get("/api/integrations/github").json()["access"] == "read_write"
    r = client.put("/api/integrations/github/access", json={"access": "read"})
    assert r.status_code == 200 and r.json()["access"] == "read"
    assert core.config.integrations.read_only == ["github"]
    hidden = {t.name for t in core.registry.tools(include_hidden=True) if core.registry.is_blocked(t)}
    assert "github_create_issue" in hidden and "github_search_issues" not in hidden
    assert client.put("/api/integrations/github/access", json={"access": "all"}).status_code == 422
    assert client.put("/api/integrations/nope/access", json={"access": "read"}).status_code == 404
    # a built-in connects at once with the chosen access
    r = client.post("/api/integrations/charts/connect", json={"fields": {}, "access": "read"})
    assert r.json()["access"] == "read" and "charts" in core.config.integrations.read_only
    client.put("/api/integrations/github/access", json={"access": "read_write"})
    assert core.config.integrations.read_only == ["charts"]


def test_access_routes_for_mcp_servers(client):
    core = client.core  # type: ignore[attr-defined]
    body = {"name": "Larder", "transport": "stdio", "command": sys.executable, "args": [str(ECHO_SERVER)],
            "access": "read"}
    server = client.post("/api/integrations/mcp", json=body).json()
    assert server["access"] == "read" and core.config.integrations.read_only == ["mcp_larder"]
    deadline = time.monotonic() + 60
    while server["status"] != "connected" and time.monotonic() < deadline:
        time.sleep(0.2)
        server = next(x for x in client.get("/api/integrations/mcp").json() if x["name"] == "Larder")
    assert server["status"] == "connected", server
    offered = {t["function"]["name"] for t in core.registry.openai_schemas()}
    assert "mcp_larder_echo" in offered and "mcp_larder_add_numbers" not in offered
    r = client.post("/api/integrations/mcp/Larder/access", json={"access": "read_write"})
    assert r.status_code == 200 and r.json()["access"] == "read_write" and core.config.integrations.read_only == []
    assert "mcp_larder_add_numbers" in {t["function"]["name"] for t in core.registry.openai_schemas()}
    client.post("/api/integrations/mcp/Larder/access", json={"access": "read"})
    assert client.post("/api/integrations/mcp/Nope/access", json={"access": "read"}).status_code == 404
    assert client.delete("/api/integrations/mcp/Larder").json() == {"ok": True}
    assert core.config.integrations.read_only == []
