"""Chat subagents (docs/API.md section 10)."""

from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.events import ToolProgress, ToolResultEvent
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call


def make_plugin(pid: str, *tools_) -> ToolPlugin:
    class P(ToolPlugin):
        id = pid
        display_name = pid.title()

    P.tools = list(tools_)
    return P()


async def start(config, isolated_home, llm, name: str) -> SentientApp:
    return await SentientApp(config, llm=llm, db_path=isolated_home / f"{name}.db", enable_background=False).start()


async def test_foreground_subagent_streams_progress_and_returns_summary(config, isolated_home):
    llm = FakeProvider(replies=[
        [tool_call("delegate_task", goal="Find out what time it is")],  # parent
        [tool_call("current_datetime")],                                  # subagent round 1
        "It is noon.",                                                    # subagent summary
        "Your helper says it is noon.",                                   # parent final
    ])
    s = await start(config, isolated_home, llm, "fg")
    try:
        sid = await s.store.create_session(channel="cli")
        async with s.bus.subscribe() as q:
            events = [ev async for ev in s.agent.run_turn(sid, "what time is it, ask a helper", channel="cli")]
            published = [q.get_nowait() for _ in range(q.qsize())]
        result = next(e for e in events if isinstance(e, ToolResultEvent) and e.name == "delegate_task")
        assert result.result["status"] == "completed" and result.result["summary"] == "It is noon."
        progress = [e for e in events if isinstance(e, ToolProgress)]
        assert progress and all(p.kind == "subagent" and p.call_id == "call_delegate_task" for p in progress)
        sub_id = result.result["subagent_id"]
        assert all(p.data["subagent_id"] == sub_id and p.data["message"] == p.text for p in progress)
        assert events.index(progress[-1]) < events.index(result)

        sub_call = llm.calls[1]
        assert sub_call["role"] == "executor" and "You are a subagent" in sub_call["messages"][0]["content"]
        assert "Find out what time it is" in sub_call["messages"][1]["content"]
        offered = {t["function"]["name"] for t in sub_call["tools"]}
        assert "current_datetime" in offered and not offered & {"delegate_task", "delegate_tasks"}

        subs = await s.subagents.list_for_session(sid)
        assert len(subs) == 1 and subs[0]["parent_call_id"] == "call_delegate_task" and subs[0]["tool_calls"] == 1
        kinds = [e["message"]["type"] for e in subs[0]["events"]]
        assert "tool_call" in kinds and "tool_result" in kinds and kinds[-1] == "final_answer"
        updates = [e["data"] for e in published if e["type"] == "subagent.updated"]
        assert [u["status"] for u in updates] == ["running", "completed"] and "events" not in updates[-1]
        # the parent's transcript does not include the subagent's messages
        assert [r["role"] for r in await s.store.recent_messages(sid, 20)] == ["user", "assistant", "tool", "assistant"]
    finally:
        await s.stop()


async def test_subagent_policy_refuses_send_approval_and_delegation(config, isolated_home):
    bought: list[str] = []
    noted: list[str] = []

    @tool("buy_thing", risk=Risk.write, risk_fn=lambda a, c: Risk.send)
    async def buy_thing(ctx: ToolContext) -> str:
        """Buy something."""
        bought.append("x")
        return "bought"

    @tool("note_outside", risk=Risk.write)
    async def note_outside(ctx: ToolContext) -> str:
        """Write a note in another app."""
        noted.append("x")
        return "noted"

    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[
        [
            {"id": "c1", "name": "buy_thing", "arguments": {}},
            {"id": "c2", "name": "note_outside", "arguments": {}},
            {"id": "c3", "name": "delegate_task", "arguments": {"goal": "x"}},
        ],
        "I could not buy it.",
    ])
    s = await start(config, isolated_home, llm, "policy")
    try:
        s.registry.register(make_plugin("shop", buy_thing, note_outside))
        out = await s.subagents.delegate("Buy the thing", tools=["shop", "delegate_task"])
        assert out["status"] == "completed" and out["summary"] == "I could not buy it."
        assert bought == [] and noted == []
        sub = await s.subagents.get(out["subagent_id"])
        results = [e["message"] for e in sub["events"] if e["message"]["type"] == "tool_result"]
        assert all(r["is_error"] for r in results)
        text = " ".join(r["result"] for r in results)
        assert "risk send" in text and "needs the user's approval" in text and "unknown tool delegate_task" in text
        refusal = s.subagents.policy(None)(s.registry.get("delegate_task"), Risk.write, {})
        assert "cannot start other subagents" in refusal
        # the tool itself also refuses when called from inside a subagent
        ctx = s.agent.tool_context(None, "subagent")
        ctx.extra["subagent_id"] = "abc"
        assert "cannot start" in (await s.registry.get("delegate_task").call(ctx, {"goal": "x"}))["error"]
    finally:
        await s.stop()


async def test_background_subagent_posts_to_chat_and_notifies(config, isolated_home):
    llm = FakeProvider(replies=["Background summary here."])
    s = await start(config, isolated_home, llm, "bg")
    try:
        sid = await s.store.create_session(channel="cli")
        out = await s.subagents.delegate("Summarize the news", session_id=sid, background=True)
        assert out["status"] == "running"
        task = s.subagents._tasks[out["subagent_id"]]
        await asyncio.wait_for(task, 5)
        sub = await s.subagents.get(out["subagent_id"])
        assert sub["status"] == "completed" and sub["background"] and sub["finished_at"]
        last = (await s.store.recent_messages(sid, 5))[-1]
        assert last["role"] == "assistant" and "Background summary here." in last["content"]
        rows = await s.store.fetchall("SELECT kind, payload FROM notifications")
        payloads = [json.loads(r["payload"]) for r in rows if r["kind"] == "info"]
        assert {"subagent_id": out["subagent_id"], "session_id": sid} in payloads
    finally:
        await s.stop()


async def test_cancel_and_timeout(config, isolated_home):
    started = asyncio.Event()

    @tool("wait_forever", risk=Risk.read)
    async def wait_forever(ctx: ToolContext) -> str:
        """Wait."""
        started.set()
        await asyncio.sleep(60)
        return "never"

    llm = FakeProvider(replies=[[tool_call("wait_forever")], [tool_call("wait_forever")]])
    s = await start(config, isolated_home, llm, "cancel")
    try:
        s.registry.register(make_plugin("waiting", wait_forever))
        sid = await s.store.create_session(channel="cli")
        out = await s.subagents.delegate("Wait", session_id=sid, background=True, tools=["wait_forever"])
        await asyncio.wait_for(started.wait(), 5)
        sub = await s.subagents.cancel(out["subagent_id"])
        assert sub["status"] == "cancelled" and sub["finished_at"]
        assert await s.store.recent_messages(sid, 5) == []  # a cancelled subagent does not post to the chat

        config.subagents.timeout_minutes = 0.002  # about 0.1 s
        res = await s.subagents.delegate("Wait again", tools=["wait_forever"])
        assert res["status"] == "error" and "Stopped after" in res["error"]
        assert await s.subagents.cancel("nope") is None
    finally:
        await s.stop()


async def test_concurrency_limit_and_delegate_tasks(config, isolated_home):
    state = {"active": 0, "max": 0}

    @tool("probe", risk=Risk.read)
    async def probe(ctx: ToolContext) -> str:
        """Probe."""
        state["active"] += 1
        state["max"] = max(state["max"], state["active"])
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return "ok"

    config.subagents.max_concurrent = 1
    llm = FakeProvider(replies=[
        [tool_call("delegate_tasks", tasks=[{"goal": "one"}, {"goal": "two"}, {"context": "no goal"}])],
        [tool_call("probe")], "summary A", [tool_call("probe")], "summary B",
        "both done",
    ])
    s = await start(config, isolated_home, llm, "many")
    try:
        s.registry.register(make_plugin("probes", probe))
        sid = await s.store.create_session(channel="cli")
        events = [ev async for ev in s.agent.run_turn(sid, "do two things", channel="cli")]
        res = next(e for e in events if isinstance(e, ToolResultEvent) and e.name == "delegate_tasks").result
        assert [r["status"] for r in res["results"]] == ["completed", "completed"]
        assert sorted(r["summary"] for r in res["results"]) == ["summary A", "summary B"]
        assert state["max"] == 1
        assert len(await s.subagents.list_for_session(sid)) == 2
    finally:
        await s.stop()


async def test_disabled_subagents_are_hidden(config, isolated_home):
    config.subagents.enabled = False
    s = await start(config, isolated_home, FakeProvider(), "off")
    try:
        assert s.registry.is_hidden("subagents")
        out = await s.subagents.delegate("anything")
        assert out["status"] == "error" and "turned off" in out["error"]
    finally:
        await s.stop()


async def test_running_rows_marked_after_restart(config, isolated_home):
    s = await start(config, isolated_home, FakeProvider(), "restart")
    await s.store.execute(
        "INSERT INTO subagents(id, goal, status, started_at) VALUES('old', 'g', 'running', '2026-01-01T00:00:00')"
    )
    await s.stop()
    s = await start(config, isolated_home, FakeProvider(), "restart")
    try:
        sub = await s.subagents.get("old")
        assert sub["status"] == "error" and "restarted" in sub["error"]
    finally:
        await s.stop()


def test_subagent_routes(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider(replies=[[tool_call("delegate_task", goal="Say hi")], "hi from helper", "parent done"])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "routes.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        sid = c.post("/api/sessions").json()["session_id"]
        lines = [json.loads(x) for x in c.post("/api/chat", json={"text": "ask a helper", "session_id": sid}).text.splitlines()]
        assert lines[-1]["type"] == "done" and any(x["type"] == "tool_progress" for x in lines)
        subs = c.get(f"/api/sessions/{sid}/subagents").json()
        assert len(subs) == 1 and subs[0]["status"] == "completed" and subs[0]["summary"] == "hi from helper"
        one = c.get(f"/api/subagents/{subs[0]['subagent_id']}").json()
        assert one["goal"] == "Say hi" and one["events"]
        assert c.post(f"/api/subagents/{subs[0]['subagent_id']}/cancel").json()["status"] == "completed"
        assert c.get("/api/subagents/missing").status_code == 404
        assert c.post("/api/subagents/missing/cancel").status_code == 404
        c.delete(f"/api/sessions/{sid}")
        assert c.get(f"/api/sessions/{sid}/subagents").json() == []


async def test_subagent_stops_on_repeated_calls_and_on_its_token_limit(config, isolated_home):
    from tests.test_loop_breaker import RecordingProvider, read

    llm = RecordingProvider(replies=[read("notes.txt", i) for i in range(4)])
    s = await start(config, isolated_home, llm, "loop")
    try:
        res = await s.subagents.delegate("Read my notes", tools=["file_read"])
        assert res["status"] == "error" and "file_read ran 3 times with the same details" in res["error"]
        assert len(llm.seen) == 3
    finally:
        await s.stop()

    config.subagents.max_tokens = 1000
    llm = RecordingProvider(replies=[read(f"n{i}.txt", i) for i in range(6)], model="openai/gpt-test", tokens=600)
    s = await start(config, isolated_home, llm, "budget")
    try:
        res = await s.subagents.delegate("Read my notes", tools=["file_read"])
        assert res["status"] == "error" and res["error"].startswith("Stopped after using 1,200 tokens")
        assert len(llm.seen) == 2
    finally:
        await s.stop()
