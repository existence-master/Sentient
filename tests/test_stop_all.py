"""Stop everything (docs/API.md section 17): chat replies, helpers, pausable services, routes, persistence."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.services import Service
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call


def _blocking_plugin(started: asyncio.Event) -> ToolPlugin:
    @tool("wait_forever", risk=Risk.read)
    async def wait_forever(ctx: ToolContext) -> str:
        """Wait."""
        started.set()
        await asyncio.Event().wait()
        return "never"

    class P(ToolPlugin):
        id = "waiting"
        display_name = "Waiting"
        tools = [wait_forever]

    return P()


@pytest.fixture
async def make(config, isolated_home):
    apps: list[SentientApp] = []

    async def factory(llm=None, name: str = "stop") -> SentientApp:
        app = SentientApp(config, llm=llm or FakeProvider(), db_path=isolated_home / f"{name}.db", enable_background=False)
        await app.start()
        apps.append(app)
        return app

    yield factory
    for app in apps:
        if app._started:
            await app.stop()


async def test_stop_cancels_a_chat_reply_during_a_tool_call(make):
    started = asyncio.Event()
    app = await make(FakeProvider(replies=[[tool_call("wait_forever")], "never"]))
    app.registry.register(_blocking_plugin(started))
    sid = await app.store.create_session(channel="desktop")

    async def consume() -> None:
        async for _ in app.agent.run_turn(sid, "wait for it"):
            pass

    turn = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 5)
    async with app.bus.subscribe() as q:
        result = await asyncio.wait_for(app.stop_all(), 3)
        events = [q.get_nowait() for _ in range(q.qsize())]
    assert result["stopped"] is True and result["source"] == "desktop" and result["stopped_at"]
    assert result["cancelled"] >= 1
    assert turn.done() and turn.cancelled()
    assert app.stopped
    stop_events = [e["data"] for e in events if e["type"] == "stop.updated"]
    assert stop_events and stop_events[0]["stopped"] is True
    last = (await app.store.recent_messages(sid, 5))[-1]
    assert last["role"] == "assistant" and "stopped" in last["content"]
    assert not app.agent._turn_tasks  # nothing left registered

    # a reply the user starts while stopped still works; resume clears the flag
    reply = [e async for e in app.agent.run_turn(sid, "hello")]
    assert reply[-1].type == "done"
    resumed = await app.resume()
    assert resumed == {"stopped": False, "stopped_at": None, "source": "desktop"} and not app.stopped


async def test_stop_cancels_background_helpers(make):
    started = asyncio.Event()
    app = await make(FakeProvider(replies=[[tool_call("wait_forever")]]))
    app.registry.register(_blocking_plugin(started))
    out = await app.subagents.delegate("Wait", background=True, tools=["wait_forever"])
    await asyncio.wait_for(started.wait(), 5)
    await asyncio.wait_for(app.stop_all(), 5)
    sub = await app.subagents.get(out["subagent_id"])
    assert sub["status"] == "cancelled" and sub["finished_at"]


async def test_stopped_state_survives_a_restart(make):
    app = await make(name="persist")
    await app.stop_all(source="telegram")
    stopped_at = app.stop_state["stopped_at"]
    await app.stop()

    again = await make(name="persist")
    assert again.stopped and again.stop_state == {"stopped": True, "stopped_at": stopped_at, "source": "telegram"}
    await again.resume()
    await again.stop()

    third = await make(name="persist")
    assert not third.stopped


async def test_stop_twice_keeps_the_first_time(make):
    app = await make()
    first = await app.stop_all(source="device")
    second = await app.stop_all(source="desktop")
    assert second["stopped_at"] == first["stopped_at"] and second["source"] == "device"


async def test_pausable_service_skips_jobs_and_halt_cancels_the_running_one(make):
    app = await make()
    runs: list[str] = []
    gate = asyncio.Event()

    class Ticker(Service):
        name = "ticker"
        pause_on_stop = True

        async def job(self) -> None:
            runs.append("start")
            await gate.wait()
            runs.append("end")

    svc = Ticker(app)
    svc.run_every(0.01, svc.job, name="tick")
    try:
        while not runs:
            await asyncio.sleep(0.01)
        await app.stop_all()  # SentientApp only halts its own services: halt this one directly
        assert await svc.halt() == 1
        await asyncio.sleep(0.1)
        assert runs == ["start"]  # cancelled mid-job, and no new job while stopped
        assert not svc._loops[0].done()  # the loop itself keeps going
        await app.resume()
        gate.set()
        while "end" not in runs:
            await asyncio.sleep(0.01)
    finally:
        await svc.stop()


async def test_dreams_suggestions_and_feeds_wait_while_stopped(make, config):
    config.dreaming.enabled = True
    config.proactivity.enabled = True
    app = await make()
    await app.stop_all()
    assert await app.dreaming.is_due() == (False, "stopped")
    assert await app.proactivity.handle_item("gmail", {"id": "m1", "subject": "Hi"}) is None
    assert await app.proactivity.poll_now() == {"ok": True, "events": 0}
    assert await app.integrations.emit_items("webhook", "webhook", [{"id": "x1"}], event="h1") == []
    await app.resume()
    published = await app.integrations.emit_items("webhook", "webhook", [{"id": "x1"}], event="h1")
    assert [i["id"] for i in published] == ["x1"]  # not claimed while stopped, so it still arrives


def test_routes_bootstrap_and_webhooks(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "routes.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        assert c.get("/api/stop").json() == {"stopped": False, "stopped_at": None, "source": None}
        assert c.get("/api/bootstrap").json()["stop"]["stopped"] is False
        hook = c.post("/api/hooks", json={"name": "Door sensor"}).json()
        with c.websocket_connect("/ws?token=test-token") as ws:
            assert ws.receive_json()["type"] == "hello"
            stopped = c.post("/api/stop-all", json={"source": "Tray!"}).json()
            assert stopped["stopped"] is True and stopped["source"] == "tray" and stopped["cancelled"] == 0
            event = ws.receive_json()
            assert event["type"] == "stop.updated" and event["data"]["stopped"] is True
        assert c.post("/api/stop-all").json()["source"] == "tray"  # no body is fine; the first source is kept
        assert c.get("/api/bootstrap").json()["stop"]["stopped"] is True
        call = c.post(f"/hooks/{hook['id']}", json={"open": True}, headers={"X-Sentient-Secret": hook["secret"]})
        assert call.status_code == 503 and call.headers["retry-after"] == "60"
        resumed = c.post("/api/resume").json()
        assert resumed == {"stopped": False, "stopped_at": None, "source": "desktop"}
        call = c.post(f"/hooks/{hook['id']}", json={"open": True}, headers={"X-Sentient-Secret": hook["secret"]})
        assert call.status_code == 200
        assert c.post("/api/stop-all", headers={"Authorization": "Bearer nope"}).status_code == 401
