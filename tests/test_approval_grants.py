"""Scoped answers to approval requests in chats (#139): once, this chat, a while, always; and what they never cover."""

from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.events import ApprovalRequest
from sentient.presence import Presence
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call


def _notes(log: list[str]) -> ToolPlugin:
    @tool("notes_add", risk=Risk.write)
    async def notes_add(ctx: ToolContext, text: str) -> dict:
        """Add a note in the notes app."""
        log.append(text)
        return {"ok": True}

    @tool("notes_run", risk=Risk.exec)
    async def notes_run(ctx: ToolContext, code: str) -> dict:
        """Run a script in the notes app."""
        log.append(f"run:{code}")
        return {"ok": True}

    class Notes(ToolPlugin):
        id = "notes"
        display_name = "Notes"
        tools = [notes_add, notes_run]

    return Notes()


async def _start(config, isolated_home, llm, *plugins: ToolPlugin) -> SentientApp:
    config.chat.tool_selection = "all"
    config.tools.approvals.mode = "ask"
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "g.db", enable_background=False).start()
    for p in plugins:
        s.registry.register(p)
    return s


async def _turn(s: SentientApp, sid: str, decision: str) -> list[ApprovalRequest]:
    asked: list[ApprovalRequest] = []
    async for ev in s.agent.run_turn(sid, "go", channel="cli"):
        if isinstance(ev, ApprovalRequest):
            asked.append(ev)
            assert s.approvals.resolve(ev.approval_id, decision)
    return asked


async def test_allow_for_a_while_skips_the_question_until_it_runs_out(config, isolated_home):
    log: list[str] = []
    add = [tool_call("notes_add", text="milk")]
    llm = FakeProvider(replies=[add, "ok", add, "ok", add, "ok"])
    s = await _start(config, isolated_home, llm, _notes(log))
    try:
        sid = await s.store.create_session(channel="cli")
        [asked] = await _turn(s, sid, "allow_timed")
        assert asked.choices == ["allow", "allow_session", "allow_timed", "allow_always", "deny"]
        assert asked.grant_minutes == 60
        other = await s.store.create_session(channel="cli")
        assert await _turn(s, other, "deny") == []  # covered in another chat too
        assert log == ["milk", "milk"]

        real = s.approvals.grants.clock
        s.approvals.grants.clock = lambda: real() + timedelta(minutes=61)
        assert len(await _turn(s, other, "deny")) == 1  # ran out: asks again
    finally:
        await s.stop()


async def test_always_allow_from_a_chat_saves_an_allow_rule(config, isolated_home):
    log: list[str] = []
    llm = FakeProvider(replies=[[tool_call("notes_add", text="eggs")], "ok"])
    s = await _start(config, isolated_home, llm, _notes(log))
    try:
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, "allow_always")
        assert s.config.tools.approvals.rules == {"notes_add": "allow"}
        assert s.config.tools.approvals.rule_origins["notes_add"].source == "approval"
    finally:
        await s.stop()


async def test_wide_answers_count_as_once_where_only_once_is_offered(config, isolated_home):
    log: list[str] = []
    llm = FakeProvider(replies=[[tool_call("notes_run", code="1+1")], "ok"])
    s = await _start(config, isolated_home, llm, _notes(log))
    try:
        sid = await s.store.create_session(channel="cli")
        async for ev in s.agent.run_turn(sid, "go", channel="cli"):
            if isinstance(ev, ApprovalRequest):
                assert ev.choices == ["allow", "deny"]  # running code: once at a time
                assert not s.approvals.resolve(ev.approval_id, "made_up")
                assert s.approvals.resolve(ev.approval_id, "allow_always")  # counts as "Allow once"
        assert log == ["run:1+1"] and await s.approvals.grants.list() == []
        assert s.config.tools.approvals.rules == {}
    finally:
        await s.stop()


async def test_a_grant_never_covers_a_higher_risk(config, isolated_home):
    s = await _start(config, isolated_home, FakeProvider(), _notes([]))
    try:
        await s.approvals.grants.add("notes_add", "timed", Risk.read)
        assert s.approvals.grants.covers("notes_add", Risk.read) is not None
        assert s.approvals.grants.covers("notes_add", Risk.write) is None
        task = await s.approvals.grants.add("notes_add", "task", Risk.write, task_id="t1")
        assert s.approvals.grants.covers("notes_add", Risk.write, "t2") is None
        assert s.approvals.grants.covers("notes_add", Risk.write, "t1")["id"] == task["id"]
    finally:
        await s.stop()


def test_grants_routes_list_and_revoke(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    s = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False)
    with TestClient(create_app(s)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        assert c.get("/api/approvals/grants").json() == {"grants": []}
        grant = c.portal.call(s.approvals.grants.add, "file_write", "timed", Risk.write)
        listed = c.get("/api/approvals/grants").json()["grants"]
        assert [(g["id"], g["scope"], g["label"], g["app"], g["risk"]) for g in listed] == [
            (grant["id"], "timed", '"File write" in Files', "Files", "write"),
        ]
        assert c.delete(f"/api/approvals/grants/{grant['id']}").json() == {"ok": True}
        assert c.delete(f"/api/approvals/grants/{grant['id']}").status_code == 404
        assert c.get("/api/approvals/grants").json() == {"grants": []}


def test_presence_counts_only_windows_that_say_they_are_in_front():
    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, kind: str, data: dict) -> None:
            events.append((kind, data))

    p = Presence(Bus())
    assert not p.in_app()
    p.set("a", False)
    assert not p.in_app() and events == []
    p.set("a", True)
    p.set("b", True)
    p.drop("a")
    assert p.in_app()
    p.set("b", False)
    assert not p.in_app()
    assert events == [("presence.changed", {"in_app": True}), ("presence.changed", {"in_app": False})]
