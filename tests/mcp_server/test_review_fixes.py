"""Reproductions for the maintainer's changes-requested review of PR #487 (review 5595264068).

1. A chat turn steered by an MCP app is outside content from the start: its text is not the user's
   words, sends and code runs ask first, and a "remember this about me" waits in Review.
2. An approval raised in such a turn also lands as a kind "approval" notification the user can
   answer from the app (the chat stream alone is invisible to MCP clients).
3. The MCP-only token reaches only the endpoints the eight tools need, never config or approvals.
5. A task an MCP app creates always waits for plan approval, even when plans auto-approve.

The second review (5609721491) adds:
6. The identity proof is keyed to the port it was asked on (a relayed proof fails), the gate
   proves again before every call, and the server follows the discovery file to a new port.
7. An MCP caller cannot shape its turn: the body's model, attachments and incognito are
   ignored, its text never starts a rule check, and the approval and plan-ready notifications
   name the app that is asking.

The final review (5624575781) adds:
8. The proof is keyed to the whole address (host and port): a relay on the same port at the
   other loopback address fails too, engine addresses are literal loopback only, and the
   discovery file records the host.
9. The app's name is cleaned (capped, control characters stripped) wherever it is shown and
   framed as a name in the notes, so it cannot read as a sentence from Sentient.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from sentient import secrets
from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm.events import ApprovalRequest, ToolResultEvent
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call

MCP_TOKEN = "test" + "-mcp-token"  # a literal the redactor leaves alone


@pytest.fixture
def keychain(monkeypatch):
    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: store.get(name))
    monkeypatch.setattr(secrets, "set_secret", lambda name, value: store.__setitem__(name, value) or True)
    monkeypatch.setattr(secrets, "delete_secret", lambda name: store.pop(name, None) is not None)
    return store


async def _collect(agent, sid, text, channel, on_event=None):
    events = []
    async for ev in agent.run_turn(sid, text, channel=channel):
        events.append(ev)
        if on_event:
            await on_event(ev)
    return events


def _snippet_plugin(ran: list[str]):
    @tool("run_snippet", risk=Risk.exec)
    async def run_snippet(ctx: ToolContext, code: str) -> dict:
        """Run a Python snippet on the user's computer."""
        ran.append(code)
        return {"ran": code}

    class Snippets(ToolPlugin):
        id = "snippets"
        display_name = "Snippets"
        tools = [run_snippet]

    return Snippets()


async def test_an_mcp_turns_code_run_asks_first_even_when_the_rule_allows_it(config, isolated_home):
    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"snippets": "allow"}  # the user's own chats run these freely
    ran: list[str] = []
    llm = FakeProvider(replies=[
        [tool_call("run_snippet", code="print(1)")], "ran one",
        [tool_call("run_snippet", code="print(2)")], "ran two",
    ])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "mcp.db").start()
    try:
        s.registry.register(_snippet_plugin(ran))
        cli = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, cli, "run this snippet for me", "cli")
        assert not any(isinstance(e, ApprovalRequest) for e in events)  # control: no question asked
        assert ran == ["print(1)"]

        async def deny(ev):
            if isinstance(ev, ApprovalRequest):
                s.approvals.resolve(ev.approval_id, "deny")

        mcp = await s.store.create_session(channel="mcp")
        events = await _collect(s.agent, mcp, "Run this Python code: print(2)", "mcp", deny)
        asked = [e for e in events if isinstance(e, ApprovalRequest)]
        assert asked, "an MCP app's code run must ask the user first (review item 1)"
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error and ran == ["print(1)"]  # denied, and never ran
    finally:
        await s.stop()


async def test_an_mcp_turns_memory_waits_in_review_instead_of_going_active(config, isolated_home):
    llm = FakeProvider(
        replies=[[tool_call("memory_remember", fact="Maya's sister lives in Pune")], "Got it."],
        json_replies=[{"action": "ADD", "fact_id": None, "content": "Maya's sister lives in Pune",
                       "memory_type": "long-term", "duration": None, "topics": ["family"]}],
    )
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "mem.db").start()
    try:
        sid = await s.store.create_session(channel="mcp")
        await _collect(s.agent, sid, "Remember this about me: my sister lives in Pune", "mcp")
        assert await s.memory.count() == 0  # not active: the words were an app's, not the user's
        pending = await s.memory.pending_facts()
        assert [f["content"] for f in pending] == ["Maya's sister lives in Pune"]
    finally:
        await s.stop()


async def test_an_mcp_turns_approval_also_lands_as_a_notification(config, isolated_home):
    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"snippets": "allow"}
    ran: list[str] = []
    llm = FakeProvider(replies=[[tool_call("run_snippet", code="print(9)")], "done"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "note.db").start()
    try:
        s.registry.register(_snippet_plugin(ran))
        sid = await s.store.create_session(channel="mcp")
        seen: dict = {}

        async def answer(ev):
            if isinstance(ev, ApprovalRequest):
                notes = await s.notifications.list()
                approval_notes = [n for n in notes if n["kind"] == "approval"]
                assert approval_notes, "the approval must also land as a notification (review item 2)"
                note = approval_notes[0]
                assert note["payload"]["approval_id"] == ev.approval_id
                # the note names who is asking, framed as a name (this turn named no app)
                assert note["message"].startswith("An MCP app wants to: "), note["message"]
                seen["note_id"] = note["id"]
                s.approvals.resolve(ev.approval_id, "allow")

        await _collect(s.agent, sid, "Run this Python code: print(9)", "mcp", answer)
        assert ran == ["print(9)"]
        note = await s.notifications.get(seen["note_id"])
        assert note["payload"]["status"] == "allow" and note["read"]  # settled once answered
    finally:
        await s.stop()


PLAN = {
    "name": "Tidy downloads",
    "description": "Tidy the downloads folder",
    "plan": [{"tool": "files", "description": "Tidy the downloads folder"}],
    "clarifying_questions": [],
}


@pytest.fixture
def engine_client(config, isolated_home, keychain):
    config.integrations.mcp_server_enabled = True  # the user turned the feature on in Settings
    keychain["mcp-server-token"] = MCP_TOKEN
    llm = FakeProvider(json_replies=[dict(PLAN)])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "g.db",
                                 enable_background=False))
    with TestClient(app) as c:
        c.gateway_token = c.app.state.token
        yield c


@pytest.fixture
def engine_client_off(config, isolated_home, keychain):
    config.integrations.mcp_server_enabled = False  # the token exists, but the feature is off
    keychain["mcp-server-token"] = MCP_TOKEN
    llm = FakeProvider(json_replies=[dict(PLAN)])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "off.db",
                                 enable_background=False))
    with TestClient(app) as c:
        c.gateway_token = c.app.state.token
        yield c


def _mcp(client) -> dict:
    return {"Authorization": f"Bearer {MCP_TOKEN}"}


def _gw(client) -> dict:
    return {"Authorization": f"Bearer {client.gateway_token}"}


def test_the_mcp_token_reaches_only_the_tool_endpoints(engine_client):
    c = engine_client
    assert c.get("/api/memories", headers=_mcp(c)).status_code == 200
    assert c.post("/api/memories", json={"content": "Maya takes the train"}, headers=_mcp(c)).status_code == 200
    assert c.get("/api/tasks", headers=_mcp(c)).status_code == 200
    assert c.post("/api/notifications", json={"message": "build done"}, headers=_mcp(c)).status_code == 200
    assert c.get("/api/mcp/status", headers=_mcp(c)).status_code == 200

    for method, url in [
        ("GET", "/api/config"), ("PATCH", "/api/config"), ("POST", "/api/approvals"),
        ("GET", "/api/memories/review"), ("DELETE", "/api/memories/1"), ("GET", "/api/sessions"),
    ]:
        r = c.request(method, url, json={} if method in ("POST", "PATCH") else None, headers=_mcp(c))
        assert r.status_code in (401, 403), f"{method} {url} must refuse the MCP token, got {r.status_code}"

    assert c.get("/api/config", headers=_gw(c)).status_code == 200  # the gateway token still reaches all


def test_a_task_created_with_the_mcp_token_waits_for_plan_approval(engine_client):
    c = engine_client
    cfg = c.get("/api/config", headers=_gw(c)).json()
    cfg["tasks"]["require_plan_approval"] = False  # plans normally approve themselves
    r = c.patch("/api/config", json={"tasks": cfg["tasks"]}, headers=_gw(c))
    assert r.status_code == 200, r.text

    made = c.post("/api/tasks", json={"prompt": "Tidy the downloads folder", "app": "Claude Code"},
                  headers=_mcp(c))
    assert made.status_code == 200, made.text
    task_id = made.json()["task_id"]
    task = made.json()
    for _ in range(50):  # the planner runs in the background; wait for it to land
        task = c.get(f"/api/tasks/{task_id}", headers=_gw(c)).json()
        if task["status"] != "planning":
            break
        time.sleep(0.1)
    assert task["original_context"]["source"] == "mcp"
    assert task["original_context"]["app"] == "Claude Code"
    assert task["status"] == "approval_pending"  # an outside app's task always waits (review item 5)
    ready = []
    for _ in range(50):  # the plan-ready note lands with the plan; wait for it
        notes = c.get("/api/notifications", headers=_gw(c)).json()["notifications"]
        ready = [n for n in notes if n.get("title") == "Plan ready for approval"]
        if ready:
            break
        time.sleep(0.1)
    assert ready and 'an app called ` Claude Code `' in ready[0]["message"]  # named, and framed as a name


def test_with_the_feature_off_the_mcp_token_opens_nothing_but_status(engine_client_off):
    c = engine_client_off
    for method, url in [("GET", "/api/memories"), ("POST", "/api/memories"), ("GET", "/api/tasks"),
                        ("POST", "/api/tasks"), ("POST", "/api/notifications")]:
        r = c.request(method, url, json={} if method == "POST" else None, headers=_mcp(c))
        assert r.status_code == 403, f"{method} {url} must refuse the MCP token while off, got {r.status_code}"
    # status stays reachable so the MCP server can say "turned off" instead of "refused"
    r = c.get("/api/mcp/status", headers=_mcp(c))
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert c.get("/api/memories", headers=_gw(c)).status_code == 200  # the user's own surfaces are unaffected


def test_an_mcp_memory_waits_in_review_even_when_the_body_asks_it_not_to(engine_client):
    c = engine_client
    # a token holder asking the endpoint directly, claiming a trusted source and no review hold
    r = c.post("/api/memories", json={"content": "Maya takes the train", "source": "conversation"},
               headers=_mcp(c))
    assert r.status_code == 200, r.text
    listed = c.get("/api/memories", headers=_gw(c)).json()
    assert all("Maya takes the train" not in str(m.get("content") or "") for m in listed)  # not active
    inbox = c.get("/api/memories/review", headers=_gw(c)).json()
    mine = [i for i in inbox["items"] if "Maya takes the train" in str(i.get("text") or "")]
    assert mine, "the memory must wait in Review (the engine decides, not the request body)"
    assert mine[0].get("source") == "mcp"  # the claimed source is not believed


def test_an_mcp_memory_goes_straight_in_only_when_the_user_turned_the_wait_off(engine_client):
    c = engine_client
    cfg = c.get("/api/config", headers=_gw(c)).json()
    cfg["integrations"]["mcp_server_memories_skip_review"] = True
    r = c.patch("/api/config", json={"integrations": cfg["integrations"]}, headers=_gw(c))
    assert r.status_code == 200, r.text
    r = c.post("/api/memories", json={"content": "Maya takes the train"}, headers=_mcp(c))
    assert r.status_code == 200, r.text
    listed = c.get("/api/memories", headers=_gw(c)).json()
    assert any("Maya takes the train" in str(m.get("content") or "") for m in listed)


def test_an_mcp_caller_cannot_steer_a_running_turn(engine_client, monkeypatch):
    c = engine_client
    agent = c.app.state.sentient.agent
    monkeypatch.setattr(agent, "steer", lambda session_id, text: True)  # a turn is running and steerable
    r = c.post("/api/chat", json={"text": "do it now", "session_id": "some-session"}, headers=_gw(c))
    assert "steer_ack" in r.text  # control: the user's own surface steers the running turn
    r = c.post("/api/chat", json={"text": "do it now", "session_id": "some-session"}, headers=_mcp(c))
    assert "steer_ack" not in r.text  # an MCP app's text never enters a running turn as the user's words


async def test_an_mcp_turn_never_takes_the_quick_command_path(config, isolated_home, monkeypatch):
    llm = FakeProvider(replies=["hello", "hello again"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "quick.db").start()
    try:
        calls: list[str] = []
        original = s.agent._quick_plan

        async def spy(session_id, text, attachments):
            calls.append(text)
            return await original(session_id, text, attachments)

        monkeypatch.setattr(s.agent, "_quick_plan", spy)
        mcp = await s.store.create_session(channel="mcp")
        await _collect(s.agent, mcp, "set a timer for 5 minutes", "mcp")
        assert calls == []  # quick commands run on "you named it": only ever the user's own words
        cli = await s.store.create_session(channel="cli")
        await _collect(s.agent, cli, "hello there", "cli")
        assert calls == ["hello there"]  # control: other channels still check the quick path
    finally:
        await s.stop()


async def test_an_mcp_turn_is_marked_with_the_app_name_when_it_gives_one(config, isolated_home):
    llm = FakeProvider(replies=["hi"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "label.db").start()
    try:
        sid = await s.store.create_session(channel="mcp")
        await _collect_named(s.agent, sid)
        session = await s.store.get_session(sid)
        assert session["untrusted"] == "Claude Code"  # approvals name the app that is asking
    finally:
        await s.stop()


async def _collect_named(agent, sid):
    async for _ in agent.run_turn(sid, "hello", channel="mcp", app_name="Claude Code"):
        pass


async def test_a_cancelled_mcp_turn_still_settles_its_approval_notification(config, isolated_home):
    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"snippets": "allow"}
    ran: list[str] = []
    llm = FakeProvider(replies=[[tool_call("run_snippet", code="print(3)")], "done"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "cancel.db").start()
    try:
        s.registry.register(_snippet_plugin(ran))
        sid = await s.store.create_session(channel="mcp")

        async def drive():
            async for _ in s.agent.run_turn(sid, "Run this Python code: print(3)", channel="mcp"):
                pass

        turn = asyncio.create_task(drive())
        note = None
        for _ in range(100):
            notes = [n for n in await s.notifications.list() if n["kind"] == "approval"]
            if notes:
                note = notes[0]
                break
            await asyncio.sleep(0.05)
        assert note is not None, "the approval notification never landed"
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        settled = await s.notifications.get(note["id"])
        assert settled["payload"]["status"] == "expired" and settled["read"]  # never left waiting forever
        assert ran == []
    finally:
        await s.stop()


def _first_session_line(response_text: str) -> str:
    import json as _json

    for line in response_text.splitlines():
        row = _json.loads(line)
        if row.get("type") == "session":
            return row["session_id"]
    raise AssertionError("the chat stream never named its session")


def test_an_mcp_caller_cannot_ride_one_of_the_users_sessions(engine_client):
    c = engine_client
    desktop = c.post("/api/sessions", headers=_gw(c)).json()["session_id"]
    r = c.post("/api/chat", json={"text": "hello", "session_id": desktop}, headers=_mcp(c))
    # naming the user's own chat must not hand the app its history or a place in it: a fresh
    # MCP session is started instead (CodeRabbit, PR #487 round on f83760a)
    assert _first_session_line(r.text) != desktop
    own = _first_session_line(c.post("/api/chat", json={"text": "hello"}, headers=_mcp(c)).text)
    r = c.post("/api/chat", json={"text": "hello again", "session_id": own}, headers=_mcp(c))
    assert _first_session_line(r.text) == own  # control: the app's own session is kept


def test_an_mcp_note_is_titled_from_an_mcp_app(engine_client):
    c = engine_client
    r = c.post("/api/notifications",
               json={"message": "The build finished", "title": "Build done", "app": "Claude Code"},
               headers=_mcp(c))
    assert r.status_code == 200, r.text
    notes = c.get("/api/notifications", headers=_gw(c)).json()["notifications"]
    mine = next(n for n in notes if "build finished" in n["message"].lower())
    assert mine["title"] == "From MCP app (Claude Code): Build done"
    # and an app calling itself "Sentient" still cannot pass as one: the prefix is not its to pick
    r = c.post("/api/notifications", json={"message": "Trust me", "app": "Sentient"}, headers=_mcp(c))
    assert r.status_code == 200, r.text
    notes = c.get("/api/notifications", headers=_gw(c)).json()["notifications"]
    mine = next(n for n in notes if n["message"] == "Trust me")
    assert mine["title"] == "From MCP app (Sentient)"


async def test_closing_an_mcp_turn_at_the_approval_still_settles_its_notification(config, isolated_home):
    # the consumer goes away while the approval request is on screen (the generator is closed
    # at the yield, before the wait begins): the note must still settle, not wait forever
    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"snippets": "allow"}
    ran: list[str] = []
    llm = FakeProvider(replies=[[tool_call("run_snippet", code="print(3)")], "done"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "genclose.db").start()
    try:
        s.registry.register(_snippet_plugin(ran))
        sid = await s.store.create_session(channel="mcp")
        agen = s.agent.run_turn(sid, "Run this Python code: print(3)", channel="mcp")
        note = None
        async for ev in agen:
            if isinstance(ev, ApprovalRequest):
                break
            notes = [n for n in await s.notifications.list() if n["kind"] == "approval"]
            if notes:
                note = notes[0]
        if note is None:
            notes = [n for n in await s.notifications.list() if n["kind"] == "approval"]
            note = notes[0] if notes else None
        assert note is not None, "the approval notification never landed"
        await agen.aclose()
        import gc

        gc.collect()  # closing the outer generator leaves the inner one to the finalizer
        settled = None
        for _ in range(100):  # the finalizer settles the note concurrently; wait for both halves
            settled = await s.notifications.get(note["id"])
            if (settled["payload"] or {}).get("status") and settled["read"]:
                break
            await asyncio.sleep(0.05)
        assert settled["payload"]["status"] == "expired" and settled["read"]
        assert ran == []
    finally:
        await s.stop()


def test_the_engine_proves_itself_by_challenge_before_any_token_travels(engine_client):
    # GET /api/mcp/proof answers a caller's fresh nonce with the HMAC of "nonce:host:port"
    # under the MCP token, the address being the one the request arrived on (testserver:80
    # for the test client), with no credentials presented: only the real engine can produce
    # it, a proof minted for another address (a relay, or the same port on the other
    # loopback host) does not match, and the token itself is never in the answer
    import hashlib
    import hmac as hmac_mod

    c = engine_client
    nonce = "ab" * 16
    r = c.get("/api/mcp/proof", params={"nonce": nonce})  # no Authorization header at all
    assert r.status_code == 200, r.text
    expected = hmac_mod.new(MCP_TOKEN.encode(), f"{nonce}:testserver:80".encode(), hashlib.sha256).hexdigest()
    assert r.json()["proof"] == expected
    assert MCP_TOKEN not in r.text
    assert c.get("/api/mcp/proof", params={"nonce": "short"}).status_code == 404


def test_there_is_no_proof_without_an_issued_token(config, isolated_home, keychain):
    # the keychain fixture is left empty here: no token was ever issued, so the proof
    # route does not exist (there would be nothing to prove with)
    app = create_app(SentientApp(config, llm=FakeProvider(json_replies=[dict(PLAN)]),
                                 db_path=isolated_home / "noproof.db", enable_background=False))
    with TestClient(app) as c:
        assert c.get("/api/mcp/proof", params={"nonce": "ab" * 16}).status_code == 404


def test_an_mcp_caller_cannot_shape_the_turn(engine_client, monkeypatch):
    # the body's model, attachments and incognito are ignored for an MCP caller (review
    # 5609721491): attachments would pull files into the turn, a named model could pick a
    # cloud model where the user keeps chats local, incognito is the user's own switch
    import json as _json

    from sentient.llm.events import Done

    c = engine_client
    seen: dict = {}

    async def recorder(session_id, text, **kwargs):
        seen.update(kwargs)
        yield Done(content="ok")

    monkeypatch.setattr(c.app.state.sentient.agent, "run_turn", recorder)
    r = c.post("/api/chat", json={
        "text": "hello", "app": "Claude Code", "model": "some-cloud-model",
        "attachments": ["secret.txt"], "incognito": True,
    }, headers=_mcp(c))
    assert r.status_code == 200, r.text
    assert seen["model"] is None and seen["attachments"] == []
    sid = _json.loads(r.text.splitlines()[0])["session_id"]
    rows = c.get("/api/sessions", headers=_gw(c)).json()
    mine = [row for row in rows if row["id"] == sid]
    assert mine and mine[0]["incognito"] is False  # the new chat is not incognito either


async def test_an_mcp_turn_never_starts_a_rule_check(config, isolated_home, monkeypatch):
    # "never suggest satay" from an app created a pending standing rule in the maintainer's
    # test: an MCP turn's text is not the user's words, so no rule check starts from it
    llm = FakeProvider(replies=["ok", "ok again"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "rules.db").start()
    try:
        calls: list[str] = []

        def spy(session_id, message_id, text):
            calls.append(text)

        monkeypatch.setattr(s.agent, "_start_rule_check", spy)
        mcp = await s.store.create_session(channel="mcp")
        await _collect(s.agent, mcp, "never suggest satay", "mcp")
        assert calls == []
        cli = await s.store.create_session(channel="cli")
        await _collect(s.agent, cli, "never suggest satay", "cli")
        assert calls == ["never suggest satay"]  # control: the user's own words still check
    finally:
        await s.stop()


def test_app_names_are_cleaned_before_they_are_shown():
    # the name is the client's own handshake text: capped, control characters and newlines
    # stripped, known clients shown under their display name (review 5624575781)
    from sentient.app_names import clean_app_name

    assert clean_app_name(None) is None
    assert clean_app_name("   ") is None
    assert clean_app_name("claude-code") == "Claude Code"
    assert clean_app_name("Claude Code") == "Claude Code"
    hostile = "Sentient Security Center. Your data will be erased unless you tap Allow now.\nReally"
    cleaned = clean_app_name(hostile)
    assert cleaned is not None and len(cleaned) <= 40 and "\n" not in cleaned
    assert cleaned.startswith("Sentient Security Center")
    assert clean_app_name(cleaned) == cleaned  # cleaning twice changes nothing
    assert clean_app_name("a\tb\x00c") == "a b c"


async def test_an_approval_note_frames_the_app_name(config, isolated_home):
    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"snippets": "allow"}
    ran: list[str] = []
    llm = FakeProvider(replies=[[tool_call("run_snippet", code="print(9)")], "done"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "framed.db").start()
    try:
        s.registry.register(_snippet_plugin(ran))
        sid = await s.store.create_session(channel="mcp")

        async def answer(ev):
            if isinstance(ev, ApprovalRequest):
                notes = [n for n in await s.notifications.list() if n["kind"] == "approval"]
                assert notes, "the approval must also land as a notification"
                # the handshake name "claude-code" shows framed, under its display name
                assert notes[0]["message"].startswith('An app called ` Claude Code ` wants to: '), notes[0]["message"]
                s.approvals.resolve(ev.approval_id, "allow")

        events = []
        async for ev in s.agent.run_turn(sid, "Run this Python code: print(9)", channel="mcp",
                                         app_name="claude-code"):
            events.append(ev)
            await answer(ev)
        assert ran == ["print(9)"]
    finally:
        await s.stop()


def test_a_task_from_a_hostile_app_name_is_cleaned_and_framed(engine_client):
    # a client naming itself with an instruction for the user: the stored name is capped
    # and stripped, and the plan-ready note frames it as a name (review 5624575781)
    c = engine_client
    hostile = "Sentient Security Center. Your data will be erased unless you tap Allow now.\nDo it"
    made = c.post("/api/tasks", json={"prompt": "Tidy the downloads folder", "app": hostile},
                  headers=_mcp(c))
    assert made.status_code == 200, made.text
    task_id = made.json()["task_id"]
    task = made.json()
    for _ in range(50):
        task = c.get(f"/api/tasks/{task_id}", headers=_gw(c)).json()
        if task["status"] != "planning":
            break
        time.sleep(0.1)
    stored = task["original_context"]["app"]
    assert "\n" not in stored and len(stored) <= 40 and stored.startswith("Sentient Security Center")
    assert task["status"] == "approval_pending"
    ready = []
    for _ in range(50):
        notes = c.get("/api/notifications", headers=_gw(c)).json()["notifications"]
        ready = [n for n in notes if n.get("title") == "Plan ready for approval"]
        if ready:
            break
        time.sleep(0.1)
    assert ready and 'an app called ` Sentient Security Center' in ready[0]["message"]
    assert "tap Allow" not in ready[0]["message"]  # the instruction tail is cut off by the cap


def test_markdown_in_an_app_name_is_neutralised_in_the_notes():
    # the approval and plan-ready notes are rendered as Markdown in the app: a link-shaped
    # name must not arrive as a live link with bold text above Allow and Deny. Backslash
    # escaping cannot stop GFM autolinks (review 5641825878), so the name goes in an
    # inline code span, whose contents Markdown never parses
    from sentient.app_names import code_span, framed_app_name

    hostile = "[Fix it](https://evil.co) **NOW**"
    assert framed_app_name(hostile) == "An app called ` [Fix it](https://evil.co) **NOW** `"
    assert framed_app_name(hostile, capital=False) == "an app called ` [Fix it](https://evil.co) **NOW** `"
    # the fence grows past any backticks in the name, so the span never breaks early
    assert framed_app_name("a `b` c") == "An app called `` a `b` c ``"
    assert framed_app_name("`x") == "An app called `` `x ``"
    # bare URLs and emails ride inside the span unparsed, not as links
    assert framed_app_name("see https://evil.co") == "An app called ` see https://evil.co `"
    assert framed_app_name("support@evil.co") == "An app called ` support@evil.co `"
    assert code_span("") == ""
    assert framed_app_name("Claude Code") == "An app called ` Claude Code `"
    assert framed_app_name(None) == "An MCP app"
    assert framed_app_name("an MCP app") == "An MCP app"
    assert framed_app_name("An MCP app", capital=False) == "an MCP app"


async def test_an_approval_note_does_not_render_a_link_shaped_app_name(config, isolated_home):
    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"snippets": "allow"}
    ran: list[str] = []
    llm = FakeProvider(replies=[[tool_call("run_snippet", code="print(9)")], "done"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "mdnote.db").start()
    try:
        s.registry.register(_snippet_plugin(ran))
        sid = await s.store.create_session(channel="mcp")

        async def answer(ev):
            if isinstance(ev, ApprovalRequest):
                notes = [n for n in await s.notifications.list() if n["kind"] == "approval"]
                assert notes, "the approval must also land as a notification"
                message = notes[0]["message"]
                # the name sits in one code span, verbatim: nothing inside it is parsed,
                # and no escape backslashes leak into the rendered text
                assert message.startswith('An app called ` [Fix it](https://evil.co) **NOW** ` wants to: '), message
                assert "\\[" not in message and "\\*" not in message
                s.approvals.resolve(ev.approval_id, "allow")

        events = []
        async for ev in s.agent.run_turn(sid, "Run this Python code: print(9)", channel="mcp",
                                         app_name="[Fix it](https://evil.co) **NOW**"):
            events.append(ev)
            await answer(ev)
        assert ran == ["print(9)"]
    finally:
        await s.stop()


def test_a_plan_ready_note_does_not_render_a_link_shaped_app_name(engine_client):
    # the plan-ready note is rendered as Markdown too, so it shares the same code span;
    # the plan's title is the app's prompt word for word, so it is code-spanned as well
    c = engine_client
    made = c.post("/api/tasks", json={"prompt": "Tidy the downloads folder",
                                      "app": "[Fix it](https://evil.co) **NOW**"},
                  headers=_mcp(c))
    assert made.status_code == 200, made.text
    task_id = made.json()["task_id"]
    task = made.json()
    for _ in range(50):
        task = c.get(f"/api/tasks/{task_id}", headers=_gw(c)).json()
        if task["status"] != "planning":
            break
        time.sleep(0.1)
    assert task["status"] == "approval_pending"
    ready = []
    for _ in range(50):
        notes = c.get("/api/notifications", headers=_gw(c)).json()["notifications"]
        ready = [n for n in notes if n.get("title") == "Plan ready for approval"]
        if ready:
            break
        time.sleep(0.1)
    assert ready, "the plan-ready note must exist"
    message = ready[0]["message"]
    assert 'an app called ` [Fix it](https://evil.co) **NOW** `' in message, message
    assert '` Tidy the downloads folder `' in message, message
    assert "\\[" not in message and "\\*" not in message
