"""Rules from chat (#130): "never delete my emails" becomes a proposed rule that only the user's click creates."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sentient.agent.chat_rules import (
    candidate_tools,
    narrow_app_keys,
    parse_detection,
    standing_text,
)
from sentient.app import SentientApp
from sentient.config.loader import load_config
from sentient.config.schema import SentientConfig
from sentient.gateway.app import create_app
from sentient.llm.events import ApprovalRequest, ToolCallEvent, ToolResultEvent
from sentient.llm.provider import ProviderError
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call

NEVER_DELETE = {"keys": ["mail_delete_email"], "rule": "never"}


def _mail(log: list[str]) -> ToolPlugin:
    @tool("mail_delete_email", risk=Risk.send)
    async def mail_delete_email(ctx: ToolContext, message_id: str) -> dict:
        """Delete an email for good."""
        log.append(f"delete:{message_id}")
        return {"deleted": message_id}

    @tool("mail_search", risk=Risk.read)
    async def mail_search(ctx: ToolContext, query: str) -> dict:
        """Search emails."""
        log.append(f"search:{query}")
        return {"found": []}

    class Mail(ToolPlugin):
        id = "mail"
        display_name = "Mail"
        tools = [mail_delete_email, mail_search]

    return Mail()


async def _start(config, isolated_home, llm, log: list[str], name: str = "rules") -> SentientApp:
    s = await SentientApp(config, llm=llm, db_path=isolated_home / f"{name}.db", enable_background=False).start()
    s.registry.register(_mail(log))
    return s


async def _turn(s: SentientApp, sid: str, text: str, decision: str = "deny") -> tuple[list[ApprovalRequest], list]:
    asked: list[ApprovalRequest] = []
    events = []
    async for ev in s.agent.run_turn(sid, text, channel="cli"):
        events.append(ev)
        if isinstance(ev, ApprovalRequest):
            asked.append(ev)
            s.approvals.resolve(ev.approval_id, decision)
    return asked, events


def _json_calls(llm: FakeProvider) -> list[dict]:
    return [c for c in llm.calls if c.get("json")]


def _offered(llm: FakeProvider) -> set[str]:
    stream_calls = [c for c in llm.calls if not c.get("json")]
    return {t["function"]["name"] for t in (stream_calls[-1]["tools"] or [])}


# ---------------------------------------------------------------------------- detection
def test_prefilter_keeps_standing_instructions_only():
    assert standing_text("never delete my emails") == "never delete my emails"
    assert standing_text("Thanks! Don\u2019t post to Slack without asking me.") == "Don\u2019t post to Slack without asking me."
    assert standing_text("always ask before sending money")
    assert standing_text("What's the weather like?") == ""
    # asking for fewer questions is never turned into a rule
    assert standing_text("Don't ask me before archiving, just do it") == ""
    assert standing_text("Stop asking me about Gmail.") == ""


async def test_never_delete_my_emails_proposes_never(config, isolated_home):
    log: list[str] = []
    llm = FakeProvider(replies=["Got it."], json_replies=[NEVER_DELETE])
    s = await _start(config, isolated_home, llm, log)
    try:
        sid = await s.store.create_session(channel="cli")
        async with s.bus.subscribe() as q:
            await _turn(s, sid, "Please never delete my emails.")
            events = [q.get_nowait() for _ in range(q.qsize())]
        [proposal] = await s.chat_rules.list(sid)
        assert proposal["rule"] == "never" and proposal["keys"] == ["mail_delete_email"]
        assert proposal["targets"] == [
            {"key": "mail_delete_email", "app": "Mail", "tool": "Delete email", "label": "Mail > Delete email"}
        ]
        assert proposal["said"] == "Please never delete my emails." and proposal["status"] == "pending"
        assert any(e["type"] == "rule_proposal.updated" and e["data"]["id"] == proposal["id"] for e in events)
        # one short fast-role prompt that lists the matching tools
        [check] = _json_calls(llm)
        assert check["role"] == "fast" and "mail_delete_email" in check["messages"][1]["content"]
        # a proposal is not a rule
        assert s.config.tools.approvals.rules == {}
    finally:
        await s.stop()


async def test_gmail_trash_is_a_candidate_for_deleting_emails(config, isolated_home):
    llm = FakeProvider(replies=["ok"], json_replies=[{"keys": ["gmail_trash"], "rule": "never"}])
    s = await _start(config, isolated_home, llm, [])
    try:
        names = s.chat_rules._app_names()
        candidates = [t.name for t in candidate_tools("never delete my emails", s.registry.tools(include_hidden=True), names)]
        assert "gmail_trash" in candidates
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, "never delete my emails")
        [proposal] = await s.chat_rules.list(sid)
        assert proposal["targets"][0]["label"] == "Gmail > Trash"
    finally:
        await s.stop()


@pytest.mark.parametrize("text", ["I never eat breakfast.", "What's on my calendar today?", "Don't ask me before deleting emails."])
async def test_unrelated_sentences_make_no_card_and_no_model_call(config, isolated_home, text):
    llm = FakeProvider(replies=["ok"], json_replies=[NEVER_DELETE])
    s = await _start(config, isolated_home, llm, [])
    try:
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, text)
        assert await s.chat_rules.list(sid, None) == []
        assert _json_calls(llm) == []
    finally:
        await s.stop()


@pytest.mark.parametrize(
    "reply",
    [
        "Sure! I think you mean emails.",
        {"keys": ["rm_rf_everything", "config"]},
        {"keys": []},
        {"rule": "never"},
        [42, None],
        None,
        {"keys": {"mail_delete_email": "never"}},
    ],
)
async def test_invalid_model_output_makes_no_card(config, isolated_home, reply):
    llm = FakeProvider(replies=["ok"], json_replies=[reply])
    s = await _start(config, isolated_home, llm, [])
    try:
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, "never delete my emails")
        assert await s.chat_rules.list(sid, None) == []
        assert s.config.tools.approvals.rules == {}
    finally:
        await s.stop()


async def test_a_failing_model_makes_no_card_and_the_turn_goes_on(config, isolated_home):
    class Broken(FakeProvider):
        async def complete_json(self, role, messages, *, model=None):
            raise ProviderError("model went away")

    llm = Broken(replies=["Fine."])
    s = await _start(config, isolated_home, llm, [])
    try:
        sid = await s.store.create_session(channel="cli")
        _, events = await _turn(s, sid, "never delete my emails")
        assert events[-1].type == "done" and events[-1].content == "Fine."
        assert await s.chat_rules.list(sid, None) == []
    finally:
        await s.stop()


def test_parse_detection_checks_keys_in_code():
    log: list[str] = []
    tools = _mail(log).tools
    for t in tools:
        t.plugin = "mail"
    assert parse_detection({"keys": ["MAIL_DELETE_EMAIL", "nope"], "rule": "Never"}, tools) == (["mail_delete_email"], "never")
    # an app key covers its tools; a made-up level is dropped (the caller decides)
    assert parse_detection('{"keys": ["mail_delete_email", "mail"], "rule": "maybe"}', tools) == (["mail"], None)
    assert parse_detection("no json here", tools) == ([], None)


@pytest.mark.parametrize(
    ("text", "reply", "keys", "label"),
    [
        # names an action: only the matching tools, even when the model names the whole app
        ("never delete my emails", {"keys": ["mail"], "rule": "never"}, ["mail_delete_email"], "Mail > Delete email"),
        # names no action: the whole app
        ("never use Slack", {"keys": ["slack"], "rule": "never"}, ["slack"], "Slack"),
        ("don't touch my Notion", {"keys": ["notion"], "rule": "never"}, ["notion"], "Notion"),
    ],
)
async def test_whole_apps_only_when_no_action_is_named(config, isolated_home, text, reply, keys, label):
    llm = FakeProvider(replies=["ok"], json_replies=[reply])
    s = await _start(config, isolated_home, llm, [])
    try:
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, text)
        [proposal] = await s.chat_rules.list(sid)
        assert proposal["keys"] == keys and proposal["targets"][0]["label"] == label
    finally:
        await s.stop()


def test_narrow_app_keys_is_deterministic():
    tools = _mail([]).tools
    for t in tools:
        t.plugin = "mail"
    assert narrow_app_keys(["mail"], "never delete my emails", tools) == ["mail_delete_email"]
    assert narrow_app_keys(["mail"], "never use my mail", tools) == ["mail"]
    # an action no tool of that app has: the app key is dropped rather than kept
    assert narrow_app_keys(["mail"], "never post anything", tools) == []
    assert narrow_app_keys(["mail_search", "mail"], "don't archive my emails", tools) == ["mail_search"]
    # an object that could be a verb ("messages") is not a second action
    assert narrow_app_keys(["mail"], "never delete my email messages", tools) == ["mail_delete_email"]


@pytest.mark.parametrize(
    ("text", "reply", "keys"),
    [
        # "messages" is what gets deleted, not a send instruction: sending stays allowed
        ("never delete my email messages", {"keys": ["gmail"], "rule": "never"}, ["gmail_trash"]),
        # each app is narrowed by its own instruction
        ("never delete my emails. never use Slack", {"keys": ["gmail", "slack"], "rule": "never"},
         ["gmail_trash", "slack"]),
        ("never delete my emails and never use Slack", {"keys": ["slack", "gmail"], "rule": "never"},
         ["slack", "gmail_trash"]),
    ],
)
async def test_each_app_is_narrowed_by_its_own_instruction(config, isolated_home, text, reply, keys):
    llm = FakeProvider(replies=["ok"], json_replies=[reply])
    s = await _start(config, isolated_home, llm, [])
    try:
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, text)
        [proposal] = await s.chat_rules.list(sid)
        assert proposal["keys"] == keys
    finally:
        await s.stop()


async def test_without_asking_makes_an_ask_rule(config, isolated_home):
    llm = FakeProvider(replies=["ok"], json_replies=[NEVER_DELETE])
    s = await _start(config, isolated_home, llm, [])
    try:
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, "Don't delete my emails without asking me first.")
        [proposal] = await s.chat_rules.list(sid)
        assert proposal["rule"] == "ask"
    finally:
        await s.stop()


async def test_no_card_when_a_rule_already_covers_it(config, isolated_home):
    config.tools.approvals.rules = {"mail": "never"}
    llm = FakeProvider(replies=["ok", "ok"], json_replies=[NEVER_DELETE, {"keys": ["mail_delete_email"], "rule": "ask"}])
    s = await _start(config, isolated_home, llm, [])
    try:
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, "never delete my emails")
        await _turn(s, sid, "always ask before you delete emails")
        assert await s.chat_rules.list(sid, None) == []
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- the user's decision
async def test_accept_creates_the_rule_and_hides_the_tool_next_turn(config, isolated_home):
    log: list[str] = []
    llm = FakeProvider(
        replies=["Got it.", "Searching.", [tool_call("mail_delete_email", message_id="m1")], "Done."],
        json_replies=[NEVER_DELETE],
    )
    s = await _start(config, isolated_home, llm, log)
    try:
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, "never delete my emails")
        [proposal] = await s.chat_rules.list(sid)
        decided = await s.chat_rules.decide(proposal["id"], "accept")
        assert decided["status"] == "accepted" and decided["decided_at"]
        approvals = s.config.tools.approvals
        assert approvals.rules == {"mail_delete_email": "never"}
        origin = approvals.rule_origins["mail_delete_email"]
        assert origin.rule == "never" and origin.said == "never delete my emails" and origin.session_id == sid
        # saved to the config file, so it survives a restart
        saved = load_config().tools.approvals
        assert saved.rules == {"mail_delete_email": "never"} and "mail_delete_email" in saved.rule_origins

        await _turn(s, sid, "find my receipts")
        assert "mail_delete_email" not in _offered(llm) and "mail_search" in _offered(llm)
        _, events = await _turn(s, sid, "clean up my inbox")
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error and "never use" in res.result["error"] and log == []
        assert await s.chat_rules.list(sid) == []  # nothing pending any more
        with pytest.raises(ValueError):
            await s.chat_rules.decide(proposal["id"], "decline")
    finally:
        await s.stop()


async def test_decline_creates_nothing(config, isolated_home):
    log: list[str] = []
    llm = FakeProvider(replies=["Got it.", [tool_call("mail_delete_email", message_id="m1")], "Deleted."], json_replies=[NEVER_DELETE])
    s = await _start(config, isolated_home, llm, log)
    try:
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, "never delete my emails")
        [proposal] = await s.chat_rules.list(sid)
        assert (await s.chat_rules.decide(proposal["id"], "decline"))["status"] == "declined"
        assert s.config.tools.approvals.rules == {} and s.config.tools.approvals.rule_origins == {}
        # declined: the chat no longer asks because of it (approvals are off in this test)
        asked, _ = await _turn(s, sid, "delete m1")
        assert asked == [] and log == ["delete:m1"]
    finally:
        await s.stop()


async def test_until_decided_the_tool_asks_even_under_an_allow_rule(config, isolated_home):
    """The model calls the tool in the same reply: the pending proposal already makes it ask, every time, for the
    rest of that chat (also after a restart), and only in that chat."""
    log: list[str] = []
    config.tools.approvals.rules = {"mail": "allow"}
    llm = FakeProvider(
        replies=[
            [tool_call("mail_delete_email", message_id="m1")], "Not deleted.",
            [tool_call("mail_delete_email", message_id="m2")], "Not deleted.",
            [tool_call("mail_delete_email", message_id="m3")], "Deleted.",
        ],
        json_replies=[NEVER_DELETE],
    )
    s = await _start(config, isolated_home, llm, log)
    try:
        sid = await s.store.create_session(channel="cli")
        asked, _ = await _turn(s, sid, "never delete my emails, and tidy up my inbox", decision="allow_session")
        assert [a.name for a in asked] == ["mail_delete_email"]
        # "Allow for this chat" does not cover it either
        asked, _ = await _turn(s, sid, "delete m2", decision="deny")
        assert [a.name for a in asked] == ["mail_delete_email"]
        other = await s.store.create_session(channel="cli")
        asked, _ = await _turn(s, other, "delete m3")
        assert asked == [] and log == ["delete:m1", "delete:m3"]
    finally:
        await s.stop()

    llm2 = FakeProvider(replies=[[tool_call("mail_delete_email", message_id="m4")], "Not deleted."])
    s = await _start(config, isolated_home, llm2, log)  # same database: the pending proposal is still there
    try:
        asked, _ = await _turn(s, sid, "delete m4")
        assert [a.name for a in asked] == ["mail_delete_email"] and log == ["delete:m1", "delete:m3"]
    finally:
        await s.stop()


async def test_a_never_rule_saved_during_the_chat_lookup_is_refused_not_asked(config, isolated_home, monkeypatch):
    log: list[str] = []
    llm = FakeProvider(replies=[[tool_call("mail_delete_email", message_id="m1")], "Could not."])
    s = await _start(config, isolated_home, llm, log)
    try:
        sid = await s.store.create_session(channel="cli")

        async def pending_while_never_is_saved(tool, session_id):
            s.approvals.config.rules = {"mail_delete_email": "never"}  # Settings saved Never meanwhile
            return "ask"

        monkeypatch.setattr(s.chat_rules, "chat_rule", pending_while_never_is_saved)
        asked, events = await _turn(s, sid, "delete m1", decision="allow")
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert asked == [] and "never use" in res.result["error"] and log == []
    finally:
        await s.stop()


async def test_a_slow_check_fails_closed_until_it_finishes(config, isolated_home, monkeypatch):
    """The reply stops waiting for a slow rule check; the tools that check is weighing still ask meanwhile."""
    import asyncio

    import sentient.agent.loop as loop_mod

    monkeypatch.setattr(loop_mod, "RULE_CHECK_WAIT_S", 0.05)
    release = asyncio.Event()

    class Slow(FakeProvider):
        async def complete_json(self, role, messages, *, model=None):
            await release.wait()
            return NEVER_DELETE

    log: list[str] = []
    config.tools.approvals.rules = {"mail": "allow"}
    llm = Slow(replies=[[tool_call("mail_delete_email", message_id="m1")], "Not deleted."])
    s = await _start(config, isolated_home, llm, log)
    try:
        sid = await s.store.create_session(channel="cli")
        asked, _ = await _turn(s, sid, "never delete my emails, and tidy up my inbox", decision="deny")
        assert [a.name for a in asked] == ["mail_delete_email"] and log == []
        release.set()
        await s.agent.drain()
        [proposal] = await s.chat_rules.list(sid)  # the proposal still arrives
        assert proposal["keys"] == ["mail_delete_email"] and s.chat_rules._checking == {}
    finally:
        await s.stop()


async def test_a_steer_message_is_checked_before_the_next_round(config, isolated_home):
    log: list[str] = []
    llm = FakeProvider(
        replies=[[tool_call("mail_search", query="old")], [tool_call("mail_delete_email", message_id="m1")], "Kept it."],
        json_replies=[NEVER_DELETE],
    )
    s = await _start(config, isolated_home, llm, log)
    try:
        sid = await s.store.create_session(channel="cli")
        asked: list[ApprovalRequest] = []
        async for ev in s.agent.run_turn(sid, "tidy up my inbox", channel="cli"):
            if isinstance(ev, ToolCallEvent) and ev.name == "mail_search":
                assert s.agent.steer(sid, "Actually, never delete my emails.")
            if isinstance(ev, ApprovalRequest):
                asked.append(ev)
                s.approvals.resolve(ev.approval_id, "deny")
        assert [a.name for a in asked] == ["mail_delete_email"] and log == ["search:old"]
        [proposal] = await s.chat_rules.list(sid)
        assert proposal["said"] == "Actually, never delete my emails." and proposal["message_id"]
    finally:
        await s.stop()


async def test_a_long_summarized_chat_still_enforces_the_rule(config, isolated_home):
    log: list[str] = []
    config.chat.history_window = 4
    config.chat.compress_after_messages = 10
    filler = 12
    llm = FakeProvider(
        replies=["Got it.", *["ok"] * filler, [tool_call("mail_delete_email", message_id="m1")], "Could not."],
        json_replies=[NEVER_DELETE],
    )
    s = await _start(config, isolated_home, llm, log)
    try:
        sid = await s.store.create_session(channel="cli")
        await _turn(s, sid, "never delete my emails")
        [proposal] = await s.chat_rules.list(sid)
        await s.chat_rules.decide(proposal["id"], "accept")
        for i in range(filler):
            await _turn(s, sid, f"note number {i}")
        await s.agent.drain()
        session = await s.store.get_session(sid)
        assert session["context_summary"]  # the early messages were folded into a summary
        stream_calls = [c for c in llm.calls if not c.get("json")]
        assert all("never delete" not in json.dumps(c["messages"][1:]) for c in stream_calls[-1:])
        _, events = await _turn(s, sid, "clean up my inbox")
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error and "never use" in res.result["error"] and log == []
        assert "mail_delete_email" not in _offered(llm)
    finally:
        await s.stop()


async def test_the_model_cannot_create_a_rule_without_the_click(config, isolated_home):
    llm = FakeProvider(
        replies=[[tool_call("accept_rule_proposal", decision="accept"), tool_call("config_set", key="rules")], "Done, it's a rule now."],
        json_replies=[NEVER_DELETE],
    )
    s = await _start(config, isolated_home, llm, [])
    try:
        sid = await s.store.create_session(channel="cli")
        _, events = await _turn(s, sid, "never delete my emails")
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert all(r.is_error and "unknown tool" in r.result["error"] for r in results)
        [proposal] = await s.chat_rules.list(sid)
        assert proposal["status"] == "pending"
        assert s.config.tools.approvals.rules == {} and load_config().tools.approvals.rules == {}
        assert not any("rule" in t.name and "propos" in t.name for t in s.registry.tools(include_hidden=True))
    finally:
        await s.stop()


def test_origin_note_goes_away_when_the_rule_changes():
    data = {"tools": {"approvals": {
        "rules": {"mail_delete_email": "never", "mail": "ask"},
        "rule_origins": {"mail_delete_email": {"rule": "never", "said": "never delete my emails", "at": "2026-10-10T09:00:00+00:00"},
                         "mail": {"rule": "never", "said": "never touch my mail"}},
    }}}
    cfg = SentientConfig.model_validate(data)
    assert list(cfg.tools.approvals.rule_origins) == ["mail_delete_email"]  # "mail" is now Ask, not Never
    data["tools"]["approvals"]["rules"] = {}
    assert SentientConfig.model_validate(data).tools.approvals.rule_origins == {}


# ---------------------------------------------------------------------------- REST
@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider(replies=["Got it."], json_replies=[NEVER_DELETE])
    core = SentientApp(config, llm=llm, db_path=isolated_home / "r.db", enable_background=False)
    app = create_app(core)
    with TestClient(app) as c:
        core.registry.register(_mail([]))
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def test_rest_lists_and_decides_proposals(client):
    lines = client.post("/api/chat", json={"text": "never delete my emails"}).text.splitlines()
    sid = json.loads(lines[0])["session_id"]
    [proposal] = client.get(f"/api/sessions/{sid}/rule-proposals").json()
    assert proposal["targets"][0]["label"] == "Mail > Delete email"
    assert client.post(f"/api/rule-proposals/{proposal['id']}", json={"decision": "maybe"}).status_code == 422
    done = client.post(f"/api/rule-proposals/{proposal['id']}", json={"decision": "accept"}).json()
    assert done["status"] == "accepted"
    approvals = client.get("/api/config").json()["tools"]["approvals"]
    assert approvals["rules"] == {"mail_delete_email": "never"}
    assert approvals["rule_origins"]["mail_delete_email"]["said"] == "never delete my emails"
    assert client.post(f"/api/rule-proposals/{proposal['id']}", json={"decision": "decline"}).status_code == 409
    assert client.post("/api/rule-proposals/nope", json={"decision": "accept"}).status_code == 404
    assert client.get(f"/api/sessions/{sid}/rule-proposals").json() == []
    assert len(client.get(f"/api/sessions/{sid}/rule-proposals", params={"status": "all"}).json()) == 1
    # changing the rule in Settings drops the origin note
    patched = client.patch("/api/config", json={"tools": {"approvals": {"rules": {"mail_delete_email": "ask"}}}}).json()
    assert patched["config"]["tools"]["approvals"]["rule_origins"] == {}
