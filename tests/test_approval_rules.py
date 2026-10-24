"""Lasting approval rules per tool or app: allow, ask, never (ADR 0016)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sentient.agent.loop import LoopResult
from sentient.app import SentientApp
from sentient.config.loader import load_config
from sentient.config.schema import SentientConfig
from sentient.gateway.app import create_app
from sentient.llm.events import ApprovalRequest, ToolResultEvent
from sentient.tasks.executor import select_tools
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.rules import pending_card_message
from tests.conftest import FakeProvider, tool_call


def _postcards(log: list[str]) -> ToolPlugin:
    @tool("send_postcard", risk=Risk.send)
    async def send_postcard(ctx: ToolContext, text: str) -> dict:
        """Post a postcard to someone outside Sentient."""
        log.append(f"send:{text}")
        return {"posted": text}

    @tool("find_postcard", risk=Risk.read)
    async def find_postcard(ctx: ToolContext, query: str) -> dict:
        """Look up a postcard."""
        log.append(f"find:{query}")
        return {"found": query}

    class Postcards(ToolPlugin):
        id = "postcards"
        display_name = "Postcards"
        tools = [send_postcard, find_postcard]

    return Postcards()


def _shop(log: list[str]) -> ToolPlugin:
    """A browser-like click tool: risk_fn raises buttons that buy or post to send, describe_fn names them."""

    def risk(arguments, ctx):
        label = str(arguments.get("label", "")).lower()
        return Risk.send if ("order" in label or "post" in label) else None

    def describe(arguments, ctx):
        label = str(arguments.get("label", ""))
        kind = "Purchase" if "order" in label.lower() else "Posts publicly" if "post" in label.lower() else "Clicks"
        return {"risk_label": kind, "target": label}

    @tool("shop_click", risk=Risk.write, risk_fn=risk, describe_fn=describe)
    async def shop_click(ctx: ToolContext, label: str) -> dict:
        """Click a button on the shop page."""
        log.append(label)
        return {"clicked": label}

    class Shop(ToolPlugin):
        id = "shop"
        display_name = "Shop"
        tools = [shop_click]

    return Shop()


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
            assert s.approvals.resolve(ev.approval_id, decision)
    return asked, events


def _offered(llm: FakeProvider, call: int = 0) -> set[str]:
    return {t["function"]["name"] for t in (llm.calls[call]["tools"] or [])}


# ---------------------------------------------------------------------------- allow
async def test_allow_runs_without_asking_even_in_always_mode(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "always"
    config.tools.approvals.rules = {"send_postcard": "allow"}
    llm = FakeProvider(replies=[[tool_call("send_postcard", text="hi")], "sent"])
    s = await _start(config, isolated_home, llm, "allow", _postcards(log))
    try:
        sid = await s.store.create_session(channel="cli")
        asked, _ = await _turn(s, sid, "send it", decision="deny")
        assert asked == [] and log == ["send:hi"]
        # without the rule, mode "always" asks for the read tool of the same app
        assert s.approvals.needs_approval(s.registry.get("find_postcard"), sid)
    finally:
        await s.stop()


async def test_allow_still_asks_for_purchases(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"shop": "allow"}
    llm = FakeProvider(replies=[
        [tool_call("shop_click", label="Next")], "ok1",
        [tool_call("shop_click", label="Post review")], "ok2",
        [tool_call("shop_click", label="Place order")], "ok3",
        [tool_call("shop_click", label="Place order")], "ok4",
        [tool_call("shop_click", label="Place order")], "ok5",
    ])
    s = await _start(config, isolated_home, llm, "purchase", _shop(log))
    try:
        sid = await s.store.create_session(channel="cli")
        assert (await _turn(s, sid, "next"))[0] == []
        assert (await _turn(s, sid, "post"))[0] == []  # a send that is not a purchase: allowed by the rule
        asked, events = await _turn(s, sid, "buy", decision="deny")
        assert [a.risk_label for a in asked] == ["Purchase"]
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error and log == ["Next", "Post review"]
        # "Allow for this chat" (or any wider answer) is never offered for a purchase; one sent anyway is "once" (#139)
        async for ev in s.agent.run_turn(sid, "buy", channel="cli"):
            if isinstance(ev, ApprovalRequest):
                assert ev.choices == ["allow", "deny"]
                assert s.approvals.resolve(ev.approval_id, "allow_session")  # counts as "Allow once"
        assert len((await _turn(s, sid, "buy again", decision="allow"))[0]) == 1
        assert log == ["Next", "Post review", "Place order", "Place order"]
    finally:
        await s.stop()


async def test_purchases_ask_even_with_approvals_off(config, isolated_home):
    """Switching approvals off quiets everyday questions, never spending money (browser.confirm_purchases is
    the one switch for that); an allow rule does not change it either."""
    s = await _start(config, isolated_home, FakeProvider(), "purchase_off", _shop([]))
    try:
        click = s.registry.get("shop_click")
        s.approvals.config.mode = "off"
        assert await s.approvals.decide(click, "s1", Risk.send, {"label": "Place order"}, None)
        assert not await s.approvals.decide(click, "s1", Risk.send, {"label": "Post review"}, None)
        s.approvals.config.rules = {"shop_click": "allow"}
        assert await s.approvals.decide(click, "s1", Risk.send, {"label": "Place order"}, None)
        s.approvals.config.mode = "ask"
        assert await s.approvals.decide(click, "s1", Risk.send, {"label": "Place order"}, None)
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- ask
async def test_ask_overrides_approvals_off_and_read_only_tools(config, isolated_home):
    log: list[str] = []
    assert config.tools.approvals.mode == "off"
    config.tools.approvals.rules = {"find_postcard": "ask"}
    llm = FakeProvider(replies=[[tool_call("find_postcard", query="paris")], "declined"])
    s = await _start(config, isolated_home, llm, "ask_off", _postcards(log))
    try:
        sid = await s.store.create_session(channel="cli")
        asked, events = await _turn(s, sid, "find it", decision="deny")
        assert [a.name for a in asked] == ["find_postcard"] and asked[0].risk == "read"
        assert next(e for e in events if isinstance(e, ToolResultEvent)).is_error
        assert log == []
        # the app's other tool has no rule and follows mode "off"
        assert not s.approvals.needs_approval(s.registry.get("send_postcard"), sid)
    finally:
        await s.stop()


async def test_ask_overrides_allow_for_this_chat(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"postcards": "ask"}
    llm = FakeProvider(replies=[
        [tool_call("send_postcard", text="a")], "ok1",
        [tool_call("send_postcard", text="b")], "ok2",
    ])
    s = await _start(config, isolated_home, llm, "ask_session", _postcards(log))
    try:
        sid = await s.store.create_session(channel="cli")
        first, _ = await _turn(s, sid, "send a", decision="allow_session")
        second, _ = await _turn(s, sid, "send b", decision="allow")
        assert len(first) == 1 and len(second) == 1
        assert log == ["send:a", "send:b"]
    finally:
        await s.stop()


async def test_ask_in_unattended_runs_is_refused(config, isolated_home):
    """Task runs and other loops without approvals cannot stop to ask, so an "ask" tool does not run."""
    log: list[str] = []
    config.tools.approvals.rules = {"postcards": "ask"}
    llm = FakeProvider(replies=[[tool_call("send_postcard", text="x")], "done"])
    s = await _start(config, isolated_home, llm, "ask_task", _postcards(log))
    try:
        ctx = s.agent.tool_context(None, "task")
        messages = [{"role": "system", "content": "run"}, {"role": "user", "content": "go"}]
        result = LoopResult()
        events = [e async for e in s.agent.run_loop(messages, ctx, result=result, use_approvals=False, source="task")]
        assert not any(isinstance(e, ApprovalRequest) for e in events)
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        stopped = "Postcards is set to Ask, and this work can't stop to ask. Change it in Settings > Approvals & safety."
        assert res.is_error and res.result["error"] == stopped
        assert result.stopped_by_rule == result.error == stopped  # the run ends at the refused call
        assert len(llm.calls) == 1 and log == []
        # subagents refuse it too, instead of running it unasked
        refusal = s.subagents.policy("s1")(s.registry.get("find_postcard"), Risk.read, {})
        assert refusal and "approval" in refusal
    finally:
        await s.stop()


async def test_pending_rule_guess_does_not_stop_a_helper_file_write(config, isolated_home):
    """#491: on a fresh home, an undecided Ask rule proposal for file_write in the chat must not make a helper's
    file_write fail with "can't stop to ask". A pending guess only ever adds a question; the helper has nobody
    to ask, so an internal write goes ahead. (A rule the user actually set still stops it: see
    test_ask_in_unattended_runs_is_refused. A pending Never card still stops it: see the next test.)"""
    llm = FakeProvider(
        replies=[[tool_call("file_write", name="note.md", content="hi")], "done"],
        json_replies=[{"keys": ["file_write"], "rule": "ask"}],
    )
    s = await _start(config, isolated_home, llm, "guess_helper")
    try:
        sid = await s.store.create_session(channel="cli")
        # the chat said something that proposed an Ask rule for file_write; the user hasn't decided yet
        await s.chat_rules.check(sid, None, "always ask me before writing files")
        assert [p["keys"] for p in await s.chat_rules.list(sid)] == [["file_write"]]
        # the helper runs the way SubagentManager._run does: no approvals UI, the subagent policy, can't pause
        ctx = s.agent.tool_context(sid, "subagent", origin="user")
        ctx.extra["subagent_id"] = "sub1"
        messages = [{"role": "system", "content": "run"}, {"role": "user", "content": "Goal: write a short file"}]
        result = LoopResult()
        events = [
            e async for e in s.agent.run_loop(
                messages, ctx, result=result, role="fast", tool_names=["file_write"],
                use_approvals=False, policy=s.subagents.policy(sid), source="subagent",
            )
        ]
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert not res.is_error, res.result
        assert res.result["saved"] == "note.md"
        assert not result.stopped_by_rule and result.error is None
    finally:
        await s.stop()


async def test_pending_never_card_stops_a_helper_file_write(config, isolated_home):
    """#491 review: an undecided Never rule proposal for file_write still stops a helper's file_write, exactly as
    main does, and the message names the suggested-rule card waiting for the user's answer in the chat instead of
    claiming something is set to Ask in Settings."""
    llm = FakeProvider(
        replies=[[tool_call("file_write", name="note.md", content="hi")], "done"],
        json_replies=[{"keys": ["file_write"], "rule": "never"}],
    )
    s = await _start(config, isolated_home, llm, "never_guess_helper")
    try:
        sid = await s.store.create_session(channel="cli")
        # the chat said something that proposed a Never rule for file_write; the user hasn't decided yet
        await s.chat_rules.check(sid, None, "do not write any files yourself")
        assert [p["keys"] for p in await s.chat_rules.list(sid)] == [["file_write"]]
        # the helper runs the way SubagentManager._run does: no approvals UI, the subagent policy, can't pause
        ctx = s.agent.tool_context(sid, "subagent", origin="user")
        ctx.extra["subagent_id"] = "sub1"
        messages = [{"role": "system", "content": "run"}, {"role": "user", "content": "Goal: write a short file"}]
        result = LoopResult()
        events = [
            e async for e in s.agent.run_loop(
                messages, ctx, result=result, role="fast", tool_names=["file_write"],
                use_approvals=False, policy=s.subagents.policy(sid), source="subagent",
            )
        ]
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        tool = s.registry.get("file_write")
        stopped = pending_card_message(s.agent.approvals.label(tool, s.agent.registry))
        assert "Change it in Settings" not in stopped  # not the old "set to Ask" wording
        assert res.is_error and res.result["error"] == stopped
        assert result.stopped_by_rule == result.error == stopped  # the run ends at the refused call
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- never
async def test_never_hides_the_tool_and_refuses_a_call(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.rules = {"send_postcard": "never"}
    llm = FakeProvider(replies=[[tool_call("send_postcard", text="x")], "ok"])
    s = await _start(config, isolated_home, llm, "never", _postcards(log))
    try:
        sid = await s.store.create_session(channel="cli")
        asked, events = await _turn(s, sid, "send it")
        offered = _offered(llm)
        assert "send_postcard" not in offered and "find_postcard" in offered
        assert asked == [] and log == []
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error
        assert res.result["error"] == (
            "You've set Sentient to never use \"Send postcard\" in Postcards. Change this in Settings > Approvals & safety."
        )
        # planners and task runs do not see it either; the Settings catalog still lists it
        assert "send_postcard" not in {t.name for t in s.registry.tools()}
        planner = next(p for p in s.registry.catalog() if p["id"] == "postcards")
        assert [t["name"] for t in planner["tools"]] == ["find_postcard"]
        settings = next(p for p in s.registry.catalog(include_blocked=True) if p["id"] == "postcards")
        assert {t["name"] for t in settings["tools"]} == {"send_postcard", "find_postcard"}
        names, _, _ = select_tools(s.registry, ["postcards"], include_core=False)
        assert names == ["find_postcard"]
    finally:
        await s.stop()


async def test_never_applies_to_every_tool_of_an_app_and_in_task_runs(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.rules = {"postcards": "never"}
    llm = FakeProvider(replies=[[tool_call("find_postcard", query="x")], "done"])
    s = await _start(config, isolated_home, llm, "never_app", _postcards(log))
    try:
        ctx = s.agent.tool_context(None, "task")
        messages = [{"role": "user", "content": "go"}]
        result = LoopResult()
        events = [
            e async for e in s.agent.run_loop(
                messages, ctx, result=result, tool_names=["find_postcard"], use_approvals=False, source="task"
            )
        ]
        assert llm.calls[0]["tools"] is None  # nothing left to offer
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.result["error"].startswith("You've set Sentient to never use Postcards.")
        assert log == []
    finally:
        await s.stop()


async def test_never_set_while_an_approval_waits_still_wins(config, isolated_home):
    """The rule is checked again right before the tool runs, so approving an old request cannot bypass it."""
    log: list[str] = []
    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[[tool_call("send_postcard", text="x")], "ok"])
    s = await _start(config, isolated_home, llm, "never_pending", _postcards(log))
    try:
        sid = await s.store.create_session(channel="cli")
        events = []
        async for ev in s.agent.run_turn(sid, "send it", channel="cli"):
            events.append(ev)
            if isinstance(ev, ApprovalRequest):
                s.approvals.config.rules = {"send_postcard": "never"}  # changed in Settings meanwhile
                s.approvals.resolve(ev.approval_id, "allow")
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error and res.result["error"].startswith("You've set Sentient to never use")
        assert log == []
    finally:
        await s.stop()


async def test_rule_changes_apply_without_restart(config, isolated_home):
    s = await _start(config, isolated_home, FakeProvider(), "live", _postcards([]))
    try:
        assert "send_postcard" in {t.name for t in s.registry.tools()}
        cfg = s.config.model_copy(deep=True)
        cfg.tools.approvals.rules = {"send_postcard": "never"}
        s.save_config(cfg)
        assert "send_postcard" not in {t.name for t in s.registry.tools()}
        assert load_config().tools.approvals.rules == {"send_postcard": "never"}  # persisted
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- precedence
async def test_tool_rule_beats_app_rule(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"postcards": "never", "send_postcard": "allow"}
    llm = FakeProvider(replies=[[tool_call("send_postcard", text="hi")], "sent"])
    s = await _start(config, isolated_home, llm, "precedence", _postcards(log))
    try:
        sid = await s.store.create_session(channel="cli")
        asked, _ = await _turn(s, sid, "send", decision="deny")
        assert asked == [] and log == ["send:hi"]
        assert _offered(llm) >= {"send_postcard"} and "find_postcard" not in _offered(llm)
        s.approvals.config.rules = {"postcards": "allow", "find_postcard": "ask"}
        assert s.approvals.needs_approval(s.registry.get("find_postcard"), sid)
        assert not s.approvals.needs_approval(s.registry.get("send_postcard"), sid)
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- config
def test_rules_config_validation():
    cfg = SentientConfig.model_validate({"tools": {"approvals": {"rules": {" gmail ": "Allow", "slack": "ask"}}}})
    assert cfg.tools.approvals.rules == {"gmail": "allow", "slack": "ask"}
    assert SentientConfig().tools.approvals.rules == {}
    with pytest.raises(ValidationError):
        SentientConfig.model_validate({"tools": {"approvals": {"rules": {"gmail": "sometimes"}}}})
    with pytest.raises(ValidationError):
        SentientConfig.model_validate({"tools": {"approvals": {"rules": {"  ": "allow"}}}})


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    app = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def test_patch_config_sets_and_removes_rules(client):
    rules = {"tools": {"approvals": {"rules": {"gmail_send_email": "ask", "slack": "never"}}}}
    cfg = client.patch("/api/config", json=rules).json()["config"]
    assert cfg["tools"]["approvals"]["rules"] == {"gmail_send_email": "ask", "slack": "never"}
    cfg = client.patch("/api/config", json={"tools": {"approvals": {"rules": {"slack": None}}}}).json()["config"]
    assert cfg["tools"]["approvals"]["rules"] == {"gmail_send_email": "ask"}
    bad = client.patch("/api/config", json={"tools": {"approvals": {"rules": {"gmail": "maybe"}}}})
    assert bad.status_code == 422
    assert client.get("/api/config").json()["tools"]["approvals"]["rules"] == {"gmail_send_email": "ask"}


def test_tools_catalog_keeps_never_tools_for_settings(client):
    client.patch("/api/config", json={"tools": {"approvals": {"rules": {"current_datetime": "never"}}}})
    names = {t["name"] for p in client.get("/api/tools").json() for t in p["tools"]}
    assert "current_datetime" in names


async def test_allow_rule_lets_a_free_signup_through_but_a_paid_subscribe_still_asks(config, isolated_home):
    """#378: "Signs up" follows the normal rules; a subscribe that shows a price stays a purchase and always asks."""
    from sentient.browser import safety
    from tests.browser.test_signup_label import PAID_BUTTON

    log: list[str] = []

    def describe(arguments, ctx):
        label = str(arguments.get("label", ""))
        return {"risk_label": safety.risk_label_for(label, "click"), "target": label}

    @tool("signup_click", risk=Risk.send, describe_fn=describe)
    async def signup_click(ctx: ToolContext, label: str) -> dict:
        """Click a button on the newsletter page."""
        log.append(label)
        return {"clicked": label}

    class Signup(ToolPlugin):
        id = "signup"
        display_name = "Signup"
        tools = [signup_click]

    config.tools.approvals.mode = "ask"
    config.tools.approvals.rules = {"signup": "allow"}
    llm = FakeProvider(replies=[
        [tool_call("signup_click", label="Subscribe to newsletter")], "ok1",
        [tool_call("signup_click", label=PAID_BUTTON)], "ok2",
    ])
    s = await _start(config, isolated_home, llm, "signup", Signup())
    try:
        sid = await s.store.create_session(channel="cli")
        assert (await _turn(s, sid, "subscribe me"))[0] == []
        asked, _ = await _turn(s, sid, "subscribe to pro", decision="deny")
        assert [a.risk_label for a in asked] == ["Purchase"] and asked[0].choices == ["allow", "deny"]
        assert log == ["Subscribe to newsletter"]
    finally:
        await s.stop()
