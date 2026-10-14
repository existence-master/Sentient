"""Rule levels "Only if I asked" and "Hand it to me" (#140, ADR 0016)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentient.agent import rule_levels
from sentient.agent.chat_rules import level_for, standing_text
from sentient.agent.loop import CAN_PAUSE_KEY, LoopResult
from sentient.app import SentientApp
from sentient.config.schema import SentientConfig
from sentient.llm.events import ApprovalRequest, ToolResultEvent
from sentient.llm.provider import ToolCall
from sentient.sandbox.policy import BridgePolicy, Refused
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call


def _mailer(log: list[str]) -> ToolPlugin:
    @tool("mailer_send_email", risk=Risk.send)
    async def mailer_send_email(ctx: ToolContext, to: str, subject: str, body: str) -> dict:
        """Send an email."""
        log.append(f"send:{to}")
        return {"sent": True}

    @tool("mailer_search", risk=Risk.read)
    async def mailer_search(ctx: ToolContext, query: str) -> dict:
        """Search emails."""
        log.append(f"search:{query}")
        return {"found": []}

    @tool("mailer_delete_email", risk=Risk.send)
    async def mailer_delete_email(ctx: ToolContext, message_id: str) -> dict:
        """Delete an email."""
        log.append(f"delete:{message_id}")
        return {"deleted": message_id}

    class Mailer(ToolPlugin):
        id = "mailer"
        display_name = "Mailer"
        tools = [mailer_send_email, mailer_search, mailer_delete_email]

    return Mailer()


def _shop(log: list[str]) -> ToolPlugin:
    def describe(arguments, ctx):
        return {"risk_label": "Purchase", "target": str(arguments.get("item"))}

    @tool("shop_buy", risk=Risk.send, describe_fn=describe)
    async def shop_buy(ctx: ToolContext, item: str) -> dict:
        """Buy something."""
        log.append(f"buy:{item}")
        return {"bought": item}

    class Shop(ToolPlugin):
        id = "shop"
        display_name = "Shop"
        tools = [shop_buy]

    return Shop()


SEND = {"to": "priya@example.com", "subject": "Notes", "body": "Hi Priya, here are the notes."}


async def _start(config, isolated_home, llm, name: str, *plugins: ToolPlugin) -> SentientApp:
    s = await SentientApp(config, llm=llm, db_path=isolated_home / f"{name}.db", enable_background=False).start()
    for p in plugins:
        s.registry.register(p)
    return s


async def _turn(s: SentientApp, sid: str, text: str, decision: str = "deny") -> tuple[list[ApprovalRequest], list]:
    asked: list[ApprovalRequest] = []
    events = []
    async for ev in s.agent.run_turn(sid, text, channel="desktop"):
        events.append(ev)
        if isinstance(ev, ApprovalRequest):
            asked.append(ev)
            assert s.approvals.resolve(ev.approval_id, decision)
    return asked, events


def _result(events: list, name: str) -> ToolResultEvent:
    return next(e for e in events if isinstance(e, ToolResultEvent) and e.name == name)


# ---------------------------------------------------------------------------- what counts as asking
@pytest.mark.parametrize("words", [
    "Send Priya an email with the notes",
    "please email priya the notes",
    "Can you send the notes to priya@example.com?",
    "Reply to the email from Sam and say yes",
    "draft it, then send the email",
    "Yes, send it",  # after Sentient showed the draft: the action is named and no other app is
])
def test_words_that_ask_for_sending_an_email(words):
    t = _mailer([]).tools[0]
    assert rule_levels.asked_for(t, [words], "Mailer", Risk.send, ["slack", "Slack"])


@pytest.mark.parametrize("words", [
    "Summarize my latest emails",
    "Don't send the email yet",
    "Ask me before you send the email",
    "Send the notes to Priya on Slack",  # another app
    "Email Priya the Slack thread",  # names the email, but also another app: it asks
    "Delete the spam emails",  # another action of the same app
    "",
])
def test_words_that_do_not_ask_for_sending_an_email(words):
    t = _mailer([]).tools[0]
    assert not rule_levels.asked_for(t, [words], "Mailer", Risk.send, ["slack", "Slack"])


def test_an_email_address_never_asks_for_another_kind_of_send():
    @tool("slack_send_message", risk=Risk.send)
    async def slack_send_message(ctx: ToolContext, channel: str, text: str) -> dict:
        """Post to Slack."""
        return {}

    slack_send_message.plugin = "slack"
    assert not rule_levels.asked_for(slack_send_message, ["send it to priya@example.com"], "Slack")
    assert rule_levels.asked_for(slack_send_message, ["ok, send it"], "Slack", other_apps=["gmail", "Gmail"])
    assert not rule_levels.asked_for(slack_send_message, ["send it with gmail"], "Slack", other_apps=["gmail", "Gmail"])


def test_a_noun_is_not_a_verb_and_look_ups_need_only_their_subject():
    @tool("slack_send_message", risk=Risk.send)
    async def slack_send_message(ctx: ToolContext, channel: str, text: str) -> dict:
        """Post to Slack."""
        return {}

    slack_send_message.plugin = "slack"
    assert not rule_levels.asked_for(slack_send_message, ["Summarize my Slack messages"], "Slack")
    assert rule_levels.asked_for(slack_send_message, ["Message the team on Slack that I'm late"], "Slack")
    search = _mailer([]).tools[1]
    assert rule_levels.asked_for(search, ["What's in my inbox?"], "Mailer", Risk.read)
    assert not rule_levels.asked_for(search, ["What's the weather?"], "Mailer", Risk.read)


# ---------------------------------------------------------------------------- "Only if I asked" in a chat
async def test_only_if_asked_runs_unasked_when_the_message_asks_for_it(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "always"
    config.tools.approvals.undo_seconds = 0
    config.tools.approvals.second_look_model = False
    config.tools.approvals.rules = {"mailer_send_email": "if_asked"}
    llm = FakeProvider(replies=[[tool_call("mailer_send_email", **SEND)], "Sent."])
    s = await _start(config, isolated_home, llm, "asked", _mailer(log))
    try:
        sid = await s.store.create_session(channel="desktop")
        asked, events = await _turn(s, sid, "Email priya@example.com the notes please")
        assert asked == [] and log == ["send:priya@example.com"]
        assert not _result(events, "mailer_send_email").is_error
    finally:
        await s.stop()


async def test_only_if_asked_asks_when_the_message_asked_for_something_else(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "off"  # it asks even with approvals off
    config.tools.approvals.rules = {"mailer": "if_asked"}
    llm = FakeProvider(replies=[[tool_call("mailer_send_email", **SEND)], "I didn't send it."])
    s = await _start(config, isolated_home, llm, "not_asked", _mailer(log))
    try:
        sid = await s.store.create_session(channel="desktop")
        asked, events = await _turn(s, sid, "Summarize the notes from today's meeting", decision="deny")
        assert [a.name for a in asked] == ["mailer_send_email"] and log == []
        assert _result(events, "mailer_send_email").result.get("declined") is True
        # what the user asked for in the same chat later still counts only for that turn
        stored = await s.store.recent_messages(sid, 20)
        approval = next(m for m in stored if m.get("approval"))["approval"]
        assert approval["rule"] == "if_asked"
    finally:
        await s.stop()


async def test_only_if_asked_never_skips_purchases_or_outside_content(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.second_look_model = False
    config.tools.approvals.rules = {"shop_buy": "if_asked"}
    llm = FakeProvider(replies=[[tool_call("shop_buy", item="a lamp")], "ok"])
    s = await _start(config, isolated_home, llm, "asked_purchase", _shop(log))
    try:
        sid = await s.store.create_session(channel="desktop")
        asked, _ = await _turn(s, sid, "Buy a lamp from the shop", decision="deny")
        assert [a.risk_label for a in asked] == ["Purchase"] and log == []
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- "Only if I asked" elsewhere
async def test_only_if_asked_pauses_a_task_and_stops_unattended_work(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.rules = {"mailer_send_email": "if_asked"}
    llm = FakeProvider(replies=[[tool_call("mailer_send_email", **SEND)], [tool_call("mailer_send_email", **SEND)]])
    s = await _start(config, isolated_home, llm, "asked_task", _mailer(log))
    try:
        # a task whose brief says "send the email" is not the user's chat message: it pauses and asks
        ctx = s.agent.tool_context(None, "task")
        ctx.extra.update({"task_id": "t1", CAN_PAUSE_KEY: True})
        messages = [{"role": "system", "content": "Send Priya an email with the notes."}, {"role": "user", "content": "go"}]
        result = LoopResult()
        async for _ in s.agent.run_loop(messages, ctx, result=result, use_approvals=False, source="task",
                                        stop=lambda: result.needs_ok is not None):
            pass
        assert log == [] and result.needs_ok is not None and result.needs_ok["approval"] is True
        assert "go ahead only when you ask for it in a chat" in result.needs_ok["question"]
        # a helper that can't stop to ask stops there
        ctx = s.agent.tool_context(None, "task")
        result = LoopResult()
        async for _ in s.agent.run_loop([{"role": "user", "content": "go"}], ctx, result=result,
                                        use_approvals=False, source="task"):
            pass
        assert log == [] and result.stopped_by_rule and "Only if I asked" in result.stopped_by_rule
        # scripts are never the user's words either
        policy = BridgePolicy.build(approvals_mode="off", allowed_tools=None, read_only=False, max_tool_calls=5,
                                    rules={"mailer_send_email": "if_asked"})
        with pytest.raises(Refused, match="only when you ask for it in a chat"):
            await policy.check(s.registry.get("mailer_send_email"), "mailer_send_email", SEND, ctx, 0)
    finally:
        await s.stop()


async def test_only_if_asked_tools_are_left_out_of_briefs_and_proactive_look_ups(config, isolated_home):
    config.tools.approvals.rules = {"mailer": "if_asked"}
    s = await _start(config, isolated_home, FakeProvider(), "asked_proactive", _mailer([]))
    try:
        assert "mailer_search" not in await s.proactivity._read_tool_names()
        assert await s.proactivity.brief.read("mailer_search", {"query": "x"}) is None
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- "Hand it to me"
async def test_hand_it_to_me_prepares_and_never_runs_in_a_chat(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "off"
    config.tools.approvals.rules = {"mailer": "hand_off"}
    llm = FakeProvider(replies=[
        [tool_call("mailer_search", query="notes")], [tool_call("mailer_send_email", **SEND)], "It's ready for you.",
    ])
    s = await _start(config, isolated_home, llm, "hand_off", _mailer(log))
    try:
        sid = await s.store.create_session(channel="desktop")
        asked, events = await _turn(s, sid, "Find the notes and send Priya an email with them")
        assert asked == [] and log == ["search:notes"]  # the look-up ran, the send didn't
        res = _result(events, "mailer_send_email")
        assert not res.is_error and rule_levels.is_handed_off(res.result)
        assert res.result["preview"]["kind"] == "email"
        fields = {f["label"]: f["value"] for f in res.result["preview"]["fields"]}
        assert fields == {"To": "priya@example.com", "Subject": "Notes"}
        assert res.result["preview"]["body"] == SEND["body"]
        assert res.result["message"].startswith('NOT DONE by Sentient: Mailer is set to "Hand it to me"')
        # the model read the same, and the transcript keeps the preview for the window
        stored = await s.store.recent_messages(sid, 20)
        assert any(m["role"] == "tool" and '"handed_off": true' in (m["content"] or "") for m in stored)
        # the activity timeline says it was handed over, not done
        items = (await s.timeline.past(kinds=["chat"]))["items"]
        handed = next(i for i in items if i["title"].startswith("Handed to you"))
        assert handed["title"] == "Handed to you: Send email in Mailer" and handed["status"] == "stopped"
    finally:
        await s.stop()


async def test_hand_it_to_me_beats_allow_for_this_chat_and_a_rule_set_while_waiting(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.mode = "always"
    llm = FakeProvider(replies=[[tool_call("mailer_send_email", **SEND)], "ok"])
    s = await _start(config, isolated_home, llm, "hand_off_wait", _mailer(log))
    try:
        sid = await s.store.create_session(channel="desktop")
        events = []
        async for ev in s.agent.run_turn(sid, "send priya an email", channel="desktop"):
            events.append(ev)
            if isinstance(ev, ApprovalRequest):
                s.approvals.config.rules = {"mailer_send_email": "hand_off"}  # changed in Settings meanwhile
                s.approvals.resolve(ev.approval_id, "allow")
        assert log == [] and rule_levels.is_handed_off(_result(events, "mailer_send_email").result)
    finally:
        await s.stop()


async def test_hand_it_to_me_is_never_swapped_for_the_trash(config, isolated_home):
    """A delete for good normally becomes its app's trash call (#142); handed over, it stays what the model called."""
    log: list[str] = []

    @tool("mailer_trash_email", risk=Risk.write)
    async def mailer_trash_email(ctx: ToolContext, message_id: str) -> dict:
        """Move an email to the trash."""
        log.append(f"trash:{message_id}")
        return {"trashed": message_id}

    config.tools.approvals.rules = {"mailer_delete_email": "hand_off"}
    llm = FakeProvider(replies=[[tool_call("mailer_delete_email", message_id="m1")], "ok"])
    s = await _start(config, isolated_home, llm, "hand_off_trash", _mailer(log))
    try:
        s.registry.plugin("mailer").tools.append(mailer_trash_email)
        s.registry.register(s.registry.plugin("mailer"), replace=True)
        sid = await s.store.create_session(channel="desktop")
        _, events = await _turn(s, sid, "delete that email")
        assert log == [] and rule_levels.is_handed_off(_result(events, "mailer_delete_email").result)
    finally:
        await s.stop()


async def test_hand_it_to_me_in_a_task_and_in_scripts(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.rules = {"mailer_send_email": "hand_off"}
    s = await _start(config, isolated_home, FakeProvider(), "hand_off_task", _mailer(log))
    try:
        task = await s.tasks.create_approved_call("Email Priya the notes", "mailer_send_email", SEND,
                                                  step="Send the notes to Priya")
        await s.tasks.drain(10)
        task = await s.tasks.get(task.get("id") or task["task_id"])
        assert log == [] and task["status"] == "completed"
        notes = await s.notifications.list(limit=10)
        ready = next(n for n in notes if n["title"] == "Ready for you: Email")
        assert "priya@example.com" in ready["message"] and ready["payload"]["hand_off"]["handed_off"] is True
        # a script can't hand anything over, so it is refused; its look-ups still run
        policy = BridgePolicy.build(approvals_mode="off", allowed_tools=None, read_only=False, max_tool_calls=5,
                                    rules={"mailer": "hand_off"})
        ctx = s.agent.tool_context(None, "task")
        with pytest.raises(Refused, match="Hand it to me"):
            await policy.check(s.registry.get("mailer_send_email"), "mailer_send_email", SEND, ctx, 0)
        assert await policy.check(s.registry.get("mailer_search"), "mailer_search", {"query": "x"}, ctx, 0) == Risk.read
    finally:
        await s.stop()


async def test_hand_it_to_me_in_an_executor_run_notifies(config, isolated_home):
    log: list[str] = []
    config.tools.approvals.rules = {"mailer_send_email": "hand_off"}
    llm = FakeProvider(replies=[[tool_call("mailer_send_email", **SEND)], "Prepared the email for you."])
    s = await _start(config, isolated_home, llm, "hand_off_loop", _mailer(log))
    try:
        ctx = s.agent.tool_context(None, "task")
        result = LoopResult()
        async for _ in s.agent.run_loop([{"role": "user", "content": "go"}], ctx, result=result,
                                        use_approvals=False, source="task"):
            pass
        assert log == [] and result.text == "Prepared the email for you."
        assert [h["label"] for h in result.handed_off] == ['"Send email" in Mailer']
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- config and rules from chat
def test_the_new_levels_are_valid_rules():
    cfg = SentientConfig.model_validate({"tools": {"approvals": {"rules": {"gmail": "If_Asked", "x_send": "hand_off"}}}})
    assert cfg.tools.approvals.rules == {"gmail": "if_asked", "x_send": "hand_off"}
    with pytest.raises(ValidationError):
        SentientConfig.model_validate({"tools": {"approvals": {"rules": {"gmail": "maybe"}}}})


def test_chat_words_choose_the_level_in_code():
    assert standing_text("Only send emails when I ask you to.")
    assert standing_text("I'll send them myself.")
    assert level_for("don't send emails unless I ask") == "if_asked"
    assert level_for("Only post to Slack when I tell you to") == "if_asked"
    assert level_for("Don't send emails yourself, just draft them") == "hand_off"
    assert level_for("never send my emails, I'll send them myself", "never") == "hand_off"
    assert level_for("don't post to Slack without asking me") == "ask"
    assert level_for("never delete my emails", "hand_off") == "never"  # the model can only pick Never or Ask
    assert level_for("never delete photos of myself", "never") == "never"  # "myself" alone hands nothing over
    assert level_for("I'll send them myself", "never") == "hand_off"


@pytest.mark.parametrize(("said", "level"), [
    ("Please don't send emails unless I ask you to.", "if_asked"),
    ("Don't send emails yourself, just draft them and I'll send them myself.", "hand_off"),
])
async def test_rules_from_chat_propose_the_new_levels(config, isolated_home, said, level):
    log: list[str] = []
    llm = FakeProvider(replies=["Got it."], json_replies=[{"keys": ["mailer_send_email"], "rule": "never"}])
    s = await _start(config, isolated_home, llm, f"propose_{level}", _mailer(log))
    try:
        sid = await s.store.create_session(channel="desktop")
        await _turn(s, sid, said)
        [proposal] = await s.chat_rules.list(sid)
        assert proposal["rule"] == level and proposal["keys"] == ["mailer_send_email"]
        await s.chat_rules.decide(proposal["id"], "accept")
        assert s.config.tools.approvals.rules == {"mailer_send_email": level}
        assert s.config.tools.approvals.rule_origins["mailer_send_email"].rule == level
    finally:
        await s.stop()


async def test_a_stricter_rule_is_never_loosened_by_a_proposal(config, isolated_home):
    config.tools.approvals.rules = {"mailer_send_email": "ask"}
    llm = FakeProvider(replies=["ok"], json_replies=[{"keys": ["mailer_send_email"], "rule": "ask"}])
    s = await _start(config, isolated_home, llm, "no_loosen", _mailer([]))
    try:
        sid = await s.store.create_session(channel="desktop")
        await _turn(s, sid, "Don't send emails unless I ask you to.")
        assert await s.chat_rules.list(sid) == []  # Ask is stricter than Only if I asked
    finally:
        await s.stop()


async def test_a_hand_off_at_run_time_is_counted_and_a_restore_handed_off_is_no_restore(config, isolated_home):
    """A rule set to Hand it to me while a call waited still reaches the run's hand-offs (and its notification)."""
    log: list[str] = []
    llm = FakeProvider(replies=[[tool_call("mailer_send_email", **SEND)], "done"])
    s = await _start(config, isolated_home, llm, "hand_off_late", _mailer(log))
    try:
        res, is_error, _ = await s.agent.run_tool(
            ToolCall(id="c1", name="mailer_send_email", arguments=SEND), s.agent.tool_context(None, "task")
        )
        assert log == ["send:priya@example.com"] and not is_error  # no rule yet: it runs
        s.approvals.config.rules = {"mailer_send_email": "hand_off"}
        res, is_error, _ = await s.agent.run_tool(
            ToolCall(id="c2", name="mailer_send_email", arguments=SEND), s.agent.tool_context(None, "task")
        )
        assert rule_levels.is_handed_off(res) and not is_error and log == ["send:priya@example.com"]
        result = LoopResult()
        async for _ in s.agent.run_loop([{"role": "user", "content": "go"}], s.agent.tool_context(None, "task"),
                                        result=result, use_approvals=False, source="task"):
            pass
        assert len(result.handed_off) == 1 and log == ["send:priya@example.com"]
    finally:
        await s.stop()
