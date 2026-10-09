"""Work nobody asked for can only read (#129, ADR 0017).

Proactive checks, the heartbeat, follow-up scans, dreaming and anything they start (subagents) run with an
unprompted origin. Whatever the approval mode or lasting rules say, such a run may only look things up and make
Sentient-internal changes; sending, deleting, buying, running code, writing outside Sentient or creating tasks is
refused in code. Work the user asked for (chats, tasks, approved suggestions) is unaffected.
"""

from __future__ import annotations

import pytest

from sentient.agent.loop import Agent, LoopResult
from sentient.app import SentientApp
from sentient.tools.base import (
    UNPROMPTED_ORIGINS,
    Risk,
    ToolContext,
    ToolPlugin,
    is_unprompted,
    tool,
)
from sentient.tools.rules import unprompted_allows
from tests.conftest import FakeProvider, tool_call

RAN: list[str] = []


def _escalate(arguments, ctx):
    return Risk.send if arguments.get("buy") else None


@tool("vault_lookup", risk=Risk.read, risk_fn=_escalate)
async def vault_lookup(ctx: ToolContext, query: str = "", buy: bool = False) -> dict:
    """Look something up in the vault (buying through it is a purchase)."""
    RAN.append("vault_lookup:buy" if buy else "vault_lookup")
    return {"found": query}


@tool("vault_note", risk=Risk.write, internal=True)
async def vault_note(ctx: ToolContext, text: str = "") -> dict:
    """Keep a note inside Sentient."""
    RAN.append("vault_note")
    return {"ok": True}


@tool("vault_edit", risk=Risk.write)
async def vault_edit(ctx: ToolContext, text: str = "") -> dict:
    """Change a document outside Sentient."""
    RAN.append("vault_edit")
    return {"ok": True}


@tool("vault_send", risk=Risk.send)
async def vault_send(ctx: ToolContext, text: str = "") -> dict:
    """Send a message to someone."""
    RAN.append("vault_send")
    return {"sent": True}


@tool("vault_run", risk=Risk.exec)
async def vault_run(ctx: ToolContext, code: str = "") -> dict:
    """Run code."""
    RAN.append("vault_run")
    return {"ok": True}


class Vault(ToolPlugin):
    id = "vault"
    display_name = "Vault"
    tools = [vault_lookup, vault_note, vault_edit, vault_send, vault_run]


ALL_CALLS = [
    tool_call("vault_lookup", query="q"), tool_call("vault_lookup", query="q", buy=True), tool_call("vault_note"),
    tool_call("vault_edit"), tool_call("vault_send"), tool_call("vault_run"),
]
NAMES = ["vault_lookup", "vault_note", "vault_edit", "vault_send", "vault_run"]


@pytest.fixture(autouse=True)
def _clear():
    RAN.clear()


@pytest.fixture
async def app(config, isolated_home):
    """Approvals off and an Allow rule for the whole app: the most permissive setup there is."""
    config.tools.approvals.mode = "off"
    config.tools.approvals.rules = {"vault": "allow"}
    llm = FakeProvider()
    a = await SentientApp(config, llm=llm, db_path=isolated_home / "unprompted.db", enable_background=False).start()
    a.registry.register(Vault())
    a.fake = llm
    yield a
    await a.stop()


async def _loop(app, ctx, *, source: str, use_approvals: bool = False, calls: list | None = None) -> LoopResult:
    app.fake.replies[:] = [list(ALL_CALLS if calls is None else calls), "done"]
    result = LoopResult()
    messages = [{"role": "user", "content": "go"}]
    async for _ in app.agent.run_loop(messages, ctx, result=result, tool_names=NAMES, use_approvals=use_approvals,
                                      source=source):
        pass
    return result


# ---------------------------------------------------------------------------- the rule
def test_only_reads_and_internal_changes_are_allowed():
    assert unprompted_allows(vault_lookup, Risk.read)
    assert unprompted_allows(vault_note, Risk.write)
    assert not unprompted_allows(vault_lookup, Risk.send)  # a look-up its risk_fn raised to a purchase
    assert not unprompted_allows(vault_edit, Risk.write)
    assert not unprompted_allows(vault_send, Risk.send)
    assert not unprompted_allows(vault_run, Risk.exec)

    @tool("plan_more_work", risk=Risk.write, internal=True)
    async def plan_more_work(ctx: ToolContext) -> dict:
        """Create a task."""
        return {}

    plan_more_work.plugin = "tasks"  # a task runs its plan later on its own: not for unprompted work
    assert not unprompted_allows(plan_more_work, Risk.write)
    assert {"proactive", "heartbeat", "followups", "dreaming"} <= UNPROMPTED_ORIGINS
    assert is_unprompted("proactive") and not is_unprompted("user") and not is_unprompted(None)


@pytest.mark.parametrize("origin", sorted(UNPROMPTED_ORIGINS))
async def test_unprompted_runs_only_read_whatever_the_rules_say(app, origin):
    ctx = app.agent.tool_context(None, "background", origin=origin)
    result = await _loop(app, ctx, source="test")
    assert RAN == ["vault_lookup", "vault_note"]
    held = [h["tool"] for h in result.held]
    assert held == ["vault_lookup", "vault_edit", "vault_send", "vault_run"]
    refusals = [m["content"] for m in result.messages if m.get("role") == "tool" and "Nobody asked" in m["content"]]
    assert len(refusals) == 4 and "suggest it to the user" in refusals[0]


@pytest.mark.parametrize("source", ["proactive", "heartbeat", "followups", "dreaming"])
async def test_an_unprompted_source_is_enough(app, source):
    ctx = app.agent.tool_context(None, "web")  # the context says user; the run's source says otherwise
    await _loop(app, ctx, source=source, use_approvals=True)
    assert RAN == ["vault_lookup", "vault_note"]


async def test_the_proactive_channel_marks_the_context(app):
    assert app.agent.tool_context(None, "proactive").origin == "proactive"
    assert app.agent.tool_context(None, "task").origin == "user"
    assert app.agent.tool_context("s1", "web").origin == "user"


@pytest.mark.parametrize("channel, source", [("web", "chat"), ("task", "task"), ("subagent", "subagent")])
async def test_work_the_user_asked_for_is_unaffected(app, channel, source):
    ctx = app.agent.tool_context(None, channel)
    result = await _loop(app, ctx, source=source, calls=[c for c in ALL_CALLS if not c["arguments"].get("buy")])
    assert RAN == ["vault_lookup", "vault_note", "vault_edit", "vault_send", "vault_run"]
    assert result.held == []


# ---------------------------------------------------------------------------- entry points
async def test_proactive_lookups_cannot_act_and_the_held_action_reaches_the_reasoner(app):
    app.config.proactivity.context_agent_rounds = 3
    app.fake.replies[:] = [[tool_call("vault_lookup", query="invoice"), tool_call("vault_lookup", buy=True)],
                          [tool_call("vault_send", text="hi")], "Found the invoice."]
    text = await app.proactivity.live_search("gmail", "new_email", {"id": "m1", "subject": "Invoice"},
                                             {"q": "is the invoice paid"})
    assert RAN == ["vault_lookup"]
    assert text and "vault_lookup" in text and "needs the user's approval" in text
    offered = {t["function"]["name"] for t in app.fake.calls[0]["tools"]}
    assert "vault_lookup" in offered and not offered & {"vault_edit", "vault_send", "vault_run", "vault_note"}


async def test_a_subagent_started_by_unprompted_work_is_unprompted_too(app):
    app.config.subagents.enabled = True
    ctx = app.agent.tool_context(None, "proactive")
    app.fake.replies[:] = [[tool_call("delegate_task", goal="look into the invoice", tools=["vault"])],
                           [tool_call("vault_edit"), tool_call("vault_lookup", query="x")], "sub done", "all done"]
    result = LoopResult()
    async for _ in app.agent.run_loop([{"role": "user", "content": "go"}], ctx, result=result,
                                      tool_names=["delegate_task"], use_approvals=False, source="proactive"):
        pass
    assert "vault_edit" not in RAN and RAN == ["vault_lookup"]
    delegated = next(m["content"] for m in result.messages if m.get("role") == "tool")
    assert '"status": "completed"' in delegated and "sub done" in delegated


async def test_every_unprompted_entry_point_runs_unprompted(app, monkeypatch):
    """Walk the background jobs that use a model and record every tool-calling run they start."""
    runs: list[tuple[str, str]] = []
    original = Agent.run_loop

    def spy(self, messages, ctx, **kw):
        runs.append((getattr(ctx, "origin", "user"), kw.get("source", "chat")))
        return original(self, messages, ctx, **kw)

    monkeypatch.setattr(Agent, "run_loop", spy)
    app.config.proactivity.context_agent_rounds = 2
    app.fake.replies[:] = [[tool_call("vault_send", text="hi")], "nothing"] * 4
    item = {"id": "m9", "thread_id": "t9", "from": "Jane <jane@acme.example>", "sender_email": "jane@acme.example",
            "to": "sarthak@example.com", "subject": "Can we meet Tuesday?", "snippet": "Can we meet Tuesday at 2?",
            "body": "Hi Sarthak, can we meet Tuesday at 2pm to go over the plan? Jane", "labels": ["INBOX"],
            "date": "2026-10-09T08:00:00+00:00"}
    await app.proactivity.process_event("gmail", "new_email", item)
    await app.proactivity.heartbeat()
    await app.proactivity.run_followups()
    await app.dreaming.run()
    assert runs, "the proactive pipeline should have looked something up"
    assert all(is_unprompted(origin) or is_unprompted(source) for origin, source in runs)
    assert "vault_send" not in RAN
