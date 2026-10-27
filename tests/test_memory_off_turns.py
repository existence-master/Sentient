"""A chat turn without memory (#525): what it starts (helpers, the reader, scripts) is never wider than the turn.

The turn decides once (``Agent.memory_allowed``) and carries the answer in ``ToolContext.memory_allowed``. Without
memory, no memory or history tool is offered or runs, and no memory reaches a helper's prompt, whoever builds the
tool list.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from sentient.app import SentientApp
from sentient.llm.events import ToolResultEvent
from sentient.sandbox.policy import BridgePolicy, Refused
from sentient.tools.base import MEMORY_TOOLS, ToolContext
from tests.conftest import FakeProvider, tool_call
from tests.memory.helpers import add_fact

FACT = "Maya Rao's sister Anika lives in Pune"


@pytest.fixture
async def app(config, isolated_home):
    config.memory.min_similarity = 0.0
    config.sandbox.backend = "process"
    llm = FakeProvider()
    a = SentientApp(config, llm=llm, db_path=isolated_home / "memoff.db", enable_background=False)
    await a.start()
    a.fake = llm
    await add_fact(a.memory, FACT, source="manual")
    yield a
    await a.stop()


def _offered(call: dict) -> set[str]:
    return {t["function"]["name"] for t in call.get("tools") or []}


def _said(app) -> str:
    """Everything any model was shown in this test."""
    return json.dumps([c["messages"] for c in app.fake.calls], default=str)


async def _private_chat(app) -> str:
    """An incognito chat with memory off: the case that exists today."""
    app.config.chat.incognito_uses_memory = False
    return await app.store.create_session(channel="desktop", incognito=True)


async def _turn(app, sid: str, text: str) -> list:
    events = [ev async for ev in app.agent.run_turn(sid, text, channel="desktop")]
    await app.agent.drain(timeout=5)
    return events


def _result(events, name: str):
    return next(e for e in events if isinstance(e, ToolResultEvent) and e.name == name)


def _helper_calls(app) -> list[dict]:
    return [c for c in app.fake.calls if "You are a subagent" in str((c.get("messages") or [{}])[0].get("content"))]


async def test_a_helper_of_a_chat_without_memory_gets_no_memory_tools_or_context(app):
    sid = await _private_chat(app)
    app.fake.replies.extend([
        [tool_call("delegate_task", goal="Find everything Sentient knows about Maya's family")],  # the chat
        [tool_call("memory_recall", query="Maya sister")],  # the helper tries anyway
        "I could not find anything.",  # the helper's report
        "I can't see that from here.",  # the chat's reply
    ])
    events = await _turn(app, sid, "Use a helper to find what you know about Maya's sister")

    helpers = _helper_calls(app)
    assert len(helpers) == 2
    for helper in helpers:
        assert not MEMORY_TOOLS & _offered(helper)  # the helper's list is never wider than the turn's
        assert not {"memory_remember", "skill_save"} & _offered(helper)  # nor does it save anything (#154)
        assert "Maya" not in helper["messages"][0]["content"]  # no name, profile or memories in its prompt
    subs = await app.subagents.list_for_session(sid)
    [recall] = [e["message"] for e in subs[0]["events"] if e["message"].get("tool_name") == "memory_recall"
                and e["message"]["type"] == "tool_result"]
    assert recall["is_error"]
    assert _result(events, "delegate_task").result["status"] == "completed"
    assert "Anika" not in _said(app)  # the stored fact reached no model


async def test_parallel_and_background_helpers_inherit_the_turn(app):
    sid = await _private_chat(app)
    app.fake.replies.extend([
        [tool_call("delegate_tasks", tasks=[{"goal": "Search past chats for Maya"}, {"goal": "Ask the user model"}])],
        [tool_call("memory_search_history", query="Maya")],
        [tool_call("user_model_ask", question="What does Maya like?")],
        "Nothing.", "Nothing.",
        "Neither helper found anything.",
    ])
    await _turn(app, sid, "Ask two helpers what you know about me")
    helpers = _helper_calls(app)
    assert len(helpers) == 4 and all(not MEMORY_TOOLS & _offered(c) for c in helpers if c.get("tools"))
    assert "Anika" not in _said(app)

    # a background helper started from the same turn, straight through the manager, keeps the turn's answer too
    app.fake.replies.extend([[tool_call("history_time_search", start_date="2020-01-01", end_date="2030-01-01")],
                             "Done."])
    out = await app.subagents.delegate("Look through old chats", session_id=sid, memory_allowed=False)
    assert out["status"] == "completed"
    later = _helper_calls(app)[-2:]
    assert all(not MEMORY_TOOLS & _offered(c) for c in later)
    assert "Anika" not in _said(app)


async def test_a_helper_of_an_incognito_chat_with_memory_reads_but_never_saves(app):
    app.config.chat.incognito_uses_memory = True
    sid = await app.store.create_session(channel="desktop", incognito=True)
    app.fake.replies.extend([[tool_call("delegate_task", goal="Recall Maya's family")], "Done.", "Done."])
    await _turn(app, sid, "Ask a helper about my family")
    [helper] = _helper_calls(app)
    assert "memory_recall" in _offered(helper) and not {"memory_remember", "skill_save"} & _offered(helper)


async def test_the_reader_refuses_a_memory_tool_in_a_chat_without_memory(app):
    sid = await _private_chat(app)
    app.fake.replies.extend([
        [tool_call("read_with_reader", tool="memory_recall", arguments={"query": "Maya"},
                   fields={"where_her_sister_lives": "text"})],
        "I can't look that up here.",
    ])
    events = await _turn(app, sid, "Use the reader to find where Maya's sister lives")
    res = _result(events, "read_with_reader")
    assert "isn't available in this chat" in res.result["error"]
    assert await app.subagents.list_for_session(sid) == []  # no reader ran
    assert not [c for c in app.fake.calls if c.get("json")] and "Anika" not in _said(app)


async def test_a_normal_chat_still_hands_memory_on(app):
    sid = await app.store.create_session(channel="desktop")
    app.fake.replies.extend([
        [tool_call("delegate_task", goal="Find what Sentient knows about Maya's sister")],
        [tool_call("memory_recall", query="Maya sister")],
        "Her sister Anika lives in Pune.",
        "Anika lives in Pune.",
    ])
    await _turn(app, sid, "Use a helper to find what you know about Maya's sister")
    [helper, report] = _helper_calls(app)
    assert {"memory_recall", "memory_search_history"} <= _offered(helper)
    assert "Maya" in helper["messages"][0]["content"]
    assert "Anika" in json.dumps(report["messages"], default=str)  # the helper's recall found the fact
    ctx = app.agent.tool_context(sid, "desktop")
    from sentient.tools.builtin.subagents_tool import _refuse_source

    assert _refuse_source(app, "memory_recall", ctx) == ""  # the reader takes it as before


async def test_another_rule_for_a_turn_without_memory_plugs_in_once(app, monkeypatch):
    """The hook #479's /v1 switch uses: whatever makes ``Agent.memory_allowed`` False covers the turn's prompt, its own
    tools and everything it starts, in a chat that isn't incognito."""
    monkeypatch.setattr(app.agent, "memory_allowed", lambda session, channel: False)
    sid = await app.store.create_session(channel="desktop")
    app.fake.replies.extend([
        [tool_call("memory_recall", query="Maya")],  # not offered: refused
        [tool_call("delegate_task", goal="Recall Maya's family")],
        [tool_call("memory_recall", query="Maya")],
        "Nothing.",
        "I don't know.",
    ])
    events = await _turn(app, sid, "Where does my sister live?")
    chat = app.fake.calls[0]
    assert not MEMORY_TOOLS & _offered(chat) and "Anika" not in chat["messages"][0]["content"]
    assert _result(events, "memory_recall").is_error
    [helper, _] = _helper_calls(app)
    assert not MEMORY_TOOLS & _offered(helper) and "Maya" not in helper["messages"][0]["content"]
    assert "Anika" not in _said(app)


async def test_a_memory_tool_refuses_to_run_for_a_run_without_memory(app):
    sid = await app.store.create_session(channel="desktop")  # not incognito: only the run's own answer says no
    recall = app.registry.get("memory_recall")
    ctx = app.agent.tool_context(sid, "desktop")
    assert any(f["fact"] == FACT for f in await recall.call(ctx, {"query": "Maya sister"}))
    ctx.memory_allowed = False
    out = await recall.call(ctx, {"query": "Maya sister"})
    assert "doesn't use what Sentient knows" in out["error"]
    assert ctx.withholds("memory_search_history") and not ctx.withholds("current_datetime")


async def test_a_script_gets_no_memory_tools_from_a_turn_without_memory(app):
    policy = BridgePolicy.build(approvals_mode="ask", allowed_tools=None, read_only=False, max_tool_calls=10,
                                memory_allowed=False)
    recall = app.registry.get("memory_recall")
    assert not policy.is_available(recall)
    with pytest.raises(Refused, match="doesn't use what Sentient knows"):
        await policy.check(recall, "memory_recall", {"query": "x"}, ToolContext(store=None, config=None, llm=None), 0)

    sid = await app.store.create_session(channel="desktop")
    ctx = app.agent.tool_context(sid, "desktop")
    ctx.memory_allowed = False
    script = textwrap.dedent(
        """
        from sentient_tools import tools, result, ToolRefused
        try:
            got = tools.memory_recall(query="Maya sister")
        except ToolRefused as exc:
            got = "refused: " + str(exc)
        result({"got": got, "available": tools.available()})
        """
    )
    res = await app.sandbox.run(script, session_id=sid, channel="desktop", ctx=ctx)
    assert res["ok"], res
    assert str(res["result"]["got"]).startswith("refused") and not MEMORY_TOOLS & set(res["result"]["available"])
    assert "Anika" not in json.dumps(res)
