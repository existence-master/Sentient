"""docs/API.md section 10: steering, tool progress, dynamic risk, parallel reads, large results,
prompt caching, hooks, transcript repair, selection triggers and prompt guidance."""

from __future__ import annotations

import asyncio

from sentient.agent.loop import LoopResult, history_to_openai
from sentient.agent.prompt import build_system_prompt
from sentient.agent.toolselect import ALWAYS_TOOLS, ToolSelector, triggered_prefixes
from sentient.app import SentientApp
from sentient.llm.events import (
    ApprovalRequest,
    Done,
    TextDelta,
    ToolCallEvent,
    ToolProgress,
    ToolResultEvent,
    UserInterjection,
)
from sentient.llm.provider import apply_prompt_cache
from sentient.tools.base import Risk, ToolContext, ToolPlugin, effective_risk, tool
from tests.conftest import FakeProvider, tool_call


def make_plugin(pid: str, *tools_, hint: str = "") -> ToolPlugin:
    class P(ToolPlugin):
        id = pid
        display_name = pid.title()
        selection_hint = hint

    P.tools = list(tools_)
    return P()


async def start(config, isolated_home, llm, name: str) -> SentientApp:
    return await SentientApp(config, llm=llm, db_path=isolated_home / f"{name}.db", enable_background=False).start()


# ---------------------------------------------------------------------------- steering
async def test_steer_during_tool_round_reaches_next_model_call(config, isolated_home):
    llm = FakeProvider(replies=[[tool_call("current_datetime")], "It is noon, in metric."])
    s = await start(config, isolated_home, llm, "steer1")
    try:
        sid = await s.store.create_session(channel="cli")
        events = []
        assert not s.agent.steer(sid, "too early")  # nothing running yet
        async for ev in s.agent.run_turn(sid, "what time is it", channel="cli"):
            events.append(ev)
            if isinstance(ev, ToolCallEvent):
                assert s.agent.is_running(sid)
                assert s.agent.steer(sid, "and use metric units")
        inter = [e for e in events if isinstance(e, UserInterjection)]
        assert len(inter) == 1 and inter[0].text == "and use metric units" and inter[0].turn_id
        assert events.index(inter[0]) > next(i for i, e in enumerate(events) if isinstance(e, ToolResultEvent))
        second = llm.calls[1]["messages"]
        assert second[-1] == {"role": "user", "content": "and use metric units"} and second[-2]["role"] == "tool"
        rows = await s.store.recent_messages(sid, 20)
        assert [r["role"] for r in rows] == ["user", "assistant", "tool", "user", "assistant"]
        assert not s.agent.is_running(sid) and not s.agent.steer(sid, "late")
        assert sum(isinstance(e, Done) for e in events) == 1
    finally:
        await s.stop()


async def test_steer_while_final_answer_streams_continues_same_turn(config, isolated_home):
    llm = FakeProvider(replies=["First answer here.", "Second answer."])
    s = await start(config, isolated_home, llm, "steer2")
    try:
        sid = await s.store.create_session(channel="cli")
        events = []
        async for ev in s.agent.run_turn(sid, "hi", channel="cli"):
            events.append(ev)
            if isinstance(ev, TextDelta) and len(events) == 1:
                assert s.agent.steer(sid, "actually, shorter")
        done = [e for e in events if isinstance(e, Done)]
        assert len(done) == 1 and done[0].content == "Second answer."
        rows = await s.store.recent_messages(sid, 20)
        assert [(r["role"], r["content"]) for r in rows] == [
            ("user", "hi"), ("assistant", "First answer here."), ("user", "actually, shorter"), ("assistant", "Second answer."),
        ]
    finally:
        await s.stop()


async def test_steer_after_last_round_starts_new_turn(config, isolated_home):
    config.models.max_tool_rounds = 1
    llm = FakeProvider(replies=["Answer one.", "Answer two."])
    s = await start(config, isolated_home, llm, "steer3")
    try:
        sid = await s.store.create_session(channel="cli")
        events = []
        async for ev in s.agent.run_turn(sid, "hi", channel="cli"):
            events.append(ev)
            if isinstance(ev, TextDelta) and len(events) == 1:
                assert s.agent.steer(sid, "one more thing")
        done = [e for e in events if isinstance(e, Done)]
        assert [d.content for d in done] == ["Answer one.", "Answer two."]
        assert done[0].turn_id != done[1].turn_id
        rows = await s.store.recent_messages(sid, 20)
        assert [(r["role"], r["content"]) for r in rows] == [
            ("user", "hi"), ("assistant", "Answer one."), ("user", "one more thing"), ("assistant", "Answer two."),
        ]
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- tool progress
async def test_tool_progress_streams_while_tool_runs(config, isolated_home):
    gate = asyncio.Event()

    @tool("long_job", risk=Risk.read)
    async def long_job(ctx: ToolContext) -> dict:
        """A long job."""
        ctx.progress({"kind": "stdout", "text": "line 1"})
        await ctx.progress({"kind": "status", "text": "halfway", "percent": 50})
        await asyncio.to_thread(lambda: ctx.progress({"kind": "stderr", "text": "from a thread"}))
        await asyncio.wait_for(gate.wait(), 5)  # only released once the progress reached the consumer
        return {"done": True, "call_id": ctx.call_id}

    llm = FakeProvider(replies=[[tool_call("long_job")], "finished"])
    s = await start(config, isolated_home, llm, "progress")
    try:
        s.registry.register(make_plugin("jobs", long_job))
        sid = await s.store.create_session(channel="cli")
        events = []

        async def run():
            async for ev in s.agent.run_turn(sid, "run the job", channel="cli"):
                events.append(ev)
                if isinstance(ev, ToolProgress):
                    gate.set()

        await asyncio.wait_for(run(), 10)
        progress = [e for e in events if isinstance(e, ToolProgress)]
        assert [p.kind for p in progress] == ["stdout", "status", "stderr"]
        assert progress[1].data == {"percent": 50} and progress[0].call_id == "call_long_job"
        assert progress[0].session_id == sid and progress[0].turn_id
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert events.index(progress[-1]) < events.index(result)
        assert result.result == {"done": True, "call_id": "call_long_job"}
        assert progress[0].model_dump()["type"] == "tool_progress"
    finally:
        await s.stop()


async def test_progress_outside_loop_is_noop():
    ctx = ToolContext(store=None, config=None, llm=None)
    ctx.progress({"text": "nobody listens"})
    await ctx.progress({"text": "still fine"})
    assert ctx.call_id is None


# ---------------------------------------------------------------------------- dynamic risk
async def test_effective_risk_variants():
    ctx = ToolContext(store=None, config=None, llm=None)

    def make(fn, risk=Risk.write):
        @tool("t", risk=risk, risk_fn=fn)
        async def t(ctx: ToolContext) -> str:
            """t"""
            return ""

        return t

    async def async_send(args, ctx):
        return Risk.send

    def boom(args, ctx):
        raise RuntimeError("bad")

    assert await effective_risk(make(None), {}, ctx) == Risk.write
    assert await effective_risk(make(lambda a, c: None), {}, ctx) == Risk.write
    assert await effective_risk(make(lambda a, c: Risk.read), {}, ctx) == Risk.read
    assert await effective_risk(make(async_send), {}, ctx) == Risk.send
    assert await effective_risk(make(lambda a, c: "send"), {}, ctx) == Risk.send
    assert await effective_risk(make(boom), {}, ctx) == Risk.exec


async def test_risk_fn_drives_approvals_and_session_allowance(config, isolated_home):
    clicked: list[str] = []

    def click_risk(arguments, ctx):
        return Risk.send if "order" in str(arguments.get("label", "")).lower() else None

    @tool("web_click", risk=Risk.write, risk_fn=click_risk)
    async def web_click(ctx: ToolContext, label: str) -> dict:
        """Click a button on a web page."""
        clicked.append(label)
        return {"clicked": label}

    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[
        [tool_call("web_click", label="Next")], "ok1",
        [tool_call("web_click", label="Next page")], "ok2",
        [tool_call("web_click", label="Place order")], "ok3",
    ])
    s = await start(config, isolated_home, llm, "risk")
    try:
        s.registry.register(make_plugin("web", web_click))
        sid = await s.store.create_session(channel="cli")
        asked: list[str] = []

        async def turn(text):
            async for ev in s.agent.run_turn(sid, text, channel="cli"):
                if isinstance(ev, ApprovalRequest):
                    asked.append(ev.risk)
                    s.approvals.resolve(ev.approval_id, "allow_session")

        await turn("click next")
        await turn("click next page")   # allowed for this chat at risk write: no prompt
        await turn("place the order")   # effective risk send: asks again despite the allowance
        assert asked == ["write", "send"]
        assert clicked == ["Next", "Next page", "Place order"]
    finally:
        await s.stop()


async def test_internal_tool_with_send_risk_still_asks(config, isolated_home):
    s = await start(config, isolated_home, FakeProvider(), "internal")
    try:
        @tool("wipe_notes", risk=Risk.write, internal=True, risk_fn=lambda a, c: Risk.send if a.get("all") else None)
        async def wipe_notes(ctx: ToolContext, all: bool = False) -> str:
            """Wipe notes."""
            return ""

        s.approvals.config.mode = "ask"
        assert not s.approvals.needs_approval(wipe_notes, "s1", Risk.write)
        assert s.approvals.needs_approval(wipe_notes, "s1", Risk.send)
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- parallel reads
async def test_read_tools_in_one_round_run_concurrently_and_keep_order(config, isolated_home):
    started: list[str] = []
    both = asyncio.Event()
    log: list[str] = []

    def lookup(name: str):
        @tool(name, risk=Risk.read)
        async def fn(ctx: ToolContext) -> dict:
            """Look something up."""
            started.append(name)
            if len(started) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), 3)  # deadlocks (times out) if the calls ran one after another
            if name == "lookup_a":
                await asyncio.sleep(0.05)  # finishes last, but its result must still come first
            log.append(name)
            return {"from": name}

        return fn

    @tool("save_it", risk=Risk.write, internal=True)
    async def save_it(ctx: ToolContext) -> dict:
        """Save."""
        log.append("save_it")
        return {"saved": True}

    calls = [
        {"id": "c1", "name": "lookup_a", "arguments": {}},
        {"id": "c2", "name": "lookup_b", "arguments": {}},
        {"id": "c3", "name": "save_it", "arguments": {}},
    ]
    llm = FakeProvider(replies=[calls, "done"])
    s = await start(config, isolated_home, llm, "parallel")
    try:
        s.registry.register(make_plugin("lookups", lookup("lookup_a"), lookup("lookup_b"), save_it))
        messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
        result = LoopResult()
        events = [ev async for ev in s.agent.run_loop(messages, s.agent.tool_context(None, "cli"), result=result)]
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert [r.call_id for r in results] == ["c1", "c2", "c3"] and not any(r.is_error for r in results)
        assert log == ["lookup_b", "lookup_a", "save_it"]  # the write ran after both reads finished
        assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["c1", "c2", "c3"]
        assert result.tool_calls == 3 and not result.tool_errors
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- large results
async def test_large_tool_result_is_cut_and_saved(config, isolated_home):
    @tool("big_dump", risk=Risk.read)
    async def big_dump(ctx: ToolContext) -> str:
        """Return a lot of text."""
        return "x" * 5000

    config.chat.tool_result_max_chars = 1000
    llm = FakeProvider(replies=[[tool_call("big_dump")], "ok"])
    s = await start(config, isolated_home, llm, "big")
    try:
        s.registry.register(make_plugin("dump", big_dump))
        sid = await s.store.create_session(channel="cli")
        events = [ev async for ev in s.agent.run_turn(sid, "dump", channel="cli")]
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.result == "x" * 5000  # the window still gets everything
        tool_msg = llm.calls[1]["messages"][-1]
        assert len(tool_msg["content"]) < 1400 and "files/outputs/tool-call_big_dump.txt" in tool_msg["content"]
        saved = isolated_home / "files" / "outputs" / "tool-call_big_dump.txt"
        assert saved.read_text(encoding="utf-8") == "x" * 5000
        rows = await s.store.recent_messages(sid, 10)
        assert next(r for r in rows if r["role"] == "tool")["content"] == tool_msg["content"]
    finally:
        await s.stop()


def with_window(llm: FakeProvider, tokens: int | None) -> FakeProvider:
    async def context_window(role: str, model: str | None = None) -> int | None:
        return tokens

    llm.context_window = context_window
    return llm


async def test_tool_result_limit_is_a_share_of_the_context(config, isolated_home):
    """#264: on an 8,192-token local model one result may fill about a quarter of the context, not 16,000 chars."""
    s = await start(config, isolated_home, FakeProvider(), "limit")
    try:
        assert await s.agent.tool_result_limit("primary") == 16000  # context length unknown: the fixed cap
        with_window(s.agent.llm, 8192)
        assert await s.agent.tool_result_limit("primary") == 6144
        with_window(s.agent.llm, 200_000)
        assert await s.agent.tool_result_limit("primary") == 16000  # a big cloud window keeps the fixed cap
        with_window(s.agent.llm, 1024)
        assert await s.agent.tool_result_limit("primary") == 1500  # never below a useful minimum
        config.chat.tool_result_context_share = 0.5
        with_window(s.agent.llm, 8192)
        assert await s.agent.tool_result_limit("primary") == 12288
    finally:
        await s.stop()


async def test_long_result_is_cut_to_the_context_share(config, isolated_home):
    @tool("big_page", risk=Risk.read)
    async def big_page(ctx: ToolContext) -> str:
        """Return a long page."""
        return "y" * 13000

    llm = with_window(FakeProvider(replies=[[tool_call("big_page")], "ok"]), 8192)
    s = await start(config, isolated_home, llm, "share")
    try:
        s.registry.register(make_plugin("page", big_page))
        sid = await s.store.create_session(channel="cli")
        events = [ev async for ev in s.agent.run_turn(sid, "page", channel="cli")]
        assert next(e for e in events if isinstance(e, ToolResultEvent)).result == "y" * 13000
        content = llm.calls[1]["messages"][-1]["content"]
        assert content.startswith('"yyy') and "y" * 6143 in content and "y" * 6200 not in content
        assert "[Result cut: this is only the first 6144 of 13002 characters." in content
        assert "files/outputs/tool-call_big_page.txt" in content
    finally:
        await s.stop()


async def test_a_tools_shortener_replaces_the_plain_cut(config, isolated_home):
    """A tool with ``shorten_fn`` (Composio's search, #264) gives the model its main parts instead of the start."""

    @tool("search_catalog", risk=Risk.read)
    async def search_catalog(ctx: ToolContext) -> dict:
        """Search the catalog."""
        return {"plan": ["call LIST_EVENTS"], "schemas": "z" * 20000}

    search_catalog.shorten_fn = lambda res: {"plan": res["plan"]}
    llm = with_window(FakeProvider(replies=[[tool_call("search_catalog")], "ok"]), 8192)
    s = await start(config, isolated_home, llm, "shorten")
    try:
        s.registry.register(make_plugin("catalog", search_catalog))
        sid = await s.store.create_session(channel="cli")
        _ = [ev async for ev in s.agent.run_turn(sid, "search", channel="cli")]
        content = llm.calls[1]["messages"][-1]["content"]
        assert content.startswith('{"plan": ["call LIST_EVENTS"]}')
        assert "[Result shortened from 20045 characters to its main parts." in content and "zzz" not in content
        saved = isolated_home / "files" / "outputs" / "tool-call_search_catalog.txt"
        assert "z" * 20000 in saved.read_text(encoding="utf-8")
    finally:
        await s.stop()


async def test_a_shortener_that_fails_falls_back_to_the_cut(config, isolated_home):
    @tool("odd_tool", risk=Risk.read)
    async def odd_tool(ctx: ToolContext) -> str:
        """Return a long odd result."""
        return "q" * 9000

    def broken(res):
        raise ValueError("unexpected shape")

    odd_tool.shorten_fn = broken
    llm = with_window(FakeProvider(replies=[[tool_call("odd_tool")], "ok"]), 8192)
    s = await start(config, isolated_home, llm, "broken")
    try:
        s.registry.register(make_plugin("odd", odd_tool))
        sid = await s.store.create_session(channel="cli")
        _ = [ev async for ev in s.agent.run_turn(sid, "odd", channel="cli")]
        assert "[Result cut: this is only the first 6144 of 9002 characters." in llm.calls[1]["messages"][-1]["content"]
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- prompt caching
def test_prompt_cache_only_for_anthropic():
    messages = [{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]
    m2, t2 = apply_prompt_cache("anthropic/claude-sonnet-5", messages, tools)
    assert m2[0]["content"] == [{"type": "text", "text": "persona", "cache_control": {"type": "ephemeral"}}]
    assert m2[1] == messages[1] and "cache_control" not in t2[0] and t2[-1]["cache_control"] == {"type": "ephemeral"}
    assert messages[0]["content"] == "persona" and "cache_control" not in tools[-1]  # inputs untouched
    m3, t3 = apply_prompt_cache("ollama_chat/qwen3:8b", messages, tools)
    assert m3 is messages and t3 is tools
    m4, t4 = apply_prompt_cache("anthropic/claude-sonnet-5", messages, None)
    assert t4 is None and m4[0]["content"][0]["cache_control"]


# ---------------------------------------------------------------------------- hooks
async def test_memory_flush_runs_before_compression(config, isolated_home):
    order: list[str] = []

    class Recorder(FakeProvider):
        async def complete_text(self, role, messages, *, model=None):
            order.append("summary")
            return "Running summary."

    config.chat.history_window = 2
    config.chat.compress_after_messages = 10
    s = await start(config, isolated_home, Recorder(), "flush")
    try:
        assert s.memory is not None
        flushed: list[tuple[str, str]] = []

        async def flush_conversation(transcript: str, user_name: str) -> list[dict]:
            order.append("flush")
            flushed.append((transcript, user_name))
            return [{"action": "ADD", "id": 1, "content": "Sarthak likes tea"}]

        s.memory.flush_conversation = flush_conversation
        sid = await s.store.create_session(channel="cli")
        for i in range(20):
            await s.store.add_message(sid, "user" if i % 2 == 0 else "assistant", f"message number {i}")
        async with s.bus.subscribe() as q:
            await s.agent._maybe_compress(sid)
            published = [q.get_nowait() for _ in range(q.qsize())]
        assert order == ["flush", "summary"]
        assert "message number 0" in flushed[0][0] and flushed[0][1] == "Sarthak"
        # flush_conversation publishes its own changes; the agent does not send them a second time
        assert not any(e["type"] == "memory.updated" for e in published)
        assert (await s.store.get_session(sid))["context_summary"] == "Running summary."

        async def broken(transcript, user_name):
            raise RuntimeError("memory offline")

        s.memory.flush_conversation = broken
        for i in range(20):
            await s.store.add_message(sid, "user", f"later message {i}")
        await s.agent._maybe_compress(sid)  # a failing flush must not stop compression
        assert order.count("summary") == 2
    finally:
        await s.stop()


async def test_user_model_context_in_prompt_with_timeout(config, isolated_home, monkeypatch):
    llm = FakeProvider(replies=["one", "two"])
    s = await start(config, isolated_home, llm, "usermodel")
    try:
        seen: list[str] = []

        async def context_with_sources(text: str) -> tuple[str, list[dict]]:
            seen.append(text)
            return "- Prefers metric units", []

        s.user_model.context_with_sources = context_with_sources
        sid = await s.store.create_session(channel="cli")
        async for _ in s.agent.run_turn(sid, "how far is Mumbai", channel="cli"):
            pass
        system = llm.calls[-1]["messages"][0]["content"]
        assert seen == ["how far is Mumbai"] and "Prefers metric units" in system

        async def slow(text: str) -> tuple[str, list[dict]]:
            await asyncio.sleep(5)
            return "- never shown", []

        monkeypatch.setattr("sentient.agent.loop.USER_MODEL_TIMEOUT_S", 0.05)
        s.user_model.context_with_sources = slow
        async for _ in s.agent.run_turn(sid, "again", channel="cli"):
            pass
        assert "never shown" not in llm.calls[-1]["messages"][0]["content"]
    finally:
        await s.stop()


async def test_turn_completed_carries_errors_and_skills(config, isolated_home):
    calls = [
        {"id": "c1", "name": "skill_view", "arguments": {"name": "demo-skill"}},
        {"id": "c2", "name": "no_such_tool", "arguments": {}},
    ]
    llm = FakeProvider(replies=[calls, "Done with it."])
    s = await start(config, isolated_home, llm, "completed")
    try:
        s.skills.write("demo-skill", "A demo procedure", "## Procedure\n1. Do it.", author="user", pending=False)
        s.skills.reload()
        assert s.skills.get("demo-skill") is not None
        sid = await s.store.create_session(channel="cli")
        async with s.bus.subscribe() as q:
            async for _ in s.agent.run_turn(sid, "use the demo skill", channel="cli"):
                pass
            published = [q.get_nowait() for _ in range(q.qsize())]
        done = next(e["data"] for e in published if e["type"] == "chat.turn_completed")
        assert done["session_id"] == sid and done["turn_id"] and done["tool_calls"] == 2
        assert done["skills_viewed"] == ["demo-skill"]
        assert done["tool_errors"] == [{"name": "no_such_tool", "error": "unknown tool no_such_tool"}]
        assert done["user_text"] == "use the demo skill" and done["reply"] == "Done with it."
    finally:
        await s.stop()


async def test_invalid_arguments_get_schema_back(config, isolated_home):
    llm = FakeProvider(replies=[[{"id": "c1", "name": "memory_recall", "arguments": {"wrong": 1}}], "ok"])
    s = await start(config, isolated_home, llm, "invalid")
    try:
        messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
        events = [ev async for ev in s.agent.run_loop(messages, s.agent.tool_context(None, "cli"), result=LoopResult())]
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error and "Invalid arguments for memory_recall" in res.result["error"]
        assert "query" in res.result["schema"]["properties"]
    finally:
        await s.stop()


# ---------------------------------------------------------------------------- transcript repair
def test_history_to_openai_repairs_broken_transcripts():
    rows = [
        {"role": "tool", "tool_call_id": "old", "name": "x", "content": "{}"},  # leading orphan
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": None},  # empty: dropped
        {"role": "assistant", "content": "checking", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "f", "arguments": {"q": 1}}},
            {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "name": "f", "content": "A"},  # b never answered (stopped)
        {"role": "user", "content": "go on"},
        {"role": "tool", "tool_call_id": "zzz", "name": "h", "content": "orphan"},  # orphan in the middle
        {"role": "assistant", "content": "partial", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ]},
        {"role": "user", "content": "stop"},
    ]
    out = history_to_openai(rows)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "user", "assistant", "user"]
    assert out[1]["tool_calls"] == [{"id": "a", "type": "function", "function": {"name": "f", "arguments": '{"q": 1}'}}]
    assert out[4] == {"role": "assistant", "content": "partial"}


# ---------------------------------------------------------------------------- selection and prompt guidance
async def test_selection_triggers_surface_new_capabilities():
    from types import SimpleNamespace

    from sentient.config.schema import SentientConfig
    from sentient.tools.registry import ToolRegistry
    from tests.test_toolselect import KeywordOnlyProvider, plugin

    reg = ToolRegistry()
    reg.register(plugin("core", "clock memory", *ALWAYS_TOOLS))
    reg.register(plugin("browser", "", "browser_open", "browser_snapshot", "browser_click"))
    reg.register(plugin("sandbox", "", "execute_code"))
    reg.register(plugin("devices", "", "device_take_photo", "device_get_location"))
    reg.register(plugin("subagents", "", "delegate_task", "delegate_tasks"))
    reg.register(plugin("weather", "weather, forecast", "weather_current", "weather_forecast"))
    reg.register(plugin("gmail", "email, inbox", "gmail_search", "gmail_send", "gmail_read"))
    cfg = SentientConfig()
    cfg.chat.max_tools_local = 9
    sel = ToolSelector(SimpleNamespace(registry=reg, llm=KeywordOnlyProvider(), config=cfg))
    model = "ollama_chat/qwen3:8b"
    assert "browser_click" in await sel.select("log in to the airline site for me", model=model)
    assert "execute_code" in await sel.select("what is the median of 3, 9 and 4", model=model)
    assert "device_take_photo" in await sel.select("what am I looking at right now", model=model)
    assert "delegate_tasks" in await sel.select("research these three laptops in parallel", model=model)
    assert await sel.select("hello there, how are you", model=model) == list(ALWAYS_TOOLS)
    assert triggered_prefixes("in order to relax") == set()


def test_capability_rules_follow_available_tools():
    base = dict(snapshot={}, facts=[], skills_index="", assistant_name="Sentient", user_name="Sarthak",
                timezone="UTC", channel="desktop")
    plain = build_system_prompt(**base, tool_names=["memory_recall"])
    assert "Open browser button" not in plain and "execute_code" not in plain
    rich = build_system_prompt(
        **base, tool_names=["browser_open", "execute_code", "delegate_task", "device_take_photo"],
        user_context="- Likes short answers",
    )
    assert "Open browser button" in rich and "Never type passwords" in rich
    assert "Use execute_code" in rich and "delegate_task" in rich and "device_*" in rich
    assert "## How the user likes things\n- Likes short answers" in rich


async def test_read_tool_escalated_by_risk_fn_asks_with_wording(config, isolated_home):
    shots: list[str] = []

    async def photo_risk(arguments, ctx):
        return Risk.send

    @tool("test_cam_photo", risk=Risk.read, risk_fn=photo_risk,
          describe_fn=lambda a, c: {"risk_label": "Uses the camera", "target": a.get("device") or "phone"})
    async def device_take_photo(ctx: ToolContext, device: str = "") -> dict:
        """Take a photo."""
        shots.append(device)
        return {"file": "photo.jpg"}

    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[[tool_call("test_cam_photo", device="glasses")], "Nice view."])
    s = await start(config, isolated_home, llm, "escalate")
    try:
        s.registry.register(make_plugin("camtest", device_take_photo))
        sid = await s.store.create_session(channel="cli")
        requests = []
        async for ev in s.agent.run_turn(sid, "what am I looking at", channel="cli"):
            if isinstance(ev, ApprovalRequest):
                requests.append(ev)
                s.approvals.resolve(ev.approval_id, "allow")
        assert len(requests) == 1 and requests[0].risk == "send"
        assert requests[0].risk_label == "Uses the camera" and requests[0].target == "glasses"
        assert shots == ["glasses"]

        ctx = s.agent.tool_context(sid, "cli")
        assert await s.approvals.requires_approval(device_take_photo, {}, ctx) == (True, Risk.send)
        # the sync helper cannot await an async risk_fn: it assumes at least send
        assert s.approvals.needs_approval(device_take_photo, sid, arguments={}, ctx=ctx)
        assert not s.approvals.needs_approval(device_take_photo, sid)  # no arguments: static risk read

        @tool("plain_send", risk=Risk.send)
        async def plain_send(ctx: ToolContext) -> str:
            """Send."""
            return ""

        from sentient.tools.base import describe_call
        assert await describe_call(plain_send, {}, ctx, Risk.send) == {"risk_label": "Sends", "target": None}
    finally:
        await s.stop()


async def test_steer_messages_are_marked_and_package_schemas_listed(config, isolated_home):
    from sentient.store.db import PACKAGE_SCHEMAS

    assert {"nodes", "memory", "channels"} <= set(PACKAGE_SCHEMAS)
    llm = FakeProvider(replies=[[tool_call("current_datetime")], "ok"])
    s = await start(config, isolated_home, llm, "marked")
    try:
        sid = await s.store.create_session(channel="cli")
        async for ev in s.agent.run_turn(sid, "time?", channel="cli"):
            if isinstance(ev, ToolCallEvent):
                s.agent.steer(sid, "in UTC please")
        rows = await s.store.recent_messages(sid, 20)
        assert [(r["role"], r["interjection"]) for r in rows if r["role"] == "user"] == [("user", False), ("user", True)]
    finally:
        await s.stop()


async def test_messaging_and_phone_channels(config, isolated_home):
    base = dict(snapshot={}, facts=[], skills_index="", assistant_name="Sentient", user_name="Sarthak", timezone="UTC")
    assert "chatting in Telegram" in build_system_prompt(**base, channel="telegram")
    assert "No wide tables" in build_system_prompt(**base, channel="discord")
    llm = FakeProvider(replies=["Hi."])
    s = await start(config, isolated_home, llm, "phone")
    try:
        sid = await s.store.create_session(channel="phone")
        async for _ in s.agent.run_turn(sid, "hello", channel="phone"):
            pass
        assert llm.calls[-1]["role"] == "voice"
        assert "speaking through the user's phone" in llm.calls[-1]["messages"][0]["content"]
    finally:
        await s.stop()


async def test_allow_session_never_covers_escalated_risk():
    from sentient.agent.approvals import ApprovalBroker
    from sentient.config.schema import ApprovalsConfig

    @tool("web_click", risk=Risk.write, risk_fn=lambda a, c: Risk.send if a.get("order") else None)
    async def web_click(ctx: ToolContext, order: bool = False) -> str:
        """Click."""
        return ""

    @tool("send_mail", risk=Risk.send)
    async def send_mail(ctx: ToolContext) -> str:
        """Send."""
        return ""

    broker = ApprovalBroker(ApprovalsConfig(mode="ask"))

    async def allow_session(t, risk):
        broker.create("a")
        broker.resolve("a", "allow_session")
        await broker.wait("a", "s1", t.name, risk)

    await allow_session(web_click, Risk.send)  # the user allowed a "Place order" click for this chat
    assert broker.needs_approval(web_click, "s1", Risk.send)  # the next escalated click still asks
    assert not broker.needs_approval(web_click, "s1", Risk.write)  # ordinary clicks do not
    await allow_session(send_mail, Risk.send)
    assert not broker.needs_approval(send_mail, "s1", Risk.send)  # declared send risk: allowance applies
    assert broker.needs_approval(send_mail, "s2", Risk.send)


async def test_skills_rechecked_after_services_register_plugins(config, isolated_home):
    from sentient import paths

    @tool("late_tool", risk=Risk.read)
    async def late_tool(ctx: ToolContext) -> str:
        """Late."""
        return ""

    s = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "late.db", enable_background=False)

    async def sandbox_start():
        s.registry.register(make_plugin("late", late_tool))

    s.sandbox.start = sandbox_start
    paths.ensure_layout()
    s.skills.write("late-skill", "Needs a late plugin", "## Procedure\n1. Go.", author="user",
                   requires_tools=["late"], pending=False)
    s.skills.write("late-tool-skill", "Needs a late tool", "## Procedure\n1. Go.", author="user",
                   requires_tools=["late_tool"], pending=False)
    await s.start()
    try:
        assert s.skills.get("late-skill") is not None and s.skills.get("late-tool-skill") is not None
    finally:
        await s.stop()
