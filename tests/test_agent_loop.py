
from sentient.app import SentientApp
from sentient.llm.events import ApprovalRequest, Done, TextDelta, ToolResultEvent
from tests.conftest import FakeProvider, tool_call


async def _collect(agent, sid, text, on_event=None):
    events = []
    async for ev in agent.run_turn(sid, text, channel="cli"):
        events.append(ev)
        if on_event:
            await on_event(ev)
    return events


async def test_plain_reply_persists_and_streams(config, isolated_home):
    llm = FakeProvider(replies=["Hello Sarthak!"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "a.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "hi")
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Hello Sarthak!"
        assert isinstance(events[-1], Done) and events[-1].content == "Hello Sarthak!"
        msgs = await s.store.recent_messages(sid, 10)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        system = llm.calls[0]["messages"][0]["content"]
        assert "Sarthak" in system and "## Working rules" in system
        assert llm.calls[0]["tools"], "tools must be offered to the model"
    finally:
        await s.stop()


async def test_tool_round_trip(config, isolated_home):
    llm = FakeProvider(replies=[[tool_call("current_datetime")], "It is now."])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "b.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "what time is it")
        kinds = [type(e).__name__ for e in events]
        assert "ToolCallEvent" in kinds and "ToolResultEvent" in kinds
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert not res.is_error and "iso" in res.result
        msgs = await s.store.recent_messages(sid, 10)
        assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
        # the second model call must see the tool result
        second = llm.calls[1]["messages"]
        assert second[-1]["role"] == "tool" and second[-2]["tool_calls"]
    finally:
        await s.stop()


async def test_approval_gate_deny_and_allow(config, isolated_home):
    from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool

    sent: list[str] = []

    @tool("send_postcard", risk=Risk.write)
    async def send_postcard(ctx: ToolContext, text: str) -> dict:
        """Post a postcard to someone outside Sentient."""
        sent.append(text)
        return {"posted": text}

    class Postcards(ToolPlugin):
        id = "postcards"
        display_name = "Postcards"
        tools = [send_postcard]

    config.tools.approvals.mode = "ask"
    llm = FakeProvider(
        replies=[
            [tool_call("send_postcard", text="x")], "declined noted",
            [tool_call("send_postcard", text="y")], "posted",
        ]
    )
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "c.db").start()
    try:
        s.registry.register(Postcards())
        sid = await s.store.create_session(channel="cli")

        async def deny(ev):
            if isinstance(ev, ApprovalRequest):
                assert ev.risk == "write"
                s.approvals.resolve(ev.approval_id, "deny")

        events = await _collect(s.agent, sid, "send a postcard", deny)
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error and "declined" in str(res.result)
        assert sent == []

        async def allow(ev):
            if isinstance(ev, ApprovalRequest):
                s.approvals.resolve(ev.approval_id, "allow_session")

        events = await _collect(s.agent, sid, "send another", allow)
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert not res.is_error and sent == ["y"]
        # session allowance now skips the prompt for the same tool
        assert not s.approvals.needs_approval(s.registry.get("send_postcard"), sid)
    finally:
        await s.stop()


async def test_internal_tools_do_not_prompt_in_ask_mode(config, isolated_home):
    config.tools.approvals.mode = "ask"
    llm = FakeProvider(replies=[[tool_call("file_write", name="note.txt", content="hi")], "saved"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "internal.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "save a note")
        assert not any(isinstance(e, ApprovalRequest) for e in events)
        assert (isolated_home / "files" / "note.txt").read_text() == "hi"
        for name in ("memory_remember", "file_write", "skill_save"):
            tool = s.registry.get(name)
            assert tool is not None and tool.internal, name
            assert not s.approvals.needs_approval(tool, sid)
        # irreversible internal actions still ask, and "always" asks for everything
        assert s.approvals.needs_approval(s.registry.get("memory_forget"), sid)
        s.approvals.config.mode = "always"
        assert s.approvals.needs_approval(s.registry.get("file_write"), sid)
    finally:
        await s.stop()

async def test_memory_remember_via_tool_then_recalled_in_prompt(config, isolated_home):
    # this test checks that recalled facts reach the prompt, not the similarity threshold
    # (the fake bag-of-words embedder scores paraphrases low)
    config.memory.min_similarity = 0.0
    llm = FakeProvider(
        replies=[[tool_call("memory_remember", fact="Sarthak's sister lives in Pune")], "Got it.", "Pune."],
        json_replies=[{"action": "ADD", "fact_id": None, "content": "Sarthak's sister lives in Pune",
                       "memory_type": "long-term", "duration": None, "topics": ["family"]}],
    )
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "d.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        await _collect(s.agent, sid, "my sister lives in Pune")
        assert await s.memory.count() == 1
        await _collect(s.agent, sid, "where does my sister live")
        system = llm.calls[-1]["messages"][0]["content"]
        assert "Sarthak's sister lives in Pune" in system, "recalled facts must be injected into the prompt"
    finally:
        await s.stop()


async def test_background_extraction_runs(config, isolated_home):
    config.memory.extract_after_turn = True
    llm = FakeProvider(
        replies=["Nice, noted."],
        json_replies=[{"facts": ["Sarthak drinks black coffee"]},
                      {"action": "ADD", "fact_id": None, "content": "Sarthak drinks black coffee",
                       "memory_type": "long-term", "duration": None, "topics": ["preferences"]}],
    )
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "e.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        await _collect(s.agent, sid, "I only ever drink black coffee, no sugar")
        await s.agent.drain()
        assert await s.memory.count() == 1
    finally:
        await s.stop()


async def test_voice_channel_uses_voice_role(config, isolated_home):
    llm = FakeProvider(replies=["Sure thing."])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "v.db").start()
    try:
        sid = await s.store.create_session(channel="voice")
        async for _ in s.agent.run_turn(sid, "what's up", channel="voice"):
            pass
        assert llm.calls[-1]["role"] == "voice"
        assert "You are speaking out loud" in llm.calls[-1]["messages"][0]["content"]
    finally:
        await s.stop()


async def test_model_without_tool_support_degrades_gracefully(config, isolated_home):
    from sentient.llm.events import Error
    from sentient.llm.provider import ProviderError

    class NoToolsProvider(FakeProvider):
        async def stream(self, role, messages, tools=None, *, model=None):
            if tools:
                self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
                raise ProviderError("Ollama_chatException - invalid character '<' looking for beginning of value")
            async for chunk in super().stream(role, messages, None, model=model):
                yield chunk

    llm = NoToolsProvider(replies=["It is sunny."])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "nt.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "weather?")
        errors = [e for e in events if isinstance(e, Error)]
        assert len(errors) == 1 and errors[0].recoverable and "can't use tools" in errors[0].message
        assert isinstance(events[-1], Done) and events[-1].content == "It is sunny."
        assert llm.calls[0]["tools"] and llm.calls[-1]["tools"] is None
    finally:
        await s.stop()


async def test_stopped_reply_is_saved(config, isolated_home):
    llm = FakeProvider(replies=["Once upon a time there was a very long story"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "stop.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        gen = s.agent.run_turn(sid, "tell me a story", channel="cli")
        async for ev in gen:
            if isinstance(ev, TextDelta):
                break
        await gen.aclose()
        msgs = await s.store.recent_messages(sid, 10)
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-1]["content"].startswith("Once") and "(stopped)" in msgs[-1]["content"]
    finally:
        await s.stop()


async def test_malformed_tool_arguments_get_clear_error(config, isolated_home):
    bad = [{"id": "c1", "name": "memory_recall", "arguments": {"_raw": "{query: sister"}}]
    llm = FakeProvider(replies=[bad, "ok"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "raw.db").start()
    try:
        sid = await s.store.create_session(channel="cli")
        events = await _collect(s.agent, sid, "who is my sister")
        res = next(e for e in events if isinstance(e, ToolResultEvent))
        assert res.is_error and "not valid JSON" in res.result["error"]
        assert res.result["schema"]["properties"]["query"]["type"] == "string"
    finally:
        await s.stop()
