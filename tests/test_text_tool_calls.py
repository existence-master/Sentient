"""Real qwen3:8b sometimes writes a tool call out as text instead of making it (#532): the call never ran and the
user saw the raw call. The loop now holds such text back while it streams, runs a written-out call of a known tool
through the usual path (so risk rules and approvals apply), nudges once when it can't run, and leaves code in a
normal prose answer alone."""

import pytest

from sentient.agent import text_calls
from sentient.app import SentientApp
from sentient.llm.events import ApprovalRequest, Done, TextDelta, ToolCallEvent, ToolResultEvent
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call

FACT = "Maya runs every Sunday at 6 a.m."
CALL = '{"name": "memory_remember", "arguments": {"fact": "Maya runs every Sunday at 6 a.m."}}'
RAW_BITS = ('"arguments"', "<tool_call>", "memory_remember", '"fact"', "inside content")

SHAPES = {
    "bare json": CALL,
    "tool_call tags": f"<tool_call>\n{CALL}\n</tool_call>",
    "code fence": f"```json\n{CALL}\n```",
    "call with a json object": f'memory_remember({{"fact": "{FACT}"}})',
    "call with keywords": f'memory_remember(fact="{FACT}")',
    "openai shape": '{"type": "function", "function": {"name": "memory_remember", "arguments": "{\\"fact\\": '
                    f'\\"{FACT}\\"}}"}}}}',
    # the exact reply real qwen3:8b gave while #531 was tested: a sentence, then the call in a made-up marker
    "qwen marker after a sentence": "I'll note your Sunday run. \n\n[inside content CODE\n"
                                    '{"type": "memory_remember", "arguments": {"fact": "Maya runs every Sunday at 6 a.m."}}'
                                    "\n[end of inside content CODE]",
    "a claim after the call": f"{CALL}\nDone, I saved that!",
}


class RecordingProvider(FakeProvider):
    """Keeps a copy of each call's message contents (the loop removes its nudge from the list afterwards)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen: list[list[str]] = []

    async def stream(self, role, messages, tools=None, *, model=None):
        self.seen.append([str(m.get("content")) for m in messages])
        async for chunk in super().stream(role, messages, tools, model=model):
            yield chunk


async def _turn(config, replies, text="Keep in mind that I run every Sunday at 6 a.m.", setup=None, on_event=None):
    llm = RecordingProvider(replies=replies)
    app = await SentientApp(config, llm=llm, enable_background=False).start()
    try:
        if setup is not None:
            setup(app)
        sid = await app.store.create_session(channel="desktop")
        events = []
        async for ev in app.agent.run_turn(sid, text):
            events.append(ev)
            if on_event is not None:
                await on_event(app, ev)
        rows = await app.store.recent_messages(sid, 50)
        memories = [m["content"] for m in await app.memory.list_facts()]
        return llm, events, rows, memories
    finally:
        await app.stop()


def _shown(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def _done(events) -> Done:
    return next(e for e in events if isinstance(e, Done))


def _nudged(llm: RecordingProvider) -> int:
    """How many nudges about a written-out call the model was sent."""
    return max((sum(1 for c in contents if c.startswith("You wrote the ")) for contents in llm.seen), default=0)


@pytest.mark.parametrize("reply", SHAPES.values(), ids=SHAPES.keys())
async def test_a_written_out_call_runs_as_a_real_call(config, reply):
    llm, events, rows, memories = await _turn(config, [reply, "Noted, I'll keep Sunday mornings free."])
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert [(c.name, c.arguments) for c in calls] == [("memory_remember", {"fact": FACT})]
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert not result.is_error, result.result
    assert any(FACT in m for m in memories)
    shown = _shown(events)
    assert not any(bit in shown for bit in RAW_BITS), shown  # the raw call never reaches the user
    assert "Done, I saved that!" not in shown  # a claim written before the call ran is dropped with it
    assert _done(events).content == "Noted, I'll keep Sunday mornings free."
    # the transcript keeps it as a real call, so the model reads it back as one
    call_rows = [r for r in rows if r["role"] == "assistant" and r.get("tool_calls")]
    assert len(call_rows) == 1 and not any(bit in (call_rows[0]["content"] or "") for bit in RAW_BITS)
    assert _nudged(llm) == 0


async def test_the_sentence_before_a_written_call_stays(config):
    _, events, rows, _ = await _turn(config, [SHAPES["qwen marker after a sentence"], "Got it."])
    assert _shown(events).startswith("I'll note your Sunday run.")
    call_row = next(r for r in rows if r["role"] == "assistant" and r.get("tool_calls"))
    assert call_row["content"] == "I'll note your Sunday run."


PROSE = {
    "json config": 'Here is a config you can use:\n```json\n{"name": "my-app", "version": 2}\n```\nSave it as app.json.',
    "python code": "Try this:\n```python\nprint(sorted(items))\n```\nIt prints the list in order.",
    "a mention": "I used memory_remember (the memory tool) to keep that.",
    "a long answer with an example call": "Sentient's memory works like this. " * 15
                                          + f"\n```json\n{CALL}\n```\nThat is what a call looks like.",
    "an unknown tool": '{"name": "book_flight", "arguments": {"to": "Pune"}}',
}


@pytest.mark.parametrize("reply", PROSE.values(), ids=PROSE.keys())
async def test_code_in_a_normal_answer_is_not_a_call(config, reply):
    llm, events, _, _ = await _turn(config, [reply, "unused"], text="Show me an example")
    assert not [e for e in events if isinstance(e, ToolCallEvent)]
    assert _shown(events) == reply and _done(events).content == reply
    assert len(llm.seen) == 1 and _nudged(llm) == 0


async def test_an_unknown_tool_in_tool_call_tags_is_nudged_not_shown(config):
    reply = '<tool_call>{"name": "book_flight", "arguments": {"to": "Pune"}}</tool_call>'
    llm, events, rows, _ = await _turn(config, [reply, "I can't book flights, sorry."])
    assert not [e for e in events if isinstance(e, ToolCallEvent)]
    assert _nudged(llm) == 1
    assert "book_flight" in llm.seen[1][-1]
    assert _shown(events) == "I can't book flights, sorry."
    assert all("You wrote the " not in (r.get("content") or "") for r in rows)  # the nudge is never saved


async def test_a_call_that_does_not_fit_its_tool_gets_one_nudge(config):
    bad = '{"name": "memory_remember", "arguments": {"note": "Maya runs on Sundays"}}'  # no such argument
    llm, events, _, memories = await _turn(config, [bad, [tool_call("memory_remember", fact=FACT)], "Saved."])
    assert _nudged(llm) == 1
    assert [e.name for e in events if isinstance(e, ToolCallEvent)] == ["memory_remember"]
    assert any(FACT in m for m in memories)
    assert not any(bit in _shown(events) for bit in RAW_BITS)
    assert _done(events).content == "Saved."


async def test_a_cut_off_call_gets_one_nudge_and_never_a_second(config):
    cut = '{"name": "memory_remember", "arguments": {"fact": "Maya runs'
    llm, events, _, _ = await _turn(config, [cut, cut, "unused"])
    assert _nudged(llm) == 1 and len(llm.seen) == 2
    assert not [e for e in events if isinstance(e, ToolCallEvent)]
    assert _done(events).content == text_calls.NOT_RUN_REPLY
    assert not any(bit in _shown(events) for bit in RAW_BITS)


async def test_a_written_out_risky_call_still_asks(config):
    sent: list[str] = []

    @tool("send_postcard", risk=Risk.send)
    async def send_postcard(ctx: ToolContext, to: str, text: str) -> dict:
        """Post a postcard to someone outside Sentient."""
        sent.append(text)
        return {"posted": text}

    class Postcards(ToolPlugin):
        id = "postcards"
        display_name = "Postcards"
        tools = [send_postcard]

    asked: list[ApprovalRequest] = []

    async def deny(app, ev):
        if isinstance(ev, ApprovalRequest):
            asked.append(ev)
            app.approvals.resolve(ev.approval_id, "deny")

    config.tools.approvals.mode = "ask"
    reply = '<tool_call>\n{"name": "send_postcard", "arguments": {"to": "Arjun", "text": "Hi"}}\n</tool_call>'
    _, events, _, _ = await _turn(
        config, [reply, "I didn't send it."], text="Send Arjun a postcard saying hi",
        setup=lambda app: app.registry.register(Postcards()), on_event=deny,
    )
    assert len(asked) == 1 and asked[0].name == "send_postcard" and asked[0].risk == "send"
    assert sent == []
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.is_error


async def test_a_real_call_with_a_written_copy_shows_neither_copy(config):
    reply_calls = [tool_call("memory_remember", fact=FACT)]

    class Both(RecordingProvider):
        """One answer with text that repeats the call and the real call itself."""

        async def stream(self, role, messages, tools=None, *, model=None):
            from sentient.llm.provider import StreamChunk, ToolCall

            if not self.seen:
                self.seen.append([])
                yield StreamChunk(text=f"Saving it.\n{CALL}", model="fake")
                yield StreamChunk(done=True, tool_calls=[ToolCall(**tc) for tc in reply_calls], model="fake")
                return
            async for chunk in super().stream(role, messages, tools, model=model):
                yield chunk

    llm = Both(replies=["Saved."])
    app = await SentientApp(config, llm=llm, enable_background=False).start()
    try:
        sid = await app.store.create_session(channel="desktop")
        events = [ev async for ev in app.agent.run_turn(sid, "remember my Sunday run")]
    finally:
        await app.stop()
    assert [e.name for e in events if isinstance(e, ToolCallEvent)] == ["memory_remember"]
    shown = _shown(events)
    assert shown.startswith("Saving it.") and shown.endswith("Saved.")
    assert not any(bit in shown for bit in RAW_BITS), shown


async def test_a_model_without_tools_streams_text_untouched(config):
    from sentient.agent.loop import LoopResult

    llm = FakeProvider(replies=[CALL])
    app = await SentientApp(config, llm=llm, enable_background=False).start()
    try:
        result = LoopResult()
        ctx = ToolContext(store=app.store, config=config, llm=llm)
        events = [e async for e in app.agent.run_loop([{"role": "user", "content": "hi"}], ctx, result=result,
                                                      tool_names=[])]
    finally:
        await app.stop()
    assert _shown(events) == CALL and result.text == CALL


# ---------------------------------------------------------------------- the helpers on their own
SCHEMAS = {"memory_remember": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]}}


def _stream(reply: str, step: int = 3) -> str:
    """What the loop would have shown of ``reply`` while it streamed in pieces of ``step`` characters."""
    acc, shown = "", 0
    for i in range(0, len(reply), step):
        acc += reply[i : i + step]
        shown = max(shown, text_calls.held_from(acc, SCHEMAS, text_calls.line_start(acc, shown)))
    return acc[:shown]


@pytest.mark.parametrize("reply", SHAPES.values(), ids=SHAPES.keys())
def test_a_written_call_is_held_back_while_it_streams(reply):
    shown = _stream(reply)
    assert not any(bit in shown for bit in RAW_BITS), shown


@pytest.mark.parametrize("reply", [p for k, p in PROSE.items() if "example" not in k])
def test_ordinary_text_streams_at_once(reply):
    assert _stream(reply) == reply


def test_find_reads_the_call_and_the_sentence_before_it():
    found = text_calls.find(SHAPES["qwen marker after a sentence"], SCHEMAS.get)
    assert found is not None and found.runnable
    assert found.calls == [("memory_remember", {"fact": FACT})]
    assert found.before == "I'll note your Sunday run."


def test_find_ignores_unknown_tools_and_mentions():
    assert text_calls.find(PROSE["an unknown tool"], SCHEMAS.get) is None
    assert text_calls.find(PROSE["a mention"], SCHEMAS.get) is None
    example = text_calls.find(PROSE["a long answer with an example call"], SCHEMAS.get)
    assert example is not None and not example.is_action and not example.runnable


# ---------------------------------------------------------------------- review follow-ups
async def test_a_call_after_prose_on_the_same_line_runs(config):
    reply = f"I will remember this: {CALL}"
    _, events, _, memories = await _turn(config, [reply, "Saved."])
    assert [e.name for e in events if isinstance(e, ToolCallEvent)] == ["memory_remember"]
    assert any(FACT in m for m in memories)
    shown = _shown(events)
    assert shown.startswith("I will remember this:") and not any(bit in shown for bit in RAW_BITS), shown


def test_a_name_last_call_stays_held_however_long():
    reply = '{"arguments": {"fact": "' + "Maya likes long walks. " * 30 + '"}, "name": "memory_remember"}'
    assert _stream(reply) == ""
    found = text_calls.find(reply, SCHEMAS.get)
    assert found is not None and found.runnable


def test_tool_call_tags_in_a_long_answer_are_not_run():
    reply = "Here is how tool calls look in Sentient. " * 15 + f"\n<tool_call>{CALL}</tool_call>\nThat's the format."
    found = text_calls.find(reply, SCHEMAS.get)
    assert found is not None and not found.is_action and not found.runnable


def test_an_unclosed_tag_runs_only_a_complete_call():
    complete = text_calls.find(f"<tool_call>\n{CALL}", SCHEMAS.get)
    assert complete is not None and complete.runnable
    cut = text_calls.find('<tool_call>\n{"name": "memory_remember", "arguments": {"fact": "Maya ru', SCHEMAS.get)
    assert cut is not None and cut.is_action and not cut.runnable  # nudged, never run with a cut-off value


def test_a_real_call_keeps_an_unrelated_example_in_the_text():
    reply = 'Saving it. For reference: {"name": "web_search", "arguments": {"query": "x"}}\n' + CALL
    schemas = {**SCHEMAS, "web_search": {"type": "object", "properties": {"query": {"type": "string"}}}}
    found = text_calls.find(reply, schemas.get, only={"memory_remember"})
    assert found is not None and "web_search" in found.cleaned and "memory_remember" not in found.cleaned


def test_id_and_type_stay_when_the_tool_takes_them():
    schemas = {"open_ticket": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}
    found = text_calls.find('{"name": "open_ticket", "id": "T-7"}', schemas.get)
    assert found is not None and found.calls == [("open_ticket", {"id": "T-7"})]


async def test_the_nudge_is_gone_once_the_model_calls_the_tool(config):
    bad = '{"name": "memory_remember", "arguments": {"note": "Maya runs on Sundays"}}'
    llm, _, _, _ = await _turn(config, [bad, [tool_call("memory_remember", fact=FACT)], "Saved."])
    assert any(c.startswith("You wrote the ") for c in llm.seen[1])
    assert not any(c.startswith("You wrote the ") for c in llm.seen[2])  # not sent again after the real call


def test_a_closing_tag_must_match_its_opening_tag():
    cut = '<tool_calls>\n{"name": "memory_remember", "arguments": {"fact": "Maya ru</tool_call>'
    found = text_calls.find(cut, SCHEMAS.get)
    assert found is not None and not found.runnable  # unclosed: never repaired into a cut-off fact


def test_an_unknown_type_does_not_hide_the_name_after_it():
    reply = '{"type": "tool_call", "name": "memory_remember", "arguments": {"fact": "Maya runs every Sunday at 6 a.m."}}'
    assert _stream(reply) == ""
    found = text_calls.find(reply, SCHEMAS.get)
    assert found is not None and found.runnable and found.calls == [("memory_remember", {"fact": FACT})]


def test_an_unclosed_tag_keeps_a_complete_call_followed_by_prose():
    found = text_calls.find(f"<tool_call>\n{CALL}\nI saved it.", SCHEMAS.get)
    assert found is not None and found.runnable and found.calls == [("memory_remember", {"fact": FACT})]
