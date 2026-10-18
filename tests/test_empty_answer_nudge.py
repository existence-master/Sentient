"""Real qwen3:8b sometimes ended a round with only hidden thinking (empty reply) after a tool call.
The loop asks once for a reply instead of showing the user nothing, and if the reply is still empty it answers
from what the tools returned (#322)."""

from sentient.agent import findings
from sentient.agent.loop import EMPTY_ANSWER_NUDGE, FINAL_ANSWER_NUDGE
from sentient.app import SentientApp
from sentient.llm.events import Done
from tests.conftest import FakeProvider


def tc(tool_name: str, **arguments):
    return {"id": f"call_{tool_name}", "name": tool_name, "arguments": arguments}


class RecordingProvider(FakeProvider):
    """Keeps a copy of each call's message contents (the loop later removes the nudge from its list)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen: list[list[str]] = []

    async def stream(self, role, messages, tools=None, *, model=None):
        self.seen.append([str(m.get("content")) for m in messages])
        async for chunk in super().stream(role, messages, tools, model=model):
            yield chunk


async def _turn(config, replies):
    llm = RecordingProvider(replies=replies)
    app = await SentientApp(config, llm=llm, enable_background=False).start()
    try:
        sid = await app.store.create_session(channel="desktop")
        done = None
        async for ev in app.agent.run_turn(sid, "What time is it?"):
            if isinstance(ev, Done):
                done = ev
        rows = await app.store.recent_messages(sid, 50)
        return llm, done, rows
    finally:
        await app.stop()


def _nudged_calls(llm: RecordingProvider, nudge: str = FINAL_ANSWER_NUDGE) -> int:
    return sum(1 for contents in llm.seen if nudge in contents)


async def test_empty_reply_after_tool_is_nudged_once(config):
    llm, done, rows = await _turn(config, [[tc("current_datetime")], "", "It is 9:41."])
    assert done is not None and done.content == "It is 9:41."
    assert _nudged_calls(llm) == 1
    assert all(r.get("content") not in (EMPTY_ANSWER_NUDGE, FINAL_ANSWER_NUDGE) for r in rows)  # never persisted


async def test_still_empty_after_the_nudge_answers_from_the_tool_results(config):
    llm, done, rows = await _turn(config, [[tc("current_datetime")], "", ""])
    assert _nudged_calls(llm) == 1
    assert len(llm.seen) == 3  # no third model call
    assert done is not None and done.content.startswith(findings.INTRO)
    line = done.content.splitlines()[1]
    assert line.startswith("- Date & time: iso: ") and "weekday: " in line
    assert rows[-1]["role"] == "assistant" and rows[-1]["content"] == done.content  # the chat keeps it


async def test_empty_reply_without_tools_keeps_the_general_nudge_and_no_summary(config):
    llm, done, _ = await _turn(config, ["", ""])
    assert _nudged_calls(llm, EMPTY_ANSWER_NUDGE) == 1 and _nudged_calls(llm) == 0
    assert done is not None and done.content == ""  # nothing was looked up, so there is nothing to list


async def test_failed_tool_calls_are_not_listed(config):
    _, done, _ = await _turn(config, [[tc("no_such_tool")], "", ""])
    assert done is not None and done.content == ""


def test_summary_lists_what_each_tool_found():
    text = findings.summarize([
        ("memory_recall", "Memory", [{"id": 4, "fact": "Sapiens cost Rs. 599 last time", "similarity": 0.8}]),
        ("web_fetch", "Web", {"url": "http://shop.test/s", "content": "Sapiens   price:\n Rs. 549"}),
        ("task_notepad", "Notes", {"ok": True}),  # nothing readable: left out
        ("web_fetch", "Web", {"url": "http://shop.test/s", "content": "Sapiens price: Rs. 499"}),  # latest wins
    ])
    assert text.splitlines() == [
        findings.INTRO,
        "- Memory: Sapiens cost Rs. 599 last time",
        "- Web: Sapiens price: Rs. 499",
    ]


def test_summary_is_short_and_says_so_when_nothing_was_found():
    assert findings.summarize([("memory_recall", "Memory", [])]) == findings.NOTHING
    long = findings.summarize([("web_fetch", "Web", {"content": "word " * 500})])
    line = long.splitlines()[1]
    assert len(line) <= len("- Web: ") + findings.MAX_CHARS and line.endswith("...")
    rows = findings.summarize([("memory_recall", "Memory", [{"fact": f"fact {i}"} for i in range(9)])])
    assert "fact 4" in rows and "fact 5" not in rows and "(and 4 more)" in rows
