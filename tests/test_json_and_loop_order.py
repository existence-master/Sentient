from sentient.agent.loop import LoopResult
from sentient.app import SentientApp
from sentient.llm.events import ToolResultEvent
from sentient.llm.provider import parse_json_loose
from sentient.tools.builtin.time_tool import resolve_tz
from tests.conftest import FakeProvider, tool_call


def test_parse_json_loose_recovers_corrupted_ollama_json():
    assert parse_json_loose('{"{"name": "Haiku", "priority": 1}') == {"name": "Haiku", "priority": 1}
    assert parse_json_loose("Sure! Here you go:\n```json\n{\"facts\": [\"a\"]}\n```") == {"facts": ["a"]}
    assert parse_json_loose("<think>hmm {not json}</think>[1, 2]") == [1, 2]


def test_parse_json_loose_prefers_expected_keys():
    text = 'first {"note": "x"} then {"plan": [{"tool": "files"}], "name": "n"}'
    assert parse_json_loose(text, expect_keys=("plan",))["name"] == "n"


def test_resolve_auto_timezone_is_iana_when_possible():
    tz = resolve_tz("auto")
    assert tz is not None
    assert "/" in str(tz) or str(tz) == "UTC" or not hasattr(tz, "key")


async def test_tool_message_recorded_before_result_event(config, isolated_home):
    llm = FakeProvider(replies=[[tool_call("current_datetime")], "done"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "order.db").start()
    try:
        messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "time?"}]
        ctx = s.agent.tool_context(None, "cli")
        result = LoopResult()
        async for ev in s.agent.run_loop(messages, ctx, result=result, use_approvals=False):
            if isinstance(ev, ToolResultEvent):
                assert messages[-1]["role"] == "tool" and messages[-1]["tool_call_id"] == ev.call_id
        assert not result.hit_step_limit
    finally:
        await s.stop()


async def test_step_limit_flag(config, isolated_home):
    llm = FakeProvider(replies=[[tool_call("current_datetime")]] * 3)
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "limit.db").start()
    try:
        messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "loop"}]
        result = LoopResult()
        async for _ in s.agent.run_loop(messages, s.agent.tool_context(None, "cli"), result=result,
                                        use_approvals=False, max_rounds=2):
            pass
        assert result.hit_step_limit
    finally:
        await s.stop()


def test_reasoning_none_only_sent_to_local_models():
    from sentient.config.schema import SentientConfig
    from sentient.llm.provider import LiteLLMProvider

    cfg = SentientConfig()
    prov = LiteLLMProvider(cfg)
    assert prov._kwargs_for("ollama_chat/qwen3:8b", "fast")["reasoning_effort"] == "none"
    assert "reasoning_effort" not in prov._kwargs_for("anthropic/claude-sonnet-5", "fast")
    assert prov._kwargs_for("anthropic/claude-sonnet-5", "primary")["reasoning_effort"] == "medium"
    assert cfg.models.reasoning["planner"] == "none" and cfg.models.reasoning["executor"] == "none"


async def test_interrupt_stops_between_calls_never_inside_one(config, isolated_home):
    """``interrupt`` (a paused task, #408) ends the loop at the next safe point: the call that ran keeps its result,
    the next call of the same answer and the next model call never happen."""
    writes = [{"id": f"call_{n}", "name": "file_write", "arguments": {"name": f"{n}.txt", "content": "x"}} for n in (1, 2)]
    llm = FakeProvider(replies=[writes, "done"])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "interrupt.db").start()
    try:
        messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "write"}]
        result = LoopResult()
        paused = {"now": False}
        async for ev in s.agent.run_loop(messages, s.agent.tool_context(None, "cli"), result=result,
                                        use_approvals=False, interrupt=lambda: paused["now"]):
            if isinstance(ev, ToolResultEvent):
                paused["now"] = True  # paused while the first call ran
        assert result.interrupted and len(llm.calls) == 1
        assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["call_1"]
        assert (isolated_home / "files" / "1.txt").exists() and not (isolated_home / "files" / "2.txt").exists()
    finally:
        await s.stop()
