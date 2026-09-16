"""Real qwen3:8b called result(...) without importing it; scripts get tools and result for free."""

from sentient.app import SentientApp
from tests.conftest import FakeProvider


async def test_result_and_tools_work_without_import(config):
    config.sandbox.backend = "process"
    app = await SentientApp(config, llm=FakeProvider(), enable_background=False).start()
    try:
        res = await app.sandbox.run("result(50 * 51 * 101 // 6)")
        assert res["ok"], res
        assert res["result"] == 42925

        res = await app.sandbox.run("now = tools.current_datetime()\nresult(bool(now))")
        assert res["ok"], res
        assert res["result"] is True and res["tool_calls"] == 1

        res = await app.sandbox.run("from sentient_tools import tools, result\nresult(1 + 1)")
        assert res["ok"] and res["result"] == 2
    finally:
        await app.stop()
