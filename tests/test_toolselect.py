from types import SimpleNamespace

from sentient.agent.toolselect import ALWAYS_TOOLS, ToolSelector, is_local_model
from sentient.config.schema import SentientConfig
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.registry import ToolRegistry
from tests.conftest import FakeProvider


def plugin(pid: str, hint: str, *names: str) -> ToolPlugin:
    tools = []
    for n in names:

        @tool(n, risk=Risk.read)
        async def _fn(ctx: ToolContext) -> str:
            """A tool."""
            return ""

        tools.append(_fn)

    class P(ToolPlugin):
        id = pid
        display_name = pid.title()
        description = f"{pid} integration"
        selection_hint = hint

    P.tools = tools
    return P()


class KeywordOnlyProvider(FakeProvider):
    """No embeddings available: selection falls back to keyword scoring, which is deterministic."""

    async def embed(self, texts, *, model=None):
        raise RuntimeError("embedding model unavailable")


def owner(budget_local: int = 9) -> SimpleNamespace:
    reg = ToolRegistry()
    reg.register(plugin("core", "clock memory", *ALWAYS_TOOLS))
    reg.register(plugin("weather", "weather, forecast, temperature, rain", "weather_current", "weather_forecast"))
    reg.register(plugin("gmail", "email, inbox, mail, send email", "gmail_search", "gmail_send", "gmail_read"))
    reg.register(plugin("github", "code, repositories, issues, pull requests", "github_issues", "github_prs"))
    reg.register(plugin("news", "news, headlines", "news_top", "news_search"))
    cfg = SentientConfig()
    cfg.chat.max_tools_local = budget_local
    return SimpleNamespace(registry=reg, llm=KeywordOnlyProvider(), config=cfg)


def test_local_detection():
    assert is_local_model("ollama_chat/qwen3:8b") and is_local_model("lm_studio/foo")
    assert not is_local_model("anthropic/claude-sonnet-5")


async def test_relevant_plugin_selected_within_budget():
    o = owner(budget_local=9)
    names = await ToolSelector(o).select("will it rain tomorrow? check the weather forecast", model="ollama_chat/qwen3:8b")
    assert names is not None and len(names) <= 9
    assert set(ALWAYS_TOOLS) <= set(names)
    assert {"weather_current", "weather_forecast"} <= set(names)
    assert "gmail_send" not in names


async def test_recent_plugins_stay_available():
    o = owner(budget_local=8)
    names = await ToolSelector(o).select("and the other one?", model="ollama_chat/qwen3:8b", recent_plugins={"github"})
    assert {"github_issues", "github_prs"} <= set(names)


async def test_cloud_and_all_mode_offer_everything():
    o = owner(budget_local=9)
    assert await ToolSelector(o).select("weather", model="anthropic/claude-sonnet-5") is None
    o.config.chat.tool_selection = "all"
    assert await ToolSelector(o).select("weather", model="ollama_chat/qwen3:8b") is None


async def test_under_budget_offers_everything():
    o = owner(budget_local=50)
    assert await ToolSelector(o).select("hello", model="ollama_chat/qwen3:8b") is None


async def test_small_talk_offers_only_core_tools():
    o = owner(budget_local=9)
    names = await ToolSelector(o).select("hello there, how are you", model="ollama_chat/qwen3:8b")
    assert names == list(ALWAYS_TOOLS)
