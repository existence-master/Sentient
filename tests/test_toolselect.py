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


# ---------------------------------------------------------------------------- large MCP servers
COMPOSIO = {
    "COMPOSIO_SEARCH_TOOLS": "Discover the right tools across 500+ apps for a use case before executing them.",
    "COMPOSIO_MULTI_EXECUTE_TOOL": "Execute one or more app tools in parallel with their arguments.",
    "COMPOSIO_GET_TOOL_SCHEMAS": "Get the input schemas of tools by their slugs.",
    "COMPOSIO_MANAGE_CONNECTIONS": "Create or check the user's connections to apps.",
    "COMPOSIO_REMOTE_BASH_TOOL": "Run bash commands in a remote sandbox.",
    "COMPOSIO_REMOTE_WORKBENCH": "Run Python in a remote workbench for bulk processing.",
    "COMPOSIO_SUBMIT_FEEDBACK": "Send feedback about tools to Composio.",
    "COMPOSIO_WAIT_FOR_CONNECTIONS": "Wait until the user finishes connecting apps.",
    "COMPOSIO_MANAGE_SKILL": "Save or delete a reusable skill.",
    "COMPOSIO_SEARCH_SKILLS": "Find saved skills for a task.",
    "COMPOSIO_USE_SKILL": "Run a saved skill.",
}


def mcp_plugin(server: str, tools: dict[str, str]) -> ToolPlugin:
    """Like MCPServerPlugin: tools named mcp_<server>_<tool>, the server name as display name."""
    from sentient.integrations.mcp import mcp_tool_name, plugin_id_for

    made = []
    for name, desc in tools.items():

        @tool(mcp_tool_name(server, name), risk=Risk.write)
        async def _fn(ctx: ToolContext) -> str:
            return ""

        _fn.description = desc
        made.append(_fn)

    class P(ToolPlugin):
        id = plugin_id_for(server)
        display_name = server
        description = f"Tools from the external MCP server '{server}'."
        selection_hint = f"tools provided by the {server} MCP server"
        category = "utilities"

    P.tools = made
    return P()


def owner_with_composio(budget_local: int = 12) -> SimpleNamespace:
    o = owner(budget_local)
    o.registry.register(mcp_plugin("composio", COMPOSIO))
    return o


def composio_names(names: list[str]) -> set[str]:
    return {n.removeprefix("mcp_composio_composio_") for n in names if n.startswith("mcp_composio_")}


async def test_named_large_mcp_server_offers_its_entry_points():
    o = owner_with_composio(12)
    names = await ToolSelector(o).select(
        "Using Composio, list the events on my Google Calendar for today", model="ollama_chat/qwen3:8b"
    )
    assert names is not None and len(names) <= 12
    assert set(ALWAYS_TOOLS) <= set(names)
    picked = composio_names(names)
    assert {"search_tools", "multi_execute_tool"} <= picked
    assert len(picked) == 12 - len(ALWAYS_TOOLS)  # fills the room, not all 11


async def test_relevant_large_mcp_server_offers_best_tools_without_naming_it():
    o = owner_with_composio(10)
    names = await ToolSelector(o).select("search for the right tools to execute", model="ollama_chat/qwen3:8b")
    picked = composio_names(names)
    assert {"search_tools", "multi_execute_tool"} <= picked
    assert len(names) <= 10


async def test_large_mcp_server_leaves_room_for_other_relevant_plugins():
    o = owner_with_composio(12)
    names = await ToolSelector(o).select(
        "with composio, and also check the weather forecast", model="ollama_chat/qwen3:8b"
    )
    assert {"weather_current", "weather_forecast"} <= set(names)
    assert {"search_tools", "multi_execute_tool"} <= composio_names(names)
    assert len(names) <= 12


async def test_large_mcp_server_does_not_crowd_out_unrelated_turns():
    o = owner_with_composio(12)
    model = "ollama_chat/qwen3:8b"
    weather = await ToolSelector(o).select("will it rain tomorrow? check the weather forecast", model=model)
    assert {"weather_current", "weather_forecast"} <= set(weather)
    assert not composio_names(weather)
    assert await ToolSelector(o).select("hello there, how are you", model=model) == list(ALWAYS_TOOLS)


def test_search_and_execute_tools_stay_together():
    from sentient.agent.toolselect import _best_tools, _tool_scores

    p = mcp_plugin("composio", COMPOSIO)
    entry = {
        "id": p.id, "display_name": p.display_name,
        "tools": [{"name": t.name, "description": t.description} for t in p.tools],
    }
    # the message matches two discovery tools and no action tool: the execute tool still comes along
    scored = _tool_scores(entry, {"discover", "saved"})
    assert [round(s[0], 2) for s in scored[:1]] == [0.1] and round(scored[9][0], 2) == 0.1
    picked = _best_tools(entry["tools"], scored, 2)
    assert {n.removeprefix("mcp_composio_composio_") for n in picked} == {"search_tools", "multi_execute_tool"}


def test_named_plugins_match_display_and_server_names():
    from sentient.agent.toolselect import named_plugins

    catalog = [
        {"id": "mcp_composio", "display_name": "composio", "category": "utilities"},
        {"id": "gcalendar", "display_name": "Google Calendar", "category": "productivity"},
        {"id": "memory", "display_name": "Memory", "category": "core"},
    ]
    assert named_plugins("Using Composio, list my events", catalog) == {"mcp_composio"}
    assert named_plugins("what's on my google calendar? check memory", catalog) == {"gcalendar"}
    assert named_plugins("compositions of music", catalog) == set()


async def test_named_server_keeps_its_connect_tool_and_room_over_weaker_matches():
    o = owner_with_composio(9)  # room for 4 tools after the core ones
    o.registry.register(plugin("files", "files, folders", "file_list", "file_read", "file_write"))
    names = await ToolSelector(o).select(
        "Using Composio, list the events on my Google Calendar for today", model="ollama_chat/qwen3:8b"
    )
    # "list" matches the files plugin only weakly: the named server keeps the room, connect tool included
    assert composio_names(names) >= {"search_tools", "multi_execute_tool", "manage_connections"}
    assert "file_list" not in names


# ---------------------------------------------------------------------------- links and named tools (#252)
BROWSER = (
    "browser_open", "browser_snapshot", "browser_click", "browser_type", "browser_select", "browser_press",
    "browser_scroll", "browser_back", "browser_tabs", "browser_switch_tab", "browser_extract", "browser_screenshot",
    "browser_close",
)


def owner_with_browser(budget_local: int = 12) -> SimpleNamespace:
    o = owner(budget_local)
    o.registry.register(plugin("browser", "operate a website step by step", *BROWSER))
    o.registry.register(plugin("web", "open and read a specific page", "web_fetch"))
    return o


async def test_link_offers_web_fetch_next_to_the_big_browser_plugin():
    o = owner_with_browser(12)
    names = await ToolSelector(o).select(
        "read http://127.0.0.1:8123/recipe.html and tell me the recipe name", model="ollama_chat/qwen3:8b"
    )
    assert "web_fetch" in names and "browser_open" in names
    assert len(names) <= 12


async def test_a_tool_named_exactly_is_always_offered():
    o = owner_with_browser(7)  # no room beyond the core tools and one more
    model = "ollama_chat/qwen3:8b"
    names = await ToolSelector(o).select("Use web_fetch to read http://example.com and the weather forecast", model=model)
    assert "web_fetch" in names and len(names) <= 7
    o = owner_with_composio(7)
    names = await ToolSelector(o).select("call COMPOSIO_USE_SKILL with my skill", model=model)
    assert "mcp_composio_composio_use_skill" in names


def test_named_tools_need_the_exact_name():
    from sentient.agent.toolselect import named_tools

    catalog = [
        {"id": "web", "tools": [{"name": "web_fetch"}]},
        {"id": "mcp_composio", "tools": [{"name": "mcp_composio_composio_search_tools"}]},
        {"id": "weather", "tools": [{"name": "weather_current"}]},
    ]
    assert named_tools("Use web_fetch on this", catalog) == ["web_fetch"]
    assert named_tools("run COMPOSIO_SEARCH_TOOLS first", catalog) == ["mcp_composio_composio_search_tools"]
    assert named_tools("fetch the web page, current weather", catalog) == []
    assert named_tools("use web_fetcher", catalog) == []
