import pytest

from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from sentient.tools.registry import ToolRegistry


def make_plugin(pid: str, *names: str) -> ToolPlugin:
    tools = []
    for n in names:

        @tool(n, risk=Risk.read)
        async def _fn(ctx: ToolContext, q: str = "") -> str:
            """A test tool."""
            return q

        tools.append(_fn)

    class P(ToolPlugin):
        id = pid
        display_name = pid.title()

    P.tools = tools
    return P()


def test_register_duplicate_is_atomic():
    reg = ToolRegistry()
    reg.register(make_plugin("a", "alpha", "beta"))
    with pytest.raises(ValueError):
        reg.register(make_plugin("b", "gamma", "beta"))
    # the failed plugin left nothing behind
    assert reg.plugin("b") is None and not reg.has_tool("gamma")


def test_unregister_removes_plugin_and_tools():
    reg = ToolRegistry()
    reg.register(make_plugin("a", "alpha"))
    reg.register(make_plugin("mcp_x", "mcp_x_one", "mcp_x_two"))
    assert reg.unregister("mcp_x") is not None
    assert reg.plugin("mcp_x") is None
    assert {t.name for t in reg.tools()} == {"alpha"}
    # re-registering the same names now works
    reg.register(make_plugin("mcp_x", "mcp_x_one"))
    assert reg.has_tool("mcp_x_one")


def test_hidden_plugins_not_offered_but_still_resolvable():
    reg = ToolRegistry()
    reg.register(make_plugin("gmail", "gmail_search"))
    reg.register(make_plugin("time", "current_datetime"))
    reg.set_hidden("gmail", True)
    offered = {s["function"]["name"] for s in reg.openai_schemas()}
    assert offered == {"current_datetime"}
    assert reg.openai_schemas(["gmail_search"]) == []
    assert reg.get("gmail_search") is not None  # an explicit call still reaches the tool
    assert [p["id"] for p in reg.catalog()] == ["time"]
    assert {p["id"] for p in reg.catalog(include_hidden=True)} == {"gmail", "time"}
    assert {t.name for t in reg.tools(include_hidden=True)} == {"gmail_search", "current_datetime"}
    reg.set_hidden("gmail", False)
    assert {s["function"]["name"] for s in reg.openai_schemas()} == {"gmail_search", "current_datetime"}


def test_replace_and_disabled():
    reg = ToolRegistry(disabled=["beta", "blocked"])
    reg.register(make_plugin("a", "alpha", "beta"))
    assert not reg.has_tool("beta")
    reg.register(make_plugin("blocked", "x"))
    assert reg.plugin("blocked") is None
    reg.register(make_plugin("a", "alpha2"), replace=True)
    assert {t.name for t in reg.tools()} == {"alpha2"}
