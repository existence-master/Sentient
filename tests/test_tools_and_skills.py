from sentient import paths
from sentient.skills.loader import SkillLibrary
from sentient.tools.base import Risk, ToolContext, tool
from sentient.tools.registry import ToolRegistry


@tool("echo", risk=Risk.read)
async def echo(ctx: ToolContext, text: str, times: int = 1) -> str:
    """Repeat text."""
    return text * times


def test_tool_schema_from_signature():
    schema = echo.openai_schema()
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["description"] == "Repeat text."
    props = schema["function"]["parameters"]["properties"]
    assert props["text"]["type"] == "string"
    assert props["times"]["default"] == 1
    assert schema["function"]["parameters"]["required"] == ["text"]


async def test_tool_call_validates_arguments():
    ctx = ToolContext(store=None, config=None, llm=None)
    assert await echo.call(ctx, {"text": "ab", "times": "2"}) == "abab"


def test_builtin_registry_loads(isolated_home):
    reg = ToolRegistry()
    reg.load_builtin()
    names = {t.name for t in reg.tools()}
    assert {"memory_recall", "memory_remember", "current_datetime", "skill_view", "skill_save", "file_write"} <= names
    assert reg.get("memory_forget").risk == Risk.send
    assert any(p["id"] == "memory" for p in reg.catalog())


def test_skill_roundtrip_and_pending(isolated_home):
    paths.ensure_layout()
    lib = SkillLibrary()
    lib.write("weekly-review", "Compile the weekly review", "# Weekly review\n## Procedure\n1. ...", tags=["productivity"])
    lib.write("secret-sauce", "Learned procedure", "## Procedure\n1. x", author="assistant", pending=True)
    lib.reload()
    assert [s.name for s in lib.list()] == ["weekly-review"]
    assert [s.name for s in lib.list_pending()] == ["secret-sauce"]
    assert "weekly-review: Compile the weekly review" in lib.prompt_index()
    lib.approve_pending("secret-sauce")
    lib.reload()
    assert {s.name for s in lib.list()} == {"weekly-review", "secret-sauce"}
    assert lib.get("secret-sauce").author == "assistant"


def test_skill_requires_tools_filter(isolated_home):
    paths.ensure_layout()
    lib = SkillLibrary()
    lib.write("mail-triage", "Triage inbox", "## Procedure", requires_tools=["gmail"])
    lib.reload(available_plugins={"memory"})
    assert lib.get("mail-triage") is None
    lib.reload(available_plugins={"memory", "gmail"})
    assert lib.get("mail-triage") is not None
