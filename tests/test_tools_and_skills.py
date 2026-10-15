from sentient import paths
from sentient.skills.loader import SKILL_SECTIONS, SkillLibrary, parse_skill_file
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


def test_new_skills_get_the_six_field_shape_and_legacy_skills_still_load(isolated_home):
    paths.ensure_layout()
    lib = SkillLibrary()
    path = lib.write("new-skill", "A new procedure", "# New skill\n## Procedure\n1. Do it.")
    new = parse_skill_file(path)
    assert new is not None
    assert [f"## {section}" in new.body for section in SKILL_SECTIONS] == [True] * 6

    legacy_path = lib.root / "legacy-skill" / "SKILL.md"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        "---\nname: legacy-skill\ndescription: An older skill.\n---\n\n"
        "# Legacy skill\n## Procedure\n1. Keep working.\n",
        encoding="utf-8",
    )
    legacy = parse_skill_file(legacy_path)
    assert legacy is not None
    assert legacy.body == "# Legacy skill\n## Procedure\n1. Keep working."


def test_skill_section_headings_must_match_the_complete_name(isolated_home):
    paths.ensure_layout()
    lib = SkillLibrary()
    path = lib.write(
        "heading-boundaries",
        "Test section heading boundaries",
        "# Heading boundaries\n## Approval required\n- Ask first.\n## Output ##\n- A report.",
    )
    skill = parse_skill_file(path)
    assert skill is not None
    assert "## Approval required" in skill.body
    assert "\n## Approval\n" in skill.body
    assert skill.body.count("## Output") == 1


def test_skill_requires_tools_filter(isolated_home):
    paths.ensure_layout()
    lib = SkillLibrary()
    lib.write("mail-triage", "Triage inbox", "## Procedure", requires_tools=["gmail"])
    lib.reload(available_plugins={"memory"})
    assert lib.get("mail-triage") is None
    lib.reload(available_plugins={"memory", "gmail"})
    assert lib.get("mail-triage") is not None
