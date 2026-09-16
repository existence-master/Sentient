import pytest

from sentient.llm.provider import ProviderError
from sentient.tasks.jsonio import parse_json_tolerant
from tests.conftest import FakeProvider


def test_parse_json_tolerant_handles_corrupted_prefix_and_fences():
    assert parse_json_tolerant('{"{"name": "Haiku", "priority": 1}}') == {"name": "Haiku", "priority": 1}
    assert parse_json_tolerant('```json\n{"plan": []}\n```') == {"plan": []}
    assert parse_json_tolerant('<think>hmm {</think>Sure: {"a": 1} trailing') == {"a": 1}
    assert parse_json_tolerant('[{"x": 1}]') == [{"x": 1}]
    with pytest.raises(ValueError):
        parse_json_tolerant("no json here {")


class BrokenJsonModeProvider(FakeProvider):
    """Simulates Ollama JSON mode corrupting the reply; plain text mode works."""

    async def complete_json(self, role, messages, *, model=None):
        self.calls.append({"role": role, "messages": messages, "json": True})
        raise ProviderError("All models failed for role 'planner': Model did not return JSON: '{\"{\"name\"'")


async def test_planning_survives_malformed_json_mode(make_app):
    llm = BrokenJsonModeProvider()
    llm.text_replies = [
        '{"{"name": "Monsoon haiku", "description": "Write a haiku", "priority": 1, "schedule": {"type": "once", "run_at": null}}}',
        '```json\n{"name": "Monsoon haiku", "description": "Write it", "plan": [{"tool": "files", "description": "Save monsoon.txt"}]}\n```',
    ]
    app = await make_app(llm)
    task = await app.tasks.create_task("Write a haiku about the monsoon and save it to monsoon.txt")
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "approval_pending", task["error"]
    assert task["name"] == "Monsoon haiku"
    assert task["plan"] == [{"tool": "files", "description": "Save monsoon.txt"}]


class WrongShapeProvider(FakeProvider):
    """JSON mode returns valid JSON of the wrong shape (seen with qwen3:4b: a list of strings)."""

    async def complete_json(self, role, messages, *, model=None):
        self.calls.append({"role": role, "messages": messages, "json": True})
        return ["A task to compose a short haiku."]


async def test_wrong_json_shape_falls_back_to_text(make_app):
    llm = WrongShapeProvider()
    llm.text_replies = [
        '{"name": "Monsoon haiku", "description": "d", "priority": 1, "schedule": {"type": "once", "run_at": null}}',
        '{"plan": [{"tool": "files", "description": "Save monsoon.txt"}]}',
    ]
    app = await make_app(llm)
    task = await app.tasks.create_task("Write a haiku")
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "approval_pending", task["error"]
    assert task["plan"][0]["tool"] == "files"


async def test_ollama_roles_ask_for_text_first():
    from sentient.tasks.jsonio import complete_json_object

    class OllamaLike(FakeProvider):
        def model_for(self, role):
            return "ollama_chat/qwen3:4b"

        async def complete_json(self, role, messages, *, model=None):
            raise AssertionError("JSON mode should not be used first for Ollama")

    llm = OllamaLike()
    llm.text_replies = ['{\n{\n  "name": "x", "plan": []}']
    assert await complete_json_object(llm, "planner", []) == {"name": "x", "plan": []}


def test_key_hints_prefer_the_whole_object_over_inner_fragments():
    # missing comma: only inner objects decode with raw_decode; json_repair (or hints) recover the outer one
    broken = '{"name": "Haiku" "description": "d", "schedule": {"type": "once", "run_at": null}}'
    parsed = parse_json_tolerant(broken, (dict,), keys=("name", "schedule"))
    assert "schedule" in parsed
    step_only = 'Plan: {"tool": "files", "description": "save"} and {"name": "x", "steps": [{"tool": "time", "description": "d"}]}'
    assert parse_json_tolerant(step_only, (dict,), keys=("plan", "steps"))["name"] == "x"


async def test_planner_steps_alias_is_accepted(make_app):
    llm = FakeProvider(json_replies=[
        {"name": "Digest", "description": "d", "priority": 1, "schedule": {"type": "once", "run_at": None}},
        {"name": "Digest", "description": "d", "steps": [{"tool": "time", "description": "Get the date"}]},
    ])
    app = await make_app(llm)
    task = await app.tasks.create_task("Tell me the date")
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "approval_pending", task["error"]
    assert task["plan"] == [{"tool": "time", "description": "Get the date"}]


async def test_real_provider_outage_still_errors(make_app):
    class DownProvider(FakeProvider):
        async def complete_json(self, role, messages, *, model=None):
            raise ProviderError("All models failed for role 'planner': connection refused")

    app = await make_app(DownProvider())
    task = await app.tasks.create_task("Write a haiku")
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "error" and "unavailable" in task["error"]
