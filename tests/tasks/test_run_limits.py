"""A task run that loops or runs over its step, time, token or cost limit fails with a plain reason and the usual
"Task failed" notification; nothing stops silently (issue #133)."""

from __future__ import annotations

import asyncio

from sentient.llm.provider import StreamChunk, ToolCall
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider
from tests.tasks.conftest import RESULT, stream_calls


def read(name: str, n: int) -> list[dict]:
    return [{"id": f"call_{n}", "name": "file_read", "arguments": {"name": name}}]


class PricedProvider(FakeProvider):
    """Reports usage (and optionally a price) on every reply, tool calls included."""

    def __init__(self, *args, tokens: int = 0, cost: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens, self.cost = tokens, cost

    async def stream(self, role, messages, tools=None, *, model=None):
        self.calls.append({"role": role, "messages": messages, "tools": tools, "model": model})
        reply = self.replies.pop(0) if self.replies else "ok"
        usage = {"prompt_tokens": self.tokens, "completion_tokens": 0}
        calls = [ToolCall(**tc) for tc in reply] if isinstance(reply, list) else []
        if not calls:
            yield StreamChunk(text=reply, model="openai/gpt-test")
        yield StreamChunk(done=True, tool_calls=calls, usage=usage, model="openai/gpt-test", cost=self.cost)


async def run_task(make_app, llm, plan_tool: str = "files", plugin: ToolPlugin | None = None):
    app = await make_app(llm)
    if plugin is not None:
        app.registry.register(plugin)
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Read my notes", "description": "Read notes.txt", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None}, "plan": [{"tool": plan_tool, "description": "Read notes.txt"}],
        "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    failures = [n for n in await app.notifications.list() if n["payload"].get("event") == "run_failed"]
    return task, failures


def assert_failed(task: dict, failures: list[dict], reason: str) -> str:
    run = task["runs"][0]
    assert task["status"] == "error" and run["status"] == "error", run
    assert reason in run["error"]
    assert len(failures) == 1 and failures[0]["title"] == "Task failed" and reason in failures[0]["message"]
    return run["error"]


async def test_identical_calls_fail_the_run(make_app):
    llm = FakeProvider(replies=[read("notes.txt", i) for i in range(6)], json_replies=[dict(RESULT)])
    task, failures = await run_task(make_app, llm)
    error = assert_failed(task, failures, "file_read ran 3 times with the same details and got the same result")
    assert error.endswith("Edit the task to add what it needs, or retry it.")
    assert len(stream_calls(llm)) == 3  # no nudge for an unattended run


async def test_different_arguments_finish_normally(make_app):
    llm = FakeProvider(
        replies=[*(read(f"notes-{i}.txt", i) for i in range(4)), "None of the notes exist."],
        json_replies=[dict(RESULT)],
    )
    task, failures = await run_task(make_app, llm)
    assert task["runs"][0]["status"] == "completed" and not failures


async def test_step_limit_fails_with_a_plain_message(make_app, config):
    config.tasks.max_tool_rounds = 2
    llm = FakeProvider(replies=[read(f"notes-{i}.txt", i) for i in range(4)], json_replies=[dict(RESULT)])
    task, failures = await run_task(make_app, llm)
    assert_failed(task, failures, "Stopped after 2 steps without finishing. The limits for one run are in Settings > Tasks.")


async def test_time_limit_fails_with_a_plain_message(make_app, config):
    @tool("slow_lookup", risk=Risk.read)
    async def slow_lookup(ctx: ToolContext) -> str:
        """Look something up slowly."""
        await asyncio.sleep(30)
        return "late"

    class Slow(ToolPlugin):
        id = "slow"
        display_name = "Slow"
        tools = [slow_lookup]

    config.tasks.run_timeout_minutes = 0.002  # about 0.1 s
    llm = FakeProvider(replies=[[{"id": "c1", "name": "slow_lookup", "arguments": {}}]], json_replies=[dict(RESULT)])
    task, failures = await run_task(make_app, llm, plan_tool="slow", plugin=Slow())
    assert_failed(task, failures, "minutes without finishing. The limits for one run are in Settings > Tasks.")


async def test_token_limit_fails_the_run(make_app, config):
    config.tasks.max_tokens_per_run = 1000
    llm = PricedProvider(replies=[read(f"notes-{i}.txt", i) for i in range(6)], json_replies=[dict(RESULT)], tokens=400)
    task, failures = await run_task(make_app, llm)
    assert_failed(task, failures, "Stopped after using 1,200 tokens without finishing. The limit for one run is 1,000.")
    assert len(stream_calls(llm)) == 3


async def test_cost_limit_fails_the_run(make_app, config):
    config.tasks.max_cost_per_run_usd = 1.0
    llm = PricedProvider(
        replies=[read(f"notes-{i}.txt", i) for i in range(6)], json_replies=[dict(RESULT)], tokens=10, cost=0.5
    )
    task, failures = await run_task(make_app, llm)
    assert_failed(task, failures, "Stopped after spending about $1.00 on the model without finishing.")
    assert len(stream_calls(llm)) == 2


async def test_default_limits_leave_a_normal_cloud_run_alone(make_app):
    llm = PricedProvider(
        replies=[*(read(f"notes-{i}.txt", i) for i in range(3)), "Read them all."],
        json_replies=[dict(RESULT)], tokens=20_000, cost=0.05,
    )
    task, failures = await run_task(make_app, llm)
    assert task["runs"][0]["status"] == "completed" and not failures
