"""Limits on a task run (issue #133).

A run that reaches its step, time, token or cost limit pauses and asks "Keep going" or "Stop here" through the
``waiting_for_user`` machinery. Keep going raises that limit by its original amount for this run (kept on the run,
so it survives a restart); any other answer fails the run with a plain message and the usual "Task failed"
notification. Time counts only while the run works. A run that repeats the same call fails at once.
"""

from __future__ import annotations

import asyncio

import pytest

from sentient.llm.provider import StreamChunk, ToolCall
from sentient.tasks import executor
from sentient.tasks.limits import KEEP_GOING, STOP_HERE
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider
from tests.tasks.conftest import RESULT, GatedProvider, stream_calls

HINT = "The limits for one run are in Settings > Tasks."


def read(name: str, n: int) -> list[dict]:
    return [{"id": f"call_{n}", "name": "file_read", "arguments": {"name": name}}]


def reads(start: int, count: int) -> list[list[dict]]:
    return [read(f"notes-{i}.txt", i) for i in range(start, start + count)]


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


@tool("slow_lookup", risk=Risk.read)
async def slow_lookup(ctx: ToolContext) -> str:
    """Look something up slowly."""
    await asyncio.sleep(30)
    return "late"


class Slow(ToolPlugin):
    id = "slow"
    display_name = "Slow"
    tools = [slow_lookup]


async def start_task(app, plan_tool: str = "files") -> str:
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Read my notes", "description": "Read notes.txt", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None}, "plan": [{"tool": plan_tool, "description": "Read notes.txt"}],
        "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    return task_id


async def notes(app, event: str) -> list[dict]:
    return [n for n in await app.notifications.list() if n["payload"].get("event") == event]


async def assert_waiting(app, task_id: str, question: str) -> str:
    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert task["status"] == "waiting_for_user" and run["status"] == "waiting_for_user", run["error"]
    assert run["pending_question"]["question"] == question
    assert run["pending_question"]["options"] == [KEEP_GOING, STOP_HERE]
    asked = [n for n in await notes(app, "question") if n["payload"].get("run_id") == run["run_id"]]
    assert any(n["message"] == question for n in asked)
    return run["run_id"]


async def answer(app, task_id: str, run_id: str, text: str) -> dict:
    await app.tasks.answer_question(task_id, run_id, text)
    await app.tasks.drain()
    return await app.tasks.get(task_id)


async def assert_failed(app, task: dict, error: str) -> None:
    run = task["runs"][-1]
    assert task["status"] == "error" and run["status"] == "error" and run["error"] == error
    [failure] = await notes(app, "run_failed")
    assert failure["title"] == "Task failed" and error in failure["message"]


# ---------------------------------------------------------------------- each limit asks, and Stop here fails
@pytest.mark.parametrize(
    ("kind", "llm_kwargs", "question", "error"),
    [
        (
            "steps", {},
            "This task has used 2 steps and isn't finished yet. Keep going for another 2 steps, or stop here?",
            f"Stopped after 2 steps without finishing. {HINT}",
        ),
        (
            "tokens", {"tokens": 400},
            "This task has used 1,200 tokens of your 1,000 token limit and isn't finished yet. "
            "Keep going for another 1,000 tokens, or stop here?",
            f"Stopped after using 1,200 tokens without finishing. The limit for one run is 1,000 tokens. {HINT}",
        ),
        (
            "cost", {"tokens": 10, "cost": 0.5},
            "This task has used $1.00 of your $1.00 limit and isn't finished yet. Keep going for another $1.00, "
            "or stop here?",
            f"Stopped after spending about $1.00 on the model without finishing. The limit for one run is $1.00. {HINT}",
        ),
    ],
)
async def test_limit_asks_and_stop_here_fails(make_app, config, kind, llm_kwargs, question, error):
    if kind == "steps":
        config.tasks.max_tool_rounds = 2
    elif kind == "tokens":
        config.tasks.max_tokens_per_run = 1000
    else:
        config.tasks.max_cost_per_run_usd = 1.0
    llm = PricedProvider(replies=reads(0, 6), json_replies=[dict(RESULT)], **llm_kwargs)
    app = await make_app(llm)
    task_id = await start_task(app)
    run_id = await assert_waiting(app, task_id, question)
    assert not await notes(app, "run_failed")
    calls = len(stream_calls(llm))

    task = await answer(app, task_id, run_id, STOP_HERE)
    await assert_failed(app, task, error)
    assert len(stream_calls(llm)) == calls  # nothing more ran


async def waiting_question(app, task_id: str) -> tuple[str, str]:
    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert task["status"] == "waiting_for_user" and run["status"] == "waiting_for_user", run["error"]
    assert run["pending_question"]["options"] == [KEEP_GOING, STOP_HERE]
    return run["run_id"], run["pending_question"]["question"]


async def test_hung_tool_hits_the_hard_deadline_and_the_run_resumes(make_app, config, monkeypatch):
    monkeypatch.setattr(executor, "HARD_DEADLINE_GRACE_S", 0.3)
    config.tasks.run_timeout_minutes = 0.02  # 1.2 s of work, then 0.3 s of grace for the call in flight
    llm = FakeProvider(
        replies=[[{"id": "c1", "name": "slow_lookup", "arguments": {}}], "Looked it up."], json_replies=[dict(RESULT)]
    )
    app = await make_app(llm)
    app.registry.register(Slow())
    task_id = await start_task(app, plan_tool="slow")
    run_id, question = await waiting_question(app, task_id)
    assert question.startswith("This task has been working for ")
    assert question.endswith("Keep going for another 0.02 minutes, or stop here?")
    run = await app.tasks.repo.get_run(run_id)
    assert 1.4 < run["limits"]["used"]["seconds"] < 2.4  # the hung call was cancelled at limit + grace
    assert len(stream_calls(llm)) == 1

    await asyncio.sleep(3.0)  # waiting for the answer is longer than the whole limit, and is not counted
    task = await answer(app, task_id, run_id, KEEP_GOING)
    assert task["status"] == "completed", task["runs"][-1]["error"]
    run = await app.tasks.repo.get_run(run_id)
    assert run["limits"]["max"]["seconds"] == pytest.approx(2.4)
    # resumed from the saved transcript: the cancelled call (no result) was dropped, nothing else was lost
    resumed = stream_calls(llm)[1]["messages"]
    assert resumed[0]["role"] == "system" and not any(m.get("tool_calls") for m in resumed)


async def test_time_limit_lets_the_call_in_flight_finish_then_asks(make_app, config):
    @tool("steady_lookup", risk=Risk.read)
    async def steady_lookup(ctx: ToolContext) -> str:
        """Look something up, taking a moment."""
        await asyncio.sleep(1.0)
        return "found it"

    class Steady(ToolPlugin):
        id = "steady"
        display_name = "Steady"
        tools = [steady_lookup]

    config.tasks.run_timeout_minutes = 0.01  # 0.6 s: passed while the lookup runs
    llm = FakeProvider(
        replies=[[{"id": "c1", "name": "steady_lookup", "arguments": {}}], "Done."], json_replies=[dict(RESULT)]
    )
    app = await make_app(llm)
    app.registry.register(Steady())
    task_id = await start_task(app, plan_tool="steady")
    run_id, question = await waiting_question(app, task_id)
    assert question.startswith("This task has been working for ")
    assert len(stream_calls(llm)) == 1  # no model call after the limit
    saved = (await app.tasks.repo.get_run(run_id))["messages"]
    assert any(m.get("role") == "tool" and "found it" in m.get("content", "") for m in saved)  # the call finished

    task = await answer(app, task_id, run_id, STOP_HERE)
    run = task["runs"][-1]
    assert run["status"] == "error" and run["error"].startswith("Stopped after 0.0") and run["error"].endswith(HINT)


# ---------------------------------------------------------------------- Keep going
async def test_keep_going_continues_and_can_finish(make_app, config):
    config.tasks.max_tool_rounds = 2
    llm = FakeProvider(replies=[*reads(0, 2), *reads(2, 1), "Read all the notes."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id = await start_task(app)
    run_id = await assert_waiting(
        app, task_id, "This task has used 2 steps and isn't finished yet. Keep going for another 2 steps, or stop here?"
    )
    task = await answer(app, task_id, run_id, "keep going")
    run = task["runs"][-1]
    assert task["status"] == "completed" and run["status"] == "completed", run["error"]
    assert len(stream_calls(llm)) == 4 and not await notes(app, "run_failed")
    infos = [u["message"]["content"] for u in run["progress_updates"] if u["message"]["type"] == "info"]
    assert "You answered: keep going" in infos


async def test_any_other_answer_stops_the_run(make_app, config):
    config.tasks.max_tokens_per_run = 1000
    llm = PricedProvider(replies=reads(0, 6), json_replies=[dict(RESULT)], tokens=600)
    app = await make_app(llm)
    task_id = await start_task(app)
    run_id = (await app.tasks.get(task_id))["runs"][-1]["run_id"]
    task = await answer(app, task_id, run_id, "not sure, maybe later")
    await assert_failed(
        app, task,
        f"Stopped after using 1,200 tokens without finishing. The limit for one run is 1,000 tokens. {HINT}",
    )


async def test_raised_limit_survives_a_restart(make_app, config):
    config.tasks.max_tool_rounds = 2
    llm = GatedProvider(replies=[*reads(0, 2), *reads(2, 1)], json_replies=[dict(RESULT)], block_on_call=3)
    app = await make_app(llm, db_name="limits-restart.db")
    task_id = await start_task(app)
    run_id = (await app.tasks.get(task_id))["runs"][-1]["run_id"]
    await app.tasks.answer_question(task_id, run_id, KEEP_GOING)
    await asyncio.wait_for(llm.blocked.wait(), 5)
    await app.stop()  # simulated quit while the run continues with the raised limit

    llm2 = FakeProvider(replies=reads(10, 4), json_replies=[dict(RESULT)])
    app2 = await make_app(llm2, db_name="limits-restart.db")
    report = await app2.tasks.recover_interrupted()
    await app2.tasks.drain()
    assert run_id in report["resumed"]
    assert len(stream_calls(llm2)) == 2  # 2 steps used before, limit raised to 4
    await assert_waiting(
        app2, task_id, "This task has used 4 steps and isn't finished yet. Keep going for another 2 steps, or stop here?"
    )


# ---------------------------------------------------------------------- loops fail at once; defaults leave runs alone
async def test_identical_calls_fail_the_run_without_asking(make_app):
    llm = FakeProvider(replies=[read("notes.txt", i) for i in range(6)], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id = await start_task(app)
    await assert_failed(
        app, await app.tasks.get(task_id),
        "Stopped because the same step kept repeating: file_read ran 3 times with the same details and got the same "
        "result each time. Edit the task to add what it needs, or retry it.",
    )
    assert len(stream_calls(llm)) == 3 and not await notes(app, "question")


async def test_different_arguments_finish_normally(make_app):
    llm = FakeProvider(replies=[*reads(0, 4), "None of the notes exist."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id = await start_task(app)
    task = await app.tasks.get(task_id)
    assert task["runs"][0]["status"] == "completed" and not await notes(app, "run_failed")


async def test_default_limits_leave_a_normal_cloud_run_alone(make_app):
    llm = PricedProvider(
        replies=[*reads(0, 3), "Read them all."], json_replies=[dict(RESULT)], tokens=20_000, cost=0.05
    )
    app = await make_app(llm)
    task_id = await start_task(app)
    task = await app.tasks.get(task_id)
    assert task["runs"][0]["status"] == "completed" and not await notes(app, "question")
