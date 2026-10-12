"""A tool set to Ask pauses a task run and asks, and the answer says how wide the yes is (#139, ADR 0016).

"Allow once" runs only the held call, "Allow for this task" lets later calls of that tool in the same task through,
"Allow for 1 hour" covers every task until it runs out, "Always allow" saves an Allow rule, "Deny" stops the run, and
a request past its time can't be approved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sentient.agent import grants as grant_rules
from sentient.tasks import ask
from sentient.tasks.service import TaskConflict
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import RESULT, stream_calls

OPTIONS = ["Allow once", "Allow for this task", "Allow for 1 hour", "Always allow", "Deny"]
WRITE_A = [tool_call("file_write", name="a.txt", content="one")]
WRITE_B = [tool_call("file_write", name="b.txt", content="two")]


async def _task(app, name: str = "Save notes") -> str:
    now = app.tasks.now_iso()
    return await app.tasks.repo.insert_task({
        "name": name, "description": name, "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None}, "plan": [{"tool": "files", "description": "Save the notes"}],
        "created_at": now, "updated_at": now,
    })


async def _start(app, name: str = "Save notes") -> tuple[str, str]:
    task_id = await _task(app, name)
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    return task_id, task["runs"][-1]["run_id"]


async def _pending(app, run_id: str) -> dict:
    return (await app.tasks.repo.get_run(run_id))["pending_question"]


async def test_ask_rule_pauses_a_task_run_and_asks_how_wide(make_app, config, isolated_home):
    config.tools.approvals.rules = {"files": "ask"}
    llm = FakeProvider(replies=[WRITE_A, "Saved."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id, run_id = await _start(app)

    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert task["status"] == "waiting_for_user" and run["status"] == "waiting_for_user"
    q = run["pending_question"]
    assert q["kind"] == "approval" and q["options"] == OPTIONS
    assert q["question"].startswith('This task wants to use "File write" in Files (name: a.txt')
    assert q["expires_at"]
    assert not (isolated_home / "files" / "a.txt").exists()  # nothing written while it waits
    assert len(stream_calls(llm)) == 1  # no model call while waiting
    note = next(n for n in await app.notifications.list() if (n.get("payload") or {}).get("event") == "question")
    assert note["payload"]["approval"]["choices"] == ["allow", "allow_task", "allow_timed", "allow_always", "deny"]
    assert note["payload"]["expires_at"] == (await _pending(app, run_id))["expires_at"]


async def test_allow_for_this_task_lets_later_calls_through(make_app, config, isolated_home):
    config.tools.approvals.rules = {"files": "ask"}
    llm = FakeProvider(replies=[WRITE_A, WRITE_B, "Saved both."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id, run_id = await _start(app)

    await app.tasks.answer_question(task_id, run_id, "Allow for this task", via="telegram")
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "completed", task["runs"][-1].get("error")
    assert (isolated_home / "files" / "a.txt").read_text() == "one"
    assert (isolated_home / "files" / "b.txt").read_text() == "two"  # the second write did not ask
    grants = await app.approvals.grants.list()
    assert [(g["tool"], g["scope"], g["task_id"], g["via"]) for g in grants] == [("file_write", "task", task_id, "telegram")]

    # another task still asks: the grant was for that task only
    llm.replies[:] = [WRITE_A, "Saved."]
    other_id, _ = await _start(app, "Other notes")
    assert (await app.tasks.get(other_id))["status"] == "waiting_for_user"

    # deleting the task ends its grant
    await app.tasks.delete(task_id)
    assert await app.approvals.grants.list() == []


async def test_allow_once_runs_only_that_call(make_app, config, isolated_home):
    config.tools.approvals.rules = {"files": "ask"}
    llm = FakeProvider(replies=[WRITE_A, WRITE_B, "Saved."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id, run_id = await _start(app)
    await app.tasks.answer_question(task_id, run_id, "yes")
    await app.tasks.drain()

    assert (isolated_home / "files" / "a.txt").exists()
    assert not (isolated_home / "files" / "b.txt").exists()
    assert (await app.tasks.get(task_id))["status"] == "waiting_for_user"  # the next write asks again
    assert await app.approvals.grants.list() == []


async def test_allow_for_a_while_covers_every_task_until_it_runs_out(make_app, config, isolated_home):
    config.tools.approvals.rules = {"files": "ask"}
    llm = FakeProvider(replies=[WRITE_A, "Saved.", WRITE_B, "Saved."], json_replies=[dict(RESULT), dict(RESULT)])
    app = await make_app(llm)
    task_id, run_id = await _start(app)
    await app.tasks.answer_question(task_id, run_id, "hour")
    await app.tasks.drain()
    assert (await app.tasks.get(task_id))["status"] == "completed"
    [grant] = await app.approvals.grants.list()
    assert grant["scope"] == "timed" and grant["expires_at"]

    other_id, _ = await _start(app, "Other notes")
    assert (await app.tasks.get(other_id))["status"] == "completed"  # covered, no question
    assert (isolated_home / "files" / "b.txt").exists()

    # an hour later the grant is gone and the tool asks again
    real = app.approvals.grants.clock
    app.approvals.grants.clock = lambda: real() + timedelta(minutes=61)
    llm.replies[:] = [WRITE_A, "Saved."]
    third_id, _ = await _start(app, "Third notes")
    assert (await app.tasks.get(third_id))["status"] == "waiting_for_user"
    assert await app.approvals.grants.list() == []


async def test_always_allow_saves_an_allow_rule(make_app, config, isolated_home):
    config.tools.approvals.rules = {"files": "ask"}
    llm = FakeProvider(replies=[WRITE_A, "Saved."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id, run_id = await _start(app)
    await app.tasks.answer_question(task_id, run_id, "Always allow")
    await app.tasks.drain()
    rules = app.config.tools.approvals
    assert rules.rules == {"files": "ask", "file_write": "allow"}
    origin = rules.rule_origins["file_write"]
    assert origin.rule == "allow" and origin.source == "approval" and origin.via == "desktop"


async def test_deny_stops_the_run(make_app, config, isolated_home):
    config.tools.approvals.rules = {"files": "ask"}
    llm = FakeProvider(replies=[WRITE_A, "Saved."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id, run_id = await _start(app)
    await app.tasks.answer_question(task_id, run_id, "No")
    task = await app.tasks.get(task_id)
    assert task["runs"][-1]["status"] == "error" and task["runs"][-1]["error"] == ask.DECLINED_STOP
    assert not (isolated_home / "files" / "a.txt").exists()


async def test_unclear_answer_keeps_waiting(make_app, config):
    config.tools.approvals.rules = {"files": "ask"}
    app = await make_app(FakeProvider(replies=[WRITE_A, "Saved."], json_replies=[dict(RESULT)]))
    task_id, run_id = await _start(app)
    with pytest.raises(ValueError, match="Answer with one of"):
        await app.tasks.answer_question(task_id, run_id, "maybe later, what is it for?")
    assert (await app.tasks.get(task_id))["status"] == "waiting_for_user"


async def test_expired_request_cant_be_approved_late(make_app, config, isolated_home):
    config.tools.approvals.rules = {"files": "ask"}
    app = await make_app(FakeProvider(replies=[WRITE_A, "Saved."], json_replies=[dict(RESULT)]))
    task_id, run_id = await _start(app)
    pending = await _pending(app, run_id)
    past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    await app.tasks.repo.update_run(run_id, {"pending_question": {**pending, "expires_at": past}})

    with pytest.raises(TaskConflict, match="expired"):
        await app.tasks.answer_question(task_id, run_id, "Allow once")
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "error" and run["error"] == ask.EXPIRED_STOP
    assert not (isolated_home / "files" / "a.txt").exists()
    note = next(n for n in await app.notifications.list() if (n.get("payload") or {}).get("event") == "question")
    assert note["payload"]["status"] == "expired"


async def test_scheduler_stops_runs_whose_request_expired(make_app, config):
    config.tools.approvals.rules = {"files": "ask"}
    app = await make_app(FakeProvider(replies=[WRITE_A, "Saved."], json_replies=[dict(RESULT)]))
    task_id, run_id = await _start(app)
    assert await app.tasks.expire_approvals() == 0  # not yet
    pending = await _pending(app, run_id)
    past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    await app.tasks.repo.update_run(run_id, {"pending_question": {**pending, "expires_at": past}})
    assert await app.tasks.expire_approvals() == 1
    assert (await app.tasks.get(task_id))["runs"][-1]["error"] == ask.EXPIRED_STOP


def _code_plugin(log: list[str]) -> ToolPlugin:
    @tool("lab_run", risk=Risk.exec)
    async def lab_run(ctx: ToolContext, code: str) -> dict:
        """Run some code."""
        log.append(code)
        return {"ok": True}

    class Lab(ToolPlugin):
        id = "lab"
        display_name = "Lab"
        tools = [lab_run]

    return Lab()


async def test_running_code_only_offers_once(make_app, config):
    config.tools.approvals.rules = {"lab": "ask"}
    log: list[str] = []
    llm = FakeProvider(replies=[[tool_call("lab_run", code="print(1)")], "Ran."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    app.registry.register(_code_plugin(log))
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Run it", "description": "Run it", "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None}, "plan": [{"tool": "lab", "description": "Run"}],
        "created_at": now, "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    run_id = task["runs"][-1]["run_id"]
    assert task["runs"][-1]["pending_question"]["options"] == ["Allow once", "Deny"]
    with pytest.raises(ValueError):  # a wider answer is not one of the choices
        await app.tasks.answer_question(task_id, run_id, "Allow for this task")
    await app.tasks.answer_question(task_id, run_id, "Allow once")
    await app.tasks.drain()
    assert log == ["print(1)"] and await app.approvals.grants.list() == []


def test_grant_choices_and_answers():
    @tool("x_send", risk=Risk.send)
    async def x_send(ctx: ToolContext) -> dict:
        """Send."""
        return {}

    assert grant_rules.choices(x_send, Risk.send, task=True) == [
        "allow", "allow_task", "allow_timed", "allow_always", "deny",
    ]
    assert grant_rules.choices(x_send, Risk.send, purchase=True, task=True) == ["allow", "deny"]
    assert grant_rules.choices(x_send, Risk.send, untrusted=True, chat=True) == ["allow", "deny"]
    assert grant_rules.choices(x_send, Risk.exec) == ["allow", "deny"]  # a risk_fn raised it to exec
    offered = grant_rules.choices(x_send, Risk.send, task=True)
    assert grant_rules.parse_answer("task", offered) == "allow_task"
    assert grant_rules.parse_answer("1 hour please", offered) == "allow_timed"
    assert grant_rules.parse_answer("ALWAYS", offered) == "allow_always"
    assert grant_rules.parse_answer("don't allow", offered) == "deny"
    assert grant_rules.parse_answer("ok", offered) == "allow"
    assert grant_rules.parse_answer("send it to bob", offered) is None
    assert grant_rules.parse_answer("task", ["allow", "deny"]) is None
    assert grant_rules.decision_label("allow_timed", 30) == "Allow for 30 minutes"
    assert grant_rules.decision_label("allow_timed", 120) == "Allow for 2 hours"
