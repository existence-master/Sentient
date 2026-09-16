import asyncio
import json
from datetime import UTC, datetime

from sentient import paths
from sentient.tasks.prompts import RESUME_NOTE
from sentient.tools.base import Risk
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import (
    PLAN_FILES,
    PLAN_TIME,
    REFINE_DAILY,
    REFINE_ONCE,
    RESULT,
    GatedProvider,
    json_calls,
    stream_calls,
)


async def _planned(app, prompt="Write a haiku and save it to haiku.txt", **kwargs):
    task = await app.tasks.create_task(prompt, **kwargs)
    await app.tasks.drain()
    return await app.tasks.get(task["task_id"])


def _types(run):
    return [u["message"]["type"] for u in run["progress_updates"]]


async def test_full_lifecycle_to_completed(make_app):
    llm = FakeProvider(
        replies=[[tool_call("file_write", name="haiku.txt", content="rain on tin roofs")], "I saved the haiku to haiku.txt."],
        json_replies=[REFINE_ONCE, PLAN_FILES, RESULT],
    )
    app = await make_app(llm)
    seen = []
    async with app.bus.subscribe() as q:
        created = await app.tasks.create_task("Write a haiku and save it to haiku.txt")
        assert created["status"] == "planning"
        assert created["original_context"] == {"source": "manual_creation"}
        await app.tasks.drain()
        task = await app.tasks.get(created["task_id"])
        assert task["status"] == "approval_pending"
        assert task["name"] == "Haiku file"
        assert task["plan"] == [{"tool": "files", "description": "Save the haiku to haiku.txt"}]
        assert task["schedule"] == {"type": "once", "run_at": None, "timezone": "Asia/Kolkata"}
        assert any("I've created a new plan" in n["message"] for n in await app.notifications.list())

        approved = await app.tasks.approve(task["task_id"])
        assert approved["status"] == "processing" and approved["runs"][0]["status"] == "processing"
        await app.tasks.drain()
        while not q.empty():
            seen.append(q.get_nowait())

    task = await app.tasks.get(created["task_id"])
    assert task["status"] == "completed" and task["next_execution_at"] is None
    run = task["runs"][0]
    assert run["status"] == "completed" and run["finished_at"] and run["error"] is None
    types = _types(run)
    assert types[0] == "info"
    assert {"tool_call", "tool_result", "final_answer"} <= set(types)
    assert types.count("final_answer") == 1
    call = next(u["message"] for u in run["progress_updates"] if u["message"]["type"] == "tool_call")
    assert call["tool_name"] == "file_write" and call["parameters"]["name"] == "haiku.txt"
    assert run["result"]["summary"] == "Saved a haiku."
    assert run["result"]["files_created"] == [{"filename": "haiku.txt", "description": ""}]
    assert run["result"]["tools_used"] == ["files"]
    assert (paths.files_dir() / "haiku.txt").read_text(encoding="utf-8") == "rain on tin roofs"

    # executor: plan tools + core helpers, never task tools or send-risk helpers
    first = stream_calls(llm)[0]
    names = {t["function"]["name"] for t in first["tools"]}
    assert {"file_write", "current_datetime", "memory_recall"} <= names
    assert "create_task_from_prompt" not in names and "memory_forget" not in names
    system = first["messages"][0]["content"]
    assert "Save the haiku to haiku.txt" in system and "Pune, India" in system and "Sarthak" in system
    # planner catalogue is dynamic and excludes the tasks plugin
    planner_system = json_calls(llm, "planner")[1]["messages"][0]["content"]
    assert '"files"' in planner_system and '"tasks"' not in planner_system
    assert json_calls(llm, "fast"), "result generator uses the fast role"

    kinds = {e["type"] for e in seen}
    assert {"task.updated", "task.run_progress", "notification.new"} <= kinds
    assert any("has finished with status: completed" in n["message"] for n in await app.notifications.list())
    events = await app.tasks.run_events(task["task_id"], run["run_id"])
    assert [e["message"]["type"] for e in events] == types


async def test_require_plan_approval_off_runs_immediately(make_app, config):
    config.tasks.require_plan_approval = False
    llm = FakeProvider(replies=["It is Tuesday."], json_replies=[REFINE_ONCE, PLAN_TIME, RESULT])
    app = await make_app(llm)
    task = await _planned(app)
    assert task["status"] == "completed" and len(task["runs"]) == 1


async def test_recurring_schedule_tick_and_reschedule(make_app, clock):
    llm = FakeProvider(replies=["Today is 2026-09-15."], json_replies=[REFINE_DAILY, PLAN_TIME, RESULT])
    app = await make_app(llm, clock=clock)  # 07:30 IST
    task = await _planned(app, "Every day at 9am tell me the date")
    assert task["schedule"]["type"] == "recurring" and task["schedule"]["timezone"] == "Asia/Kolkata"
    task = await app.tasks.approve(task["task_id"])
    assert task["status"] == "active" and task["enabled"] is True
    assert task["next_execution_at"] == "2026-09-15T03:30:00+00:00"
    assert task["runs"] == []

    assert await app.tasks.tick() == []
    clock.dt = datetime(2026, 9, 15, 3, 31, tzinfo=UTC)
    run_ids = await app.tasks.tick()
    assert len(run_ids) == 1
    assert await app.tasks.tick() == []  # already claimed
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "active"
    assert task["next_execution_at"] == "2026-09-16T03:30:00+00:00"
    assert task["last_execution_at"] == "2026-09-15T03:31:00+00:00"
    assert [r["status"] for r in task["runs"]] == ["completed"]


async def test_once_in_future_is_pending_until_due(make_app, clock):
    refine = {**REFINE_ONCE, "schedule": {"type": "once", "run_at": "2026-09-16T09:00"}}
    llm = FakeProvider(replies=["done"], json_replies=[refine, PLAN_TIME, RESULT])
    app = await make_app(llm, clock=clock)
    task = await _planned(app, "Tomorrow at 9am tell me the date")
    task = await app.tasks.approve(task["task_id"])
    assert task["status"] == "pending"
    assert task["next_execution_at"] == "2026-09-16T03:30:00+00:00"
    clock.dt = datetime(2026, 9, 16, 3, 29, tzinfo=UTC)
    assert await app.tasks.tick() == []
    clock.dt = datetime(2026, 9, 16, 3, 30, tzinfo=UTC)
    assert len(await app.tasks.tick()) == 1
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "completed" and task["next_execution_at"] is None


async def test_triggered_task_matches_and_dedupes(make_app):
    refine = {
        "name": "Invoice alert", "description": "Log invoices", "priority": 1,
        "schedule": {"type": "triggered", "source": "gmail", "event": "new_email",
                     "filter": {"subject": {"$contains": "invoice"}}},
    }
    llm = FakeProvider(replies=["Logged the invoice."], json_replies=[refine, PLAN_TIME, RESULT])
    app = await make_app(llm)
    task = await _planned(app, "Whenever I get an invoice email, log it")
    task = await app.tasks.approve(task["task_id"])
    assert task["status"] == "active" and task["next_execution_at"] is None

    email = {"id": "m1", "from": "Jane <jane@x.com>", "sender_email": "jane@x.com", "subject": "Invoice #42"}
    assert await app.tasks.handle_event("gcalendar", "new_event", {"id": "e1", "summary": "Invoice"}) == []
    assert await app.tasks.handle_event("gmail", "new_email", {**email, "id": "m0", "subject": "Hello"}) == []
    run_ids = await app.tasks.handle_event("gmail", "new_email", email, event_id="m1")
    assert len(run_ids) == 1
    assert await app.tasks.handle_event("gmail", "new_email", email, event_id="m1") == []  # dedupe
    await app.tasks.drain()

    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "active"
    run = task["runs"][0]
    assert run["trigger_event_data"] == email and run["status"] == "completed"
    assert "Invoice #42" in stream_calls(llm)[0]["messages"][0]["content"]
    # declined tasks never fire
    await app.tasks.decline(task["task_id"])
    assert await app.tasks.handle_event("gmail", "new_email", {**email, "id": "m2"}) == []


async def test_change_request_replans_with_previous_result(make_app):
    plan2 = {"name": "ignored", "description": "x", "plan": [{"tool": "files", "description": "Rewrite haiku.txt about winter"}]}
    llm = FakeProvider(replies=["Saved."], json_replies=[REFINE_ONCE, PLAN_FILES, RESULT, plan2])
    app = await make_app(llm)
    task = await _planned(app)
    await app.tasks.approve(task["task_id"])
    await app.tasks.drain()

    task = await app.tasks.chat(task["task_id"], "Make it about winter instead")
    assert task["status"] == "planning"
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "approval_pending"
    assert task["plan"] == plan2["plan"]
    assert task["name"] == "Haiku file"  # change requests keep the name (v2)
    assert [m["role"] for m in task["chat_history"]] == ["user", "assistant"]
    prompt = json_calls(llm, "planner")[-1]["messages"][1]["content"]
    assert "previous_result" in prompt and "Saved a haiku." in prompt and "winter" in prompt


async def test_clarifying_questions_then_resume_planning(make_app):
    asks = {"name": "Send report", "description": "x", "plan": [], "clarifying_questions": ["Which address should get the report?"]}
    llm = FakeProvider(json_replies=[REFINE_ONCE, asks, PLAN_FILES])
    app = await make_app(llm)
    task = await _planned(app, "Send the report to my accountant")
    assert task["status"] == "clarification_pending"
    assert task["clarifying_questions"] == [
        {"question_id": "q1", "text": "Which address should get the report?", "answer": None}
    ]
    assert any("more information" in n["message"] for n in await app.notifications.list())

    task = await app.tasks.answer_clarifications(task["task_id"], [{"question_id": "q1", "answer_text": "cpa@example.com"}])
    assert task["status"] == "planning"
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "approval_pending" and task["clarifying_questions"][0]["answer"] == "cpa@example.com"
    assert "cpa@example.com" in json_calls(llm, "planner")[-1]["messages"][1]["content"]


async def test_cancel_running_run(make_app):
    llm = GatedProvider(replies=[[tool_call("current_datetime")], "never"], json_replies=[REFINE_ONCE, PLAN_TIME], block_on_call=2)
    app = await make_app(llm)
    task = await _planned(app)
    task = await app.tasks.approve(task["task_id"])
    await asyncio.wait_for(llm.blocked.wait(), 5)
    run_id = task["runs"][0]["run_id"]
    task = await app.tasks.cancel_run(task["task_id"], run_id)
    assert task["status"] == "cancelled"
    assert task["runs"][0]["status"] == "cancelled"
    await app.tasks.drain()
    task = await app.tasks.get(task["task_id"])
    assert task["status"] == "cancelled" and task["runs"][0]["status"] == "cancelled"
    assert "Run cancelled by user." in [u["message"].get("content") for u in task["runs"][0]["progress_updates"]]


async def _interrupt_after_first_tool(make_app, db_name):
    llm = GatedProvider(
        replies=[[tool_call("file_write", name="part1.txt", content="one")], "unused"],
        json_replies=[REFINE_ONCE, PLAN_FILES], block_on_call=2,
    )
    app = await make_app(llm, db_name=db_name)
    task = await _planned(app)
    task = await app.tasks.approve(task["task_id"])
    await asyncio.wait_for(llm.blocked.wait(), 5)
    await app.stop()  # simulated quit mid-run
    return task["task_id"], task["runs"][0]["run_id"]


async def test_restart_resumes_from_checkpoint(make_app):
    task_id, run_id = await _interrupt_after_first_tool(make_app, "restart.db")
    llm2 = FakeProvider(replies=["Finished: part1.txt is written."], json_replies=[RESULT])
    app2 = await make_app(llm2, db_name="restart.db")
    run = await app2.tasks.repo.get_run(run_id)
    assert run["status"] == "processing"
    assert any(m["role"] == "tool" and m["name"] == "file_write" for m in run["messages"])

    report = await app2.tasks.recover_interrupted()
    assert report["resumed"] == [run_id]
    await app2.tasks.drain()
    task = await app2.tasks.get(task_id)
    assert task["status"] == "completed" and task["runs"][0]["status"] == "completed"
    messages = stream_calls(llm2)[0]["messages"]
    assert messages[-1] == {"role": "user", "content": RESUME_NOTE}
    assert any(m["role"] == "tool" for m in messages)
    contents = [u["message"].get("content") for u in task["runs"][0]["progress_updates"]]
    assert "Resuming the run after a restart." in contents
    assert len([c for c in llm2.calls if c.get("tools") is not None]) == 1  # the tool was not re-run


async def test_restart_without_resume_marks_error(make_app, config):
    task_id, run_id = await _interrupt_after_first_tool(make_app, "restart2.db")
    config.tasks.resume_interrupted_runs = False
    app2 = await make_app(FakeProvider(), db_name="restart2.db")
    report = await app2.tasks.recover_interrupted()
    assert report["failed"] == [run_id]
    task = await app2.tasks.get(task_id)
    assert task["status"] == "error" and task["runs"][0]["error"] == "Interrupted by restart"
    assert any("interrupted by a restart" in n["message"] for n in await app2.notifications.list())


async def test_swarm_runs_workers_in_parallel_and_aggregates(make_app):
    llm = FakeProvider(
        replies=["alpha summary", "beta summary", '{"topic": "gamma"}'],
        json_replies=[
            {"items": ["alpha", "beta", "gamma"]},
            {"workers": [{"item_indices": [0, 1, 2], "worker_prompt": "Summarize the topic", "required_tools": ["time", "tasks"]}]},
            {"summary": "Three topics summarized."},
        ],
    )
    app = await make_app(llm)
    created = await app.tasks.create_task("Research alpha, beta and gamma", is_swarm=True)
    assert created["task_type"] == "swarm" and created["swarm_details"]["goal"] == "Research alpha, beta and gamma"
    await app.tasks.drain()
    task = await app.tasks.get(created["task_id"])
    assert task["status"] == "completed"
    details = task["swarm_details"]
    assert details["items"] == ["alpha", "beta", "gamma"]
    assert details["total_agents"] == 3 and details["completed_agents"] == 3
    assert sorted(map(json.dumps, details["aggregated_results"])) == sorted(
        map(json.dumps, ["alpha summary", "beta summary", {"topic": "gamma"}])
    )
    assert {u["status"] for u in details["progress_updates"]} >= {"processing", "completed", "aggregating"}
    run = task["runs"][0]
    assert run["status"] == "completed" and run["result"]["summary"] == "Three topics summarized."
    assert run["plan"][0]["worker_prompt"] == "Summarize the topic"
    for call in stream_calls(llm):
        assert {t["function"]["name"] for t in call["tools"]} == {"current_datetime"}
    assert any("Swarm task" in n["message"] for n in await app.notifications.list())


async def test_disable_tasks_for_plugin(make_app, clock):
    llm = FakeProvider(json_replies=[REFINE_DAILY, PLAN_TIME, REFINE_ONCE, PLAN_FILES])
    app = await make_app(llm, clock=clock)
    daily = await _planned(app, "Every day at 9am tell me the date")
    await app.tasks.approve(daily["task_id"])
    other = await _planned(app)
    assert await app.tasks.disable_tasks_for_plugin("time") == 1
    daily = await app.tasks.get(daily["task_id"])
    assert daily["enabled"] is False and daily["status"] == "active"
    assert (await app.tasks.get(other["task_id"]))["enabled"] is True
    assert any("was disabled because" in n["message"] for n in await app.notifications.list())
    clock.dt = datetime(2026, 9, 20, tzinfo=UTC)
    assert await app.tasks.tick() == []
    assert await app.tasks.disable_tasks_for_plugin("time") == 0
    # re-enabling recomputes the next run from now
    daily = await app.tasks.update(daily["task_id"], {"enabled": True})
    assert daily["enabled"] is True and daily["next_execution_at"] == "2026-09-20T03:30:00+00:00"


async def test_rerun_archive_delete(make_app):
    llm = FakeProvider(json_replies=[REFINE_ONCE, PLAN_FILES, PLAN_TIME])
    app = await make_app(llm)
    task = await _planned(app)
    copy = await app.tasks.rerun(task["task_id"])
    assert copy["task_id"] != task["task_id"] and copy["status"] == "planning" and copy["name"] == task["name"]
    await app.tasks.drain()
    assert (await app.tasks.get(copy["task_id"]))["plan"] == PLAN_TIME["plan"]
    assert (await app.tasks.archive(task["task_id"]))["status"] == "archived"
    async with app.bus.subscribe() as q:
        assert await app.tasks.delete(task["task_id"]) == {"ok": True}
        assert q.get_nowait() is not None
    assert len(await app.tasks.list()) == 1
    assert all(n.get("task_id") != task["task_id"] for n in await app.notifications.list())


async def test_task_tools(make_app):
    llm = FakeProvider(json_replies=[REFINE_ONCE, PLAN_FILES])
    app = await make_app(llm)
    create = app.registry.get("create_task_from_prompt")
    assert create is not None and create.risk == Risk.write
    assert app.registry.get("search_tasks").risk == Risk.read
    plugin = next(p for p in app.registry.catalog() if p["id"] == "tasks")
    assert plugin["icon"] == "IconChecklist" and plugin["category"] == "core"

    ctx = app.agent.tool_context("session-1", "desktop")
    res = await create.call(ctx, {"prompt": "Write a haiku and save it to haiku.txt"})
    assert res["status"] == "success"
    await app.tasks.drain()
    task = await app.tasks.get(res["task_id"])
    assert task["original_context"] == {"source": "chat", "session_id": "session-1"}

    found = await app.registry.get("search_tasks").call(ctx, {"query": "haiku"})
    assert [t["task_id"] for t in found["tasks"]] == [res["task_id"]]
    assert (await app.registry.get("search_tasks").call(ctx, {"status": "active"}))["tasks"] == []
    status = await app.registry.get("get_task_status").call(ctx, {"task_id": res["task_id"]})
    assert status["task"]["status"] == "approval_pending" and status["task"]["latest_run"] is None
    missing = await app.registry.get("get_task_status").call(ctx, {"task_id": "nope"})
    assert missing["status"] == "failure"
