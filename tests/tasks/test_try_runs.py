"""Try runs ("Try it first", #142): a task run that notes what it would send, change or run instead of doing it."""

from __future__ import annotations

import pytest

from sentient import paths
from sentient.agent import dry_run
from sentient.tasks.service import TaskConflict
from sentient.tools.base import Risk, ToolContext, ToolPlugin, tool
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import json_calls, stream_calls


def _mail(log: list[str]) -> ToolPlugin:
    @tool("mailer_list_inbox", risk=Risk.read)
    async def mailer_list_inbox(ctx: ToolContext) -> dict:
        """List the inbox."""
        log.append("list")
        return {"emails": [{"from": "sam@example.com", "subject": "Lunch?"}]}

    @tool("mailer_send_email", risk=Risk.send)
    async def mailer_send_email(ctx: ToolContext, to: str, subject: str = "", body: str = "") -> dict:
        """Send an email."""
        log.append(f"send:{to}")
        return {"sent": True}

    class Mailer(ToolPlugin):
        id = "mailer"
        display_name = "Mailer"
        tools = [mailer_list_inbox, mailer_send_email]

    return Mailer()


LIST = [tool_call("mailer_list_inbox")]
SEND_AND_WRITE = [
    tool_call("mailer_send_email", to="sam@example.com", subject="Lunch", body="Friday at 1 works."),
    {"id": "call_write", "name": "file_write", "arguments": {"name": "reply.txt", "content": "Friday at 1 works."}},
]


async def _task(app, *, status: str = "approval_pending", schedule: dict | None = None) -> str:
    now = app.tasks.now_iso()
    return await app.tasks.repo.insert_task({
        "name": "Answer lunch emails", "description": "Reply to lunch invites and keep a copy",
        "status": status, "schedule": schedule or {"type": "recurring", "frequency": "daily", "time": "09:00"},
        "plan": [{"tool": "mailer", "description": "Read the inbox and reply"},
                 {"tool": "files", "description": "Save a copy of the reply"}],
        "created_at": now, "updated_at": now,
    })


async def test_try_run_notes_sends_and_writes_but_looks_things_up_for_real(make_app):
    log: list[str] = []
    llm = FakeProvider(replies=[LIST, SEND_AND_WRITE, "Replied to Sam and saved a copy."])
    app = await make_app(llm)
    app.registry.register(_mail(log))
    task_id = await _task(app)

    data = await app.tasks.dry_run(task_id)
    assert data["status"] == "approval_pending"  # the task itself is left as it was
    await app.tasks.drain()

    assert log == ["list"]  # the look-up ran, the email did not
    assert not (paths.files_dir() / "reply.txt").exists()
    task = await app.tasks.get(task_id)
    assert task["status"] == "approval_pending" and task["next_execution_at"] is None
    assert task["last_execution_at"] is None
    run = task["runs"][-1]
    assert run["dry_run"] is True and run["status"] == "completed"
    kinds = [(a["tool"], a["kind"]) for a in run["would_do"]]
    assert kinds == [("mailer_send_email", "email"), ("file_write", "change")]
    email = run["would_do"][0]
    assert email["app"] == "Mailer" and email["preview"]["body"] == "Friday at 1 works."
    assert {"label": "To", "value": "sam@example.com"} in email["preview"]["fields"]
    assert email["summary"].startswith("Send an email in Mailer: To sam@example.com")
    # the report is written in code, never by the fast model
    assert run["result"]["summary"].startswith("Try run finished. Nothing was sent, changed or run.")
    assert "- File write in Files: reply.txt" in run["result"]["summary"]
    assert json_calls(llm, "fast") == []
    # the model was told it was only noted
    tool_results = [m for m in stream_calls(llm)[-1]["messages"] if m.get("role") == "tool"]
    assert any("TRY RUN" in str(m.get("content")) for m in tool_results)
    events = [e["message"] for e in await app.tasks.run_events(task_id, run["run_id"])]
    assert events[0]["content"] == dry_run.OPENING
    note = next(n for n in await app.notifications.list() if (n.get("payload") or {}).get("event") == "dry_run_finished")
    assert "would do 2 things" in note["message"]


async def test_try_run_leaves_an_active_task_and_its_schedule_alone(make_app):
    llm = FakeProvider(replies=[LIST, "Nothing to answer."])
    app = await make_app(llm)
    app.registry.register(_mail([]))
    task_id = await _task(app, status="active")
    await app.tasks.repo.update_task(task_id, {"next_execution_at": "2030-01-01T03:30:00+00:00"})

    await app.tasks.dry_run(task_id)
    await app.tasks.drain()

    task = await app.tasks.get(task_id)
    assert task["status"] == "active" and task["next_execution_at"] == "2030-01-01T03:30:00+00:00"
    run = task["runs"][-1]
    assert run["would_do"] == []
    assert run["result"]["summary"].startswith("Try run finished. A real run would only look things up")
    assert await app.tasks.repo.latest_run(task_id) is None  # a try run is never the task's last result


async def test_never_rules_still_refuse_and_ask_rules_are_noted_without_pausing(make_app):
    log: list[str] = []
    llm = FakeProvider(replies=[SEND_AND_WRITE, "Done."])
    app = await make_app(llm)
    app.registry.register(_mail(log))
    app.config.tools.approvals.rules = {"mailer_send_email": "ask", "file_write": "never"}
    task_id = await _task(app)

    await app.tasks.dry_run(task_id)
    await app.tasks.drain()

    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["status"] == "completed"  # an Ask rule doesn't stop a try run
    assert [a["tool"] for a in run["would_do"]] == ["mailer_send_email"]  # Never is refused, not noted
    assert run["would_do"][0]["asks"] is True  # a real run would ask first
    assert log == []


async def test_try_run_conflicts(make_app):
    app = await make_app(FakeProvider())
    now = app.tasks.now_iso()
    planning = await app.tasks.repo.insert_task({"name": "x", "status": "planning", "created_at": now, "updated_at": now})
    with pytest.raises(TaskConflict):
        await app.tasks.dry_run(planning)
    script = await app.tasks.repo.insert_task({
        "name": "check", "status": "active", "task_type": "script", "script": {"code": "result(1)"},
        "created_at": now, "updated_at": now,
    })
    with pytest.raises(TaskConflict):
        await app.tasks.dry_run(script)
    task_id = await _task(app)
    await app.tasks.repo.insert_run(task_id, now=now, dry_run=True)  # one is already going
    with pytest.raises(TaskConflict, match="already going"):
        await app.tasks.dry_run(task_id)


async def test_try_run_of_an_approved_call_notes_it(make_app):
    log: list[str] = []
    app = await make_app(FakeProvider())
    app.registry.register(_mail(log))
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Email Sam", "status": "pending", "schedule": {"type": "once", "run_at": "2030-01-01T00:00:00+00:00"},
        "original_context": {"fixed_call": {"tool": "mailer_send_email", "arguments": {"to": "sam@example.com"}}},
        "created_at": now, "updated_at": now,
    })
    await app.tasks.dry_run(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert log == [] and run["status"] == "completed" and task["status"] == "pending"
    assert [a["tool"] for a in run["would_do"]] == ["mailer_send_email"]


def test_report_and_summary_lines():
    assert dry_run.report([]).startswith("Try run finished. A real run would only look things up")
    entry = {"title": "Run", "kind": "run", "app": "Terminal", "arguments": {"command": "git push"}}
    assert dry_run.summary_line(entry) == "Run in Terminal: git push"
    text = dry_run.report([{**entry, "summary": dry_run.summary_line(entry)}])
    assert text.endswith("A real run would do 1 thing:\n- Run in Terminal: git push")
