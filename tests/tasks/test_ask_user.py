"""A running task can ask the user a question (ask_user), wait without calling the model, and carry on
with the answer, also after a restart (issue #108)."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.tasks import ask
from sentient.tasks.prompts import RESUME_NOTE
from sentient.tasks.service import TaskConflict
from tests.conftest import FakeProvider, tool_call
from tests.tasks.conftest import PLAN_FILES, REFINE_ONCE, RESULT, stream_calls

FLIGHTS = ["IndiGo 6E 123 at 07:10", "Air India AI 456 at 09:40"]
ASK = [tool_call("ask_user", question="Which flight should I book?", options=FLIGHTS)]


async def _waiting_task(app, name: str = "Book a flight to Goa") -> tuple[str, str]:
    """Insert an approved-ready task, approve it, and wait until its run has asked its question."""
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": name,
        "description": name,
        "status": "approval_pending",
        "schedule": {"type": "once", "run_at": None},
        "plan": [{"tool": "time", "description": "Check today's date"}],
        "created_at": now,
        "updated_at": now,
    })
    await app.tasks.approve(task_id)
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    return task_id, task["runs"][-1]["run_id"]


def _question_notes(notes: list[dict]) -> list[dict]:
    return [n for n in notes if n["kind"] == "task" and (n.get("payload") or {}).get("event") == "question"]


async def test_ask_user_pauses_then_the_answer_resumes_the_run(make_app):
    llm = FakeProvider(replies=[ASK, "Booked the IndiGo flight you picked."], json_replies=[dict(RESULT)])
    app = await make_app(llm)
    task_id, run_id = await _waiting_task(app)

    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert task["status"] == "waiting_for_user"
    assert run["status"] == "waiting_for_user" and run["finished_at"] is None
    assert run["pending_question"]["question"] == "Which flight should I book?"
    assert run["pending_question"]["options"] == FLIGHTS and run["pending_question"]["asked_at"]
    # the loop stopped cleanly: one model call, no further call while waiting
    assert len(stream_calls(llm)) == 1
    offered = {t["function"]["name"] for t in stream_calls(llm)[0]["tools"]}
    assert "ask_user" in offered

    [note] = _question_notes(await app.notifications.list())
    assert note["title"] == "Book a flight to Goa needs your answer"
    assert note["message"] == "Which flight should I book?"
    assert note["payload"]["run_id"] == run_id and note["payload"]["options"] == FLIGHTS

    data = await app.tasks.answer_question(task_id, run_id, FLIGHTS[0])
    assert data["status"] == "processing"
    await app.tasks.drain()

    task = await app.tasks.get(task_id)
    run = task["runs"][-1]
    assert task["status"] == "completed" and run["status"] == "completed", run["error"]
    assert run["pending_question"] is None
    resumed = stream_calls(llm)[1]["messages"]
    last = resumed[-1]
    assert last["role"] == "tool" and last["tool_call_id"] == "call_ask_user"
    assert json.loads(last["content"])["answer"] == FLIGHTS[0]
    assert not any(m.get("content") == RESUME_NOTE for m in resumed)
    contents = [u["message"].get("content") for u in run["progress_updates"]]
    assert f"You answered: {FLIGHTS[0]}" in contents and "Got your answer. Carrying on with the task." in contents
    assert "Booked the IndiGo flight you picked." in contents
    assert "task_questions" not in (run["result"] or {}).get("tools_used", [])

    [note] = _question_notes(await app.notifications.list())
    assert note["payload"]["status"] == "answered" and note["payload"]["answer"] == FLIGHTS[0] and note["read"]


async def test_cancel_while_waiting_clears_the_question(make_app):
    llm = FakeProvider(replies=[ASK])
    app = await make_app(llm)
    task_id, run_id = await _waiting_task(app)

    task = await app.tasks.cancel_run(task_id, run_id)
    run = task["runs"][-1]
    assert task["status"] == "cancelled" and run["status"] == "cancelled" and run["pending_question"] is None
    stored = await app.tasks.repo.get_run(run_id)
    assert stored["pending_question"] is None
    tool_msg = next(m for m in stored["messages"] if m.get("role") == "tool" and m.get("name") == "ask_user")
    assert ask.CANCELLED_NOTE in tool_msg["content"]
    [note] = _question_notes(await app.notifications.list())
    assert note["payload"]["status"] == "cancelled"
    assert await app.tasks.waiting_questions() == []
    with pytest.raises(TaskConflict):
        await app.tasks.answer_question(task_id, run_id, "too late")
    assert len(stream_calls(llm)) == 1


async def test_triggered_task_keeps_firing_and_returns_to_waiting(make_app):
    llm = FakeProvider(replies=[ASK, "Handled item 2.", "Handled item 1."], json_replies=[dict(RESULT), dict(RESULT)])
    app = await make_app(llm)
    now = app.tasks.now_iso()
    task_id = await app.tasks.repo.insert_task({
        "name": "Sort new orders", "description": "Sort each new order", "status": "active",
        "schedule": {"type": "triggered", "source": "webhook", "event": "orders", "filter": {}},
        "plan": [{"tool": "time", "description": "Check the date"}], "created_at": now, "updated_at": now,
    })
    [first] = await app.tasks.handle_event("webhook", "orders", {"id": "1"})
    await app.tasks.drain()
    assert (await app.tasks.get(task_id))["status"] == "waiting_for_user"

    [second] = await app.tasks.handle_event("webhook", "orders", {"id": "2"})
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert {r["run_id"]: r["status"] for r in task["runs"]} == {first: "waiting_for_user", second: "completed"}
    assert task["status"] == "waiting_for_user"

    await app.tasks.answer_question(task_id, first, FLIGHTS[0])
    await app.tasks.drain()
    task = await app.tasks.get(task_id)
    assert task["status"] == "active" and all(r["status"] == "completed" for r in task["runs"])


async def test_waiting_run_survives_a_restart(make_app):
    app = await make_app(FakeProvider(replies=[ASK]), db_name="ask-restart.db")
    task_id, run_id = await _waiting_task(app)
    await app.stop()  # simulated quit while the task waits

    llm2 = FakeProvider(replies=["Booked the Air India flight."], json_replies=[dict(RESULT)])
    app2 = await make_app(llm2, db_name="ask-restart.db")
    report = await app2.tasks.recover_interrupted()
    await app2.tasks.drain()
    assert run_id not in report["resumed"] and run_id not in report["failed"]
    task = await app2.tasks.get(task_id)
    assert task["status"] == "waiting_for_user" and task["runs"][-1]["status"] == "waiting_for_user"
    assert stream_calls(llm2) == []
    [waiting] = await app2.tasks.waiting_questions()
    assert waiting["run_id"] == run_id and waiting["options"] == FLIGHTS

    await app2.tasks.answer_question(task_id, run_id, FLIGHTS[1])
    await app2.tasks.drain()
    task = await app2.tasks.get(task_id)
    assert task["status"] == "completed", task["runs"][-1]["error"]
    sent = stream_calls(llm2)[0]["messages"]
    assert json.loads(sent[-1]["content"])["answer"] == FLIGHTS[1]
    assert not any(m.get("content") == RESUME_NOTE for m in sent)


async def test_busy_actions_are_refused_while_waiting(make_app):
    app = await make_app(FakeProvider(replies=[ASK]))
    task_id, run_id = await _waiting_task(app)
    for call in (app.tasks.run_now(task_id), app.tasks.approve(task_id), app.tasks.chat(task_id, "Change it")):
        with pytest.raises(TaskConflict, match="waiting for your answer"):
            await call
    with pytest.raises(ValueError):
        await app.tasks.answer_question(task_id, run_id, "   ")


async def test_ask_user_is_only_offered_inside_task_runs(make_app):
    llm = FakeProvider(replies=["Hello!"])
    app = await make_app(llm)
    registry = app.registry
    assert registry.get("ask_user") is not None
    assert "ask_user" not in {t.name for t in registry.tools()}
    assert "ask_user" not in {s["function"]["name"] for s in registry.openai_schemas()}
    assert ask.PLUGIN_ID not in {p["id"] for p in registry.catalog()}
    assert [s["function"]["name"] for s in registry.openai_schemas(["ask_user"])] == ["ask_user"]

    session_id = await app.store.create_session()
    async for _ in app.agent.run_turn(session_id, "hi"):
        pass
    offered = {t["function"]["name"] for t in llm.calls[-1]["tools"] or []}
    assert offered and "ask_user" not in offered

    ctx = app.agent.tool_context(None, "chat")
    assert "error" in await registry.get("ask_user").call(ctx, {"question": "Which one?"})


async def test_one_question_per_round_and_options_are_cleaned(make_app):
    two = [
        {"id": "call_a", "name": "ask_user", "arguments": {"question": "First?", "options": [" Yes ", "yes", "", "No"]}},
        {"id": "call_b", "name": "ask_user", "arguments": {"question": "Second?"}},
    ]
    llm = FakeProvider(replies=[two])
    app = await make_app(llm)
    task_id, run_id = await _waiting_task(app)
    run = (await app.tasks.get(task_id))["runs"][-1]
    assert run["pending_question"]["question"] == "First?" and run["pending_question"]["options"] == ["Yes", "No"]
    stored = await app.tasks.repo.get_run(run_id)
    assert stored["pending_question"]["tool_call_id"] == "call_a"
    second = next(m for m in stored["messages"] if m.get("tool_call_id") == "call_b")
    assert "already asked" in second["content"]


# ---------------------------------------------------------------------------- REST
@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider(replies=[ASK, "Booked it."], json_replies=[REFINE_ONCE, PLAN_FILES, RESULT])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "ask-routes.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def _wait(client, task_id, statuses, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in statuses:
            return task
        time.sleep(0.05)
    raise AssertionError(f"task stuck in {task['status']}")


def test_answer_route(client):
    task_id = client.post("/api/tasks", json={"prompt": "Book my flight to Goa"}).json()["task_id"]
    _wait(client, task_id, {"approval_pending"})
    client.post(f"/api/tasks/{task_id}/approve")
    task = _wait(client, task_id, {"waiting_for_user"})
    run = task["runs"][-1]
    assert run["status"] == "waiting_for_user" and run["pending_question"]["options"] == FLIGHTS
    url = f"/api/tasks/{task_id}/runs/{run['run_id']}/answer"

    assert client.post(url, json={"answer": "  "}).status_code == 400
    assert client.post(f"/api/tasks/{task_id}/runs/nope/answer", json={"answer": "x"}).status_code == 404
    answered = client.post(url, json={"answer": FLIGHTS[0]})
    assert answered.status_code == 200 and answered.json()["status"] == "processing"
    assert client.post(url, json={"answer": FLIGHTS[1]}).status_code == 409
    task = _wait(client, task_id, {"completed"})
    assert task["runs"][-1]["pending_question"] is None
    assert client.post(url, json={"answer": "again"}).status_code == 409
