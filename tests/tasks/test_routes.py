import time

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider
from tests.tasks.conftest import PLAN_FILES, REFINE_ONCE, RESULT

TASK_KEYS = {
    "task_id", "name", "description", "status", "priority", "assignee", "task_type", "schedule", "plan", "runs",
    "chat_history", "clarifying_questions", "swarm_details", "enabled", "model", "original_context", "error",
    "next_execution_at", "last_execution_at", "created_at", "updated_at", "script", "browser_profile",
}
RUN_KEYS = {
    "run_id", "status", "created_at", "execution_start_time", "finished_at", "plan", "trigger_event_data",
    "progress_updates", "result", "error", "retry_of", "pending_question", "last_activity_at",
}


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    plan2 = {"plan": [{"tool": "files", "description": "Rewrite the haiku"}]}
    llm = FakeProvider(replies=["Saved haiku.txt."], json_replies=[REFINE_ONCE, REFINE_ONCE, PLAN_FILES, RESULT, plan2])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "routes.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def wait_for(client, task_id, statuses, timeout=10.0, until=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in statuses and (until is None or until(task)):
            return task
        time.sleep(0.05)
    raise AssertionError(f"task stuck in {task['status']}")


def test_task_routes_end_to_end(client):
    assert client.get("/api/tasks", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/tasks").json() == []

    preview = client.post("/api/tasks/preview", json={"prompt": "Write a haiku"}).json()
    assert set(preview) == {"name", "description", "priority", "schedule"}
    assert preview["schedule"]["type"] == "once"

    created = client.post("/api/tasks", json={"prompt": "Write a haiku and save it", "assignee": "ai"}).json()
    assert set(created) == TASK_KEYS and created["status"] == "planning"
    task_id = created["task_id"]
    task = wait_for(client, task_id, {"approval_pending"})

    patched = client.patch(f"/api/tasks/{task_id}", json={"name": "Renamed", "priority": 0, "bogus": 1}).json()
    assert patched["name"] == "Renamed" and patched["priority"] == 0
    assert client.patch(f"/api/tasks/{task_id}", json={"status": "weird"}).status_code == 400
    assert client.post(f"/api/tasks/{task_id}/clarifications", json={"answers": []}).status_code == 409

    approved = client.post(f"/api/tasks/{task_id}/approve").json()
    assert approved["status"] in {"processing", "completed"}
    # the task settles before its report is written (service._finish_run), so wait for the report too
    task = wait_for(client, task_id, {"completed"}, until=lambda t: t["runs"] and t["runs"][0].get("result"))
    run = task["runs"][0]
    assert set(run) == RUN_KEYS and run["status"] == "completed"
    assert run["result"]["summary"] == "Saved a haiku."
    events = client.get(f"/api/tasks/{task_id}/runs/{run['run_id']}/events").json()
    assert events[-1]["message"]["type"] == "final_answer" or "final_answer" in [e["message"]["type"] for e in events]
    assert client.post(f"/api/tasks/{task_id}/runs/{run['run_id']}/cancel").status_code == 409
    assert client.get(f"/api/tasks/{task_id}/runs/nope/events").status_code == 404

    chatted = client.post(f"/api/tasks/{task_id}/chat", json={"message": "Make it longer"}).json()
    assert chatted["status"] == "planning"
    task = wait_for(client, task_id, {"approval_pending"})
    assert task["plan"] == [{"tool": "files", "description": "Rewrite the haiku"}]

    assert client.post(f"/api/tasks/{task_id}/archive").json()["status"] == "archived"
    copy = client.post(f"/api/tasks/{task_id}/rerun").json()
    assert copy["task_id"] != task_id and copy["status"] == "planning"
    assert client.post(f"/api/tasks/{copy['task_id']}/decline").json()["status"] == "declined"
    assert client.post(f"/api/tasks/{copy['task_id']}/run-now").status_code == 409  # no plan

    assert {t["task_id"] for t in client.get("/api/tasks").json()} == {task_id, copy["task_id"]}
    assert client.delete(f"/api/tasks/{task_id}").json() == {"ok": True}
    assert client.get(f"/api/tasks/{task_id}").status_code == 404
    assert client.post("/api/tasks/nope/approve").status_code == 404
    assert client.post("/api/tasks", json={"prompt": "   "}).status_code == 400
