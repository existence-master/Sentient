from __future__ import annotations

import functools

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    llm = FakeProvider()
    core = SentientApp(config, llm=llm, db_path=isolated_home / "um.db", enable_background=False)
    with TestClient(create_app(core)) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        c.llm = llm
        c.core = core
        yield c


def test_user_model_routes(client):
    llm, core = client.llm, client.core
    empty = client.get("/api/user-model").json()
    assert empty == {"summary": "", "updated_at": None, "insights": [], "questions": []}

    assert client.post("/api/user-model/insights", json={"statement": ""}).status_code == 400
    ins = client.post("/api/user-model/insights", json={"statement": "Sarthak prefers tea", "dimension": "preferences"}).json()
    assert (ins["status"], ins["source"], ins["dimension"]) == ("confirmed", "user", "preferences")
    assert set(ins) == {"id", "dimension", "statement", "confidence", "status", "source", "evidence", "created_at", "updated_at"}

    patched = client.patch(f"/api/user-model/insights/{ins['id']}", json={"statement": "Sarthak prefers green tea"}).json()
    assert patched["statement"] == "Sarthak prefers green tea" and patched["status"] == "confirmed"
    assert client.patch(f"/api/user-model/insights/{ins['id']}", json={"status": "sleepy"}).status_code == 400
    assert client.patch("/api/user-model/insights/missing", json={"status": "retired"}).status_code == 404

    assert client.post("/api/user-model/refresh").json() == {"added": 0, "updated": 0, "disputed": 0, "questions": 0}
    sid = client.portal.call(functools.partial(core.store.create_session))
    client.portal.call(functools.partial(core.store.add_message, sid, "user", "switched to black coffee this month"))
    llm.json_replies.append(
        {"operations": [{"op": "contradict", "id": ins["id"], "evidence": ["m1"], "question": "Coffee over tea now?"}],
         "summary": "Sarthak likes hot drinks."}
    )
    assert client.post("/api/user-model/refresh").json() == {"added": 0, "updated": 0, "disputed": 0, "questions": 1}
    state = client.get("/api/user-model").json()
    assert state["summary"] == "Sarthak likes hot drinks." and state["updated_at"]
    [question] = state["questions"]
    assert set(question) == {"id", "question", "insight_id", "created_at"} and question["insight_id"] == ins["id"]

    assert client.post(f"/api/user-model/questions/{question['id']}", json={"answer": ""}).status_code == 400
    llm.json_replies.append({"verdict": "rewrite", "statement": "Sarthak drinks black coffee and green tea."})
    answered = client.post(f"/api/user-model/questions/{question['id']}", json={"answer": "Both, honestly"}).json()
    assert answered["ok"] is True and answered["insight"]["statement"] == "Sarthak drinks black coffee and green tea."
    assert client.post(f"/api/user-model/questions/{question['id']}", json={"answer": "x"}).status_code == 404
    assert client.delete(f"/api/user-model/questions/{question['id']}").status_code == 404

    assert client.delete(f"/api/user-model/insights/{ins['id']}").json() == {"ok": True}
    assert client.delete(f"/api/user-model/insights/{ins['id']}").status_code == 404


def test_dream_routes(client):
    assert client.get("/api/memories/dreams").json() == []
    started = client.post("/api/memories/dreams/run").json()
    assert started["status"] == "running" and started["trigger"] == "manual"
    assert set(started) == {"id", "started_at", "finished_at", "status", "trigger", "stats", "journal_md", "error"}
    listed = client.get("/api/memories/dreams", params={"limit": 5}).json()
    assert listed[0]["id"] == started["id"]
    one = client.get(f"/api/memories/dreams/{started['id']}").json()
    assert one["id"] == started["id"]
    assert client.get("/api/memories/dreams/nope").status_code == 404
    # the background dream finishes on the portal loop
    for _ in range(50):
        one = client.get(f"/api/memories/dreams/{started['id']}").json()
        if one["status"] != "running":
            break
    assert one["status"] == "completed" and "stats" in one
