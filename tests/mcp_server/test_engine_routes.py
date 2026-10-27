"""The engine side of the MCP server (issue #443, ADR 0034): a memory an outside app adds waits in
Review, and an outside caller can leave the user a plain info notification and nothing more.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider


@pytest.fixture
def client(config, isolated_home):
    app = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "g.db",
                                 enable_background=False))
    with TestClient(app) as c:
        c.token = c.app.state.token
        yield c


def _h(client) -> dict:
    return {"Authorization": f"Bearer {client.token}"}


def test_mcp_server_is_off_in_the_default_config(client):
    cfg = client.get("/api/config", headers=_h(client)).json()
    assert cfg["integrations"]["mcp_server_enabled"] is False
    assert cfg["integrations"]["mcp_server_memories_skip_review"] is False


def test_a_memory_held_for_review_waits_in_the_inbox_and_is_not_listed(client):
    r = client.post(
        "/api/memories",
        json={"content": "Maya's office is in Lisbon", "source": "mcp", "hold_for_review": True},
        headers=_h(client),
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "pending"

    listed = client.get("/api/memories", params={"q": "Lisbon"}, headers=_h(client)).json()
    assert listed == []  # a held memory is never listed or searched (ADR 0021)

    inbox = client.get("/api/memories/review", headers=_h(client)).json()
    held = [i for i in inbox["items"] if i["text"] == "Maya's office is in Lisbon"]
    assert len(held) == 1 and held[0]["from"] == "an MCP app"


def test_a_memory_saved_without_the_hold_is_active(client):
    r = client.post(
        "/api/memories", json={"content": "Maya drinks tea in the morning", "source": "mcp"},
        headers=_h(client),
    )
    assert r.status_code == 200, r.text
    assert r.json().get("status") != "pending"
    listed = client.get("/api/memories", params={"q": "tea"}, headers=_h(client)).json()
    assert [m["content"] for m in listed] == ["Maya drinks tea in the morning"]


def test_an_outside_caller_can_only_leave_an_info_notification(client):
    r = client.post(
        "/api/notifications",
        json={"message": "Claude Code finished the report", "title": "Report", "kind": "approval"},
        headers=_h(client),
    )
    assert r.status_code == 200, r.text
    note = r.json()
    assert note["kind"] == "info"  # the caller's kind is never trusted
    assert note["message"] == "Claude Code finished the report"

    listed = client.get("/api/notifications", headers=_h(client)).json()
    assert [n["id"] for n in listed["notifications"]] == [note["id"]]
    assert listed["unread"] == 1


def test_a_notification_needs_a_message(client):
    r = client.post("/api/notifications", json={"message": "  "}, headers=_h(client))
    assert r.status_code == 400
