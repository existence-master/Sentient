from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider
from tests.sandbox.conftest import KitPlugin


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    config.tools.approvals.mode = "off"  # the route must stay read-only even when approvals are off
    config.sandbox.backend = "process"
    core = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False)
    app = create_app(core)
    with TestClient(app) as c:
        core.registry.register(KitPlugin())
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def test_status(client):
    body = client.get("/api/sandbox/status").json()
    assert body["enabled"] is True
    assert body["backend"] == "process"
    assert isinstance(body["docker_available"], bool)
    assert body["python_version"].startswith("3.")


def test_run_is_read_only(client):
    code = (
        "from sentient_tools import tools, result, ToolRefused\n"
        "try:\n    tools.kit_email_send(to='x')\n    sent = True\nexcept ToolRefused:\n    sent = False\n"
        "print('looked up', tools.kit_lookup(q='z')['echo'])\nresult({'sent': sent})\n"
    )
    body = client.post("/api/sandbox/run", json={"code": code}).json()
    assert body["ok"], body
    assert body["result"] == {"sent": False}
    assert "looked up z" in body["stdout"]
    assert body["tool_calls"] == 1


def test_run_requires_token(client):
    r = client.post("/api/sandbox/run", json={"code": "print(1)"}, headers={"Authorization": "Bearer nope"})
    assert r.status_code in {401, 403}
