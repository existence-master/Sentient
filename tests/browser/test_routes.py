"""Gateway routes and plugin visibility (no browser launch)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.browser import service as browser_service
from sentient.gateway.app import create_app
from tests.conftest import FakeProvider


@pytest.fixture
def make_client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    clients = []

    def build(**browser_cfg):
        for k, v in browser_cfg.items():
            setattr(config.browser, k, v)
        app = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "b.db", enable_background=False))
        c = TestClient(app)
        c.__enter__()
        c.headers.update({"Authorization": "Bearer test-token"})
        clients.append(c)
        return c

    yield build
    for c in clients:
        c.__exit__(None, None, None)


def test_disabled_browser_hides_plugin(make_client):
    c = make_client(enabled=False)
    st = c.get("/api/browser/status").json()
    assert st["available"] is False and st["running"] is False and st["tabs"] == []
    assert "turned off" in st["error"]
    core = c.app.state.sentient
    assert core.registry.is_hidden("browser") and core.registry.get("browser_click") is not None
    assert c.get("/api/browser/screenshot").status_code == 409
    r = c.post("/api/browser/open", json={"url": "https://example.com"})
    assert r.status_code == 409 and "turned off" in r.json()["detail"]
    assert c.post("/api/browser/close").json()["running"] is False
    assert c.get("/api/browser/status", headers={"Authorization": "Bearer nope"}).status_code in {401, 403}


def test_missing_browser_explained(make_client, monkeypatch):
    monkeypatch.setattr(browser_service, "installed_engines", lambda pref="auto": [])
    c = make_client()
    st = c.get("/api/browser/status").json()
    assert st["available"] is False and "Install Microsoft Edge or Google Chrome" in st["error"]
    assert c.app.state.sentient.registry.is_hidden("browser")


def test_enabled_browser_visible_with_risk_fns(make_client, monkeypatch):
    monkeypatch.setattr(browser_service, "installed_engines", lambda pref="auto": ["msedge"])
    c = make_client()
    core = c.app.state.sentient
    assert not core.registry.is_hidden("browser")
    st = c.get("/api/browser/status").json()
    assert st == {"available": True, "running": False, "engine": "msedge", "headless": True, "tabs": [], "error": None}
    names = {t.name for t in core.registry.tools() if t.plugin == "browser"}
    assert {"browser_open", "browser_snapshot", "browser_click", "browser_type", "browser_select", "browser_press",
            "browser_scroll", "browser_back", "browser_tabs", "browser_switch_tab", "browser_extract",
            "browser_screenshot", "browser_close"} <= names
    click = core.registry.get("browser_click")
    assert callable(getattr(click, "risk_fn", None))
    assert "never complete a purchase" in click.description
    # toggling the setting off hides the tools again
    r = c.patch("/api/config", json={"browser": {"enabled": False}})
    assert r.status_code == 200
    for _ in range(50):
        if core.registry.is_hidden("browser"):
            break
        c.get("/api/health")
    assert core.registry.is_hidden("browser")
