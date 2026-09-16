from __future__ import annotations

import pytest

from sentient import secrets
from sentient.app import SentientApp
from sentient.tools.base import ToolContext
from tests.conftest import FakeProvider


@pytest.fixture(autouse=True)
def keychain(monkeypatch) -> dict[str, str]:
    """In-memory stand-in for the OS keychain so tests never touch the real one."""
    store: dict[str, str] = {}

    def get_secret(name: str, env_var: str | None = None) -> str | None:
        return store.get(name)

    def set_secret(name: str, value: str) -> bool:
        store[name] = value
        return True

    def delete_secret(name: str) -> bool:
        return store.pop(name, None) is not None

    monkeypatch.setattr(secrets, "get_secret", get_secret)
    monkeypatch.setattr(secrets, "set_secret", set_secret)
    monkeypatch.setattr(secrets, "delete_secret", delete_secret)
    return store


@pytest.fixture
async def app(config, isolated_home):
    a = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "integrations.db", enable_background=False)
    await a.start()
    try:
        yield a
    finally:
        await a.stop()


@pytest.fixture
def ctx(app) -> ToolContext:
    return app.agent.tool_context(None, "test")
