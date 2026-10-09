"""Fixtures for the terminal package: a started app with the terminal on and one allowed project folder."""

from __future__ import annotations

import sys

import pytest

from sentient.app import SentientApp
from sentient.terminal import guard
from tests.conftest import FakeProvider


def py(code: str) -> str:
    """A shell command that runs ``code`` with this Python. ``code`` must not contain double quotes, $ or `."""
    exe = sys.executable
    return f"& '{exe}' -c \"{code}\"" if guard.IS_WINDOWS else f"'{exe}' -c \"{code}\""


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    return folder.resolve()


@pytest.fixture
async def make(config, isolated_home, project):
    """``await make(replies)`` starts an app whose scripted model makes ``replies``."""
    apps: list[SentientApp] = []
    if guard.find_shell() is None:
        pytest.skip("no command shell on this machine")
    config.tools.approvals.mode = "ask"
    config.terminal.enabled = True
    config.terminal.allowed_folders = [str(project)]

    async def factory(replies: list | None = None, name: str = "term") -> SentientApp:
        app = SentientApp(config, llm=FakeProvider(replies=replies or []), db_path=isolated_home / f"{name}.db",
                          enable_background=False)
        await app.start()
        apps.append(app)
        return app

    yield factory
    for app in apps:
        if app._started:
            await app.stop()
