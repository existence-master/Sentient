"""Fixtures for importing from Hermes: a copy of ``fixtures/hermes`` with traps that fail the test if read."""

from __future__ import annotations

import builtins
import hashlib
import io
import os
import shutil
from pathlib import Path

import pytest

from sentient import secrets
from sentient.app import SentientApp
from tests.conftest import FakeProvider

FIXTURE = Path(__file__).parent / "fixtures" / "hermes"
SECRET_NAMES = {"auth.json", ".env"}
TRAP = "TRAP-VALUE-THAT-MUST-NEVER-BE-READ"


def folder_hash(folder: Path) -> str:
    """Hermes' bundled-skill hash (tools/skills_sync.py ``_dir_hash``)."""
    h = hashlib.md5()
    for f in sorted(folder.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(folder)).encode("utf-8"))
            h.update(f.read_bytes())
    return h.hexdigest()


@pytest.fixture
def hermes_home(tmp_path) -> Path:
    """The fixture Hermes folder, with secrets files (also inside a skill) and the bundled-skill manifest:
    ``arxiv`` is bundled and unchanged, ``plan`` is bundled but edited by the user."""
    home = tmp_path / "hermes"
    shutil.copytree(FIXTURE, home)
    for trap in (home / "auth.json", home / ".env", home / "skills" / "productivity" / "weekly-review" / ".env"):
        trap.write_text(TRAP, encoding="utf-8")
    skills = home / "skills"
    (skills / ".bundled_manifest").write_text(
        f"arxiv:{folder_hash(skills / 'research' / 'arxiv')}\nplan:{hashlib.md5(b'original').hexdigest()}\n",
        encoding="utf-8",
    )
    return home


@pytest.fixture
def opened_secrets(monkeypatch) -> list[str]:
    """Fails loudly if anything opens auth.json or .env while the test runs."""
    seen: list[str] = []
    real_open, real_os_open = io.open, os.open

    def name_of(file) -> str:
        try:
            return os.path.basename(os.fsdecode(file))
        except TypeError:
            return ""

    def guarded_open(file, *args, **kwargs):
        if name_of(file) in SECRET_NAMES:
            seen.append(str(file))
            raise AssertionError(f"{file} must never be read")
        return real_open(file, *args, **kwargs)

    def guarded_os_open(path, *args, **kwargs):
        if name_of(path) in SECRET_NAMES:
            seen.append(str(path))
            raise AssertionError(f"{path} must never be read")
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    return seen


@pytest.fixture(autouse=True)
def keychain(monkeypatch) -> dict[str, str]:
    """In-memory keychain, so tests can check that nothing secret was stored."""
    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get_secret", lambda name, env_var=None: store.get(name))
    monkeypatch.setattr(secrets, "set_secret", lambda name, value: store.__setitem__(name, value) or True)
    monkeypatch.setattr(secrets, "delete_secret", lambda name: store.pop(name, None) is not None)
    return store


@pytest.fixture
async def app(config, isolated_home):
    llm = FakeProvider()
    a = SentientApp(config, llm=llm, db_path=isolated_home / "migrate.db", enable_background=False)
    await a.start()
    a.fake = llm
    try:
        yield a
    finally:
        await a.stop()
