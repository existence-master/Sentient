"""The keychain is per data folder (issue #302): the default folder keeps the old entry names, any other folder
gets its own service, and SENTIENT_KEYCHAIN_NAMESPACE overrides it. Uses a fake keyring, never the real one."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from mcp.shared.auth import OAuthToken

from sentient import secrets
from sentient.app import SentientApp
from sentient.integrations import mcp_auth
from tests.conftest import FakeProvider


class FakeKeyring:
    """keyring's module API over a dict keyed by (service, name), like every OS backend."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, name: str) -> str | None:
        return self.entries.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self.entries[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        if (service, name) not in self.entries:
            raise KeyError(name)  # keyring raises PasswordDeleteError
        del self.entries[(service, name)]

    def services(self) -> set[str]:
        return {svc for svc, _ in self.entries}


@pytest.fixture
def kr(monkeypatch) -> FakeKeyring:
    fake = FakeKeyring()
    monkeypatch.setattr(secrets, "_keyring", lambda: fake)
    monkeypatch.delenv(secrets.ENV_NAMESPACE, raising=False)
    return fake


@pytest.fixture
def user_home(tmp_path, monkeypatch) -> Path:
    """A stand-in for the user's own folder, so ``~/.sentient`` is a temporary folder too."""
    home = tmp_path / "user"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def use_home(monkeypatch, folder: Path | None) -> None:
    if folder is None:
        monkeypatch.delenv("SENTIENT_HOME", raising=False)
    else:
        monkeypatch.setenv("SENTIENT_HOME", str(folder))


def test_the_default_home_keeps_the_old_entry_names(kr, user_home, monkeypatch):
    kr.set_password("sentient", "channel_telegram_token", "tg-real")  # saved before #302
    kr.set_password("sentient", "mcp:composio:oauth", '1\n{"tokens":{"access_token":"a"}}')
    use_home(monkeypatch, None)
    assert secrets.namespace() == "default" and secrets.service() == "sentient"
    assert secrets.get_secret("channel_telegram_token") == "tg-real"
    assert secrets.load_json("mcp:composio:oauth") == {"tokens": {"access_token": "a"}}
    assert secrets.set_secret("openrouter", "or-key")
    assert kr.entries[("sentient", "openrouter")] == "or-key"
    assert kr.services() == {"sentient"}


@pytest.mark.parametrize("spelled", ["{home}/.sentient", "{home}/.sentient/", "{home}/x/../.sentient"])
def test_sentient_home_pointing_at_the_default_folder_is_the_default(kr, user_home, monkeypatch, spelled):
    use_home(monkeypatch, Path(spelled.format(home=user_home)))
    assert secrets.service() == "sentient"


def test_another_home_is_isolated(kr, user_home, monkeypatch, tmp_path):
    kr.set_password("sentient", "channel_telegram_token", "tg-real")
    use_home(monkeypatch, tmp_path / "demo")
    svc = secrets.service()
    assert svc.startswith("sentient-") and len(svc) == len("sentient-") + 12
    assert secrets.get_secret("channel_telegram_token") is None
    assert secrets.set_secret("channel_telegram_token", "tg-demo")
    assert secrets.delete_secret("channel_telegram_token")
    assert kr.entries == {("sentient", "channel_telegram_token"): "tg-real"}  # the default home's entry untouched

    secrets.set_secret("anthropic", "a-demo")
    use_home(monkeypatch, tmp_path / "other")
    assert secrets.service() not in {svc, "sentient"}
    assert secrets.get_secret("anthropic") is None


def test_a_homes_namespace_is_stable(kr, user_home, monkeypatch, tmp_path):
    folder = tmp_path / "later"
    before = secrets.service(folder)
    folder.mkdir()
    assert secrets.service(folder) == before  # the same before and after the folder exists
    assert secrets.service(Path(str(folder) + "/")) == before


def test_the_override(kr, user_home, monkeypatch, tmp_path):
    kr.set_password("sentient", "mcp:composio:oauth", '1\n{"tokens":{"access_token":"a"}}')
    use_home(monkeypatch, tmp_path / "real-test")
    monkeypatch.setenv(secrets.ENV_NAMESPACE, "default")  # a deliberate real test with the owner's sign-ins
    assert secrets.service() == "sentient"
    assert secrets.load_json("mcp:composio:oauth") == {"tokens": {"access_token": "a"}}
    assert "SENTIENT_KEYCHAIN_NAMESPACE" in secrets.describe()

    monkeypatch.setenv(secrets.ENV_NAMESPACE, "E2E test/302")
    assert secrets.service() == "sentient-e2e-test-302"
    assert secrets.load_json("mcp:composio:oauth") is None

    for odd in ("---", "!!", "default!", " Default "):  # only the word itself shares; anything else stays isolated
        monkeypatch.setenv(secrets.ENV_NAMESPACE, odd)
        assert secrets.service() != "sentient" or odd.strip().lower() == "default"

    monkeypatch.setenv(secrets.ENV_NAMESPACE, "E2E test/302")
    use_home(monkeypatch, None)  # the override applies to the default home too
    assert secrets.service() == "sentient-e2e-test-302"
    monkeypatch.setenv(secrets.ENV_NAMESPACE, "  ")  # blank means not set
    assert secrets.service() == "sentient"


def test_chunked_values_stay_in_their_namespace(kr, user_home, monkeypatch, tmp_path):
    kr.set_password("sentient", "chatgpt", "1\n{}")
    use_home(monkeypatch, tmp_path / "demo")
    svc = secrets.service()
    big = {"access_token": "x" * (secrets.CHUNK * 3), "refresh_token": "r"}
    assert secrets.save_json("chatgpt", big)
    names = {name for s, name in kr.entries if s == svc}
    assert names == {"chatgpt", "chatgpt:1", "chatgpt:2", "chatgpt:3"}
    assert secrets.load_json("chatgpt") == big
    secrets.delete_json("chatgpt")
    assert kr.entries == {("sentient", "chatgpt"): "1\n{}"}


async def test_mcp_sign_in_records_under_a_namespace(kr, user_home, monkeypatch, tmp_path):
    url = "https://mcp.example.test/mcp"
    tok_key = mcp_auth.tokens_secret("composio", url)
    real = '1\n{"tokens":{"access_token":"real"},"server_url":"' + url + '"}'
    kr.set_password("sentient", tok_key, real)
    use_home(monkeypatch, tmp_path / "demo")
    store = mcp_auth.KeychainTokenStorage("composio", url=url)
    assert await store.get_tokens() is None  # the default home's sign-in is not visible here
    await store.set_tokens(OAuthToken(access_token="t" * 2500, token_type="Bearer", expires_in=3600))
    rec = secrets.load_json(tok_key)
    assert rec["server_url"] == url and rec["expires_at"] > time.time()
    assert (await store.get_tokens()).access_token == "t" * 2500
    mcp_auth.forget_server("composio", url=url)
    assert kr.entries == {("sentient", tok_key): real}  # removing it here leaves the default home's


async def test_backup_lists_only_this_homes_sign_ins(kr, user_home, monkeypatch, config, isolated_home):
    kr.set_password("sentient", "openai", "sk-real")  # the default home's key
    app = await SentientApp(config, llm=FakeProvider(), enable_background=False).start()
    try:
        assert not [s for s in await app.backups.sign_ins() if s["kind"] == "model"]
        assert await app.backups._signed_in({"kind": "model", "id": "openai"}) is False
        secrets.set_secret("anthropic", "sk-demo")
        assert [s["id"] for s in await app.backups.sign_ins() if s["kind"] == "model"] == ["anthropic"]
    finally:
        await app.stop()


def _seed_safety():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "desktop" / "scripts" / "seed_safety.py"
    spec = importlib.util.spec_from_file_location("seed_safety", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_seed_safety_pauses_tasks_and_channel_deliveries(config, isolated_home, user_home):
    from sentient.config.loader import load_config, save_config

    save_config(config)
    app = await SentientApp(config, llm=FakeProvider(), enable_background=False).start()
    try:
        await app.tasks.repo.insert_task({"id": "demo-daily", "name": "Daily digest", "status": "active",
                                          "enabled": True, "plan": [], "created_at": "2026-10-01T00:00:00Z",
                                          "updated_at": "2026-10-01T00:00:00Z"})
        await app.channels.store.add_chat("telegram", "100200301", "Maya Rao", deliver=True, session_id=None)
    finally:
        await app.stop()

    report = _seed_safety().quiet(isolated_home)
    assert report == {"tasks_paused": 1, "plans_stopped": 0, "runs_stopped": 0, "chats_muted": 1, "channels_off": True}
    assert load_config().channels.enabled is False
    app = await SentientApp(load_config(), llm=FakeProvider(), enable_background=False).start()
    try:
        assert (await app.tasks.repo.get_task("demo-daily"))["enabled"] is False
        assert (await app.channels.store.chat("telegram", "100200301"))["deliver"] is False
    finally:
        await app.stop()


class NoModel(FakeProvider):
    """A model that must never be called: every call is recorded and fails."""

    def __init__(self) -> None:
        super().__init__()
        self.used: list[str] = []

    async def stream(self, role, messages, tools=None, *, model=None):
        self.used.append(f"stream:{role}")
        raise RuntimeError("seed_safety let a model call through")
        yield  # pragma: no cover

    async def complete_text(self, role, messages, *, model=None):
        self.used.append(f"text:{role}")
        raise RuntimeError("seed_safety let a model call through")

    async def complete_json(self, role, messages, *, model=None):
        self.used.append(f"json:{role}")
        raise RuntimeError("seed_safety let a model call through")

    async def embed(self, texts, *, model=None):
        self.used.append("embed")
        raise RuntimeError("seed_safety let a model call through")


async def test_seed_safety_keeps_planning_tasks_and_live_runs_from_calling_the_model(config, isolated_home, user_home):
    """#388: the engine plans 'planning' tasks and resumes 'processing' runs on start even when they are paused."""
    from sentient.config.loader import load_config, save_config

    save_config(config)
    now = "2026-10-01T00:00:00+00:00"
    app = await SentientApp(config, llm=FakeProvider(), enable_background=False).start()
    try:
        repo = app.tasks.repo
        await repo.insert_task({"id": "demo-planning", "name": "Find gift ideas for Anika", "status": "planning",
                                "enabled": True, "plan": [], "original_prompt": "Find gift ideas for Anika",
                                "created_at": now, "updated_at": now})
        await repo.insert_task({"id": "demo-running", "name": "Collect onboarding examples", "status": "processing",
                                "enabled": True, "plan": [{"tool": "web", "description": "Read the pages"}],
                                "schedule": {"type": "once", "run_at": None, "timezone": "UTC"},
                                "created_at": now, "updated_at": now})
        run_id = await repo.insert_run("demo-running", now=now, plan=[{"tool": "web", "description": "Read the pages"}])
        # finished but never settled, with no result: the engine would ask the model to write one on start
        await repo.insert_task({"id": "demo-unsettled", "name": "Sort the inbox", "status": "processing",
                                "enabled": True, "plan": [{"tool": "gmail", "description": "Sort it"}],
                                "schedule": {"type": "recurring", "frequency": "daily", "time": "09:00"},
                                "created_at": now, "updated_at": now})
        done = await repo.insert_run("demo-unsettled", now=now, plan=[])
        await repo.update_run(done, {"status": "completed", "finished_at": now})
    finally:
        await app.stop()

    report = _seed_safety().quiet(isolated_home)
    assert report["plans_stopped"] == 1 and report["runs_stopped"] == 1
    llm = NoModel()
    app = await SentientApp(load_config(), llm=llm, enable_background=True).start()
    try:
        await app.tasks.drain(timeout=10)
        assert llm.used == []
        planning = await app.tasks.repo.get_task("demo-planning")
        assert (planning["status"], planning["enabled"]) == ("declined", False)
        assert (await app.tasks.repo.get_task("demo-running"))["status"] == "cancelled"
        run = await app.tasks.repo.get_run(run_id)
        assert run["status"] == "cancelled" and run["finished_at"]
        assert (await app.tasks.repo.get_task("demo-unsettled"))["status"] == "active"
    finally:
        await app.stop()


def test_seed_safety_refuses_the_default_home(user_home):
    with pytest.raises(SystemExit):
        _seed_safety().quiet(user_home / ".sentient")
