"""Model presets (#212): switch every role between local and cloud in one step, save your own, undo."""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.config.schema import PRESET_CLOUD_MODELS, ModelRoles
from sentient.gateway.app import create_app
from sentient.llm import presets
from tests.conftest import FakeProvider

OLLAMA = "http://localhost:11434"
DEFAULTS = ModelRoles()
LOCAL_MAIN = DEFAULTS.primary.split("/", 1)[1]
LOCAL_EMBED = DEFAULTS.embedding.split("/", 1)[1]
ANTHROPIC = PRESET_CLOUD_MODELS["anthropic"]
OPENAI = PRESET_CLOUD_MODELS["openai"]


@pytest.fixture(autouse=True)
def keys(monkeypatch) -> dict[str, str]:
    """Provider keys by name; empty unless a test adds one. Never reads the real keychain or environment."""
    found: dict[str, str] = {}
    monkeypatch.setattr("sentient.secrets.get_secret", lambda name, env_var=None: found.get(name))
    return found


class FakeOllama:
    """``/api/tags`` on respx for every test, so nothing ever reaches a real Ollama."""

    def __init__(self) -> None:
        self.names: list[str] = [LOCAL_MAIN, f"{LOCAL_EMBED}:latest"]
        self.up = True

    def tags(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"models": [{"name": n} for n in self.names]})


@pytest.fixture(autouse=True)
def ollama():
    fake = FakeOllama()
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{OLLAMA}/api/tags").mock(side_effect=fake.tags)
        yield fake


@pytest.fixture
async def app(config, isolated_home):
    a = SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "p.db", enable_background=False)
    await a.start()
    try:
        yield a
    finally:
        await a.stop()


def by_name(listing: dict) -> dict:
    return {p["name"]: p for p in listing["presets"]}


async def test_cloud_and_mixed_are_disabled_without_a_key(app):
    items = by_name(await presets.listing(app))
    assert list(items) == ["Local only", "Cloud", "Mixed"]
    assert items["Local only"]["available"] is True
    for name in ("Cloud", "Mixed"):
        assert items[name]["available"] is False
        assert items[name]["reason"] == "Needs a key for Anthropic, OpenAI or OpenRouter."
    with pytest.raises(presets.PresetError) as err:
        await presets.apply(app, "Cloud")
    assert err.value.status == 409
    assert app.config.models.roles.primary == DEFAULTS.primary


async def test_apply_local_only_switches_every_role(app):
    app.config.models.roles = ModelRoles(primary="openai/gpt-x", fast="openai/gpt-y", planner="openai/gpt-z",
                                         voice="openai/gpt-y", embedding="openai/embed-x")
    app.config.models.fallbacks = {"primary": ["openai/gpt-y"]}
    result = await presets.apply(app, "local ONLY")  # names match without case
    roles = app.config.models.roles
    assert (roles.primary, roles.fast, roles.planner, roles.voice) == (DEFAULTS.primary, DEFAULTS.fast, None, None)
    assert roles.embedding == DEFAULTS.embedding  # a cloud embedding model is not "local only"
    assert app.config.models.fallbacks == {}
    assert result["preset"] == "Local only" and result["missing"] == [] and result["can_undo"] is True
    assert {c["role"] for c in result["changed"]} == {"primary", "fast", "planner", "voice", "embedding"}
    listing = await presets.listing(app)
    assert listing["active"] == "Local only" and listing["modified"] is False
    assert by_name(listing)["Local only"]["active"] is True
    # the switch is saved to disk, not only in memory
    from sentient.config.loader import load_config

    assert load_config().models.active_preset == "Local only"


async def test_cloud_uses_the_first_provider_with_a_key(app, keys, ollama):
    keys["openrouter"] = "sk-or"
    keys["openai"] = "sk-oa"
    ollama.names = ["mxbai-embed-large"]
    app.config.models.roles.embedding = "ollama/mxbai-embed-large"
    result = await presets.apply(app, "Cloud")
    roles = app.config.models.roles
    assert (roles.primary, roles.fast, roles.voice) == (OPENAI["main"], OPENAI["fast"], OPENAI["fast"])
    assert roles.embedding == "ollama/mxbai-embed-large"  # left alone: changing it re-indexes memory
    assert result["missing"] == []
    assert by_name(await presets.listing(app))["Cloud"]["provider"] == "openai"


async def test_mixed_keeps_background_work_and_memory_local(app, keys, ollama):
    keys["anthropic"] = "sk-ant"
    app.config.models.roles.embedding = "ollama/mxbai-embed-large"
    ollama.names = [LOCAL_MAIN, "mxbai-embed-large:latest"]
    result = await presets.apply(app, "Mixed")
    roles = app.config.models.roles
    assert roles.primary == ANTHROPIC["main"] and roles.planner is None and roles.voice == ANTHROPIC["fast"]
    assert roles.fast == DEFAULTS.fast
    assert roles.embedding == "ollama/mxbai-embed-large"  # the user's own local embedding model is kept
    assert result["missing"] == []


async def test_missing_models_and_keys_come_back_with_fixes(app, keys, ollama):
    ollama.names = []
    result = await presets.apply(app, "Local only")
    pulls = {m["model"]: m for m in result["missing"]}
    assert set(pulls) == {DEFAULTS.primary, DEFAULTS.embedding}
    main = pulls[DEFAULTS.primary]
    assert main["kind"] == "pull_model" and main["roles"] == ["primary", "fast"]
    assert main["action"] == {"kind": "pull_model", "name": LOCAL_MAIN, "label": f"Download {LOCAL_MAIN}"}

    ollama.up = False
    result = await presets.apply(app, "Local only")
    [down] = result["missing"]
    assert down["kind"] == "start_ollama" and set(down["roles"]) == {"primary", "fast", "embedding"}

    # a saved setup with a cloud model whose key was removed since
    keys["anthropic"] = "sk-ant"
    await presets.apply(app, "Cloud")
    presets.save_current(app, "Work")
    keys.clear()
    ollama.up, ollama.names = True, [LOCAL_MAIN, LOCAL_EMBED]
    result = await presets.apply(app, "Work")
    [nokey] = result["missing"]
    assert nokey["kind"] == "add_key" and nokey["provider"] == "anthropic"
    assert set(nokey["roles"]) == {"primary", "fast", "voice"}
    assert nokey["action"] == {"kind": "add_key", "provider": "anthropic", "label": "Add Anthropic key"}


async def test_save_rename_and_delete_your_own(app):
    app.config.models.roles.primary = "ollama_chat/qwen3:14b"
    app.config.models.context_length = 16384
    saved = presets.save_current(app, "  Big   local ")
    assert saved["name"] == "Big local" and saved["roles"]["primary"] == "ollama_chat/qwen3:14b"
    assert saved["context_length"] == 16384
    assert app.config.models.active_preset == "Big local"

    for bad, status in (("", 400), ("Cloud", 400), ("a/b", 400), ("big LOCAL", 409), ("x" * 41, 400)):
        with pytest.raises(presets.PresetError) as err:
            presets.save_current(app, bad)
        assert err.value.status == status, bad
    presets.save_current(app, "big local", overwrite=True)
    assert list(app.config.models.presets) == ["big local"]

    presets.rename(app, "BIG LOCAL", "Desk")
    assert list(app.config.models.presets) == ["Desk"] and app.config.models.active_preset == "Desk"
    for call in (lambda: presets.rename(app, "Cloud", "Mine"), lambda: presets.delete(app, "Local only")):
        with pytest.raises(presets.PresetError) as err:
            call()
        assert err.value.status == 400
    with pytest.raises(presets.PresetError) as err:
        presets.delete(app, "Nope")
    assert err.value.status == 404

    # switching back to a saved setup restores its other settings too
    await presets.apply(app, "Local only")
    app.config.models.context_length = 8192
    await presets.apply(app, "Desk")
    assert app.config.models.roles.primary == "ollama_chat/qwen3:14b" and app.config.models.context_length == 16384

    presets.delete(app, "desk")
    assert app.config.models.presets == {} and app.config.models.active_preset is None


async def test_undo_puts_the_previous_setup_back_once(app, keys):
    keys["anthropic"] = "sk-ant"
    app.config.models.fallbacks = {"primary": ["ollama_chat/llama3.1:8b"]}
    await presets.apply(app, "Local only")
    before = app.config.models.model_dump()
    await presets.apply(app, "Cloud")
    assert app.config.models.roles.primary == ANTHROPIC["main"]
    listing = await presets.listing(app)
    assert listing["can_undo"] is True and listing["undo_preset"] == "Local only"
    result = await presets.undo(app)
    assert result["preset"] == "Local only" and result["can_undo"] is False
    assert app.config.models.model_dump() == before
    assert {c["role"] for c in result["changed"]} == {"primary", "fast", "voice"}
    with pytest.raises(presets.PresetError) as err:
        await presets.undo(app)
    assert err.value.status == 409


async def test_changing_a_role_by_hand_marks_the_preset_modified(app):
    await presets.apply(app, "Local only")
    app.config.models.roles.fast = "ollama_chat/qwen3:4b"
    listing = await presets.listing(app)
    assert listing["active"] == "Local only" and listing["modified"] is True


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    app = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "r.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def test_preset_routes(client, keys):
    listing = client.get("/api/models/presets").json()
    assert [p["name"] for p in listing["presets"]] == ["Local only", "Cloud", "Mixed"]
    assert listing["active"] is None and listing["can_undo"] is False
    assert client.post("/api/models/presets/Cloud/apply").status_code == 409
    assert client.post("/api/models/presets/Nope/apply").status_code == 404
    assert client.post("/api/models/presets/undo").status_code == 409

    keys["anthropic"] = "sk-ant"
    r = client.post("/api/models/presets/Cloud/apply")
    assert r.status_code == 200 and r.json()["preset"] == "Cloud"
    assert client.get("/api/config").json()["models"]["roles"]["primary"] == ANTHROPIC["main"]

    saved = client.post("/api/models/presets", json={"name": "Travel"})
    assert saved.status_code == 200 and saved.json()["name"] == "Travel"
    assert client.post("/api/models/presets", json={"name": "Travel"}).status_code == 409
    assert client.patch("/api/models/presets/Travel", json={"name": "On the road"}).status_code == 200
    assert client.delete("/api/models/presets/Mixed").status_code == 400

    r = client.post("/api/models/presets/Local%20only/apply")
    assert r.status_code == 200 and r.json()["changed"]
    r = client.post("/api/models/presets/undo")
    assert r.status_code == 200 and r.json()["preset"] == "On the road"
    assert client.delete("/api/models/presets/On%20the%20road").json() == {"ok": True}
    assert client.get("/api/models/presets").json()["active"] is None
