"""Model check-up (#132): per-role reply, tool call, JSON, thinking, context and GPU checks with plain fixes."""

import asyncio
import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.gateway.app import create_app
from sentient.llm import hardware
from sentient.llm.checkup import checkup, run_checkup
from tests.conftest import FakeProvider, tool_call

OLLAMA = "http://localhost:11434"
GOOD_TOOLS = [[tool_call("find_city", name="Paris")], [tool_call("get_weather", city_id="c-42")]]


def ollama(mock, *, show=None, ps=None, tags=("qwen3:8b",), up=True):
    """Mock Ollama's native API: /api/version, /api/show, /api/ps and /api/tags."""
    if up:
        mock.get(f"{OLLAMA}/api/version").respond(200, json={"version": "0.12.0"})
    else:
        mock.get(f"{OLLAMA}/api/version").mock(side_effect=httpx.ConnectError("refused"))
    default_show = {"capabilities": ["completion", "tools", "thinking"], "model_info": {"qwen3.context_length": 40960}}
    mock.post(f"{OLLAMA}/api/show").respond(200, json=default_show if show is None else show)
    mock.get(f"{OLLAMA}/api/ps").respond(200, json={"models": ps or []})
    mock.get(f"{OLLAMA}/api/tags").respond(200, json={"models": [{"name": t} for t in tags]})


def by_id(result: dict) -> dict:
    return {c["id"]: c for c in result["checks"]}


async def test_pass_case_local_model_on_gpu(config):
    llm = FakeProvider(replies=["ready", *GOOD_TOOLS])
    with respx.mock(assert_all_called=False) as mock:
        ollama(mock, ps=[{"name": "qwen3:8b", "size": 6_000_000_000, "size_vram": 6_000_000_000}])
        done = await checkup(config, llm, {"primary": "ollama_chat/qwen3:8b"})
    [role] = done["roles"]
    checks = by_id(role)
    assert done["status"] == "pass" and role["status"] == "pass", role
    assert list(checks) == ["connection", "reply", "tools", "chain", "thinking", "context", "gpu"]
    assert "8,192" in checks["context"]["detail"] and "40,960" in checks["context"]["detail"]
    assert checks["gpu"]["detail"].startswith("Runs fully on the graphics card")
    assert not any("fix" in c for c in role["checks"])
    # the scripted tool call went to the role's own model, with both test tools
    assert llm.calls[1]["model"] == "ollama_chat/qwen3:8b" and len(llm.calls[1]["tools"]) == 2
    assert llm.calls[2]["messages"][-1]["role"] == "tool"


async def test_tool_call_failure_suggests_a_model_that_can(config):
    llm = FakeProvider(replies=["ready", "It is sunny in Paris."])  # answers in text instead of calling the tool
    with respx.mock(assert_all_called=False) as mock:
        ollama(mock, tags=("qwen3:4b", "qwen3:8b"))
        done = await checkup(config, llm, {"primary": "ollama_chat/qwen3:4b"})
    [role] = done["roles"]
    tools = by_id(role)["tools"]
    assert role["status"] == "fail" and tools["status"] == "fail"
    assert tools["fix"] == "qwen3:4b can't call tools reliably; try qwen3:8b."
    assert tools["action"] == {"kind": "use_model", "role": "primary", "model": "ollama_chat/qwen3:8b", "label": "Use qwen3:8b"}
    assert "chain" not in by_id(role)  # no second step after a failed first one


async def test_tool_failure_offers_download_when_the_better_model_is_missing(config):
    llm = FakeProvider(replies=["ready", [tool_call("get_weather", city_id="Paris")]])
    with respx.mock(assert_all_called=False) as mock:
        ollama(mock, tags=("llama3.2:1b",))
        done = await checkup(config, llm, {"executor": "ollama_chat/llama3.2:1b"})
    tools = by_id(done["roles"][0])["tools"]
    assert tools["status"] == "fail" and "get_weather" in tools["detail"]
    assert tools["action"] == {"kind": "pull_model", "name": "qwen3:8b", "label": "Download qwen3:8b"}


async def test_cpu_offload_warns_and_offers_a_shorter_context(config):
    config.models.context_length = 16384
    llm = FakeProvider(replies=["ready", [tool_call("find_city", name="Paris")]], json_replies=[{"ok": True}])
    with respx.mock(assert_all_called=False) as mock:
        ollama(mock, ps=[{"name": "qwen3:8b", "size": 8_000_000_000, "size_vram": 6_000_000_000}])
        done = await checkup(config, llm, {"fast": "ollama_chat/qwen3:8b"})
    [role] = done["roles"]
    checks = by_id(role)
    assert role["status"] == "warn" and done["status"] == "warn"
    gpu = checks["gpu"]
    assert gpu["status"] == "warn" and "25% of this model on the processor" in gpu["detail"]
    assert gpu["fix"] == "It will be slow. Pick a smaller model or a shorter context length."
    assert gpu["action"] == {"kind": "set_context_length", "value": 8192, "role": None, "label": "Use 8,192 tokens"}
    assert checks["json"]["status"] == "pass"  # fast needs JSON
    assert "chain" not in checks  # only primary and executor run two tool steps


async def test_unreachable_ollama_and_missing_model(config):
    with respx.mock(assert_all_called=False) as mock:
        ollama(mock, up=False)
        done = await checkup(config, FakeProvider(), {"primary": "ollama_chat/qwen3:8b"})
    [role] = done["roles"]
    assert role["status"] == "fail" and [c["id"] for c in role["checks"]] == ["connection"]
    assert "Start the Ollama app" in role["checks"][0]["fix"]

    with respx.mock(assert_all_called=False) as mock:
        ollama(mock)
        mock.post(f"{OLLAMA}/api/show").respond(404, json={"error": "model 'qwen3:14b' not found"})
        done = await checkup(config, FakeProvider(), {"primary": "ollama_chat/qwen3:14b"})
    connection = done["roles"][0]["checks"][0]
    assert connection["status"] == "fail" and "isn't downloaded" in connection["detail"]
    assert connection["action"] == {"kind": "pull_model", "name": "qwen3:14b", "label": "Download qwen3:14b"}


async def test_a_model_that_never_answers_times_out(config):
    class Slow(FakeProvider):
        async def stream(self, role, messages, tools=None, *, model=None):
            await asyncio.sleep(5)
            yield  # pragma: no cover

    with respx.mock(assert_all_called=False) as mock:
        ollama(mock, ps=[{"name": "qwen3:8b", "size": 6_000_000_000, "size_vram": 0}])
        done = await checkup(config, Slow(), {"voice": "ollama_chat/qwen3:8b"}, timeout_s=0.05)
    checks = by_id(done["roles"][0])
    assert checks["reply"]["status"] == "fail" and "No reply within" in checks["reply"]["fix"]
    assert "tools" not in checks
    assert "all of this model" in checks["gpu"]["detail"]  # the GPU check still explains why


async def test_cloud_role_skips_ollama_checks(config, monkeypatch):
    monkeypatch.setattr("sentient.llm.checkup.secrets.get_secret", lambda name, env=None: "sk-test" if name == "anthropic" else None)
    llm = FakeProvider(replies=["ready", *GOOD_TOOLS])
    with respx.mock(assert_all_called=False) as mock:
        done = await checkup(config, llm, {"primary": "anthropic/claude-sonnet-5"})
        assert mock.calls.call_count == 0  # no Ollama requests for a cloud model
    [role] = done["roles"]
    assert role["status"] == "pass" and role["local"] is False and role["provider"] == "anthropic"
    assert [c["id"] for c in role["checks"]] == ["connection", "reply", "tools", "chain"]

    monkeypatch.setattr("sentient.llm.checkup.secrets.get_secret", lambda name, env=None: None)
    done = await checkup(config, FakeProvider(), {"primary": "anthropic/claude-sonnet-5"})
    connection = done["roles"][0]["checks"][0]
    assert connection["status"] == "fail" and connection["fix"] == "Add your Anthropic key under Providers."


async def test_thinking_left_on_for_a_background_role(config):
    class Thinker(FakeProvider):
        async def stream(self, role, messages, tools=None, *, model=None):
            from sentient.llm.provider import StreamChunk

            yield StreamChunk(thinking="hmm", model="fake")
            async for chunk in super().stream(role, messages, tools, model=model):
                yield chunk

    config.models.reasoning.pop("executor")
    with respx.mock(assert_all_called=False) as mock:
        ollama(mock)
        done = await checkup(config, Thinker(replies=["ready", *GOOD_TOOLS]), {"executor": "ollama_chat/qwen3:8b"})
    thinking = by_id(done["roles"][0])["thinking"]
    assert thinking["status"] == "warn"
    assert thinking["action"] == {"kind": "set_reasoning", "role": "executor", "value": "none", "label": "Turn reasoning off"}


async def test_events_stream_and_optional_roles_use_primary(config):
    config.models.roles.primary = "fake/main"
    config.models.roles.fast = "fake/main"
    config.models.roles.embedding = "fake/embed"
    llm = FakeProvider(replies=["ready", *GOOD_TOOLS, "ready", GOOD_TOOLS[0]], json_replies=[{"ok": True}])
    events = [e async for e in run_checkup(config, llm)]
    assert events[0]["type"] == "start" and [r["role"] for r in events[0]["roles"]][:2] == ["primary", "fast"]
    assert any(e["type"] == "step" and e["label"] == "Trying a tool call" for e in events)
    done = events[-1]
    assert done["type"] == "done" and done["status"] == "pass"
    roles = {r["role"]: r for r in done["roles"]}
    assert roles["planner"]["inherits"] == "primary" and roles["planner"]["status"] == "skip"
    assert by_id(roles["embedding"])["embedding"]["detail"] == "Works (16 dimensions)."


@pytest.fixture
def client(config, isolated_home, monkeypatch):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    # the route sizes its hints for this computer: never run the real detection or reach Ollama here
    monkeypatch.setattr(hardware, "detect", lambda: {"os": "linux", "ram_gb": None, "gpus": [], "unified_memory": False,
                                                     "usable_vram_gb": None, "ollama_vram_gb": None})

    async def no_ollama(base):
        return None

    monkeypatch.setattr(hardware, "ollama_vram_gb", no_ollama)
    llm = FakeProvider(replies=["ready", *GOOD_TOOLS])
    app = create_app(SentientApp(config, llm=llm, db_path=isolated_home / "r.db", enable_background=False))
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


def test_checkup_route_streams_ndjson_and_never_changes_config(client):
    before = client.get("/api/config").json()
    r = client.post("/api/models/checkup", json={"roles": {"primary": "fake/main"}})
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert lines[0]["type"] == "start" and lines[-1]["type"] == "done"
    assert lines[0]["hardware"]["summary"] == "unknown"
    assert lines[-1]["roles"][0]["status"] == "pass"
    assert client.get("/api/config").json() == before
    assert client.post("/api/models/checkup", json={"roles": {"boss": "fake/x"}}).status_code == 400
    # an empty selection checks nothing (it is not "every role")
    empty = [json.loads(line) for line in client.post("/api/models/checkup", json={"roles": {}}).text.splitlines()]
    assert empty[0]["roles"] == [] and empty[-1]["roles"] == []


async def test_doctor_models_table(config):
    from types import SimpleNamespace

    from rich.console import Console

    from sentient.cli import _model_checkup

    config.models.roles.primary = "fake/main"
    config.models.roles.fast = "fake/main"
    config.models.roles.embedding = "fake/embed"
    llm = FakeProvider(replies=["ready", "no tools here", "ready", GOOD_TOOLS[0]], json_replies=[{"ok": True}])
    table = await _model_checkup(SimpleNamespace(config=config, llm=llm))
    out = Console(width=200, record=True)
    out.print(table)
    text = out.export_text()
    assert "Tool call" in text and "fail" in text and "Fix:" in text and "Uses the primary model." in text


async def test_roles_with_the_same_model_and_settings_share_results(config):
    config.models.reasoning["fast"] = "none"
    config.models.reasoning["primary"] = "none"
    llm = FakeProvider(replies=["ready", *GOOD_TOOLS], json_replies=[{"ok": True}])
    with respx.mock(assert_all_called=False) as mock:
        ollama(mock)
        done = await checkup(config, llm, {"primary": "ollama_chat/qwen3:8b", "fast": "ollama_chat/qwen3:8b"})
    primary, fast = (by_id(r) for r in done["roles"])
    # one reply, one tool call and one second step in total; fast only adds its own JSON check
    assert sum(1 for c in llm.calls if not c.get("json")) == 3 and sum(1 for c in llm.calls if c.get("json")) == 1
    assert fast["reply"]["detail"].startswith("Same as primary. Answered in")
    assert fast["tools"]["detail"] == "Same as primary. Called the test tool correctly."
    assert "chain" not in fast and fast["json"]["detail"] == "Returned clean JSON."
    assert not primary["reply"]["detail"].startswith("Same as")

    # a different reasoning setting is a different test: everything runs again
    config.models.reasoning["fast"] = "low"
    llm = FakeProvider(replies=["ready", *GOOD_TOOLS, "ready", GOOD_TOOLS[0]], json_replies=[{"ok": True}])
    with respx.mock(assert_all_called=False) as mock:
        ollama(mock)
        done = await checkup(config, llm, {"primary": "ollama_chat/qwen3:8b", "fast": "ollama_chat/qwen3:8b"})
    fast = by_id(done["roles"][1])
    assert sum(1 for c in llm.calls if not c.get("json")) == 5
    assert not fast["reply"]["detail"].startswith("Same as")


async def test_shared_tool_failure_points_its_fix_at_each_role(config):
    config.models.reasoning["executor"] = "none"
    config.models.reasoning["voice"] = "none"
    llm = FakeProvider(replies=["ready", "It is sunny."])
    with respx.mock(assert_all_called=False) as mock:
        ollama(mock, tags=("qwen3:4b", "qwen3:8b"))
        done = await checkup(config, llm, {"executor": "ollama_chat/qwen3:4b", "voice": "ollama_chat/qwen3:4b"})
    executor, voice = (by_id(r)["tools"] for r in done["roles"])
    assert executor["action"]["role"] == "executor" and voice["action"]["role"] == "voice"
    assert voice["detail"].startswith("Same as executor.") and len(llm.calls) == 2
