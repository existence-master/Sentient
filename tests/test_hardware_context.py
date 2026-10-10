"""Hardware-aware local models and the context meter (#131).

Detection is mocked everywhere: no test runs nvidia-smi, reads the registry or reaches Ollama.
"""

import subprocess
import sys

import httpx
import litellm
import pytest
import respx
from fastapi.testclient import TestClient

from sentient.app import SentientApp
from sentient.config.schema import LOCAL_MODEL_TIERS, SentientConfig
from sentient.gateway.app import create_app
from sentient.llm import hardware, meter, presets
from sentient.llm.checkup import checkup, run_checkup
from sentient.llm.events import Usage
from sentient.llm.provider import LiteLLMProvider
from sentient.tasks.executor import ProgressMapper
from tests.conftest import FakeProvider, tool_call

OLLAMA = "http://localhost:11434"
TIER = {row["id"]: row for row in LOCAL_MODEL_TIERS}


def fake_hw(vram: float | None = None, ram: float | None = 32.0, name: str = "NVIDIA GeForce RTX 4060") -> dict:
    gpus = [hardware._gpu(name, vram, "nvidia")] if vram else []
    return {"os": "windows", "ram_gb": ram, "gpus": gpus, "unified_memory": False,
            "usable_vram_gb": vram, "ollama_vram_gb": None}


@pytest.fixture
def detected(monkeypatch):
    """Replace detection with a scripted answer and count how often it runs."""
    state = {"hw": fake_hw(8.0, 16.0), "calls": 0}

    def detect():
        state["calls"] += 1
        return dict(state["hw"])

    monkeypatch.setattr(hardware, "detect", detect)
    return state


# ----------------------------------------------------------------------------- detection
def test_nvidia_smi_output_is_parsed_with_or_without_header_and_units():
    plain = "NVIDIA GeForce RTX 4060 Laptop GPU, 8188\nNVIDIA RTX A6000, 49140\n"
    gpus = hardware.parse_nvidia_smi(plain)
    assert [(g["name"], g["vram_gb"], g["vendor"], g["usable"]) for g in gpus] == [
        ("NVIDIA GeForce RTX 4060 Laptop GPU", 8.0, "nvidia", True), ("NVIDIA RTX A6000", 48.0, "nvidia", True)]
    csv = "name, memory.total [MiB]\nNVIDIA GeForce RTX 3090, 24576 MiB\n"
    assert [(g["name"], g["vram_gb"]) for g in hardware.parse_nvidia_smi(csv)] == [("NVIDIA GeForce RTX 3090", 24.0)]
    assert hardware.parse_nvidia_smi("") == [] and hardware.parse_nvidia_smi("No devices were found") == []
    [odd] = hardware.parse_nvidia_smi("Some GPU, [N/A]")
    assert odd["vram_gb"] is None


def test_missing_or_failing_tools_read_as_nothing(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    monkeypatch.setattr(hardware.os.path, "exists", lambda path: False)
    assert hardware.nvidia_gpus() == []

    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    assert hardware._run(["nvidia-smi"]) is None

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(subprocess, "run", slow)
    assert hardware._run(["nvidia-smi"]) is None


def test_detect_never_fails_when_every_probe_does(monkeypatch):
    def broken():
        raise RuntimeError("no access")

    for probe in ("total_ram_gb", "nvidia_gpus", "windows_gpus", "linux_amd_gpus"):
        monkeypatch.setattr(hardware, probe, broken)
    hw = hardware.detect()
    assert hw["ram_gb"] is None and hw["gpus"] == [] and hw["usable_vram_gb"] is None
    assert hardware.recommend(hw)["tier"] == "unknown"


def test_apple_silicon_counts_two_thirds_of_memory(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware, "total_ram_gb", lambda: 36.0)
    monkeypatch.setattr(hardware, "nvidia_gpus", list)
    monkeypatch.setattr(hardware, "_run", lambda cmd: "Apple M3 Pro\n")
    hw = hardware.detect()
    assert hw["os"] == "macos" and hw["unified_memory"] is True
    assert hw["gpus"][0]["name"] == "Apple M3 Pro" and hw["usable_vram_gb"] == 24.0
    assert hardware.recommend(hw)["tier"] == "gpu_24"
    assert hardware.summary(hw) == "Apple M3 Pro, 36 GB of memory"


def test_built_in_intel_graphics_is_not_counted():
    hw = {"gpus": [hardware._gpu("Intel(R) UHD Graphics", 0.1)], "ram_gb": 16.0}
    assert hw["gpus"][0]["usable"] is False
    assert hardware.summary(hw) == "no graphics card a local model can use, 16 GB of memory"


# ----------------------------------------------------------------------------- tiers
@pytest.mark.parametrize(
    ("vram", "ram", "tier", "runs_on"),
    [
        (8.0, 16.0, "gpu_8", "graphics"),  # qwen3:8b at 8,192 on an 8 GB card, as before
        (6.0, 32.0, "cpu", "processor"),  # too little graphics memory: sized by total memory
        (None, 32.0, "cpu", "processor"),
        (None, 7.8, "cpu", "processor"),  # an "8 GB" computer still runs qwen3:8b, slowly
        (None, 3.8, "small", "processor"),
        (12.0, 32.0, "gpu_12", "graphics"),
        (16.0, 32.0, "gpu_16", "graphics"),
        (24.0, 64.0, "gpu_24", "graphics"),
        (None, None, "unknown", "unknown"),
    ],
)
def test_tier_selection(vram, ram, tier, runs_on):
    rec = hardware.recommend(fake_hw(vram, ram))
    assert rec["tier"] == tier and rec["runs_on"] == runs_on and rec["note"]
    if tier != "unknown":
        assert rec["model"] == TIER[tier]["model"] and rec["context_length"] == TIER[tier]["context_length"]
    else:  # today's defaults
        assert rec["model"] == "ollama_chat/qwen3:8b" and rec["context_length"] == 8192


def test_little_memory_recommends_a_cloud_model_and_never_a_chat_only_one_as_main():
    rec = hardware.recommend(fake_hw(None, 3.8))
    assert rec["cloud_first"] is True and rec["name"] == "qwen3:4b"
    assert rec["summary"] == "a cloud model (qwen3:4b, reading 8,192 tokens at a time for chat only)"
    assert "cloud model is the better choice" in rec["note"] and "chat only" in rec["note"]
    assert all(not hardware.recommend(fake_hw(v, r))["cloud_first"] for v, r in ((8.0, 16.0), (None, 7.8), (None, 32.0)))
    slow = hardware.recommend(fake_hw(None, 16.0))
    assert slow["name"] == "qwen3:8b" and "slow" in slow["note"] and "cloud model is much faster" in slow["note"]


async def test_chat_only_fallback_never_reaches_presets_or_check_up_fixes(config):
    hw = with_rec(fake_hw(None, 3.8))
    local = presets.presets(config, hw)[0]
    assert local["roles"]["primary"] == "ollama_chat/qwen3:8b" and "context_length" not in local
    assert "struggle with tasks" in local["description"]
    llm = FakeProvider(replies=["ready", "It is sunny in Paris."])
    with respx.mock(assert_all_called=False) as mock:
        ollama_api(mock, tags=("llama3.2:1b",))
        done = await checkup(config, llm, {"primary": "ollama_chat/llama3.2:1b"}, hardware=hw)
    tools = {c["id"]: c for c in done["roles"][0]["checks"]}["tools"]
    assert tools["action"] == {"kind": "pull_model", "name": "qwen3:8b", "label": "Download qwen3:8b"}


def test_eight_gb_card_keeps_qwen3_8b_at_8192():
    rec = hardware.recommend(fake_hw(8.0, 16.0))
    assert (rec["name"], rec["context_length"]) == ("qwen3:8b", 8192)
    assert rec["summary"] == "qwen3:8b, reading 8,192 tokens at a time"


def test_a_model_loaded_on_the_graphics_card_proves_memory_detection_missed():
    hw = {**fake_hw(None, 32.0), "ollama_vram_gb": 9.5}
    assert hardware.recommend(hw)["tier"] == "gpu_8"


# ----------------------------------------------------------------------------- route, cache and presets
@pytest.fixture
def client(config, isolated_home, monkeypatch, detected):
    monkeypatch.setenv("SENTIENT_GATEWAY_TOKEN", "test-token")
    app = create_app(SentientApp(config, llm=FakeProvider(), db_path=isolated_home / "h.db", enable_background=False))
    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{OLLAMA}/api/ps").respond(200, json={"models": []})
        mock.get(f"{OLLAMA}/api/tags").respond(200, json={"models": [{"name": "qwen3:8b"}]})
        with TestClient(app) as c:
            c.headers.update({"Authorization": "Bearer test-token"})
            yield c


def test_hardware_route_recommends_and_caches(client, detected):
    body = client.get("/api/system/hardware").json()
    assert body["ram_gb"] == 16.0 and body["usable_vram_gb"] == 8.0 and body["gpus"][0]["vendor"] == "nvidia"
    assert body["summary"] == "NVIDIA GeForce RTX 4060 with 8 GB of graphics memory, 16 GB of memory"
    assert body["recommendation"]["model"] == "ollama_chat/qwen3:8b"
    assert body["recommendation"]["context_length"] == 8192
    client.get("/api/system/hardware")
    assert detected["calls"] == 1  # cached
    detected["hw"] = fake_hw(16.0, 32.0)
    again = client.get("/api/system/hardware", params={"refresh": "true"}).json()
    assert detected["calls"] == 2 and again["recommendation"]["tier"] == "gpu_16"


def test_hardware_route_says_unknown_when_nothing_is_found(client, detected):
    detected["hw"] = fake_hw(None, None)
    body = client.get("/api/system/hardware").json()
    assert body["summary"] == "unknown" and body["recommendation"]["tier"] == "unknown"


def test_local_only_preset_is_sized_for_this_computer(client, detected):
    plain = {p["name"]: p for p in client.get("/api/models/presets").json()["presets"]}["Local only"]
    assert "context_length" not in plain  # not checked yet: the usual defaults
    detected["hw"] = fake_hw(16.0, 32.0)
    client.get("/api/system/hardware")
    local = {p["name"]: p for p in client.get("/api/models/presets").json()["presets"]}["Local only"]
    assert local["roles"]["primary"] == local["roles"]["fast"] == "ollama_chat/qwen3:14b"
    assert local["context_length"] == 16384 and "qwen3:14b, reading 16,384 tokens" in local["description"]
    result = client.post("/api/models/presets/Local only/apply").json()
    assert result["preset"] == "Local only"
    models = client.get("/api/config").json()["models"]
    assert models["roles"]["primary"] == "ollama_chat/qwen3:14b" and models["context_length"] == 16384


def test_presets_without_hardware_keep_the_defaults(config):
    local = presets.presets(config)[0]
    assert local["roles"]["primary"] == "ollama_chat/qwen3:8b" and "context_length" not in local


# ----------------------------------------------------------------------------- check-up hints
def ollama_api(mock, *, ps=None, tags=("qwen3:8b",)):
    mock.get(f"{OLLAMA}/api/version").respond(200, json={"version": "0.12.0"})
    show = {"capabilities": ["completion", "tools"], "model_info": {"qwen3.context_length": 40960}}
    mock.post(f"{OLLAMA}/api/show").respond(200, json=show)
    mock.get(f"{OLLAMA}/api/ps").respond(200, json={"models": ps or []})
    mock.get(f"{OLLAMA}/api/tags").respond(200, json={"models": [{"name": t} for t in tags]})


def with_rec(hw: dict) -> dict:
    return {**hw, "summary": hardware.summary(hw), "recommendation": hardware.recommend(hw)}


async def test_checkup_offers_the_context_length_that_fits_this_card(config):
    config.models.context_length = 16384
    llm = FakeProvider(replies=["ready", [tool_call("find_city", name="Paris")], [tool_call("get_weather", city_id="c-42")]])
    hw = with_rec(fake_hw(8.0, 16.0))
    events = []
    with respx.mock(assert_all_called=False) as mock:
        ollama_api(mock, ps=[{"name": "qwen3:8b", "size": 8_000_000_000, "size_vram": 6_000_000_000}])
        async for event in run_checkup(config, llm, {"primary": "ollama_chat/qwen3:8b"}, hardware=hw):
            events.append(event)
    assert events[0]["type"] == "start" and events[0]["hardware"]["recommendation"]["tier"] == "gpu_8"
    checks = {c["id"]: c for c in events[-1]["roles"][0]["checks"]}
    assert checks["context"]["status"] == "warn" and "about 8,192 tokens" in checks["context"]["fix"]
    assert checks["context"]["action"] == {"kind": "set_context_length", "value": 8192, "role": None,
                                           "label": "Use 8,192 tokens"}
    assert checks["gpu"]["action"]["value"] == 8192
    assert checks["gpu"]["fix"].endswith("For this computer Sentient suggests qwen3:8b, reading 8,192 tokens at a time.")


async def test_checkup_suggests_the_model_sized_for_this_computer(config):
    llm = FakeProvider(replies=["ready", "It is sunny in Paris."])
    with respx.mock(assert_all_called=False) as mock:
        ollama_api(mock, tags=("llama3.2:1b",))
        done = await checkup(config, llm, {"primary": "ollama_chat/llama3.2:1b"}, hardware=with_rec(fake_hw(16.0)))
    tools = {c["id"]: c for c in done["roles"][0]["checks"]}["tools"]
    assert tools["fix"] == "Pick a model that handles tools well, like qwen3:14b."
    assert tools["action"] == {"kind": "pull_model", "name": "qwen3:14b", "label": "Download qwen3:14b"}


async def test_checkup_without_hardware_is_unchanged(config):
    llm = FakeProvider(replies=["ready"])
    with respx.mock(assert_all_called=False) as mock:
        ollama_api(mock)
        events = [e async for e in run_checkup(config, llm, {"embedding": None})]
    assert "hardware" not in events[0]


# ----------------------------------------------------------------------------- context meter
class WindowProvider(FakeProvider):
    """FakeProvider whose models read ``window`` tokens at once."""

    def __init__(self, window: int | None, **kwargs):
        super().__init__(**kwargs)
        self.window = window

    async def context_window(self, role, model=None):
        return self.window


async def _usage_events(app, text="hello") -> list[Usage]:
    session_id = await app.store.create_session()
    return [e async for e in app.agent.run_turn(session_id, text) if isinstance(e, Usage)]


@pytest.fixture
async def make_app(config, isolated_home):
    apps = []

    async def make(llm):
        a = SentientApp(config, llm=llm, db_path=isolated_home / f"m{len(apps)}.db", enable_background=False)
        await a.start()
        apps.append(a)
        return a

    yield make
    for a in apps:
        await a.stop()


async def test_meter_in_chat_usage_events(make_app):
    app = await make_app(WindowProvider(1_000_000, replies=["hi there"]))
    [usage] = await _usage_events(app)
    assert usage.context_length == 1_000_000 and usage.context_used > 2  # counted prompt, not just the reported 1
    assert usage.context_percent == round(100 * usage.context_used / 1_000_000)
    assert usage.context_warning is None
    assert usage.model_dump()["context_used"] == usage.context_used


async def test_meter_warns_near_the_limit_and_never_blocks(make_app):
    app = await make_app(WindowProvider(200, replies=["still answering"]))
    events = []
    session_id = await app.store.create_session()
    async for e in app.agent.run_turn(session_id, "hello"):
        events.append(e)
    usage = next(e for e in events if isinstance(e, Usage))
    assert usage.context_percent >= meter.WARN_PERCENT
    assert usage.context_warning.startswith("This chat is getting long for fake")
    assert events[-1].type == "done" and events[-1].content == "still answering"


async def test_meter_works_when_the_provider_reports_no_usage(make_app):
    class NoUsage(WindowProvider):
        async def stream(self, role, messages, tools=None, *, model=None):
            async for chunk in super().stream(role, messages, tools, model=model):
                chunk.usage = {}
                yield chunk

    app = await make_app(NoUsage(1000, replies=["hi"]))
    [usage] = await _usage_events(app)
    assert usage.prompt_tokens == 0 and usage.context_length == 1000 and usage.context_used > 0
    assert await app.store.fetchall("SELECT * FROM usage") == []  # nothing reported, nothing recorded


async def test_meter_is_left_out_when_the_context_length_is_unknown(make_app):
    for llm in (FakeProvider(replies=["a"]), WindowProvider(None, replies=["b"])):
        app = await make_app(llm)
        [usage] = await _usage_events(app)
        assert usage.context_length is None and usage.context_percent is None and usage.context_warning is None


def test_warning_threshold_and_wording():
    assert meter.meter(84, 100, "ollama_chat/qwen3:8b", "chat")["context_warning"] is None
    local = meter.meter(85, 100, "ollama_chat/qwen3:8b", "chat")["context_warning"]
    assert local == ("This chat is getting long for qwen3:8b (85% of what it reads at once). Older messages may be "
                     "left out: start a new chat, or set a longer context length in Settings > Models.")
    cloud = meter.meter(90, 100, "anthropic/claude-sonnet-5-5", "chat")["context_warning"]
    assert cloud.endswith("so a new chat works best.") and "Settings" not in cloud
    task = meter.meter(95, 100, "ollama_chat/qwen3:8b", "task")["context_warning"]
    assert task.startswith("This task's work is getting long") and "Settings > Models" in task
    assert chr(0x2014) not in local + cloud + task  # no em-dashes in plain copy


async def test_task_runs_stream_the_meter_and_log_the_warning_once():
    published, logged = [], []

    class Svc:
        app = type("A", (), {"bus": type("B", (), {"publish": staticmethod(lambda t, d: published.append((t, d)))})()})()

        async def progress(self, task_id, run_id, message):
            logged.append(message)

    mapper = ProgressMapper(Svc(), "t1", "r1")
    calm = Usage(model="m", **meter.meter(50, 100, "ollama_chat/qwen3:8b", "task"))
    full = Usage(model="m", **meter.meter(90, 100, "ollama_chat/qwen3:8b", "task"))
    for event in (calm, full, full, Usage(model="m")):
        await mapper.handle(event, [])
    # a call with an unknown context length (a fallback model) clears the meter instead of leaving the old one
    assert [d["percent"] for _, d in published] == [50, 90, 90, None]
    assert published[-1][1]["length"] is None and published[-1][1]["warning"] is None
    assert published[0] == ("task.run_context", {"task_id": "t1", "run_id": "r1", "used": 50, "length": 100,
                                                 "percent": 50, "warning": None})
    assert logged == [{"type": "info", "content": full.context_warning}]


# ----------------------------------------------------------------------------- context windows
async def test_cloud_models_use_their_known_window_or_skip(monkeypatch):
    known = {"anthropic/claude-sonnet-5-5": {"max_input_tokens": 1_000_000}, "openai/gpt-5": {"max_tokens": 272_000}}

    def info(model):
        if model not in known:
            raise ValueError("This model isn't mapped yet.")
        return known[model]

    monkeypatch.setattr(litellm, "get_model_info", info)
    prov = LiteLLMProvider(SentientConfig())
    assert await prov.context_window("primary", "anthropic/claude-sonnet-5-5") == 1_000_000
    assert await prov.context_window("primary", "openai/gpt-5") == 272_000
    assert await prov.context_window("primary", "groq/some-new-model") is None
    assert await prov.context_window("primary", "lm_studio/local-model") is None


async def test_ollama_window_is_the_context_length_sent_capped_at_the_model_maximum():
    cfg = SentientConfig()
    cfg.models.context_length = 16384
    cfg.models.context_length_per_role = {"executor": 32768}
    prov = LiteLLMProvider(cfg)
    with respx.mock() as mock:
        mock.post(f"{OLLAMA}/api/show").respond(200, json={"model_info": {"qwen3.context_length": 20000}})
        assert await prov.context_window("primary", "ollama_chat/qwen3:8b") == 16384
        assert await prov.context_window("executor", "ollama_chat/qwen3:8b") == 20000
    down = LiteLLMProvider(cfg)
    with respx.mock() as mock:
        mock.post(f"{OLLAMA}/api/show").mock(side_effect=httpx.ConnectError("refused"))
        assert await down.context_window("primary", "ollama_chat/qwen3:8b") == 16384
