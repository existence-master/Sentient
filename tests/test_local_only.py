"""Local only (#148): with ``models.local_only`` on, no request reaches a model outside this computer.

The provider refuses before anything is sent, for every path: a role's model, its fallbacks, a chat's own model, JSON
and text jobs, embeddings, Claude Code and ChatGPT plans, and voice.
"""

from __future__ import annotations

from types import SimpleNamespace

import litellm
import pytest

from sentient.app import SentientApp
from sentient.config.schema import ProviderConfig, SentientConfig
from sentient.llm import availability, claude_code
from sentient.llm.events import Done, Notice
from sentient.llm.locality import billing, runs_here
from sentient.llm.provider import LiteLLMProvider, LocalOnlyRefused
from sentient.voice.stt import FasterWhisperSTT, build_stt
from sentient.voice.tts import SystemTTS, build_tts
from tests.conftest import FakeProvider

LOCAL = "ollama_chat/qwen3:8b"
CLOUD = "openai/gpt-4o"


def local_only() -> SentientConfig:
    cfg = SentientConfig()
    cfg.models.local_only = True
    return cfg


@pytest.mark.parametrize(
    ("model", "providers", "env", "expected"),
    [
        (LOCAL, {}, {}, True),
        ("ollama/llama3.2", {}, {}, True),
        ("lm_studio/big-model", {}, {}, True),  # the default config points LM Studio at localhost
        ("vllm/meta/llama", {}, {}, True),  # LiteLLM runs vLLM inside this process
        ("llamafile/model", {}, {}, True),  # LiteLLM's default address is 127.0.0.1
        ("hosted_vllm/model", {}, {}, False),  # no address: can't prove it is here
        ("hosted_vllm/model", {"hosted_vllm": ProviderConfig(api_base="http://[::1]:8000/v1")}, {}, True),
        (LOCAL, {"ollama_chat": ProviderConfig(api_base="http://192.168.1.20:11434")}, {}, False),
        (LOCAL, {"ollama_chat": ProviderConfig(api_base="http://gpu-box:11434")}, {}, False),
        (LOCAL, {"ollama_chat": ProviderConfig(api_base=None)}, {"OLLAMA_API_BASE": "https://ollama.example.com"}, False),
        (LOCAL, {"ollama_chat": ProviderConfig(api_base=None)}, {}, True),
        ("ollama_chat/gpt-oss:120b-cloud", {}, {}, False),  # Ollama forwards cloud models to ollama.com
        ("ollama/glm-4.6:cloud", {}, {}, False),
        (CLOUD, {"openai": ProviderConfig(api_base="http://127.0.0.1:9000/v1")}, {}, False),  # may be a proxy
        ("claude-code/sonnet", {}, {}, False),
        ("chatgpt/gpt-5", {}, {}, False),
        ("anthropic/claude-sonnet-4-5", {}, {}, False),
        ("nous/hermes", {}, {}, False),
    ],
)
def test_runs_here_is_strict(monkeypatch, model, providers, env, expected):
    for var in ("OLLAMA_API_BASE", "LM_STUDIO_API_BASE", "LLAMAFILE_API_BASE", "HOSTED_VLLM_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cfg = SentientConfig()
    cfg.models.providers.update(providers)
    assert runs_here(cfg, model) is expected


def test_billing_kinds():
    assert billing(LOCAL) == "local"
    assert billing("claude-code/sonnet") == "plan" and billing("chatgpt/gpt-5") == "plan"
    assert billing(CLOUD) == "paid" and billing("ollama_chat/gpt-oss:120b-cloud") == "paid"


@pytest.fixture
def sent(monkeypatch):
    """Every request LiteLLM would send, by model. Local calls answer; nothing else may be called at all."""
    calls: list[str] = []

    async def acompletion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs.get("stream"):
            async def chunks():
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="here", reasoning_content=None))])

            return chunks()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))], usage=None)

    async def aembedding(**kwargs):
        calls.append(kwargs["model"])
        return SimpleNamespace(data=[{"embedding": [0.1]}], usage=None)

    def stream_chunk_builder(chunks, messages=None):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="here", tool_calls=None))], usage=None)

    async def no_cloud(*a, **k):
        raise AssertionError("a plan model was called")
        yield  # pragma: no cover

    async def no_limit(*a, **k):
        return None

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "aembedding", aembedding)
    monkeypatch.setattr(litellm, "stream_chunk_builder", stream_chunk_builder)
    monkeypatch.setattr(claude_code, "stream", no_cloud)
    monkeypatch.setattr(LiteLLMProvider, "_model_max_context", no_limit)
    monkeypatch.setattr("sentient.llm.provider.secrets.get_secret", lambda *a, **k: "sk-test")
    return calls


async def collect(prov, role, **kw):
    return [c async for c in prov.stream(role, [{"role": "user", "content": "hi"}], **kw)]


async def test_a_cloud_role_falls_back_to_a_local_model_and_never_calls_the_cloud(sent):
    cfg = local_only()
    cfg.models.roles.primary = CLOUD
    cfg.models.fallbacks = {"primary": ["claude-code/sonnet", "chatgpt/gpt-5", LOCAL]}
    out = await collect(LiteLLMProvider(cfg), "primary")
    assert out[-1].done and out[-1].model == LOCAL
    assert sent == [LOCAL]


async def test_every_path_refuses_a_cloud_model_with_a_plain_message(sent):
    cfg = local_only()
    cfg.models.roles.primary = CLOUD
    cfg.models.roles.planner = "anthropic/claude-sonnet-4-5"
    cfg.models.roles.embedding = "openai/text-embedding-3-small"
    prov = LiteLLMProvider(cfg)
    with pytest.raises(LocalOnlyRefused) as refused:
        await collect(prov, "primary")
    assert refused.value.code == "local_only"
    assert str(refused.value).startswith("Local only is on, so nothing was sent to gpt-4o")
    with pytest.raises(LocalOnlyRefused):
        await collect(prov, "primary", model="claude-code/sonnet")  # a chat's own model: no fallback
    with pytest.raises(LocalOnlyRefused):
        await prov.complete_json("planner", [{"role": "user", "content": "plan"}])
    with pytest.raises(LocalOnlyRefused):
        await prov.complete_text("fast", [{"role": "user", "content": "title"}], model="ollama_chat/gpt-oss:120b-cloud")
    with pytest.raises(LocalOnlyRefused):
        await prov.embed(["remember this"])
    assert sent == []


async def test_turning_it_off_lets_the_cloud_answer_again(sent):
    cfg = SentientConfig()
    cfg.models.roles.primary = CLOUD
    await collect(LiteLLMProvider(cfg), "primary")
    assert sent == [CLOUD]


async def test_a_chat_with_a_cloud_model_falls_back_with_one_notice(config, isolated_home):
    config.models.local_only = True
    llm = FakeProvider(replies=["Answered on this computer."])
    s = await SentientApp(config, llm=llm, db_path=isolated_home / "lo.db", enable_background=False).start()
    try:
        availability.forget_cache()
        sid = await s.store.create_session(channel="desktop", model=CLOUD)
        events = [ev async for ev in s.agent.run_turn(sid, "hi", channel="desktop")]
        notices = [e for e in events if isinstance(e, Notice)]
        assert len(notices) == 1 and "Local only is on, so nothing was sent to gpt-4o" in notices[0].message
        assert [c["model"] for c in llm.calls if not c.get("json")] == [None]
        assert isinstance(events[-1], Done) and events[-1].content == "Answered on this computer."
    finally:
        await s.stop()


def test_voice_stays_on_this_computer():
    cfg = local_only()
    cfg.voice.stt_provider = "openai"
    cfg.voice.stt_model = "whisper-1"
    cfg.voice.tts_provider = "elevenlabs"
    stt = build_stt(cfg)
    assert isinstance(stt, FasterWhisperSTT) and stt.model == "base"
    assert isinstance(build_tts(cfg), SystemTTS)
    cfg.models.local_only = False
    assert not isinstance(build_stt(cfg), FasterWhisperSTT)
