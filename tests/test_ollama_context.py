"""Ollama's context length (num_ctx) reaches every Ollama chat call, per role, and never a cloud call."""

from types import SimpleNamespace

import litellm
import pytest
import respx
from pydantic import ValidationError

from sentient.config.schema import SentientConfig
from sentient.llm.provider import LiteLLMProvider

SHOW = "http://localhost:11434/api/show"
CHAT_ROLES = ("primary", "fast", "planner", "executor", "vision", "voice")


@pytest.fixture
def calls(monkeypatch):
    """Capture every litellm call instead of reaching a model."""
    seen: list[dict] = []

    async def acompletion(**kwargs):
        seen.append(kwargs)
        if kwargs.get("stream"):
            async def empty():
                return
                yield

            return empty()
        content = '{"ok": true}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    async def aembedding(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2]}])

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "aembedding", aembedding)
    monkeypatch.setattr(litellm, "stream_chunk_builder", lambda chunks, messages=None: None)
    monkeypatch.setattr("sentient.llm.provider.secrets.get_secret", lambda *a, **k: None)
    return seen


async def _call_every_way(prov: LiteLLMProvider, role: str) -> None:
    msgs = [{"role": "user", "content": "hi"}]
    async for _ in prov.stream(role, msgs):
        pass
    await prov.complete_text(role, msgs)
    await prov.complete_json(role, msgs)


async def test_every_ollama_role_sends_its_context_length(calls):
    cfg = SentientConfig()
    cfg.models.roles.primary = "ollama_chat/qwen3:8b"
    cfg.models.roles.fast = "ollama/llama3.2:3b"  # the plain ollama/ prefix gets it too
    cfg.models.context_length = 8192
    cfg.models.context_length_per_role = {"executor": 16384, "fast": 4096}
    prov = LiteLLMProvider(cfg)
    expected = {"primary": 8192, "fast": 4096, "planner": 8192, "executor": 16384, "vision": 8192, "voice": 8192}
    with respx.mock() as router:
        show = router.post(SHOW).respond(json={"model_info": {"qwen3.context_length": 40960}})
        for role in CHAT_ROLES:
            calls.clear()
            await _call_every_way(prov, role)
            assert len(calls) == 3
            assert all(c["num_ctx"] == expected[role] for c in calls), role
            assert await prov.context_length(role) == expected[role]
    assert show.call_count == 2  # once per model, then remembered


async def test_cloud_roles_never_get_num_ctx(calls):
    cfg = SentientConfig()
    cfg.models.roles.primary = "anthropic/claude-sonnet-5"
    cfg.models.roles.fast = "openai/gpt-5-mini"
    cfg.models.roles.embedding = "ollama/nomic-embed-text"
    cfg.models.context_length_per_role = {"fast": 16384}
    prov = LiteLLMProvider(cfg)
    with respx.mock(assert_all_called=False) as router:
        show = router.post(SHOW).respond(json={"model_info": {}})
        for role in CHAT_ROLES:
            await _call_every_way(prov, role)
            assert await prov.context_length(role) is None
        await prov.embed(["hello"])
    assert calls and all("num_ctx" not in c for c in calls)
    assert not show.called


async def test_context_length_is_capped_at_the_models_maximum(calls):
    cfg = SentientConfig()
    cfg.models.context_length = 131072
    prov = LiteLLMProvider(cfg)
    with respx.mock() as router:
        router.post(SHOW).respond(json={"model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960}})
        await _call_every_way(prov, "primary")
    assert [c["num_ctx"] for c in calls] == [40960, 40960, 40960]


async def test_unknown_maximum_keeps_the_setting_and_asks_again(calls):
    prov = LiteLLMProvider(SentientConfig())
    with respx.mock() as router:
        show = router.post(SHOW).respond(500)
        await _call_every_way(prov, "primary")
    assert [c["num_ctx"] for c in calls] == [8192, 8192, 8192]
    assert show.call_count == 3  # a failed look-up is not remembered


def test_context_length_bounds():
    assert SentientConfig().models.context_length == 8192
    with pytest.raises(ValidationError):
        SentientConfig.model_validate({"models": {"context_length": 512}})
    with pytest.raises(ValidationError):
        SentientConfig.model_validate({"models": {"context_length_per_role": {"fast": 100}}})
