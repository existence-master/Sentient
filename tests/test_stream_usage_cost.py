"""Streamed cloud calls ask for token usage, and the final chunk carries the call's price when it is known (#133).

Without ``stream_options.include_usage`` OpenAI-style streams leave usage out, so token and cost budgets saw nothing.
"""

from types import SimpleNamespace

import litellm
import pytest

from sentient.config.schema import SentientConfig
from sentient.llm.provider import LiteLLMProvider

USAGE = SimpleNamespace(prompt_tokens=120, completion_tokens=30)


@pytest.fixture
def calls(monkeypatch):
    """Capture litellm calls; the assembled stream reports usage and the price list knows only gpt-priced."""
    seen: list[dict] = []

    async def acompletion(**kwargs):
        seen.append(kwargs)

        async def chunks():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi", reasoning_content=None))])

        return chunks()

    def stream_chunk_builder(chunks, messages=None):
        message = SimpleNamespace(content="Hi", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=USAGE)

    def completion_cost(completion_response=None, model=None):
        if model != "openai/gpt-priced":
            raise ValueError("This model isn't mapped yet.")
        return 0.0042

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "stream_chunk_builder", stream_chunk_builder)
    monkeypatch.setattr(litellm, "completion_cost", completion_cost)
    monkeypatch.setattr("sentient.llm.provider.secrets.get_secret", lambda *a, **k: None)
    return seen


async def final_chunk(prov: LiteLLMProvider, model: str):
    out = [c async for c in prov.stream("executor", [{"role": "user", "content": "hi"}], model=model)]
    assert out[-1].done
    return out[-1]


async def test_cloud_streams_ask_for_usage_and_carry_the_price(calls):
    prov = LiteLLMProvider(SentientConfig())
    done = await final_chunk(prov, "openai/gpt-priced")
    assert calls[-1]["stream_options"] == {"include_usage": True}
    assert done.usage == {"prompt_tokens": 120, "completion_tokens": 30}
    assert done.cost == pytest.approx(0.0042)

    unpriced = await final_chunk(prov, "openrouter/some/new-model")
    assert calls[-1]["stream_options"] == {"include_usage": True}
    assert unpriced.usage["prompt_tokens"] == 120 and unpriced.cost is None


async def test_ollama_streams_are_left_alone(calls, monkeypatch):
    async def no_limit(*a, **k):
        return None

    prov = LiteLLMProvider(SentientConfig())
    monkeypatch.setattr(prov, "_model_max_context", no_limit)
    await final_chunk(prov, "ollama_chat/qwen3:8b")
    assert "stream_options" not in calls[-1]
