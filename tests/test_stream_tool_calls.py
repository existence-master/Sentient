"""A streamed tool call without an id is kept (#509).

Ollama 0.9.0 sends a call with no arguments (``browser_screenshot``) without an id, and LiteLLM's
``stream_chunk_builder`` (used here for real) drops such a call: a qwen3:8b helper's screenshot was lost and the round
looked like an empty answer.
"""

import litellm
import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)

from sentient.config.schema import SentientConfig
from sentient.llm.provider import LiteLLMProvider


def _chunk(delta: Delta, finish: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(model="qwen3:8b", choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish)])


def _call(name: str | None, arguments: str, call_id: str | None, index: int = 0) -> ChatCompletionDeltaToolCall:
    return ChatCompletionDeltaToolCall(id=call_id, index=index, type="function",
                                       function=Function(name=name, arguments=arguments))


@pytest.fixture
def stream_of(monkeypatch):
    """Make the next litellm call stream ``deltas`` (the real stream_chunk_builder assembles them)."""
    replies: list[list[Delta]] = []

    async def acompletion(**kwargs):
        deltas = replies.pop(0)

        async def chunks():
            for delta in deltas:
                yield _chunk(delta)
            yield _chunk(Delta(content=None), finish="tool_calls")

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr("sentient.llm.provider.secrets.get_secret", lambda *a, **k: None)
    return replies


async def _tool_calls(prov: LiteLLMProvider):
    out = [c async for c in prov.stream("executor", [{"role": "user", "content": "go"}], model="ollama_chat/qwen3:8b")]
    assert out[-1].done
    return out[-1].tool_calls


async def test_a_call_without_arguments_or_id_is_kept(stream_of):
    stream_of.append([Delta(content=None, tool_calls=[_call("browser_screenshot", "{}", None)])])
    [call] = await _tool_calls(LiteLLMProvider(SentientConfig()))
    assert call.name == "browser_screenshot" and call.arguments == {}
    assert call.id.startswith("call_")


async def test_ids_the_model_gave_and_arguments_in_pieces_stay_as_they_were(stream_of):
    stream_of.append([
        Delta(content=None, tool_calls=[_call("browser_open", '{"url": ', "call_from_ollama")]),
        Delta(content=None, tool_calls=[_call(None, '"http://127.0.0.1/test.html"}', None)]),
    ])
    [call] = await _tool_calls(LiteLLMProvider(SentientConfig()))
    assert call.id == "call_from_ollama"
    assert call.name == "browser_open" and call.arguments == {"url": "http://127.0.0.1/test.html"}
