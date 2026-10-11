"""Known provider failures reach the user as plain sentences that say what to do; the raw text only goes to the
log (#280).

The error bodies below were recorded from the real providers on 2026-10-10 (user and request ids replaced). respx
serves them to the real LiteLLM, so the exceptions are the ones people get. The 5xx bodies follow the providers'
documented shapes.
"""

from __future__ import annotations

import json
import logging

import httpx
import litellm
import pytest
import respx

from sentient.app import SentientApp
from sentient.config.schema import SentientConfig
from sentient.llm.errors import plain_error
from sentient.llm.events import Done, Error
from sentient.llm.provider import LiteLLMProvider, PlainProviderError, ProviderError, ProviderFailed

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC = "https://api.anthropic.com/v1/messages"
OPENAI = "https://api.openai.com/v1/chat/completions"
OLLAMA = "http://localhost:11434/api/chat"
FREE = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"

FREE_PER_DAY = {"error": {
    "message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day",
    "code": 429, "metadata": {
        "headers": {"X-RateLimit-Limit": "50", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1791676800000"},
        "limit_source": "openrouter_free_tier_daily",
        "remedy_hint": "Wait for the daily reset (see X-RateLimit-Reset), or purchase credits to raise your "
                       "free-model daily limit.", "provider_name": None}}, "user_id": "user_test"}
AGENTIC_ONLY = {"error": {
    "message": "thinkingmachines/inkling:free is only available on agentic harnesses. Try plugging it into a coding "
               "agent or productivity app listed on https://openrouter.ai/apps",
    "code": 403, "metadata": {"routing_funnel": [{"step": "Initial Endpoints", "endpoint_count": 1}],
                              "failed_routing_step": "Gate Free Endpoints by Agentic Harness"}}}
NO_CREDITS = {"error": {
    "message": "This request requires more credits, or fewer max_tokens. You requested up to 32000 tokens, but can "
               "only afford 398. To increase, visit https://openrouter.ai/settings/credits and upgrade to a paid "
               "account", "code": 402, "metadata": {
        "limit_source": "openrouter_credits",
        "remedy_hint": "Add credits at https://openrouter.ai/settings/credits, or lower max_tokens / prompt size to "
                       "fit your remaining balance.", "provider_name": None}}, "user_id": "user_test"}
NOT_A_MODEL = {"error": {"message": "nosuchlab/no-such-model-9 is not a valid model ID", "code": 400},
               "user_id": "user_test"}
REVOKED_KEY = {"error": {"message": "User not found.", "code": 401}}
ANTHROPIC_BAD_KEY = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"},
                     "request_id": "req_test"}
OPENAI_BAD_KEY = {"error": {
    "message": "Incorrect API key provided: sk-proj-************************************0000. You can find your API "
               "key at https://platform.openai.com/account/api-keys.",
    "type": "invalid_request_error", "param": None, "code": "invalid_api_key"}}
OLLAMA_NOT_PULLED = {"error": 'model "no-such-model:1b" not found, try pulling it first'}
OPENROUTER_502 = {"error": {"message": "Provider returned error", "code": 502,
                            "metadata": {"raw": "upstream connect error", "provider_name": "Chutes"}}}
OPENROUTER_503 = {"error": {"message": "No instances available", "code": 503}}
ANTHROPIC_OVERLOADED = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
NO_TOOL_ENDPOINTS = {"error": {"message": "No endpoints found that support tool use. To learn more about provider "
                                          "routing, visit: https://openrouter.ai/docs/provider-routing", "code": 404}}

PICK = "pick another model in Settings > Models."
CASES = [
    ("free_per_day", FREE, OPENROUTER, 429, FREE_PER_DAY,
     "OpenRouter's free models have reached today's limit. Add credits on openrouter.ai, wait until tomorrow, or "
     + PICK),
    ("agentic_only", "openrouter/thinkingmachines/inkling:free", OPENROUTER, 403, AGENTIC_ONLY,
     "OpenRouter only lets coding tools use thinkingmachines/inkling:free, not assistants like Sentient. Please "
     + PICK),
    ("no_credits", "openrouter/anthropic/claude-opus-4.1", OPENROUTER, 402, NO_CREDITS,
     "Your OpenRouter account is out of credits. Add credits on openrouter.ai/settings/credits, or " + PICK),
    ("not_a_model", "openrouter/nosuchlab/no-such-model-9", OPENROUTER, 400, NOT_A_MODEL,
     "OpenRouter doesn't have a model called nosuchlab/no-such-model-9. Check the name, or " + PICK),
    ("revoked_key", "openrouter/openai/gpt-4o-mini", OPENROUTER, 401, REVOKED_KEY,
     "OpenRouter didn't accept your key. It may be mistyped or revoked: add a new one in Settings > Models."),
    ("anthropic_bad_key", "anthropic/claude-haiku-4-5", ANTHROPIC, 401, ANTHROPIC_BAD_KEY,
     "Anthropic didn't accept your key. It may be mistyped or revoked: add a new one in Settings > Models."),
    ("openai_bad_key", "openai/gpt-4o-mini", OPENAI, 401, OPENAI_BAD_KEY,
     "OpenAI didn't accept your key. It may be mistyped or revoked: add a new one in Settings > Models."),
    ("ollama_not_pulled", "ollama_chat/no-such-model:1b", OLLAMA, 404, OLLAMA_NOT_PULLED,
     "no-such-model:1b isn't downloaded in Ollama. Download it in Settings > Models, or pick another model there."),
    ("openrouter_502", "openrouter/some/model", OPENROUTER, 502, OPENROUTER_502,
     "OpenRouter is having trouble right now. Try again in a few minutes, or " + PICK),
    ("openrouter_503", "openrouter/some/model", OPENROUTER, 503, OPENROUTER_503,
     "OpenRouter is having trouble right now. Try again in a few minutes, or " + PICK),
    ("anthropic_overloaded", "anthropic/claude-haiku-4-5", ANTHROPIC, 529, ANTHROPIC_OVERLOADED,
     "Anthropic is having trouble right now. Try again in a few minutes, or " + PICK),
]
RAW = ("litellm", "Exception", "{", "user_test")  # what a user must never see


@pytest.fixture
def http(monkeypatch):
    """LiteLLM over plain httpx (respx can't see its aiohttp transport), with a stored key for every provider."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr("sentient.llm.provider.secrets.get_secret", lambda *a, **k: "test-key")
    litellm.in_memory_llm_clients_cache.flush_cache()
    with respx.mock(assert_all_called=False) as router:
        router.post("http://localhost:11434/api/show").mock(return_value=httpx.Response(404, json={}))
        yield router
    litellm.in_memory_llm_clients_cache.flush_cache()


def provider(model: str) -> LiteLLMProvider:
    config = SentientConfig()
    for role in ("primary", "fast", "planner", "executor"):
        setattr(config.models.roles, role, model)
    config.models.fallbacks = {}
    return LiteLLMProvider(config)


def assert_plain(message: str, expected: str) -> None:
    assert message == expected
    assert not any(raw in message for raw in RAW)


@pytest.mark.parametrize(("case", "model", "url", "status", "body", "expected"), CASES, ids=[c[0] for c in CASES])
async def test_known_failures_become_plain_sentences(http, caplog, case, model, url, status, body, expected):
    http.post(url).mock(return_value=httpx.Response(status, json=body))
    llm = provider(model)
    with caplog.at_level(logging.WARNING, logger="sentient.llm.provider"), pytest.raises(ProviderFailed) as streamed:
        [c async for c in llm.stream("primary", [{"role": "user", "content": "hi"}])]
    assert_plain(str(streamed.value), expected)
    raw = body["error"]["message"] if isinstance(body["error"], dict) else body["error"]
    assert raw.split(".")[0][:40] in caplog.text.replace('\\"', '"')  # the raw reply stays in the log

    with pytest.raises(ProviderFailed) as planned:  # background jobs (task planning, memory) say the same
        await llm.complete_json("planner", [{"role": "user", "content": "plan"}])
    assert_plain(str(planned.value), expected)


async def test_unreachable_and_slow_providers(http):
    http.post(OLLAMA).mock(side_effect=httpx.ConnectError("All connection attempts failed"))
    http.post(OPENROUTER).mock(side_effect=httpx.ReadTimeout("timed out"))
    with pytest.raises(ProviderFailed) as down:
        await provider("ollama_chat/qwen3:8b").complete_text("primary", [{"role": "user", "content": "hi"}])
    assert_plain(str(down.value), "Sentient couldn't reach Ollama. Make sure it's running, then try again.")
    with pytest.raises(ProviderFailed) as slow:
        await provider(FREE).complete_text("primary", [{"role": "user", "content": "hi"}])
    assert_plain(str(slow.value), "OpenRouter took too long to answer. Try again in a moment, or " + PICK)

    # what LiteLLM's default aiohttp transport says when Ollama isn't running (recorded)
    refused = litellm.APIConnectionError(
        message="Ollama_chatException - Cannot connect to host 127.0.0.1:11434 ssl:default "
                "[The remote computer refused the network connection]", llm_provider="ollama_chat", model="qwen3:8b")
    assert plain_error(refused, "ollama_chat/qwen3:8b") == str(down.value)
    offline = litellm.APIConnectionError(message="OpenrouterException - Cannot connect to host openrouter.ai:443",
                                         llm_provider="openrouter", model="x")
    assert plain_error(offline, FREE) == (
        "Sentient couldn't reach OpenRouter. Check your internet connection and try again.")


async def test_unknown_failures_and_tool_refusals_keep_their_own_text(http):
    from sentient.agent.loop import is_tool_format_error

    http.post(OPENROUTER).mock(return_value=httpx.Response(404, json=NO_TOOL_ENDPOINTS))
    with pytest.raises(ProviderError) as no_tools:
        [c async for c in provider(FREE).stream("primary", [{"role": "user", "content": "hi"}], tools=[{
            "type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}])]
    assert not isinstance(no_tools.value, PlainProviderError)
    assert is_tool_format_error(no_tools.value)  # chat answers without tools instead of failing

    from sentient.llm.provider import _failure
    from sentient.llm.responses import ResponsesError

    no_tools_plan = ResponsesError("This model does not support tools.", 400)
    assert _failure(no_tools_plan, "chatgpt/gpt-test") is no_tools_plan  # left for the loop to answer without tools
    assert isinstance(_failure(ResponsesError("Usage limit reached.", 429), "chatgpt/gpt-test"), ProviderFailed)

    odd = litellm.BadRequestError(message="OpenrouterException - something new", model="x", llm_provider="openrouter")
    assert plain_error(odd, FREE) is None
    assert plain_error(ValueError("Model did not return JSON: 'x'"), FREE) is None  # Sentient's own errors
    assert not is_tool_format_error(ProviderFailed("Tools are parsed... by nobody"))


async def test_a_backup_model_reports_its_own_reason(http):
    http.post(OPENROUTER).mock(return_value=httpx.Response(429, json=FREE_PER_DAY))
    http.post(OLLAMA).mock(side_effect=httpx.ConnectError("All connection attempts failed"))
    llm = provider(FREE)
    llm.config.models.fallbacks = {"primary": ["ollama_chat/qwen3:8b"]}
    with pytest.raises(ProviderFailed, match="couldn't reach Ollama"):
        [c async for c in llm.stream("primary", [{"role": "user", "content": "hi"}])]


async def test_a_reply_cut_off_midway_says_why(monkeypatch):
    async def acompletion(**kwargs):
        async def chunks():
            from types import SimpleNamespace

            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel", reasoning_content=None))])
            raise litellm.ServiceUnavailableError(message="OpenrouterException - " + json.dumps(OPENROUTER_503),
                                                  llm_provider="openrouter", model="some/model")

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr("sentient.llm.provider.secrets.get_secret", lambda *a, **k: None)
    got = []
    with pytest.raises(ProviderFailed) as err:
        async for c in provider("openrouter/some/model").stream("primary", [{"role": "user", "content": "hi"}]):
            got.append(c.text)
    assert got == ["Hel"]
    assert_plain(str(err.value), "OpenRouter is having trouble right now. Try again in a few minutes, or " + PICK)


async def test_embeddings_say_it_plainly_too(http):
    http.post("http://localhost:11434/api/embed").mock(return_value=httpx.Response(404, json={
        "error": 'model "nomic-embed-text" not found, try pulling it first'}))
    with pytest.raises(ProviderFailed, match="nomic-embed-text isn't downloaded in Ollama"):
        await provider(FREE).embed(["hello"], model="ollama/nomic-embed-text")


async def test_chat_and_tasks_show_the_plain_sentence(http, config, isolated_home):
    """End to end in the engine: the chat's error event and a task's error, with the real provider class."""
    http.post(OPENROUTER).mock(return_value=httpx.Response(429, json=FREE_PER_DAY))
    for role in ("primary", "fast", "planner", "executor"):
        setattr(config.models.roles, role, FREE)
    config.models.fallbacks = {}
    expected = ("OpenRouter's free models have reached today's limit. Add credits on openrouter.ai, wait until "
                "tomorrow, or " + PICK)
    app = SentientApp(config, llm=LiteLLMProvider(config), db_path=isolated_home / "e.db", enable_background=False)
    await app.start()
    try:
        sid = await app.store.create_session(channel="cli")
        events = [ev async for ev in app.agent.run_turn(sid, "hi", channel="cli")]
        errors = [e for e in events if isinstance(e, Error)]
        assert len(errors) == 1 and not errors[0].recoverable
        assert_plain(errors[0].message, expected)
        assert not any(isinstance(e, Done) and "litellm" in (e.content or "") for e in events)

        task = await app.tasks.create_task("Look up tomorrow's weather in Pune")
        await app.tasks.drain()
        task = await app.tasks.get(task["task_id"])
        assert task["status"] == "error"
        assert_plain(task["error"], expected)
    finally:
        await app.stop()
