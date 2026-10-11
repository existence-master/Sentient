"""Plain sentences for the ways a model provider says no (issue #280).

LiteLLM's errors carry the provider's raw reply (``litellm.RateLimitError: ... OpenrouterException - {"error": ...``).
``plain_error`` turns the failures people actually hit into one sentence that says what to do. The raw text only
goes to the log; anything unknown returns None and keeps its own message.
"""

from __future__ import annotations

SETTINGS = "Settings > Models"

LABELS = {
    "openrouter": "OpenRouter", "anthropic": "Anthropic", "openai": "OpenAI", "gemini": "Google Gemini",
    "vertex_ai": "Google Vertex AI", "groq": "Groq", "mistral": "Mistral", "deepseek": "DeepSeek", "xai": "xAI",
    "together_ai": "Together AI", "fireworks_ai": "Fireworks AI", "cohere": "Cohere", "perplexity": "Perplexity",
    "nous": "Nous Portal", "azure": "Azure OpenAI", "bedrock": "Amazon Bedrock", "ollama": "Ollama",
    "ollama_chat": "Ollama", "lm_studio": "LM Studio", "chatgpt": "ChatGPT",
}
# where each provider sells credits, for "out of credits"
BILLING = {
    "openrouter": "openrouter.ai/settings/credits", "anthropic": "console.anthropic.com",
    "openai": "platform.openai.com", "deepseek": "platform.deepseek.com", "xai": "console.x.ai",
    "mistral": "console.mistral.ai", "nous": "portal.nousresearch.com",
}
LOCAL = {"ollama", "ollama_chat", "lm_studio", "llamafile", "vllm", "hosted_vllm"}

CREDIT_HINTS = (
    "requires more credits", "insufficient credits", "insufficient_quota", "exceeded your current quota",
    "credit balance is too low", "insufficient balance", "out of credits", "payment required",
)
KEY_HINTS = ("invalid api key", "invalid x-api-key", "incorrect api key", "api key not valid", "invalid_api_key",
             "user not found", "no auth credentials")
NOT_FOUND_HINTS = ("not a valid model id", "model_not_found", "does not exist", "not found, try pulling it first",
                   "unknown model", "no such model")
UNREACHABLE_HINTS = ("cannot connect to host", "all connection attempts failed", "connection refused",
                     "refused the network connection", "connecterror", "getaddrinfo failed", "name or service not known", "nodename nor servname")


def _parts(model: str) -> tuple[str, str]:
    prefix, _, name = model.partition("/")
    return (prefix, name) if name else ("", model)


def _label(prefix: str) -> str:
    return LABELS.get(prefix) or (prefix.replace("_", " ").title() if prefix else "The model's provider")


def _status(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def is_tool_support_error(text: str) -> bool:
    """The provider turned the request down because the model can't call tools (the agent loop then answers
    without tools, so these keep their own message)."""
    m = text.lower()
    return ("does not support tools" in m or "support tool use" in m or "invalid character" in m
            or ("tool" in m and "pars" in m))


def plain_error(exc: BaseException, model: str) -> str | None:
    """A plain sentence for a known provider failure of ``model`` that says what to do, or None when unknown.

    Only LiteLLM's own errors are translated: Sentient's errors (Claude Code, the ChatGPT plan) are already plain.
    """
    if not type(exc).__module__.startswith("litellm"):
        return None
    text = str(exc)
    m = text.lower()
    if is_tool_support_error(m):
        return None
    prefix, name = _parts(model)
    label, status, kind = _label(prefix), _status(exc), type(exc).__name__
    pick = f"pick another model in {SETTINGS}"

    if "agentic harness" in m:
        return (f"{label} only lets coding tools use {name}, not assistants like Sentient. "
                f"Please {pick}.")
    if "free-models-per-day" in m:
        return (f"{label}'s free models have reached today's limit. Add credits on openrouter.ai, wait until "
                f"tomorrow, or {pick}.")
    if "free-models-per-min" in m:
        return f"{label}'s free models allow only a few requests a minute. Wait a minute and try again, or {pick}."
    if status == 402 or any(h in m for h in CREDIT_HINTS):
        where = f" on {BILLING[prefix]}" if prefix in BILLING else ""
        return f"Your {label} account is out of credits. Add credits{where}, or {pick}."
    if "no endpoints found matching your data policy" in m:
        return (f"Your {label} privacy settings don't allow any provider of {name}. Change them on "
                f"openrouter.ai/settings/privacy, or {pick}.")
    if status == 401 or kind == "AuthenticationError" or any(h in m for h in KEY_HINTS):
        return f"{label} didn't accept your key. It may be mistyped or revoked: add a new one in {SETTINGS}."
    if status == 404 or kind == "NotFoundError" or any(h in m for h in NOT_FOUND_HINTS):
        if prefix in {"ollama", "ollama_chat"}:
            return f"{name} isn't downloaded in Ollama. Download it in {SETTINGS}, or pick another model there."
        return f"{label} doesn't have a model called {name}. Check the name, or {pick}."
    if status == 429 or kind == "RateLimitError":
        return f"{label} is getting too many requests right now. Wait a minute and try again, or {pick}."
    if status == 408 or kind == "Timeout":
        return f"{label} took too long to answer. Try again in a moment, or {pick}."
    if any(h in m for h in UNREACHABLE_HINTS):
        if prefix in LOCAL:
            return f"Sentient couldn't reach {label}. Make sure it's running, then try again."
        return f"Sentient couldn't reach {label}. Check your internet connection and try again."
    if (kind in {"InternalServerError", "ServiceUnavailableError", "BadGatewayError"}
            or (kind != "APIConnectionError" and status is not None and status >= 500) or "overloaded" in m):
        return f"{label} is having trouble right now. Try again in a few minutes, or {pick}."
    return None
