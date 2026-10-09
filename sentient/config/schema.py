"""The single source of truth for everything a user can configure.

Design rules:
- Every field has a default so a brand-new install works with an empty file.
- Every field carries a description, because the desktop Settings screen is
  generated from the JSON schema of this model.
- Secrets (API keys, OAuth tokens) are NOT here. They live in the OS keychain
  via ``sentient.secrets`` and are referenced by name.

Ownership (parallel development): each feature package owns its own section
class below and may add fields to it. Do not edit another package's section.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------------------- core (owner: core)
class AssistantConfig(BaseModel):
    name: str = Field("Sentient", description="What the assistant calls itself.")
    user_name: str = Field("", description="How the assistant addresses you.")
    timezone: str = Field(
        "auto", description="IANA timezone like 'Asia/Kolkata', or 'auto' to use the system zone."
    )
    location: str = Field("", description="City, Country. Used for weather, maps and local context.")
    language: str = Field("en", description="Preferred reply language (BCP-47 code).")
    onboarding_complete: bool = Field(False, description="Set when the first-run setup finishes.")


class ProviderConfig(BaseModel):
    """Connection details for one LLM provider. Keys are looked up in the keychain
    under ``sentient/<provider>`` first, then in the named environment variable."""

    api_base: str | None = Field(None, description="Override base URL (local servers, proxies).")
    api_key_env: str | None = Field(
        None, description="Environment variable holding the API key, if not stored in the keychain."
    )


class ModelRoles(BaseModel):
    """Which model does which job. Strings are ``provider/model`` as understood by LiteLLM,
    e.g. ``ollama_chat/qwen3:8b``, ``anthropic/claude-sonnet-5``, ``openai/gpt-5``."""

    primary: str = Field("ollama_chat/qwen3:8b", description="Main conversational and tool-using model.")
    fast: str = Field(
        "ollama_chat/qwen3:8b",
        description="Cheap model for background jobs: memory extraction, summaries, triage.",
    )
    planner: str | None = Field(None, description="Task refinement and planning (defaults to primary).")
    executor: str | None = Field(None, description="Long-running task execution (defaults to primary).")
    embedding: str = Field("ollama/nomic-embed-text", description="Embedding model for memory search.")
    vision: str | None = Field(None, description="Model used when an image is attached (defaults to primary).")
    voice: str | None = Field(
        None, description="Model for spoken conversations (defaults to primary). Pick a fast one: latency matters most here."
    )


class ModelsConfig(BaseModel):
    roles: ModelRoles = Field(default_factory=ModelRoles)
    fallbacks: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-role ordered fallback models tried when the role's model fails.",
    )
    reasoning: dict[str, str] = Field(
        default_factory=lambda: {"primary": "medium", "fast": "none", "planner": "none", "executor": "none", "voice": "none"},
        description="Per-role reasoning effort: none | low | medium | high. 'none' disables thinking "
        "(fast, good for background JSON jobs). Ignored by providers that lack the feature.",
    )
    temperature: dict[str, float] = Field(
        default_factory=dict, description="Optional per-role sampling temperature."
    )
    context_length: int = Field(
        8192,
        ge=2048,
        le=1_048_576,
        description="How much text (in tokens) a local Ollama model reads at once: instructions, the conversation "
        "and tool results. Ollama's own default is only 4096 on most computers, which quietly cuts off long tasks. "
        "Bigger values let the model see more but need more graphics memory: 8192 keeps qwen3:8b fully on an "
        "8 GB graphics card, while 16384 or more can push part of it onto the processor and make replies much "
        "slower. Never goes above the model's own maximum. Cloud models ignore this.",
    )
    context_length_per_role: dict[str, Annotated[int, Field(ge=2048, le=1_048_576)]] = Field(
        default_factory=dict,
        description="Optional per-role context length for local Ollama models, replacing the value above for that "
        "role, e.g. {\"executor\": 16384}. Best used when the roles run different models: one model given different "
        "lengths in different roles is reloaded by Ollama each time the role changes.",
    )
    providers: dict[str, ProviderConfig] = Field(
        default_factory=lambda: {
            "ollama": ProviderConfig(api_base="http://localhost:11434"),
            "ollama_chat": ProviderConfig(api_base="http://localhost:11434"),
            "openai": ProviderConfig(api_key_env="OPENAI_API_KEY"),
            "anthropic": ProviderConfig(api_key_env="ANTHROPIC_API_KEY"),
            "gemini": ProviderConfig(api_key_env="GEMINI_API_KEY"),
            "openrouter": ProviderConfig(api_key_env="OPENROUTER_API_KEY"),
            "groq": ProviderConfig(api_key_env="GROQ_API_KEY"),
            "mistral": ProviderConfig(api_key_env="MISTRAL_API_KEY"),
            "deepseek": ProviderConfig(api_key_env="DEEPSEEK_API_KEY"),
            "xai": ProviderConfig(api_key_env="XAI_API_KEY"),
            "lm_studio": ProviderConfig(api_base="http://localhost:1234/v1"),
        },
        description="Provider connection settings keyed by LiteLLM provider prefix.",
    )
    max_tool_rounds: int = Field(12, ge=1, le=100, description="Max tool-call rounds per chat turn.")
    request_timeout_s: int = Field(180, ge=10, description="Per-request timeout in seconds.")


class GatewayConfig(BaseModel):
    host: str = Field("127.0.0.1", description="Bind address. The desktop app always uses loopback.")
    port: int = Field(7777, ge=0, le=65535, description="0 lets the desktop app pick a free port.")


class ChatConfig(BaseModel):
    history_window: int = Field(30, ge=2, description="Recent messages sent to the model per turn.")
    compress_after_messages: int = Field(
        60, ge=10, description="Summarize older turns of a long conversation into a running summary."
    )
    show_thinking: bool = Field(True, description="Show the model's reasoning in a collapsible block.")
    auto_title: bool = Field(True, description="Name new chats automatically from the first exchange.")
    tool_selection: Literal["auto", "all"] = Field(
        "auto",
        description="auto: offer only the tools relevant to each message when there are more than the budget "
        "(faster and more accurate on local models). all: always offer every tool.",
    )
    max_tools_local: int = Field(12, ge=3, le=200, description="Tool budget per turn for local models.")
    max_tools_cloud: int = Field(64, ge=3, le=500, description="Tool budget per turn for cloud models.")
    tool_result_max_chars: int = Field(
        16000,
        ge=1000,
        le=1_000_000,
        description="Longest tool result the model reads. Longer results are cut and the full text is saved "
        "under files/outputs/ so the assistant can open it if needed.",
    )
    parallel_read_tools: bool = Field(
        True, description="Run look-up tools the model asks for in the same step at the same time."
    )


class ApprovalsConfig(BaseModel):
    mode: Literal["off", "ask", "always"] = Field(
        "ask",
        description="off: never ask. ask: confirm actions that change things outside Sentient, send, "
        "delete or run code (its own memory, files and task list never interrupt). always: confirm every tool call.",
    )
    remember_session: bool = Field(True, description="'Allow for this chat' answers are honored.")
    timeout_s: int = Field(600, ge=10, description="Unanswered approvals are denied after this long.")
    rules: dict[str, Literal["allow", "ask", "never"]] = Field(
        default_factory=dict,
        description="Lasting rules for apps and tools. The key is a tool name (gmail_send_email) or an app id "
        "(gmail, meaning all of its tools); a tool's own rule beats its app's rule. allow: go ahead without asking "
        "(purchases still ask). ask: always ask first, whatever the setting above says. never: Sentient can't use it.",
    )

    @field_validator("rules", mode="before")
    @classmethod
    def _clean_rules(cls, value: object) -> object:
        """Trim keys, lower-case values and reject empty keys; the Literal checks the values."""
        if not isinstance(value, dict):
            return value
        out: dict[str, object] = {}
        for key, rule in value.items():
            name = str(key).strip()
            if not name:
                raise ValueError("A rule needs a tool name or an app id.")
            out[name] = rule.strip().lower() if isinstance(rule, str) else rule
        return out


class ToolsConfig(BaseModel):
    approvals: ApprovalsConfig = Field(default_factory=ApprovalsConfig)
    disabled: list[str] = Field(default_factory=list, description="Tool or plugin ids to hide.")
    repeated_call_limit: int = Field(
        3,
        ge=0,
        le=20,
        description="Stop when the assistant uses the same tool with the same details and gets the same result "
        "this many times. In chat it is first asked to try something else. 0 turns this off.",
    )


class SubagentsConfig(BaseModel):
    """Chat subagents: background or parallel workers the assistant spawns (owner: core)."""

    enabled: bool = Field(True, description="Let the assistant hand work to subagents that run alongside the chat.")
    max_concurrent: int = Field(3, ge=1, le=16, description="Subagents running at the same time.")
    max_rounds: int = Field(24, ge=1, le=200, description="Tool rounds per subagent.")
    role: Literal["executor", "primary", "fast"] = Field("executor", description="Model role subagents use.")
    timeout_minutes: int = Field(20, ge=1, description="A subagent is stopped after this long.")
    max_tokens: int = Field(
        1_000_000, ge=0, description="A subagent is stopped after using this many tokens on a cloud model "
        "(local models are not counted). 0 means no limit.",
    )
    max_cost_usd: float = Field(
        2.0, ge=0, description="A subagent is stopped after spending about this many US dollars on a cloud model, "
        "when the model's price is known. 0 means no limit.",
    )


class UIConfig(BaseModel):
    theme: Literal["system", "dark", "light"] = Field("dark", description="Window theme.")
    accent: str = Field("sentient", description="Accent color: sentient (amber), violet, blue, emerald, rose.")
    launch_at_login: bool = Field(False, description="Start Sentient when you log in.")
    minimize_to_tray: bool = Field(True, description="Closing the window keeps Sentient running in the tray.")


# ----------------------------------------------------------------------------- memory (owner: memory/evolution agent)
class MemoryConfig(BaseModel):
    facts_top_k: int = Field(8, ge=0, le=50, description="How many recalled facts to inject per turn.")
    min_similarity: float = Field(0.35, ge=0.0, le=1.0, description="Recall threshold (cosine).")
    duplicate_similarity: float = Field(
        0.98, ge=0.5, le=1.0,
        description="Above this similarity a new fact is skipped without asking the model. Keep it high: "
        "embedding models score contradictions (Pune vs Mumbai) as very similar.",
    )
    extract_after_turn: bool = Field(
        True, description="Learn facts about you from each message in the background."
    )
    workspace_budget_chars: int = Field(
        6000, ge=500, description="Max characters injected from each workspace markdown file."
    )
    graph_link_similarity: float = Field(0.80, ge=0.0, le=1.0, description="Memory graph edge threshold.")
    summarize_after_minutes: int = Field(
        60, ge=5, description="Conversation chunks older than this are summarized into episodic memory."
    )
    summary_chunk_messages: int = Field(30, ge=6, description="Messages per episodic summary.")
    summaries_enabled: bool = Field(
        True, description="Summarize older conversations into first-person episodic memories in the background."
    )
    summarize_interval_minutes: int = Field(5, ge=1, description="How often to look for conversations to summarize.")
    purge_interval_minutes: int = Field(60, ge=5, description="How often expired short-term memories are deleted.")
    import_chunk_chars: int = Field(
        4000, ge=500, le=20000, description="Characters per chunk when learning facts from an imported document."
    )
    flush_enabled: bool = Field(
        True, description="Before old turns of a long chat are compressed, save the lasting facts they contain."
    )
    flush_max_chars: int = Field(
        6000, ge=500, le=50000, description="Most transcript characters read when saving facts before compression."
    )
    flush_max_facts: int = Field(8, ge=0, le=50, description="Most facts saved from one compression flush.")
    keyword_weight: float = Field(
        0.15, ge=0.0, le=1.0,
        description="How much keyword matches add to embedding similarity when ranking recalled facts.",
    )


# ----------------------------------------------------------------------------- tasks (owner: tasks agent)
class TasksConfig(BaseModel):
    tick_seconds: int = Field(30, ge=5, description="Scheduler poll interval.")
    max_concurrent_runs: int = Field(2, ge=1, le=16, description="Task runs allowed at the same time.")
    run_timeout_minutes: int = Field(
        30, ge=1, description="Minutes of work after which a run asks whether to keep going (time spent waiting for "
        "your answer doesn't count). A swarm is stopped instead.",
    )
    max_tool_rounds: int = Field(
        40, ge=1, le=200, description="Steps after which a run asks whether to keep going (a swarm worker stops)."
    )
    max_tokens_per_run: int = Field(
        2_000_000, ge=0, description="Tokens on a cloud model after which a run asks whether to keep going "
        "(local models are not counted). A swarm shares one limit and stops. 0 means no limit.",
    )
    max_cost_per_run_usd: float = Field(
        5.0, ge=0, description="US dollars spent on a cloud model after which a run asks whether to keep going, "
        "when the model's price is known. A swarm shares one limit and stops. 0 means no limit.",
    )
    require_plan_approval: bool = Field(True, description="Plans wait for your approval before running.")
    swarm_max_agents: int = Field(5, ge=1, le=50, description="Parallel sub-agents per swarm task.")
    resume_interrupted_runs: bool = Field(
        True,
        description="When Sentient restarts, continue runs that were in progress from their last checkpoint "
        "instead of marking them failed.",
    )
    stuck_after_minutes: int = Field(
        10, ge=0, description="A run that shows no sign of work for this many minutes (the AI model writes or thinks nothing, "
        "and no step reports progress or finishes) stops and asks you what to do: try again, skip the step or cancel. "
        "0 turns this off.",
    )
    stuck_after_repeated_errors: int = Field(
        5, ge=0, le=50, description="A run whose step keeps failing with the same error this many times in a row "
        "stops and asks you what to do. 0 turns this off.",
    )
    catch_up_window_hours: int = Field(
        12, ge=0, le=168, description="When the computer was off or asleep at a task's scheduled time, run it once "
        "when Sentient is back if it is less than this many hours late. Later ones are skipped and you're told. "
        "A task never runs more than once to catch up. 0 always skips.",
    )


# ----------------------------------------------------------------------------- integrations (owner: integrations agent)
class IntegrationsConfig(BaseModel):
    oauth_redirect_port: int = Field(
        0, ge=0, le=65535, description="Loopback port for OAuth callbacks (0 = pick a free port)."
    )
    search_provider: Literal["duckduckgo", "brave", "google_cse", "searxng"] = Field(
        "duckduckgo", description="Web search backend. DuckDuckGo needs no key."
    )
    searxng_url: str = Field("", description="Base URL of a SearXNG instance when search_provider is searxng.")
    weather_provider: Literal["open_meteo", "accuweather"] = Field(
        "open_meteo", description="Weather backend. Open-Meteo needs no key."
    )
    mcp_servers: dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "External MCP servers: {name: {transport: stdio|http, command, args, url, env, enabled, "
            "auth: none|headers|oauth, header_keys}}. Header values and sign-ins live in the keychain."
        ),
    )
    hide_disconnected_tools: bool = Field(
        True,
        description="Only show the model tools of integrations that are connected (keeps small local models focused).",
    )
    news_country: str = Field(
        "", description="Two-letter country code for top headlines (empty = guess from your location, else US)."
    )
    web_fetch_max_chars: int = Field(
        20000, ge=1000, le=200000, description="Longest page text the web fetch tool returns to the model."
    )
    github_oauth_client_id: str = Field(
        "", description="Optional GitHub OAuth app client id enabling 'Sign in with GitHub' (device flow) instead of a token."
    )
    mcp_timeout_s: int = Field(60, ge=5, le=600, description="Timeout for a single MCP tool call.")
    fast_sync_seconds: int = Field(
        60, ge=0, le=3600,
        description=(
            "How often connected Gmail and Google Calendar are checked with their change feeds, and how often "
            "IMAP email is checked when the server has no push (IDLE). Uses no model calls. 0 turns change feeds off."
        ),
    )
    hide_one_time_codes: bool = Field(
        True,
        description=(
            "Hide one-time codes, sign-in links and password reset links in your email before the AI reads it, so a "
            "tricky email can't get them out of Sentient. You still see them when you open the email in your mail app."
        ),
    )
    webhook_max_body_kb: int = Field(
        256, ge=1, le=10240, description="Largest request body (in KB) an inbound webhook accepts."
    )
    webhook_rate_limit_per_minute: int = Field(
        30, ge=0, le=6000,
        description=(
            "Most calls one inbound webhook accepts per minute. Extra calls get a 429 with a Retry-After header, "
            "so a leaked webhook URL can't flood Sentient with task runs. 0 turns the limit off."
        ),
    )


# ----------------------------------------------------------------------------- proactivity (owner: memory/proactivity agent)
class FollowUpsConfig(BaseModel):
    """Once a day, Sentient looks for emails still waiting on a reply, from you or from someone else."""

    enabled: bool = Field(
        True,
        description="Once a day, find emails still waiting on a reply and offer a ready draft. Nothing is sent "
        "without your approval.",
    )
    sources: list[str] = Field(
        default_factory=lambda: ["gmail", "email_imap"],
        description="Which connected email accounts follow-ups check (gmail, email_imap).",
    )
    waiting_on_you_days: int = Field(
        3, ge=1, le=60, description="Suggest a reply when an email sent to you has had no answer for this many days."
    )
    waiting_on_them_days: int = Field(
        4, ge=1, le=60,
        description="Suggest a nudge when your own question has had no answer for this many days.",
    )
    max_age_days: int = Field(
        21, ge=2, le=180, description="Leave conversations alone once they have been quiet for longer than this."
    )
    max_suggestions: int = Field(3, ge=1, le=20, description="At most this many follow-up suggestions per check.")


class DailyBriefConfig(BaseModel):
    """A short morning digest. It is an ordinary recurring task you can edit, pause or delete in Tasks."""

    sections: list[str] = Field(
        default_factory=lambda: ["calendar", "email", "tasks", "weather"],
        description="What the brief includes: calendar, email, tasks, weather, news. Leave one out to turn it off.",
    )
    evening_sections: list[str] = Field(
        default_factory=lambda: ["done", "sent", "files", "waiting", "tomorrow"],
        description="What the Evening Brief includes: done (tasks finished or failed today), sent (emails sent today), "
        "files (files made today), waiting (still waiting for you), tomorrow (tomorrow's first events).",
    )
    max_items: int = Field(7, ge=1, le=20, description="At most this many lines in one brief.")
    news_topics: list[str] = Field(
        default_factory=list,
        description="Topics for the news section, for example 'climate' or 'cricket'. No topics means no news.",
    )
    summarize_emails: bool = Field(
        True,
        description="Let the fast model write one short line for each unread email. Off: show the sender and subject.",
    )


class ProactivityConfig(BaseModel):
    enabled: bool = Field(True, description="Let Sentient watch connected apps and suggest actions.")
    poll_interval_minutes: int = Field(10, ge=1, description="How often Gmail/Calendar are checked.")
    base_confidence_threshold: float = Field(
        0.70, ge=0.0, le=1.0, description="Minimum confidence before a suggestion is shown."
    )
    quiet_hours: str = Field("", description="e.g. '22:00-07:00'. No suggestions during these hours.")
    heartbeat_minutes: int = Field(
        0, ge=0, description="Periodic check-in with no trigger (0 = off). Uses the fast model."
    )
    sources: list[str] = Field(
        default_factory=lambda: ["gmail", "gcalendar"], description="Connected apps watched for new items."
    )
    reasoner_role: Literal["fast", "primary"] = Field(
        "fast", description="Model role that decides whether an event deserves a suggestion."
    )
    context_agent_rounds: int = Field(
        3, ge=0, le=10,
        description="Tool rounds for the read-only search across connected apps before reasoning (0 = skip).",
    )
    first_poll_lookback_hours: int = Field(
        6, ge=0, le=168, description="On the first poll of a newly connected app, look back this many hours."
    )
    max_items_per_poll: int = Field(20, ge=1, le=200, description="New items processed per source per poll.")
    heartbeat_daily_cap: int = Field(
        3, ge=1, le=24, description="At most this many check-in suggestions per day."
    )
    suggestion_ttl_hours: int = Field(
        48, ge=1, le=720,
        description="Suggestions you have not acted on expire after this long (calendar ones when the event starts).",
    )
    webhook_suggestions: bool = Field(
        True, description="Also suggest actions for webhook calls that no triggered task handles."
    )
    followups: FollowUpsConfig = Field(
        default_factory=FollowUpsConfig,
        description="Notice emails waiting on a reply (from you or to you) and offer a draft. Gmail and IMAP email.",
    )
    brief: DailyBriefConfig = Field(
        default_factory=DailyBriefConfig,
        description="Your Daily Brief: today's calendar, emails that need you, tasks and the weather in a few lines.",
    )


# ----------------------------------------------------------------------------- self-evolution (owner: memory/evolution agent)
class SkillsConfig(BaseModel):
    extra_dirs: list[str] = Field(default_factory=list, description="Additional SKILL.md folders.")
    write_approval: bool = Field(
        True, description="Skills Sentient writes for itself wait for your review before activating."
    )


class EvolutionConfig(BaseModel):
    review_enabled: bool = Field(
        True, description="After conversations and task runs, look for procedures worth saving as skills."
    )
    review_idle_minutes: int = Field(10, ge=1, description="Review a chat after it has been idle this long.")
    min_tool_calls_for_review: int = Field(3, ge=1, description="Only review work that used at least this many tool calls.")
    curator_enabled: bool = Field(True, description="Periodically retire unused skills and merge duplicates.")
    curator_interval_hours: int = Field(168, ge=1, description="How often the curator runs.")
    stale_after_days: int = Field(14, ge=1, description="Skills unused this long are marked stale.")
    archive_after_days: int = Field(30, ge=1, description="Stale skills unused this long are archived.")
    curator_merge_suggestions: bool = Field(
        True, description="Let the curator propose merging near-duplicate skills (as pending reviews)."
    )
    user_profile_updates: bool = Field(
        True, description="Let Sentient keep USER.md and MEMORY.md up to date from what it learns."
    )
    profile_update_hours: int = Field(24, ge=1, description="How often MEMORY.md and USER.md are refreshed.")
    skill_repair: bool = Field(
        True,
        description="When a skill fails in use (tool errors, a failed task run, or you correct the result), "
        "draft a fix for your review.",
    )
    repair_cooldown_hours: int = Field(
        6, ge=0, le=168, description="Wait at least this long between repair proposals for the same skill."
    )


# ----------------------------------------------------------------------------- voice (owner: voice agent)
class VoiceConfig(BaseModel):
    stt_provider: Literal["faster_whisper", "openai", "deepgram", "elevenlabs"] = Field(
        "faster_whisper", description="Speech-to-text engine. faster-whisper runs locally."
    )
    stt_model: str = Field("base", description="faster-whisper size (tiny, base, small, medium, large-v3) or cloud model id.")
    stt_device: Literal["auto", "cpu", "cuda"] = Field(
        "auto",
        description="Where local STT runs. auto: tiny/base/small on the CPU (keeps the GPU free for the LLM), "
        "medium/large try the GPU first and fall back to the CPU.",
    )
    stt_language: str = Field(
        "", description="Spoken language for STT: empty follows assistant.language, 'auto' detects it."
    )
    tts_provider: Literal["system", "kokoro", "openai", "elevenlabs"] = Field(
        "system", description="Text-to-speech engine. 'system' uses the OS voices with no download."
    )
    tts_voice: str = Field("", description="Voice id/name for the chosen TTS engine (empty = default).")
    tts_speed: float = Field(1.0, ge=0.5, le=2.0, description="Speaking rate multiplier.")
    tts_model: str = Field(
        "", description="Cloud TTS model id (empty = gpt-4o-mini-tts for OpenAI, eleven_flash_v2_5 for ElevenLabs)."
    )
    kokoro_variant: Literal["int8", "fp16", "fp32"] = Field(
        "fp32",
        description="Kokoro model precision. fp32 (310 MB) is fastest on CPUs; int8 (88 MB) is smaller but ~4x slower.",
    )
    vad_silence_ms: int = Field(700, ge=200, le=3000, description="Silence that ends an utterance.")
    vad_min_speech_ms: int = Field(
        250, ge=50, le=2000, description="Shorter sounds (coughs, clicks) are ignored."
    )
    vad_max_utterance_s: float = Field(
        30.0, ge=3.0, le=120.0, description="An utterance is cut off and processed after this long."
    )
    barge_in: bool = Field(
        True, description="Talking while Sentient speaks interrupts it (needs echo cancellation or headphones)."
    )
    tts_first_clause_chars: int = Field(
        40,
        ge=0,
        le=240,
        description="Start speaking the first part of a reply at a comma or dash once it is this many characters "
        "long, instead of waiting for the whole first sentence. 0 always waits for full sentences.",
    )
    preload_on_start: bool = Field(
        False,
        description="Load the speech models when Sentient starts so the first voice reply is not slowed by loading "
        "(uses memory even if you never talk). Voice sessions also warm them up when they open.",
    )
    wake_word: str = Field(
        "hey sentient", description="Phrase that wakes hands-free voice sessions (smart glasses, talk mode)."
    )
    wake_engine: Literal["whisper", "openwakeword"] = Field(
        "whisper",
        description="How the wake word is detected. whisper: understands any phrase with a tiny local speech model. "
        "openwakeword: lighter always-on detector for its pretrained phrases (hey jarvis, alexa, hey mycroft) "
        "or a custom model.",
    )
    wake_sensitivity: float = Field(
        0.5, ge=0.0, le=1.0, description="Higher wakes more easily, including by mistake; lower needs a clearer phrase."
    )
    wake_model: str = Field(
        "",
        description="openwakeword only: pretrained model name (hey_jarvis, alexa, hey_mycroft, hey_rhasspy) or the "
        "path of a custom .onnx model. Empty derives it from the wake word.",
    )
    wake_whisper_model: Literal["tiny", "tiny.en", "base", "base.en"] = Field(
        "base", description="whisper wake engine: the small speech model that listens for the wake word on the CPU. base is much more reliable than tiny at hearing the greeting."
    )
    wake_earcon: bool = Field(True, description="Play a short chime when the wake word is heard.")
    follow_up_seconds: float = Field(
        8.0,
        ge=0.0,
        le=120.0,
        description="After a spoken reply, keep listening this many seconds for a follow-up without the wake word.",
    )


# ----------------------------------------------------------------------------- sandbox (owner: sandbox agent)
class SandboxConfig(BaseModel):
    enabled: bool = Field(True, description="Let the assistant write and run short Python scripts that call its tools.")
    backend: Literal["auto", "process", "docker"] = Field(
        "auto", description="auto: Docker when it is running, else a separate local process."
    )
    timeout_s: int = Field(120, ge=5, le=3600, description="A script is stopped after this long.")
    max_output_chars: int = Field(
        20_000, ge=1_000, le=1_000_000, description="Printed output kept from a script (each of output and errors)."
    )
    max_tool_calls: int = Field(
        200, ge=0, le=10_000, description="Most tool calls one script may make before further calls are refused."
    )
    max_concurrent_runs: int = Field(2, ge=1, le=16, description="Scripts that may run at the same time.")
    max_files: int = Field(50, ge=0, le=1_000, description="Most files copied from one script run to files/outputs.")
    max_file_mb: int = Field(25, ge=1, le=2_048, description="Files a script creates larger than this are not copied.")
    keep_workdirs: bool = Field(
        False, description="Keep each run's working folder under ~/.sentient/sandbox for troubleshooting."
    )
    docker_image: str = Field("python:3.12-slim", description="Container image used by the Docker backend.")
    docker_memory_mb: int = Field(512, ge=64, le=65_536, description="Memory limit for a script in Docker.")
    docker_cpus: float = Field(1.0, ge=0.1, le=64, description="CPU limit for a script in Docker.")
    allow_network_in_docker: bool = Field(
        False,
        description="Let scripts in Docker reach the network. When off the container has no network and "
        "talks to Sentient's tools through its working folder.",
    )


# ----------------------------------------------------------------------------- terminal (owner: terminal)
class TerminalConfig(BaseModel):
    """Commands on this computer (ADR 0019). Off until the user turns it on and adds a folder."""

    enabled: bool = Field(
        False,
        description="Let Sentient run commands on this computer, like git or a build script. It asks first unless "
        "the command is in the list below or you set an Allow rule.",
    )
    allowed_folders: list[str] = Field(
        default_factory=list,
        description="Folders commands may start in (and the folders inside them). Nothing runs until you add one.",
    )
    default_folder: str = Field(
        "",
        description="Where commands start when Sentient doesn't pick a folder. Empty uses the first allowed folder.",
    )
    allowed_commands: list[str] = Field(
        default_factory=lambda: ["git status", "git diff", "git log", "ls", "dir", "pwd"],
        description="Commands that never need asking. A command matches when it is exactly one of these or starts "
        "with one followed by a space, and has no ; & | < > ` $ ( ) { } or line breaks.",
    )
    timeout_s: int = Field(180, ge=5, le=3600, description="A command is stopped after this long.")
    max_output_chars: int = Field(
        8_000,
        ge=1_000,
        le=1_000_000,
        description="Output kept for the answer (each of output and errors). Longer output is saved to a file.",
    )


# ----------------------------------------------------------------------------- browser (owner: browser agent)
class BrowserConfig(BaseModel):
    enabled: bool = Field(True, description="Let the assistant use a web browser for sites without an integration.")
    engine: Literal["auto", "msedge", "chrome", "chromium"] = Field(
        "auto", description="Browser to drive. auto picks an installed Edge or Chrome."
    )
    headless: bool = Field(True, description="Run the browser hidden. The live view still shows what it does.")
    idle_minutes: float = Field(
        10, ge=0, le=1_440, description="Close the hidden browser after this many minutes without use (0 keeps it open)."
    )
    allow_domains: list[str] = Field(
        default_factory=list,
        description="Only let the assistant visit these sites (for example amazon.in). Empty allows every site.",
    )
    block_domains: list[str] = Field(
        default_factory=list, description="Sites the assistant must never open. Subdomains are blocked too."
    )
    max_snapshot_chars: int = Field(
        12_000, ge=1_000, le=200_000, description="Longest page description (buttons, fields, text) sent to the model."
    )
    max_extract_chars: int = Field(
        20_000, ge=1_000, le=500_000, description="Longest readable page text returned when reading a page."
    )
    confirm_purchases: bool = Field(
        True,
        description="Ask before clicking anything that looks like buying, paying, sending, posting or deleting.",
    )
    live_view: bool = Field(True, description="Stream small pictures of the page while the assistant uses the browser.")


# ----------------------------------------------------------------------------- devices (owner: nodes agent)
class NodesConfig(BaseModel):
    enabled: bool = Field(True, description="Allow phones, glasses and other devices to connect to Sentient.")
    lan_enabled: bool = Field(
        False, description="Accept devices on your local network (encrypted, pairing code required)."
    )
    lan_port: int = Field(7778, ge=1024, le=65535, description="Port devices on your network connect to.")
    mdns_enabled: bool = Field(
        True,
        description="Announce Sentient on your network (mDNS) so glasses and phones find it without typing an address.",
    )
    camera_requires_approval: bool = Field(
        True, description="Ask before the assistant takes a photo or captures a screen with one of your devices."
    )
    allow_desktop_screen: bool = Field(
        True, description="Let the assistant capture this computer's screen through the desktop app when you ask."
    )
    invoke_timeout_s: int = Field(20, ge=2, le=120, description="How long to wait for a device to answer a request.")
    idle_timeout_s: int = Field(
        180,
        ge=30,
        le=3600,
        description="A device that sends nothing for this long is treated as disconnected (devices ping about "
        "every third of this).",
    )


# ----------------------------------------------------------------------------- channels (owner: channels agent)
class ChannelAppConfig(BaseModel):
    """Settings for one messaging app (Telegram, Discord, WhatsApp)."""

    enabled: bool = Field(True, description="Allow this messaging app to be connected and used.")
    deliver_default: bool = Field(
        True, description="Newly paired chats also receive task results, plans to approve and suggestions."
    )
    stream_edits: bool = Field(
        True, description="Show the reply as it is written by updating the message, instead of sending it at the end."
    )
    edit_interval_s: float = Field(
        1.0, ge=0.3, le=10.0, description="Minimum seconds between message updates while a reply streams."
    )
    voice_replies: bool = Field(
        False, description="When you send a voice note, also answer with a spoken audio message."
    )
    show_tool_activity: bool = Field(
        True, description="Briefly show what the assistant is doing (for example 'Searching the web...')."
    )


class ChannelsConfig(BaseModel):
    enabled: bool = Field(True, description="Let paired messaging apps talk to Sentient.")
    telegram: ChannelAppConfig = Field(default_factory=ChannelAppConfig, description="Telegram bot settings.")
    discord: ChannelAppConfig = Field(default_factory=ChannelAppConfig, description="Discord bot (direct messages).")
    whatsapp: ChannelAppConfig = Field(
        default_factory=lambda: ChannelAppConfig(edit_interval_s=2.0),
        description="WhatsApp, linked to your own account (your 'Message yourself' chat).",
    )
    pairing_code_minutes: int = Field(
        10, ge=1, le=60, description="How long a pairing code stays valid after it is shown."
    )
    pairing_max_attempts: int = Field(
        5, ge=1, le=20, description="Wrong codes allowed before the current code is cancelled and a chat is paused."
    )
    deliver_task_results: bool = Field(True, description="Send finished and failed task results to delivery chats.")
    deliver_plans: bool = Field(True, description="Send task plans that wait for approval, with Approve buttons.")
    deliver_suggestions: bool = Field(True, description="Send proactive suggestions, with Approve and Dismiss buttons.")
    deliver_subagents: bool = Field(True, description="Send summaries when background work finishes.")
    deliver_briefs: bool = Field(True, description="Send your Daily Brief to delivery chats.")


# ----------------------------------------------------------------------------- know you (owner: memory agent)
class UserModelConfig(BaseModel):
    enabled: bool = Field(True, description="Build an evolving picture of your preferences, goals and style.")
    role: str = Field("fast", description="Model role used to refresh the user model and answer questions about you.")
    refresh_after_turns: int = Field(
        12, ge=1, le=500, description="Refresh the user model after this many completed chat turns."
    )
    min_refresh_hours: float = Field(
        24.0, ge=0.0, le=720.0,
        description="Automatic refreshes run at most this often (manual refresh and dreaming ignore it).",
    )
    max_operations: int = Field(8, ge=1, le=50, description="Most insight changes applied from one refresh.")
    max_active_insights: int = Field(80, ge=5, le=1000, description="Stop adding inferred insights beyond this many.")
    max_open_questions: int = Field(5, ge=0, le=50, description="Most open questions waiting for your answer.")
    support_step: float = Field(0.1, ge=0.0, le=0.5, description="Confidence added when new evidence supports an insight.")
    contradict_step: float = Field(
        0.2, ge=0.0, le=1.0, description="Confidence removed when new evidence contradicts an insight."
    )
    dispute_below: float = Field(
        0.35, ge=0.0, le=1.0,
        description="An insight whose confidence drops below this is marked disputed and you are asked about it.",
    )
    context_max_chars: int = Field(
        600, ge=0, le=4000, description="Budget for the user model block added to the system prompt."
    )
    context_min_similarity: float = Field(
        0.7, ge=0.0, le=1.0,
        description="An active insight counts as relevant to a message above this embedding similarity. Local "
        "embedding models score unrelated text around 0.65, so keep it high.",
    )
    recent_messages: int = Field(30, ge=0, le=200, description="Recent user messages read per refresh.")
    recent_facts: int = Field(25, ge=0, le=200, description="New or changed facts read per refresh.")
    recent_summaries: int = Field(5, ge=0, le=50, description="Conversation summaries read per refresh.")


class DreamingConfig(BaseModel):
    enabled: bool = Field(True, description="Consolidate memory overnight: merge duplicates, settle contradictions.")
    time: str = Field("03:00", description="Local time the nightly consolidation starts (HH:MM).")
    require_idle_minutes: int = Field(
        30, ge=0, le=1440, description="Only dream when you have not chatted for this many minutes."
    )
    role: str = Field("fast", description="Model role used to merge memories and confirm contradictions.")
    max_facts: int = Field(500, ge=10, le=20000, description="Most recent facts reviewed per dream.")
    max_model_calls: int = Field(20, ge=0, le=500, description="Upper bound on model calls for merging and contradictions.")
    merge_similarity: float = Field(
        0.9, ge=0.5, le=1.0,
        description="Embedding similarity above which two facts may be duplicates (word overlap is also required).",
    )
    merge_min_overlap: float = Field(
        0.6, ge=0.0, le=1.0, description="Share of meaningful words two facts must have in common to be merged."
    )
    contradiction_similarity: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Facts about the same person that share only a broad attribute (routine, ownership, health) are "
        "checked for a contradiction above this embedding similarity. Same place, job, relationship or diet is always checked.",
    )
    promote_min_recalls: int = Field(
        3, ge=1, le=1000, description="Short-term facts recalled at least this often become long-term."
    )
    refresh_user_model: bool = Field(True, description="Refresh the user model as part of each dream.")
    notify: bool = Field(True, description="Send a notification when a dream changed something meaningful.")


class SentientConfig(BaseModel):
    """Root configuration, persisted at ``~/.sentient/config.yaml``."""

    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    tasks: TasksConfig = Field(default_factory=TasksConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
    proactivity: ProactivityConfig = Field(default_factory=ProactivityConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    subagents: SubagentsConfig = Field(default_factory=SubagentsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    terminal: TerminalConfig = Field(default_factory=TerminalConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    nodes: NodesConfig = Field(default_factory=NodesConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    user_model: UserModelConfig = Field(default_factory=UserModelConfig)
    dreaming: DreamingConfig = Field(default_factory=DreamingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
