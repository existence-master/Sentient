/**
 * TypeScript mirror of docs/API.md (the desktop <-> backend contract).
 * Section numbers in comments refer to that document. When the contract changes,
 * change it here too and keep field names identical to the JSON.
 */

export type ISODate = string
export type JsonValue = string | number | boolean | null | JsonValue[] | { [k: string]: JsonValue }

/** `{"detail": "..."}` error body. */
export interface ErrorBody {
  detail: string | Array<{ loc?: Array<string | number>; msg: string; type?: string }>
}

// ============================================================================ §2 Core
export interface Health {
  ok: boolean
  version: string
  name: string
}

export const ROLE_NAMES = ['primary', 'fast', 'voice', 'planner', 'executor', 'embedding', 'vision'] as const
export type RoleName = (typeof ROLE_NAMES)[number]

export interface ModelRoles {
  primary: string
  fast: string
  /** Spoken turns (channel voice/glasses). null = primary. */
  voice: string | null
  planner: string | null
  executor: string | null
  embedding: string
  vision: string | null
}

export type ThemePreference = 'system' | 'dark' | 'light'
export const ACCENTS = ['sentient', 'violet', 'blue', 'emerald', 'rose'] as const
export type AccentName = (typeof ACCENTS)[number]

export interface AssistantConfig {
  name: string
  user_name: string
  /** IANA zone or "auto" */
  timezone: string
  location: string
  language: string
  onboarding_complete: boolean
}

export interface UIConfig {
  theme: ThemePreference
  /** One of ACCENTS; unknown values fall back to "sentient". */
  accent: string
  launch_at_login: boolean
  minimize_to_tray: boolean
}

/** §17 Stop everything. `source`: desktop | tray | hotkey | telegram | discord | device. */
export interface StopState {
  stopped: boolean
  stopped_at: ISODate | null
  source: string | null
}

/** What the local model is doing (#149): `GET /api/models/busy`, live as `model.busy`. */
export type ModelJobKind = 'chat' | 'interactive' | 'task' | 'suggestions' | 'memory' | 'skills' | 'titles' | 'background'
export interface ModelBusy {
  busy: boolean
  job: ModelJobKind | null
  model: string | null
  since: ISODate | null
  /** Jobs that go as soon as the model is free. */
  waiting: number
  /** Background jobs held back while you chat or on battery. */
  deferred: number
  /** #296: on battery only Sentient's own upkeep waits; `battery_low` at or below `battery_critical_percent`. */
  deferred_reason: 'chat' | 'battery' | 'battery_low' | null
}

export interface StopResult extends StopState {
  /** How many running jobs were cancelled. */
  cancelled: number
}

export interface Bootstrap {
  version: string
  home: string
  assistant: AssistantConfig
  models: ModelRoles
  memory_enabled: boolean
  unread_notifications: number
  ui: UIConfig
  features: { voice: boolean; proactivity: boolean; [feature: string]: boolean }
  /** §17; missing on older engines */
  stop?: StopState
}

export type PersonaId = 'friendly' | 'professional' | 'concise' | 'coach' | 'custom'

export interface OnboardingRequest {
  user_name: string
  assistant_name?: string
  timezone?: string
  location?: string
  professional_context?: string
  personal_context?: string
  persona?: PersonaId | string
  /** Set up the Daily Brief (a recurring task) as part of onboarding. */
  daily_brief?: boolean
}

export interface OkResponse {
  ok: boolean
}

/** §7 "Also forget what Sentient learned from it" (#153): how much a source taught, before and after forgetting it.
 *  ``memories`` are deleted; ``shared`` memories another source also backs stay, without this one. */
export interface ForgetCounts {
  source: string
  memories: number
  shared: number
  chats: number
  messages: number
  summaries: number
  /** what its tools returned in chats, replaced by a short note */
  results: number
  /** emails or other items proactivity kept from it */
  items: number
  /** insights about the user inferred only from what goes */
  insights: number
  item_noun: 'emails' | 'items'
}

/** A disconnect or removal with ``?forget=true`` carries what was forgotten. */
export interface Forgotten {
  forgotten?: ForgetCounts
}

// ---------------------------------------------------------------------------- config (sentient/config/schema.py)
export type ReasoningEffort = 'none' | 'low' | 'medium' | 'high'

export interface ProviderConfig {
  api_base: string | null
  api_key_env: string | null
}

export interface ModelsConfig {
  roles: ModelRoles
  fallbacks: Record<string, string[]>
  reasoning: Record<string, ReasoningEffort | string>
  temperature: Record<string, number>
  context_length: number
  context_length_per_role: Record<string, number>
  providers: Record<string, ProviderConfig>
  /** The user's own saved model setups (built-ins are not stored). */
  presets: Record<
    string,
    Omit<ModelPreset, 'name' | 'builtin' | 'available' | 'reason' | 'provider' | 'description' | 'cloud' | 'blocked_by' | 'active'>
  >
  active_preset: string | null
  /** #148: no request goes to a model outside this computer (switched from the title bar model menu). */
  local_only?: boolean
  /** #148: most to spend on paid cloud models per month in US dollars; 0 means no budget. */
  monthly_budget_usd?: number
  /** #148: at 100%, `ask` pauses paid cloud models until Keep going, `stop` until next month or a bigger budget. */
  budget_at_limit?: 'ask' | 'stop'
  /** Experimental: claude-code/<model> answers chats through the user's own Claude Code (ADR 0022). */
  experimental_claude_code?: boolean
  /** Claude Code does everything, not just chats (ADR 0032); only counts with experimental_claude_code on. */
  claude_code_everywhere?: boolean
  /** Most Claude Code runs at once; more wait their turn, chats first. */
  claude_code_max_concurrent?: number
  max_tool_rounds: number
  request_timeout_s: number
  /** #149: one local model job at a time, chats first; background waits after a chat and on battery. */
  local_queue?: boolean
  background_quiet_s?: number
  background_on_battery?: boolean
  /** #296: how upkeep behaves on battery. Tasks always run on time. */
  battery_min_percent?: number
  battery_critical_percent?: number
  battery_idle_s?: number
  battery_upkeep_every_s?: number
  battery_max_wait_s?: number
}

export interface GatewayConfig {
  host: string
  port: number
}

export interface ChatConfig {
  history_window: number
  compress_after_messages: number
  show_thinking: boolean
  auto_title: boolean
  /** Timers and reminders done at once, without the model (#150). */
  quick_commands?: boolean
  /** Incognito chats can use what Sentient already knows (#154). Nothing from them is remembered either way. */
  incognito_uses_memory?: boolean
  incognito_delete_after_minutes?: number
}

export type ApprovalMode = 'off' | 'ask' | 'always'

/**
 * Lasting rule for a tool or a whole app (ADR 0016). No entry means the normal approvals setting. `if_asked` ("Only if
 * I asked") runs without asking only when the user's own chat message asks for it; `hand_off` ("Hand it to me")
 * prepares the action and hands it to the user, never doing it (#140).
 */
export type ApprovalRule = 'allow' | 'if_asked' | 'ask' | 'hand_off' | 'never'

export interface ApprovalsConfig {
  mode: ApprovalMode
  remember_session: boolean
  timeout_s: number
  /** Key: a tool name (`gmail_send_email`) or an app id (`gmail`). A tool's own rule beats its app's rule. */
  rules: Record<string, ApprovalRule>
  /** Rules made from a chat message, by rule key (#130). Dropped by the engine when the rule changes. */
  rule_origins?: Record<string, RuleOrigin>
}

export interface RuleOrigin {
  rule: ApprovalRule
  /** The user's own words, shortened. */
  said: string
  at: ISODate | ''
  session_id: string | null
  /** `approval`: the user chose "Always allow" when Sentient asked (#139). Older engines leave it out (chat). */
  source?: 'chat' | 'approval'
  /** Where an "Always allow" answer was given: desktop, telegram, discord or whatsapp. */
  via?: string | null
}

/** A live "Allow for this task" or "Allow for 1 hour" answer (#139, `GET /api/approvals/grants`). */
export interface ApprovalGrant {
  id: string
  tool: string
  /** Plain name, e.g. `"Write file" in Files`. */
  label: string
  app: string | null
  scope: 'task' | 'timed'
  task_id: string | null
  task_name: string | null
  risk: Risk
  via: string
  created_at: ISODate
  /** Set for `timed` grants. */
  expires_at: ISODate | null
}

/** One app or tool a rule proposal covers: `label` is "Gmail > Trash", or just "Gmail" for the whole app. */
export interface RuleProposalTarget {
  key: string
  app: string | null
  tool: string | null
  label: string
}

/** "Make this a rule?" from something the user said in a chat (docs/API.md section 2, #130). */
export interface RuleProposal {
  id: string
  session_id: string
  message_id: string | null
  said: string
  rule: 'if_asked' | 'ask' | 'hand_off' | 'never'
  keys: string[]
  targets: RuleProposalTarget[]
  status: 'pending' | 'accepted' | 'declined'
  created_at: ISODate
  decided_at: ISODate | null
}

export interface ToolsConfig {
  approvals: ApprovalsConfig
  disabled: string[]
}

export interface MemoryConfig {
  facts_top_k: number
  min_similarity: number
  duplicate_similarity: number
  extract_after_turn: boolean
  workspace_budget_chars: number
  graph_link_similarity: number
  summarize_after_minutes: number
  summary_chunk_messages: number
  [key: string]: unknown
}

export interface TasksConfig {
  tick_seconds: number
  max_concurrent_runs: number
  run_timeout_minutes: number
  max_tool_rounds: number
  require_plan_approval: boolean
  swarm_max_agents: number
  [key: string]: unknown
}

export interface McpServerConfig {
  transport: 'stdio' | 'http'
  command?: string
  args?: string[]
  url?: string
  env?: Record<string, string>
  auth?: McpAuth
  header_keys?: string[]
  enabled?: boolean
}

export interface IntegrationsConfig {
  oauth_redirect_port: number
  search_provider: 'duckduckgo' | 'brave' | 'google_cse' | 'searxng'
  searxng_url: string
  weather_provider: 'open_meteo' | 'accuweather'
  mcp_servers: Record<string, McpServerConfig>
  /** App ids set to Read only (#141). */
  read_only?: string[]
  [key: string]: unknown
}

export interface ProactivityConfig {
  enabled: boolean
  poll_interval_minutes: number
  base_confidence_threshold: number
  quiet_hours: string
  heartbeat_minutes: number
  [key: string]: unknown
}

export interface SkillsConfig {
  extra_dirs: string[]
  write_approval: boolean
  [key: string]: unknown
}

export interface EvolutionConfig {
  review_enabled: boolean
  review_idle_minutes: number
  min_tool_calls_for_review: number
  curator_enabled: boolean
  curator_interval_hours: number
  stale_after_days: number
  archive_after_days: number
  user_profile_updates: boolean
  [key: string]: unknown
}

export interface VoiceConfig {
  stt_provider: 'faster_whisper' | 'openai' | 'deepgram' | 'elevenlabs' | string
  stt_model: string
  stt_device: 'auto' | 'cpu' | 'cuda'
  stt_language: string
  tts_provider: 'system' | 'kokoro' | 'openai' | 'elevenlabs' | string
  tts_model: string
  tts_voice: string
  tts_speed: number
  kokoro_variant: string
  vad_silence_ms: number
  vad_min_speech_ms: number
  vad_max_utterance_s: number
  barge_in: boolean
  wake_word: string
  /** Push to talk and dictation into any app (#169). Optional: older engines don't have it. */
  dictation?: DictationConfig
  [key: string]: unknown
}

/** The shortcuts are desktop settings (bridge `getShortcuts`: `pushToTalk`, `dictate`). */
export interface DictationConfig {
  /** raw: as heard; tidy: fillers and punctuation fixed locally; polish: also the fast model, guarded. */
  cleanup: 'raw' | 'tidy' | 'polish'
  language: string
  stop_after_silence_s: number
  speak_replies: boolean
}

export interface SentientConfig {
  assistant: AssistantConfig
  models: ModelsConfig
  gateway: GatewayConfig
  chat: ChatConfig
  memory: MemoryConfig
  tools: ToolsConfig
  skills: SkillsConfig
  evolution: EvolutionConfig
  tasks: TasksConfig
  integrations: IntegrationsConfig
  proactivity: ProactivityConfig
  voice: VoiceConfig
  ui: UIConfig
}

export type ConfigSection = keyof SentientConfig

export type DeepPartial<T> = T extends Array<infer U>
  ? Array<U>
  : T extends object
    ? { [K in keyof T]?: DeepPartial<T[K]> }
    : T

export interface ConfigPatchResponse {
  saved: boolean
  config: SentientConfig
}

/** Subset of JSON Schema emitted by pydantic (`GET /api/config/schema`). */
export interface JsonSchema {
  $ref?: string
  $defs?: Record<string, JsonSchema>
  title?: string
  description?: string
  type?: 'object' | 'array' | 'string' | 'integer' | 'number' | 'boolean' | 'null'
  properties?: Record<string, JsonSchema>
  additionalProperties?: boolean | JsonSchema
  items?: JsonSchema
  enum?: Array<string | number>
  const?: unknown
  default?: unknown
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  exclusiveMaximum?: number
  anyOf?: JsonSchema[]
  oneOf?: JsonSchema[]
  allOf?: JsonSchema[]
  format?: string
  [key: string]: unknown
}

// ---------------------------------------------------------------------------- sessions & transcript
export interface Session {
  id: string
  title: string | null
  channel: string
  created_at: ISODate
  updated_at: ISODate
  /** Nothing is learned from it and it is deleted when closed (#154). */
  incognito?: boolean
  /** The model picked for this chat (#312); null or missing follows the default model. */
  model?: string | null
  /** The memories this chat uses and saves to (#155): "personal", "work" or a project's name. */
  memory_scope?: string
  /** Who or what wrote outside content this chat read ("Gmail"), "" when clean, null before its first turn (ADR 0018). */
  untrusted?: string | null
  [key: string]: unknown
}

export interface ToolCallRef {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

/** `other`: a message someone else wrote in a paired group chat (`name` = who); `event`: something the user did to
 * the chat, e.g. trusting it again (`name: "trusted"`, content JSON `{event, was, via}`), ADR 0029. */
export type MessageRole = 'user' | 'assistant' | 'tool' | 'system' | 'other' | 'event'

/** Raw transcript row (`GET /api/sessions/{id}/messages`). */
export interface TranscriptMessage {
  id: string
  session_id?: string
  role: MessageRole
  content: string | null
  thinking?: string | null
  tool_calls?: ToolCallRef[] | null
  tool_call_id?: string | null
  name?: string | null
  attachments?: string[]
  /** Memories the final reply had in mind; `[]` on every other row. */
  memory_sources?: MemorySource[]
  sources?: AnswerSource[]
  /** What code found the final reply's calls really did (#165); `[]` on every other row. */
  checks?: OutcomeCheck[]
  created_at: ISODate
}

/**
 * §10 What code found one call really did, after it ran (#165): its result said it worked and, where the app can,
 * the effect was read back (the file, the Sent folder, the event). Never decided by a model.
 */
export interface OutcomeCheck {
  tool: string
  /** The app it went through ("Gmail", or an MCP server's name). */
  app: string
  /** `email`, `message`, `file`, `event`, `home`, or null when only the result was read. */
  kind: string | null
  /** Short label once checked: "email sent", "file written". */
  action: string
  ok: boolean
  /** Why it failed, in plain words; "" when it worked. */
  detail: string
  /** `read_back`: the effect was read back; `result`: only the call's own result was read. */
  how: 'read_back' | 'result'
  /** A later call did the same thing again and it worked, so this failure is settled. */
  settled?: boolean
  /** A failure that still stands, as the user reads it: "Sentient said it sent the email, but ...". */
  message?: string
}

/**
 * A memory a reply had in mind (§2 "Memory sources"): a fact (`id` = Memory id) or a user-model
 * insight (`id` = Insight id). `text` is how it read during that reply.
 */
export interface MemorySource {
  kind: 'fact' | 'insight'
  id: number | string
  text: string
  /** Fact source (`conversation`, `manual`, `file:<name>`...) or insight source (`user` | `inferred`). */
  source: string
  /** `prompt`: it was in front of the model; `tool`: a memory look-up returned it during the reply. */
  via: 'prompt' | 'tool'
}

/** Something a reply used (#174): a web page, an email, a file, or a connected app. Recorded from tool results. */
export interface AnswerSource {
  kind: 'web' | 'email' | 'file' | 'app'
  title: string
  /** Web page, or the email in its mail app. */
  url?: string
  /** File name under the files folder (`kind: 'file'`). */
  file?: string
  /** Full path of a note from a folder the user added (`kind: 'file'`, #176); `url` is then an obsidian:// link. */
  path?: string
  /** Name of that folder. */
  folder?: string
  /** Sender (`kind: 'email'`). */
  from?: string
  id?: string
  /** A short excerpt of what Sentient read. */
  snippet?: string
  /** Plugin id and the tool that found it. */
  app: string
  tool: string
  /** Every tool of that app the reply used (`kind: 'app'`). */
  tools?: string[]
  /** `prompt`: a piece of this note matched the message and went along with it (#354); absent: a tool read it. */
  via?: 'prompt'
}

export interface MessageSearchHit {
  session_id: string
  message_id: string
  role: 'user' | 'assistant'
  snippet: string
  created_at: ISODate
}

export interface ChatRequest {
  text: string
  session_id?: string
  attachments?: string[]
  model?: string
  channel?: string
  /** Start a new incognito chat (#154); ignored with `session_id`. */
  incognito?: boolean
}

/** How wide a yes is (#139): once, this chat, this task (task runs only), a while, or an Allow rule. */
export type ApprovalDecision = 'allow' | 'allow_session' | 'allow_task' | 'allow_timed' | 'allow_always' | 'deny'

// ---------------------------------------------------------------------------- files, tools, usage
export interface UploadedFile {
  /** Name relative to ~/.sentient/files, e.g. `uploads/report.pdf`. Pass to `attachments`. */
  name: string
  size: number
  mime: string
}

export interface FileEntry extends UploadedFile {
  modified_at: ISODate
}

export type Risk = 'read' | 'write' | 'send' | 'exec'

export interface ToolInfo {
  name: string
  description: string
  risk: Risk
}

export interface ToolPlugin {
  id: string
  display_name: string
  description: string
  category: string
  icon: string
  auth: string
  selection_hint: string
  tools: ToolInfo[]
}

export interface TokenCounts {
  prompt_tokens: number
  completion_tokens: number
}

export interface UsageReport {
  totals: TokenCounts & { cost_usd?: number }
  by_model: Array<TokenCounts & { model: string; calls: number; cost_usd?: number | null }>
  by_source: Array<TokenCounts & { source: string; calls: number }>
  by_day: Array<TokenCounts & { day: string; calls?: number }>
}

/** How a model is paid for (#148): free on this computer, on the user's Claude or ChatGPT plan, or per token. */
export type Billing = 'local' | 'plan' | 'paid'

export interface CostRow extends TokenCounts {
  calls: number
  /** US dollars from priced calls; null when none of its calls had a known price (plan, local or unpriced). */
  cost_usd: number | null
  /** Paid calls whose price LiteLLM's list doesn't know. */
  unpriced_calls: number
}

export interface CostPeriod {
  key: 'today' | 'week' | 'month'
  label: string
  since: string
  cost_usd: number
  calls: number
  tokens: number
  paid_calls: number
  unpriced_calls: number
  plan_calls: number
  plan_tokens: number
  local_calls: number
  local_tokens: number
  by_model: Array<CostRow & { model: string; billing: Billing }>
  by_role: Array<CostRow & { role: string; plan_calls: number; local_calls: number }>
  /** Claude Code calls by the role that made them, most tokens first (#444). Missing from older engines. */
  claude_code_by_role?: Array<{ role: string; calls: number; prompt_tokens: number; completion_tokens: number }>
}

export interface BudgetState {
  monthly_usd: number
  at_limit: 'ask' | 'stop'
  spent_usd: number
  month: string
  percent: number | null
  state: 'off' | 'ok' | 'warning' | 'reached'
  kept_going: boolean
  message: string | null
}

/** `GET /api/usage/cost` (#148). */
export interface CostReport {
  local_only: boolean
  periods: CostPeriod[]
  budget: BudgetState
}

// ============================================================================ §3 Models & secrets
export interface Provider {
  id: string
  label: string
  kind: 'cloud' | 'local'
  key_required: boolean
  key_set: boolean
  api_base: string | null
  docs_url: string
  suggested: string[]
  /** Connected by signing in (ChatGPT plan), never by pasting a key. `key_set` means signed in. */
  sign_in?: boolean
}

export interface LocalModel {
  name: string
  size?: number
  family?: string
  parameter_size?: string | null
  is_embedding?: boolean
}

export interface LocalRuntime {
  reachable: boolean
  models: LocalModel[]
}

export interface LocalModels {
  ollama: LocalRuntime
  lm_studio: LocalRuntime
}

/** `POST /api/models/connect/openrouter` (docs/API.md §3). */
export interface ProviderSignIn {
  auth_url: string
  state: string
}

/** `GET /api/models/connect/chatgpt` (docs/API.md §3). `available` is false when the sign-in is turned off. */
export interface ChatGPTStatus {
  available: boolean
  reason: string | null
  signed_in: boolean
  email: string | null
  manage_usage_url: string
  /** True while the one-time "You're using your ChatGPT plan" message is still to be shown. */
  welcome: boolean
}

/** `GET /api/models/connect/{openrouter|chatgpt}/{state}`. */
export interface ProviderSignInStatus {
  status: 'waiting' | 'exchanging' | 'connected' | 'failed'
  error: string | null
}

/** `POST /api/models/connect/{provider}/check`: a free request with the saved key. */
export interface ProviderKeyCheck {
  ok: boolean
  detail?: string
  error?: string
}

/** One entry of `GET /api/models/catalog/{provider}`. `id` is a full model string, e.g. `openrouter/x/y:free`. */
export interface CatalogModel {
  id: string
  label: string
  free: boolean
  tools: boolean | null
  context_length: number | null
}

/** `GET /api/models/claude-code`: Claude through the user's own Claude Code (experimental, ADR 0022). */
export interface ClaudeCodeStatus {
  enabled: boolean
  installed: boolean
  /** From `claude --version`, only asked while it is turned on. */
  version: string | null
  detail: string
  models: string[]
  /** The quick-jobs model when Claude Code does everything (claude-code/haiku). */
  fast_model?: string
  /** Claude Code does everything, not just chats (`models.claude_code_everywhere`, ADR 0032). Missing from older engines. */
  everywhere?: boolean
  /** The roles that may use a Claude Code model (#442, #444). Missing from older engines. */
  roles?: RoleName[]
  /** Most Claude Code runs at once, and how many run and wait now. */
  max_concurrent?: number
  running?: number
  waiting?: number
}

export interface ModelTestResult {
  ok: boolean
  latency_ms: number
  reply?: string
  error?: string
  supports_tools?: boolean
}

/** A model setup switched in one step (`GET /api/models/presets`, docs/API.md §3). */
export interface ModelPreset {
  name: string
  builtin: boolean
  /** False for Cloud and Mixed until a cloud key is set, or while Local only blocks it; `reason` says why. */
  available: boolean
  reason: string | null
  /** #335: the preset sets a model outside this computer. */
  cloud: boolean
  /** #335: `local_only` when Local only is on and this preset uses cloud models. */
  blocked_by: 'local_only' | null
  /** The cloud provider a built-in uses. */
  provider: string | null
  description: string | null
  /** Roles this preset sets; null = use the primary model. Roles left out keep their model. */
  roles: Partial<Record<RoleName, string | null>>
  fallbacks?: Record<string, string[]>
  reasoning?: Record<string, string>
  context_length?: number
  context_length_per_role?: Record<string, number>
  active: boolean
}

export interface ModelPresetList {
  active: string | null
  /** A role was changed by hand since the active preset was applied. */
  modified: boolean
  can_undo: boolean
  undo_preset: string | null
  presets: ModelPreset[]
}

export type PresetMissingAction = { kind: 'pull_model'; name: string; label: string } | { kind: 'add_key'; provider: string; label: string }

export interface PresetMissing {
  kind: 'pull_model' | 'add_key' | 'start_ollama' | 'sign_in'
  roles: RoleName[]
  model: string | null
  provider?: string
  detail: string
  fix: string
  action: PresetMissingAction | null
}

export interface PresetApplyResult {
  preset: string | null
  changed: { role: RoleName; from: string | null; to: string | null }[]
  missing: PresetMissing[]
  can_undo: boolean
  /** Local only after the switch (#335: applying Local only can turn it on, and undo puts it back). */
  local_only: boolean
}

/** `POST /api/models/checkup` (docs/API.md §3). */
export type CheckupStatus = 'pass' | 'warn' | 'fail' | 'skip'

export type CheckupAction =
  | { kind: 'use_model'; role: RoleName; model: string; label: string }
  | { kind: 'pull_model'; name: string; label: string }
  | { kind: 'set_reasoning'; role: RoleName; value: string; label: string }
  | { kind: 'set_context_length'; value: number; role: RoleName | null; label: string }

export interface CheckupCheck {
  id: string
  label: string
  status: CheckupStatus
  detail: string
  fix?: string
  action?: CheckupAction
}

export interface CheckupRole {
  role: RoleName
  model: string | null
  provider: string | null
  local: boolean | null
  /** Set for an optional role with no model of its own. */
  inherits: RoleName | null
  status: CheckupStatus
  checks: CheckupCheck[]
}

/** `GET /api/system/hardware` (docs/API.md §3, #131): this computer and the local model that fits it. */
export interface HardwareGpu {
  name: string
  vendor: 'nvidia' | 'amd' | 'intel' | 'apple' | 'other'
  vram_gb: number | null
  /** Ollama can run models on it (NVIDIA, AMD with its own memory, Apple silicon). */
  usable: boolean
}

export interface HardwareRecommendation {
  tier: string
  model: string
  name: string
  context_length: number
  runs_on: 'graphics' | 'processor' | 'unknown'
  /** Too little memory for a local model that can do tasks: a cloud model comes first and `model` is chat only. */
  cloud_first: boolean
  /** "qwen3:8b, reading 8,192 tokens at a time" */
  summary: string
  note: string
  /** Memory search model for Hindi and other languages besides English (#339); null when a cloud model comes first. */
  multilingual_embedding?: { model: string; name: string; size: string; note: string } | null
}

export interface Hardware {
  os: string | null
  ram_gb: number | null
  gpus: HardwareGpu[]
  unified_memory: boolean
  usable_vram_gb: number | null
  ollama_vram_gb: number | null
  /** Plain description, or "unknown" when nothing could be checked. */
  summary: string
  recommendation: HardwareRecommendation
}

/** One NDJSON line of `POST /api/models/checkup`. */
export type CheckupEvent =
  | { type: 'start'; roles: { role: RoleName; model: string | null }[]; hardware?: Hardware }
  | { type: 'step'; role: RoleName; label: string }
  | ({ type: 'role' } & CheckupRole)
  | { type: 'done'; status: CheckupStatus; roles: CheckupRole[] }

export interface EmbeddingTestResult {
  ok: boolean
  dim?: number
  error?: string
}

export interface FallbacksResponse {
  ok: boolean
  fallbacks?: Record<string, string[]>
}

/** One NDJSON line of `POST /api/models/ollama/pull`. */
export interface OllamaPullProgress {
  status: string
  digest?: string
  completed?: number
  total?: number
  error?: string
}

export interface SecretStatus {
  name: string
  set: boolean
  source: 'keychain' | 'env' | null
  /** Provider keys belong in Settings > Models; integration secrets in Integrations. */
  kind: 'provider' | 'integration'
}

// ============================================================================ §1 Live channel
interface TurnScoped {
  session_id?: string | null
  turn_id?: string | null
}

export interface HelloEvent {
  type: 'hello'
  version: string
  assistant: string
}
export interface SessionEvent {
  type: 'session'
  session_id: string
  client_id?: string | null
}
export interface ThinkingDeltaEvent extends TurnScoped {
  type: 'thinking_delta'
  text: string
}
export interface TextDeltaEvent extends TurnScoped {
  type: 'text_delta'
  text: string
}
export interface ToolCallEvent extends TurnScoped {
  type: 'tool_call'
  call_id: string
  name: string
  arguments: Record<string, unknown>
}
export interface ToolResultEvent extends TurnScoped {
  type: 'tool_result'
  call_id: string
  name: string
  result: unknown
  is_error: boolean
  duration_ms: number | null
  /** What code found the call really did (#165); absent for calls that aren't checked. */
  check?: OutcomeCheck | null
}
export interface ApprovalRequestEvent extends TurnScoped {
  type: 'approval_request'
  approval_id: string
  call_id: string
  name: string
  arguments: Record<string, unknown>
  risk: Risk
  reason: string
  /** Short plain label of the effective risk, e.g. "Purchase" (§10 `describe_fn`). */
  risk_label?: string | null
  /** Short label of what is acted on, e.g. "Place order". */
  target?: string | null
  /** Why this asks although rules would let it run: the chat read outside content (ADR 0018). */
  untrusted?: string | null
  /** What a second look found off about this call (#143), e.g. "You asked to email Arjun, but this goes to ...". */
  warning?: string | null
  /** The answers offered, in order (#139). Older engines leave it out: allow, allow_session, deny. */
  choices?: ApprovalDecision[] | null
  /** How long `allow_timed` lasts, in minutes. */
  grant_minutes?: number | null
  /** What will be sent or changed, the way a person reads it (#142). */
  preview?: ActionPreview | null
}
/** §10 A human-readable preview of an outgoing action (#142): an email's recipients, subject and text. */
export interface ActionPreview {
  kind: 'email' | 'message' | 'post' | 'delete' | 'trash' | 'purchase' | 'action' | string
  title: string
  fields: Array<{ label: string; value: string }>
  body?: string | null
  app?: string | null
}
/**
 * §10 The `tool_result` of a call a "Hand it to me" rule handed to the user instead of running (#140). It is not an
 * error: Sentient prepared it and the user takes the last step.
 */
export interface HandOff {
  handed_off: true
  message: string
  /** Plain name of what it was, e.g. `"Send email" in Gmail`. */
  label: string
  title: string
  preview: ActionPreview | null
  /** Plain details when nothing in the call reads like an email or message (a command, a click). */
  details?: Array<{ label: string; value: string }>
}
/** §10 An email, message or post waiting `seconds` before it goes, so it can still be undone (#142). */
export interface UndoWindowEvent extends TurnScoped {
  type: 'undo_window'
  undo_id: string
  call_id: string
  name: string
  seconds: number
  /** ISO time the window ends. */
  until: string
  preview?: ActionPreview | null
}
export type UndoAction = 'undo' | 'send_now'
/** Reply to `undo.respond`: `done` false means the window had already ended. */
export interface UndoAckEvent {
  type: 'undo.ack'
  undo_id: string
  action: UndoAction
  done: boolean
}
export interface UsageEvent extends TurnScoped {
  type: 'usage'
  model: string
  prompt_tokens: number
  completion_tokens: number
  /** Context meter (#131), null when the model's context length is unknown. */
  context_used?: number | null
  context_length?: number | null
  context_percent?: number | null
  /** Plain warning from 85%. */
  context_warning?: string | null
}

/** How full the model's context was after its latest call (chat `usage`, task `task.run_context`). */
export interface ContextMeter {
  used: number
  length: number
  percent: number
  warning: string | null
}
export interface ErrorEvent extends TurnScoped {
  type: 'error'
  message: string
  recoverable?: boolean
  /** #148: `local_only` (Local only kept a cloud model from being used) or `budget_reached` (monthly budget used up). */
  code?: 'local_only' | 'budget_reached' | string | null
  /** §17: a queued message was dropped by Stop everything (echoes the `client_id` it was sent with). */
  dropped?: boolean
  client_id?: string | null
}
/** §10 A plain note about this reply that is not an error, sent once before it (#312). */
export interface NoticeEvent extends TurnScoped {
  type: 'notice'
  message: string
  /** `model_fallback`: the chat's own model couldn't answer, so the default answered. */
  code?: string
  /** The chat's model that couldn't answer, and the one that answered instead. */
  model?: string | null
  used?: string | null
}
export interface DoneEvent extends TurnScoped {
  type: 'done'
  content: string
  message_id: string | null
  cancelled?: boolean
  /** §1: the turn ended in an error before any reply (the `error` event before it has the message). Nothing was saved. */
  error?: string | null
  memory_sources?: MemorySource[]
  sources?: AnswerSource[]
  /** What code found this reply's calls really did (#165). */
  checks?: OutcomeCheck[]
  /** §17: a queued message dropped by Stop everything (its text; never sent). Comes with the `client_id` it was sent with. */
  dropped?: string[]
  client_id?: string | null
}
export interface ApprovalAckEvent {
  type: 'approval.ack'
  approval_id: string
  resolved: boolean
}
export interface PongEvent {
  type: 'pong'
}
/** §10 Reply to `chat.steer`: `queued` false means no reply was running and a new turn starts. */
export interface SteerAckEvent {
  type: 'steer_ack'
  session_id: string
  queued: boolean
}
/** §10 The model received a steer message mid-reply. */
export interface UserInterjectionEvent extends TurnScoped {
  type: 'user_interjection'
  text: string
}
export type ToolProgressKind = 'stdout' | 'stderr' | 'status' | 'frame' | 'subagent'
/** §10 Streaming output of long tools (code runs, browser, subagents) before their `tool_result`. */
export interface ToolProgressEvent extends TurnScoped {
  type: 'tool_progress'
  call_id: string
  name: string
  kind: ToolProgressKind
  text?: string
  /** `data:image/jpeg;base64,...` for `frame`. */
  image?: string
  data?: Record<string, unknown>
}

/** Chat / agent events (no dot in `type`, plus `approval.ack`). */
export type ChatEvent =
  | HelloEvent
  | SessionEvent
  | ThinkingDeltaEvent
  | TextDeltaEvent
  | ToolCallEvent
  | ToolResultEvent
  | ToolProgressEvent
  | ApprovalRequestEvent
  | UndoWindowEvent
  | UsageEvent
  | ErrorEvent
  | NoticeEvent
  | DoneEvent
  | ApprovalAckEvent
  | UndoAckEvent
  | PongEvent
  | SteerAckEvent
  | UserInterjectionEvent

export type ChatEventType = ChatEvent['type']

/** Turn-scoped events as produced by `POST /api/chat` and the voice socket. */
export type AgentEvent = Exclude<ChatEvent, HelloEvent | ApprovalAckEvent | UndoAckEvent | PongEvent | SteerAckEvent>

export interface MemoryUpdatedData {
  action: 'ADD' | 'UPDATE' | 'DELETE'
  /** null for bulk changes (import, forget-by-source, expiry purge) */
  id: number | null
  content?: string
  source?: string
  /**
   * `merged` | `contradicted` | `promoted` | `expired` | `approved` | `discarded` | `review_expired` | `moved` |
   * `scope_reset` (#155), or
   * `instruction` on a held memory that reads like an instruction for how Sentient should act (#234)
   */
  reason?: string
  count?: number
  /** `pending`: the memory waits in the review inbox (ADR 0021). */
  status?: 'pending'
  /** The memory's scope (#155) on a save: where it was kept. With `reason: "moved"`, where it moved to (`previous_scope`
   * is where it was); with `reason: "scope_reset"`, the scope that was forgotten. */
  scope?: string
  previous_scope?: string
  /** An UPDATE's words before it, so the save can be undone (#234). */
  previous?: string
  /** The chat it was learned in; that chat shows "Remembered: ... Undo" (#234). */
  session_id?: string
}

export interface DomainEventMap {
  'task.updated': Task
  'task.deleted': { task_id: string }
  'tasks.away': AwayStatus
  'task.run_progress': { task_id: string; run_id: string; update: ProgressUpdate }
  'task.run_activity': { task_id: string; run_id: string; last_activity_at: ISODate }
  'task.run_context': { task_id: string; run_id: string; used: number | null; length: number | null; percent: number | null; warning: string | null }
  'notification.new': Notification
  'notification.updated': Notification
  'notification.read': { id: string | null }
  'notification.deleted': { id: string | null }
  'integration.updated': Integration
  'memory.updated': MemoryUpdatedData
  /** "Also forget what Sentient learned from it" ran for a source (#153). */
  'memory.forgotten': { id: string; source: string; label: string; counts: Omit<ForgetCounts, 'source'>; at: string }
  'skill.updated': { name: string; state: SkillState }
  /** A server's tools wait for review, or the user answered (#146). */
  'mcp.review': { server: string; reason?: 'new' | 'changed'; tools?: string[]; needed: boolean }
  /** Only the fields that changed are sent: a rename (`title`) or the chat's own model (`model`, null = default). */
  'session.updated': { session_id: string; title?: string; model?: string | null; untrusted?: string }
  /** A chat was deleted: by the user, or an incognito chat that was closed or left idle (#154). */
  'session.deleted': { session_id: string; reason?: 'incognito' }
  'rule_proposal.updated': RuleProposal
  /** A task or timed grant was added, revoked or ran out (#139); refetch the list. */
  'approval_grants.updated': { count: number }
  /** Whether anyone is looking at the window now (#139). */
  'presence.changed': { in_app: boolean }
  'config.updated': { sections: string[] }
  /** #148: the monthly budget crossed 80% or 100%, or the user chose Keep going. */
  'budget.updated': BudgetState
  'voice.state': { state: VoiceState }
  // §10-14
  'subagent.updated': Subagent
  'browser.updated': BrowserStatus
  'browser.frame': BrowserFrame
  'node.updated': DeviceNode
  'node.deleted': { node_id: string }
  'node.event': { node_id: string; event: string; data: Record<string, unknown> }
  'channel.updated': Channel
  'channel.message': ChannelMessage
  'channel.delivery': ChannelDelivery
  // §15-16
  'user_model.updated': UserModelUpdatedData
  'dream.updated': Dream
  'source.items': SourceItemsData
  // §17
  'stop.updated': StopState
  // #149
  'model.busy': ModelBusy
  // §20
  'backup.updated': BackupUpdated
  // §22
  'folders.updated': FoldersUpdated
  // §23
  'reminder.updated': Reminder
  'import.updated': MemoryImportJob
}

export type DomainEventType = keyof DomainEventMap

export type DomainEvent<K extends DomainEventType = DomainEventType> = {
  [P in K]: { type: P; data: DomainEventMap[P]; ts: ISODate }
}[K]

export type ServerMessage = ChatEvent | DomainEvent

/** Client -> server messages on `WS /ws`. */
export type ClientMessage =
  | {
      type: 'chat.send'
      session_id?: string
      text: string
      attachments?: string[]
      model?: string
      client_id?: string
      channel?: string
      /** Start a new incognito chat (#154); ignored with `session_id`. */
      incognito?: boolean
    }
  | { type: 'chat.cancel'; session_id: string }
  | { type: 'chat.steer'; session_id: string; text: string }
  | { type: 'approval.respond'; approval_id: string; decision: ApprovalDecision }
  /** §10 Undo, or send now, a send waiting in its undo window (#142). */
  | { type: 'undo.respond'; undo_id: string; action: UndoAction }
  /** The window is focused and visible, or not (#139): approval requests go to a paired chat only while it isn't. */
  | { type: 'presence'; active: boolean }
  | { type: 'ping' }

// ============================================================================ §10 Subagents
export type SubagentStatus = 'running' | 'completed' | 'error' | 'cancelled'

export interface Subagent {
  subagent_id: string
  session_id: string | null
  parent_call_id: string | null
  goal: string
  /** `reader`: the tool-less reader behind `read_with_reader` (#144); it never makes tool calls. */
  kind?: 'worker' | 'reader'
  status: SubagentStatus
  background: boolean
  summary: string | null
  error: string | null
  /** Number of tool calls the helper made. */
  tool_calls: number
  started_at: ISODate
  finished_at: ISODate | null
  /** Omitted in `subagent.updated`. */
  events?: ProgressUpdate[]
}

/** Result of `delegate_task` (foreground: final; background: `{subagent_id, status: "running"}`). */
export interface DelegateResult {
  subagent_id?: string
  status?: SubagentStatus
  summary?: string | null
  files_created?: string[]
  error?: string | null
}

// ============================================================================ §11 Code execution
export interface SandboxResult {
  ok: boolean
  backend: 'process' | 'docker' | string
  stdout: string
  stderr: string
  result: unknown
  /** Names under `files/outputs/`. */
  files_created: string[]
  tool_calls: number
  duration_ms: number
  error: string | null
}

export interface SandboxStatus {
  enabled: boolean
  backend: string
  docker_available: boolean
  python_version: string
}

// ============================================================================ §18 Terminal
/** Result of `terminal_run` (§18). */
export interface TerminalResult {
  ok: boolean
  command: string
  cwd: string | null
  shell: string | null
  exit_code: number | null
  stdout: string
  stderr: string
  timed_out: boolean
  stopped: boolean
  duration_ms: number
  /** Full output under `files/` when it was too long to keep, e.g. `outputs/terminal-<id>.txt`. */
  output_file: string | null
  error: string | null
}

export interface TerminalStatus {
  enabled: boolean
  /** `pwsh`, `powershell`, `bash`, `zsh` or `sh`; null when no shell was found. */
  shell: string | null
  shell_path: string | null
  allowed_folders: string[]
  default_folder: string | null
  /** Plain descriptions of what the built-in blocklist refuses. */
  blocked: string[]
  running: { id: string; command: string; cwd: string; started_at: string }[]
}

// ============================================================================ §12 Browser
export interface BrowserTab {
  index: number
  url: string
  title: string
  active: boolean
}

export interface BrowserStatus {
  available: boolean
  running: boolean
  engine: string | null
  headless: boolean
  tabs: BrowserTab[]
  error: string | null
  /** The running profile, or `default` when closed. */
  profile?: string
  /** True while connected to a browser the user started (an `attach` profile). */
  attached?: boolean
}

export type BrowserProfileKind = 'launch' | 'attach'

export interface BrowserProfile {
  name: string
  kind: BrowserProfileKind
  /** `''` uses the main Browser setting. */
  engine: string
  /** `attach` only: `http://127.0.0.1:<port>`. */
  endpoint: string
  notes: string
  running: boolean
}

export interface BrowserProfiles {
  active: string | null
  profiles: BrowserProfile[]
}

export interface BrowserProfileCreate {
  name: string
  kind?: BrowserProfileKind
  engine?: string
  endpoint?: string
  notes?: string
}

export type BrowserProfilePatch = Partial<Pick<BrowserProfile, 'name' | 'engine' | 'endpoint' | 'notes'>>

export interface BrowserFrame {
  url: string
  title: string
  image: string
}

// ============================================================================ §13 Devices
export type DeviceKind = 'phone' | 'glasses' | 'desktop' | 'watch' | 'custom'

export type DeviceCapability =
  | 'camera.photo'
  | 'screen.capture'
  | 'location.get'
  | 'notify.show'
  | 'display.text'
  | 'display.card'
  | 'audio.play'
  | 'speak'
  | 'mic.stream'
  | 'clipboard.read'
  | 'clipboard.write'
  | 'button.events'
  | 'battery'

/** A "node" in the contract. The UI always calls it a device. */
export interface DeviceNode {
  node_id: string
  name: string
  kind: DeviceKind | string
  platform: string
  capabilities: Array<DeviceCapability | string>
  online: boolean
  last_seen_at: ISODate | null
  /** 0..1, null when unknown. */
  battery: number | null
  created_at: ISODate
  charging?: boolean | null
  app_version?: string | null
  /** `lan` = over the home network, `local` = on this computer (gateway). */
  connection?: 'lan' | 'local' | null
  worn?: boolean | null
}

export interface DevicePairing {
  code: string
  expires_at: ISODate
  lan_enabled: boolean
  /** LAN `wss://` node URLs when the listener runs, else the gateway `ws://` URL. */
  urls: string[]
  /** Opens the web device app with the code (`https://<lan-ip>:<port>/node/#code=...` or the gateway's `/node/`). */
  web_url?: string
  /** Null when the LAN listener isn't running. */
  fingerprint?: string | null
  /** `sentient://pair?url=...&code=...&fp=...` */
  qr: string
  /** Inline SVG QR code of `web_url`. */
  qr_svg?: string
}

export interface DeviceLanInfo {
  enabled: boolean
  running?: boolean
  port: number
  urls: string[]
  web_urls?: string[]
  fingerprint: string | null
  mdns?: boolean
  error?: string | null
}

export type DeviceInvokeCode = 'offline' | 'unsupported' | 'timeout' | 'not_allowed' | 'bad_upload' | string

export interface DeviceInvokeResult {
  ok: boolean
  data?: Record<string, unknown> & { mime?: string; base64?: string; lat?: number; lon?: number; accuracy_m?: number; label?: string }
  error?: string | { code: string; message: string }
  code?: DeviceInvokeCode
}

// ============================================================================ §14 Messaging channels
export type ChannelId = 'telegram' | 'discord' | 'whatsapp'
/** `linking`: WhatsApp is waiting for its QR code (`qr`) to be scanned. */
export type ChannelStatus = 'disconnected' | 'connecting' | 'linking' | 'connected' | 'error'

export interface PairedChat {
  chat_id: string
  label: string
  paired_at: ISODate
  deliver: boolean
  session_id: string | null
  /** A group chat: only its owner (who paired it) talks to Sentient there (ADR 0029). */
  is_group?: boolean
  owner_id?: string | null
}

export interface Channel {
  id: ChannelId | string
  display_name: string
  status: ChannelStatus
  account_label: string | null
  error: string | null
  /** WhatsApp while linking: the text to show as a QR code. Changes every 20 seconds or so. */
  qr?: string | null
  paired: PairedChat[]
  setup: { fields: IntegrationSetupField[]; instructions_md: string }
}

export interface ChannelPairing {
  code: string
  expires_at: ISODate
  instructions: string
}

export interface ChannelMessage {
  channel: string
  chat_id: string
  session_id: string
  direction: 'in' | 'out'
  text: string
}

/** A notification sent to a paired chat (or that couldn't be), as the Activity screen lists it. */
export interface ChannelDelivery {
  id: string
  channel: string
  chat_id: string
  title: string
  status: 'delivered' | 'failed'
  ref: TimelineRef | null
}

// ============================================================================ §4 Tasks
export type TaskStatus =
  | 'planning'
  | 'clarification_pending'
  | 'approval_pending'
  | 'pending'
  | 'active'
  | 'processing'
  /** A run asked the user a question (`ask_user`) and waits for the answer. */
  | 'waiting_for_user'
  | 'completed'
  | 'completed_with_errors'
  | 'error'
  | 'declined'
  | 'cancelled'
  | 'archived'

export type Weekday = 'Monday' | 'Tuesday' | 'Wednesday' | 'Thursday' | 'Friday' | 'Saturday' | 'Sunday'

/** What happens to a run missed while the computer was off or asleep (§4). Absent: run once if it isn't too late. */
export type CatchUpPolicy = 'run' | 'skip'

export interface OnceSchedule {
  type: 'once'
  run_at: string | null
  timezone?: string
  catch_up?: CatchUpPolicy
}
export interface RecurringSchedule {
  type: 'recurring'
  /** `interval` runs every `interval_minutes` (minimum 5; the engine normalizes `hourly` to 60) and ignores `days`/`time`. */
  frequency: 'daily' | 'weekly' | 'interval'
  days?: Weekday[] | string[]
  time: string
  interval_minutes?: number
  /** Daily/weekly only: run at some time in this part of the day instead of at `time` (§4 "Fuzzy windows"). */
  window?: ScheduleWindow
  timezone?: string
  catch_up?: CatchUpPolicy
}
export type ScheduleWindow = 'morning' | 'afternoon' | 'evening' | 'night'

// §16 script jobs
export type ScriptCondition = 'changed' | 'noteworthy' | 'alert' | 'every_run'
export type ScriptThen = 'notify' | 'run'

export interface TaskScript {
  code: string
  condition: ScriptCondition
  then: ScriptThen
  last_result: unknown
  last_run_at: ISODate | null
  last_error: string | null
  /** Recent checks, oldest first (§16). Older engines leave it out. */
  checks?: ScriptCheck[] | null
}
export type ScriptCheckOutcome = 'baseline' | 'nothing_new' | 'changed' | 'minor' | 'alert' | 'quiet' | 'reported' | 'failed' | 'skipped'
export interface ScriptCheck {
  at: ISODate
  outcome: ScriptCheckOutcome
  note?: string
  /** A row of the same quiet outcome folds into one entry. */
  count?: number
  last_at?: ISODate
}
export interface TriggeredSchedule {
  type: 'triggered'
  source: string
  event: string
  filter?: Record<string, unknown>
}
export type TaskSchedule = OnceSchedule | RecurringSchedule | TriggeredSchedule

export interface PlanStep {
  tool: string
  description: string
}

export type ProgressMessageType = 'info' | 'thought' | 'tool_call' | 'tool_result' | 'final_answer' | 'error'

export interface ProgressUpdate {
  timestamp: ISODate
  message: {
    type: ProgressMessageType
    content?: string
    tool_name?: string
    parameters?: Record<string, unknown>
    result?: unknown
    is_error?: boolean
    /** `tool_result` only: what code found the call really did (#165). */
    check?: OutcomeCheck
  }
}

export interface LinkRef {
  url: string
  description: string
}

export interface TaskRunResult {
  summary: string
  links_created: LinkRef[]
  links_found: LinkRef[]
  files_created: Array<{ filename: string; description: string }>
  tools_used: string[]
}

/** `skipped` (#161): a routine's run that didn't start because what it reads wasn't there (`error` says why). */
export type RunStatus = 'processing' | 'waiting_for_user' | 'completed' | 'completed_with_errors' | 'error' | 'cancelled' | 'skipped'

/** The question a `waiting_for_user` run asked (§4). Answer with `POST /api/tasks/{id}/runs/{run_id}/answer`. */
export interface RunQuestion {
  question: string
  /** Suggested answers (0 to 6). A free-text answer is always allowed. */
  options: string[]
  asked_at: ISODate | null
  /** `stuck`: the run stopped getting anywhere and asks what to do (`reason` says why); `limit`: it reached a limit;
   * `approval`: it holds a call until you allow it (a tool set to Ask, or sending after outside content, #139). */
  kind?: 'question' | 'limit' | 'stuck' | 'approval'
  reason?: string | null
  /** Approvals only: after this a late answer can't approve it. */
  expires_at?: ISODate | null
  /** Approvals only (#142): what the held call would send, the way a person reads it. */
  preview?: ActionPreview | null
}

export interface Run {
  run_id: string
  status: RunStatus
  created_at: ISODate
  execution_start_time?: ISODate | null
  finished_at?: ISODate | null
  plan: PlanStep[]
  trigger_event_data?: Record<string, unknown>
  progress_updates: ProgressUpdate[]
  result?: TaskRunResult | null
  error: string | null
  /** Run id this run retries (`POST /api/tasks/{id}/runs/{run_id}/retry`). */
  retry_of?: string | null
  /** Set while `status` is `waiting_for_user`, otherwise `null`. */
  pending_question?: RunQuestion | null
  /** When the run last showed any sign of work (a step, a model reply). */
  last_activity_at?: ISODate | null
  /** Live only (never sent by the engine): how full the model's context is, from `task.run_context` (#131). */
  context?: ContextMeter | null
  /** Memories the run had in mind: facts in its instructions or found by a memory look-up (docs/API.md §4). */
  memory_sources?: MemorySource[]
  /** Pages, emails, files and apps the run used (#291), like a chat reply's sources. Older engines leave it out. */
  sources?: AnswerSource[]
  /** A try run ("Try it first", #142): it only noted what it would send, change or run. */
  dry_run?: boolean
  /** Try runs only: what a real run would do, in order. */
  would_do?: TryRunAction[]
  /** What code found the run's calls really did (#165). A failure that still stands makes it `completed_with_errors`. */
  checks?: OutcomeCheck[]
  /** A skipped run set to try again later (#161): when it does. */
  retry_at?: ISODate | null
}

/** One call a try run noted instead of making (#142). Built by the engine from the call itself. */
export interface TryRunAction {
  tool: string
  plugin?: string
  app: string | null
  /** The tool's plain name ("Send email"). */
  title: string
  kind: 'email' | 'message' | 'post' | 'delete' | 'trash' | 'purchase' | 'run' | 'change' | string
  risk: string
  preview: ActionPreview | null
  arguments: Record<string, unknown>
  /** A real run would stop and ask you first (a tool set to Ask, a purchase, sending after outside content). */
  asks: boolean
  at: ISODate | null
  /** One plain line: "Send an email in Gmail: To sam@example.com". */
  summary: string
}

export interface TaskChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: ISODate
}

export interface ClarifyingQuestion {
  question_id: string
  text: string
  answer: string | null
}

export interface SwarmProgressUpdate {
  /** `agent-1`, `agent-2`… or `aggregator` */
  worker_id: string
  timestamp: ISODate
  status: 'processing' | 'completed' | 'error' | 'aggregating' | string
  message: string
}

export interface SwarmDetails {
  goal: string
  items: unknown[]
  total_agents: number
  completed_agents: number
  progress_updates: SwarmProgressUpdate[]
  aggregated_results: unknown[]
}

export interface Task {
  task_id: string
  name: string
  description: string
  status: TaskStatus
  priority: number
  assignee: 'ai' | string
  task_type: 'single' | 'swarm' | 'script'
  schedule: TaskSchedule | null
  /** `null` unless `task_type` is `script` (§16). */
  script?: TaskScript | null
  plan: PlanStep[]
  runs: Run[]
  chat_history: TaskChatMessage[]
  clarifying_questions: ClarifyingQuestion[]
  /** `null` for single tasks. */
  swarm_details: SwarmDetails | null
  enabled: boolean
  /** Never paused while you are away (#163). Older engines leave it out (false). */
  keep_running?: boolean
  model: string | null
  /** Named browser profile its browser steps use (§12); `null` is the default one. */
  browser_profile?: string | null
  /** The memories its runs use and save to (#155): "personal", "work" or a project's name. */
  memory_scope?: string
  /** Where its notifications go besides the app (§4 "Where results go"). Older engines leave it out (default). */
  deliver_to?: TaskDeliverTo
  /** Recurring tasks: short notes its runs keep for the next run (§4 "Recurring tasks remember"). */
  notepad?: string | null
  /** Routines: the setup checklist (§4 "Setup checklist", #161). `null` for other tasks; older engines leave it out. */
  setup?: TaskSetup | null
  original_context: { source: 'manual_creation' | 'chat' | 'proactive' | 'trigger' | string; [k: string]: unknown }
  /** Last planning/run failure message (v2 `task.error`). */
  error: string | null
  next_execution_at: ISODate | null
  last_execution_at: ISODate | null
  created_at: ISODate
  updated_at: ISODate
}

/** What a routine reads (#161): a web page, a connected app, a folder, or nothing to check first. */
export interface TaskSource {
  kind: 'page' | 'app' | 'folder' | 'none'
  /** The web address, app id or folder path; empty for `none`. */
  target: string
  /** How to show it: "example.com/events", "Gmail". */
  label: string
}

/** What to do when what a routine reads isn't there: skip quietly, skip and tell me, or try again a little later. */
export type MissingSourcePolicy = 'skip' | 'notify' | 'retry'

/** A routine's setup checklist (§4 "Setup checklist", #161), filled in from its plan for the user to confirm. */
export interface TaskSetup {
  source: TaskSource
  /** What the user gets each time. */
  output: string
  /** `plan`: goes ahead with what the plan does. `ask`: asks before it sends, posts, buys, deletes or runs anything. */
  approval: 'plan' | 'ask'
  /** The plan's apps that can send or change things, which `plan` lets it use. */
  acts: string[]
  if_missing: MissingSourcePolicy
  confirmed_at: ISODate | null
}

/** `PATCH /api/tasks/{id}` `{setup}`: answers to merge in; `confirm: true` marks the checklist confirmed. */
export interface TaskSetupAnswers {
  source?: { kind: TaskSource['kind']; target?: string }
  output?: string
  approval?: TaskSetup['approval']
  if_missing?: MissingSourcePolicy
  confirm?: boolean
}

export interface TaskCreateRequest {
  prompt: string
  is_swarm?: boolean
  assignee?: 'ai'
  model?: string
  browser_profile?: string
  /** The memories its runs use (#155); personal when left out. */
  memory_scope?: string
}

export interface TaskPreview {
  name: string
  description: string
  priority: number
  schedule: TaskSchedule | null
}

/** `POST /api/tasks/preview` may also describe a script job when the planner proposes one. */
export type TaskPreviewWithScript = TaskPreview & { task_type?: string; script?: Partial<TaskScript> | null }

/** `{type: "recurring", frequency: "interval", interval_minutes}` (minimum 5). */
export interface IntervalSchedule {
  type: 'recurring'
  frequency: 'interval'
  interval_minutes: number
  timezone?: string
}

/** A paired chat a task sends to. `{channel: 'whatsapp', chat_id: 'self'}` is WhatsApp's "Message yourself" chat. */
export interface DeliveryChat {
  channel: 'telegram' | 'discord' | 'whatsapp' | string
  chat_id: string
}
/** `default`: paired chats with delivery on; `desktop`: the app only; a list: only those chats. */
export type TaskDeliverTo = 'default' | 'desktop' | DeliveryChat[]

export type TaskPatch = Partial<Pick<Task, 'name' | 'description' | 'priority' | 'schedule' | 'plan' | 'enabled' | 'status' | 'model' | 'browser_profile' | 'memory_scope' | 'deliver_to' | 'notepad' | 'keep_running'>> & {
  /** Routines (#161): answers to the setup checklist, merged in by the engine. */
  setup?: TaskSetupAnswers
}

/** `GET /api/tasks/away` (#163): routines pause after a long absence and resume with one click. */
export interface AwayStatus {
  state: 'active' | 'asked' | 'paused'
  last_active_at: ISODate | null
  /** When the question was asked (`asked`) or the routines were paused (`paused`). */
  since: ISODate | null
  reason: 'days' | 'results' | null
  /** `paused`: the tasks paused for absence. */
  task_ids: string[]
  notification_id: string | null
  /** `asked`: when they pause if you don't come back. */
  pause_at?: ISODate
  away_pause_days: number
  away_pause_after_results: number
  away_ask_hours: number
}

export interface ClarificationAnswer {
  question_id: string
  answer_text: string
}

// ============================================================================ §5 Integrations
export type IntegrationCategory = 'productivity' | 'communication' | 'knowledge' | 'development' | 'home' | 'utilities' | 'core'
export type IntegrationAuthType = 'builtin' | 'oauth' | 'api_key' | 'manual' | 'mcp'
export type IntegrationStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export interface IntegrationSetupField {
  key: string
  label: string
  secret: boolean
  required: boolean
  help?: string
}

export interface Integration {
  id: string
  display_name: string
  description: string
  category: IntegrationCategory | string
  icon: string
  auth_type: IntegrationAuthType
  connected: boolean
  account_label: string | null
  status: IntegrationStatus
  error: string | null
  setup: { fields: IntegrationSetupField[]; instructions_md: string; docs_url: string | null }
  privacy_filters: { supported: boolean; fields: string[] }
  triggers: Array<{ event: string; label: string }>
  /**
   * For optional keyed providers that replace a keyless builtin
   * (`accuweather` -> `weather`, `brave_search` -> `internet_search`...). These have no tools.
   */
  alternative_for: string | null
  /** Read only (`read`) or Read and write (`read_write`, the default). Older engines leave it out. */
  access?: ConnectionAccess
  /** #175: the integration has devices to choose from (Home Assistant). Older engines leave it out. */
  entity_picker?: boolean
  tools: ToolInfo[]
}

/** #175: one device (entity) of an integration with an entity picker. */
export interface IntegrationEntity {
  entity_id: string
  name: string
  /** light, switch, climate, lock, cover, sensor... */
  domain: string
  area: string
  state: string | null
  /** Sentient may see and use it. */
  allowed: boolean
}

/** #175: every device, or only the picked ones. */
export interface EntityChoice {
  mode: 'all' | 'picked'
  picked: string[]
}

export interface IntegrationEntities extends EntityChoice {
  entities: IntegrationEntity[]
}

/** #141: a Read only connection hides and refuses its tools that change, send, delete or run something. */
export type ConnectionAccess = 'read' | 'read_write'

/**
 * `connect` result for OAuth (`{auth_url, state}`) and GitHub device flow (`+ user_code`).
 * Open `auth_url` with `getBridge().openExternal`; show `user_code` prominently when present.
 * Completion arrives as an `integration.updated` event.
 */
export interface OAuthStart {
  auth_url: string
  state: string
  user_code?: string
}

export type ConnectResponse = Integration | OAuthStart

export interface IntegrationTestResult {
  ok: boolean
  detail: string
}

export interface PrivacyFilters {
  keywords: string[]
  emails: string[]
  labels: string[]
  [field: string]: string[]
}

export interface McpToolInfo {
  /** Sentient tool name, `mcp_<server>_<tool>`. */
  name: string
  /** The tool's name on the MCP server. */
  mcp_name: string
  description: string
  risk: Risk
  /** The risk is decided per call from the tools it runs (Composio's multi-execute). Older engines leave it out. */
  per_call?: boolean
}

/** One tool in a server's safety check (#146). */
export interface McpToolReview extends McpToolInfo {
  /** `new` and `changed` tools are not offered until the user decides. */
  status: 'approved' | 'declined' | 'new' | 'changed'
  level: VetVerdict
  findings: SkillLintFinding[]
  suggested: 'approve' | 'decline'
  /** `changed` only: the description the user approved before, and that answer. */
  previous_description?: string
  previous_decision?: 'approved' | 'declined'
}

export interface McpReview {
  /** Some tools wait for the user's answer. */
  needed: boolean
  reason: 'new' | 'changed' | null
  verdict: VetVerdict
  tools: McpToolReview[]
  scanner: string
}

export type McpServerStatus = 'connecting' | 'connected' | 'needs_sign_in' | 'error' | 'disconnected' | 'disabled'

/** How a remote server is signed in to: nothing, static headers, or the MCP OAuth sign-in. */
export type McpAuth = 'none' | 'headers' | 'oauth'

export interface McpServer {
  name: string
  transport: 'stdio' | 'http'
  command: string | null
  args: string[]
  url: string | null
  /** Env values live in the keychain and are never returned. */
  env_keys: string[]
  auth: McpAuth
  /** Header values live in the keychain and are never returned. */
  header_keys: string[]
  /** Header or env names with no saved value yet (an imported server, for example). Older engines leave it out. */
  missing_values?: string[]
  /** An OAuth sign-in is stored. */
  signed_in: boolean
  /** A browser sign-in is waiting for the user. */
  signing_in: boolean
  enabled: boolean
  /** Read only or Read and write. Older engines leave it out. */
  access?: ConnectionAccess
  status: McpServerStatus | string
  /** Only the tools Sentient offers (approved, unchanged). */
  tools: McpToolInfo[]
  /** The safety check of every tool the server listed when it last connected; null before it connects. */
  review?: McpReview | null
  error: string | null
  /** Active transport result; null when disabled, disconnected, connecting, or awaiting sign-in. */
  health?: 'healthy' | 'failing' | null
  last_check_at?: ISODate | null
  /** Number of attempted calls, including calls rejected by the MCP transport. */
  usage_count?: number
}

export interface McpCatalogEntry {
  id: string
  name: string
  description: string
  documentation_url: string
  transport: 'stdio' | 'http'
  command?: string
  args?: string[]
  required_values: string[]
  enabled: boolean
}

export interface McpServerCreate {
  name: string
  transport: 'stdio' | 'http'
  command?: string
  args?: string[]
  url?: string
  env?: Record<string, string>
  headers?: Record<string, string>
  auth?: McpAuth
  enabled?: boolean
  access?: ConnectionAccess
}

/** `POST /api/integrations/mcp/{name}/sign-in`: open `auth_url` in the browser. */
export interface McpSignInStart {
  auth_url: string
  state: string
}

export interface McpTestResult {
  ok: boolean
  /** MCP tool names. */
  tools: string[]
  error?: string
}

// ============================================================================ §6 Notifications & proactivity
export type NotificationKind = 'info' | 'task' | 'approval' | 'proactive' | 'skill' | 'error' | 'brief' | 'reminder'

/** Daily Brief (docs/API.md section 6). */
export type BriefKind = 'morning' | 'evening'
/** Morning sections, then the Evening Brief's wrap-up sections. */
export type BriefSectionId = 'calendar' | 'email' | 'tasks' | 'weather' | 'news' | 'done' | 'sent' | 'files' | 'waiting' | 'tomorrow'
export type BriefFeedback = 'up' | 'down'

export interface BriefItem {
  id: string
  section: BriefSectionId
  /** One line. */
  text: string
  /** `https://...` opens the source; `/tasks/<id>` opens a task in the app; null when there is nothing to open. */
  link: string | null
  /** Why am I seeing this. */
  why: string
  feedback: BriefFeedback | null
  /** Email items that came from a suggestion card. */
  notification_id?: string
}

export interface Brief {
  /** The notification id. */
  id: string
  kind: BriefKind
  day: string
  title: string
  status: 'active' | 'expired'
  task_id: string | null
  created_at: ISODate
  expires_at: ISODate
  sections: Array<{ id: BriefSectionId; label: string; feedback: BriefFeedback | null }>
  items: BriefItem[]
  /** Sections that found nothing, with a plain reason ("Connect Google Calendar to see today's events."). */
  skipped: Array<{ section: BriefSectionId; label: string; reason: string }>
}

/** One brief's task: its schedule and sections. */
export interface BriefTaskState {
  set_up: boolean
  task_id: string | null
  /** False while the task is paused. */
  enabled: boolean
  /** 'HH:MM' local time. */
  time: string | null
  days: string[] | null
  next_at: ISODate | null
  sections: BriefSectionId[]
}

/** The morning brief's fields at the top level, the Evening Brief's under `evening`. */
export interface BriefState {
  set_up: boolean
  task_id: string | null
  /** False while the task is paused. */
  enabled: boolean
  /** 'HH:MM' local time. */
  time: string | null
  days: string[] | null
  next_at: ISODate | null
  sections: BriefSectionId[]
  news_topics: string[]
  max_items: number
  available: Record<BriefSectionId, boolean>
  /** The brief showing now (one at a time, morning or evening). */
  today: Brief | null
  evening: BriefTaskState
}

export interface BriefSetup {
  /** Which brief: morning (default) or evening. */
  kind?: BriefKind
  /** 'HH:MM' or a word: early, morning, midday, afternoon, evening. */
  time?: string
  /** Day names, 'weekdays' or 'daily'. */
  days?: string[] | string
  sections?: BriefSectionId[]
  news_topics?: string[]
  max_items?: number
}

export interface ProactiveSuggestion {
  suggestion_type: string
  description: string
  action_details: { action_type: string; [k: string]: unknown }
  reasoning: string
  confidence: number
  source_event: { source: 'gmail' | 'gcalendar' | 'heartbeat' | string; event_type: string; summary: string; item_id?: string; url?: string }
  /** Set on follow-up suggestions (dropped email threads), docs/API.md section 6. */
  follow_up?: FollowUp
}

export interface FollowUp {
  kind: 'waiting_on_you' | 'waiting_on_them'
  person: string
  person_email: string
  to: string
  subject: string
  draft: string
  days_waiting: number
  thread_id: string
  message_id: string
  mailbox?: string
}

export interface SuggestionPayload {
  suggestion: ProactiveSuggestion
  status: 'pending' | 'approved' | 'dismissed'
  task_id: string | null
}

export interface Notification {
  id: string
  kind: NotificationKind
  title: string | null
  /** markdown */
  message: string
  payload: Partial<SuggestionPayload> & Record<string, unknown>
  task_id: string | null
  read: boolean
  created_at: ISODate
}

export interface NotificationList {
  notifications: Notification[]
  unread: number
}

export interface SuggestionActionResponse {
  ok: boolean
  task_id?: string
}

export interface ProactivitySource {
  source: string
  connected: boolean
  last_poll_at: ISODate | null
  last_error: string | null
}

export interface ProactivityStatus {
  enabled: boolean
  last_poll_at: Record<string, ISODate | null>
  sources: ProactivitySource[]
  suggestions_today: number
  /** Inside `proactivity.quiet_hours` right now (suggestions are held). */
  quiet_now?: boolean
  heartbeat_minutes?: number
  /** Daily check for emails waiting on a reply. */
  followups?: { enabled: boolean; last_run_at: ISODate | null }
}

export interface ProactivityPreference {
  suggestion_type: string
  score: number
  threshold: number
  approvals: number
  dismissals: number
}

// ============================================================================ §7 Memory
export const MEMORY_TOPICS = [
  'Personal Identity',
  'Interests & Lifestyle',
  'Work & Learning',
  'Health & Wellbeing',
  'Relationships & Social Life',
  'Financial',
  'Goals & Challenges',
  'Miscellaneous'
] as const
export type MemoryTopicName = (typeof MEMORY_TOPICS)[number]

export interface Memory {
  id: number
  content: string
  topics: string[]
  source: string
  memory_type: 'long-term' | 'short-term'
  created_at: ISODate
  updated_at: ISODate
  expires_at: ISODate | null
  /** Present on semantic search results (`q`). */
  similarity?: number
  /** Listed memories are always `active`; `pending` ones are only in the review inbox. */
  status?: 'active' | 'pending'
  review?: ReviewNote | null
  /** Where it is kept (#155): "personal" (default), "work", "always" (every chat) or a project's name. */
  scope?: string
  /** #152: how sure Sentient is (0..1), from where it came and how often it was learned again. */
  confidence?: number
  /** #152: only `current` memories go into chats and tasks. */
  state?: MemoryState
  /** The memory that replaced it (state `superseded`). */
  superseded_by?: number | null
  /** Its words, on `GET /api/memories/{id}` of a superseded memory. */
  superseded_by_content?: string | null
  /** Exactly where its own source said it. */
  ref?: MemorySourceRef | null
  /** Every source, its own first (#152). */
  sources?: MemorySourceLink[]
}

export type MemoryState = 'current' | 'disputed' | 'superseded'

/** Exactly where a source said a memory (#152): a chat message, an email or app result, a note, a file. */
export interface MemorySourceRef {
  kind?: 'message' | 'email' | 'app' | 'note' | 'file' | string
  session_id?: string
  message_id?: string
  quote?: string | null
  title?: string
  url?: string
  id?: string
  app?: string
  path?: string
  folder_id?: string
  file?: string
}

export interface MemorySourceLink {
  source: string
  ref: MemorySourceRef | null
  at: string
  /** its own source (the first one) */
  own: boolean
}

/** One memory scope (#155) from `GET /api/memories/scopes`. */
export interface MemoryScope {
  name: string
  label: string
  /** Active memories kept there. */
  count: number
  /** Always, Personal and Work; projects are the user's own. */
  built_in: boolean
}

/** Where a memory waiting for review came from (ADR 0021). */
export interface ReviewNote {
  /** Plain name of the source: "Gmail", "Hermes", "resume.pdf", "a proactive check". */
  from: string
  /** The text it was taken from, at most 400 characters. */
  snippet: string | null
  session_id: string | null
}

export interface MemoryReviewItem {
  kind: 'fact' | 'insight'
  id: number | string
  text: string
  source: string
  from: string
  snippet: string | null
  session_id: string | null
  /** An import from another assistant: the conversation, note or file it was in, its day and the export's name. */
  title?: string | null
  dated?: string | null
  export?: string | null
  /** `instruction`: it reads like an instruction for how Sentient should act, so it waits even from a clean chat (#234). */
  reason?: string | null
  created_at: ISODate
  /** When it is let go if nobody reviews it. */
  expires_at: ISODate | null
  /** A fact's memory scope (#155): it is approved into it. Null for insights. */
  scope?: string | null
}

export interface MemoryReviewInbox {
  items: MemoryReviewItem[]
  count: number
  expire_days: number
  /** Review history (#353): memories the nightly tidy-up merged into another one, newest first. Not in `count`. */
  merged?: MergedMemory[]
}

/** A memory the nightly tidy-up merged into another one because they said the same thing in two languages (#353). */
export interface MergedMemory {
  id: number
  text: string
  source: string
  merged_at: ISODate
  /** Who merged it, in plain words ("nightly tidying"). */
  by: string
  /** The memory that stayed; null when it was deleted since. */
  merged_into: { id: number; text: string } | null
}

export interface MemoryQuery {
  topic?: string
  q?: string
  source?: string
  scope?: string
  /** None: current and disputed; `all` adds superseded ones (#152). */
  state?: MemoryState | 'all'
  limit?: number
  offset?: number
}

export interface MemoryTopic {
  name: string
  description: string
  count: number
}

export interface MemoryGraphNode {
  id: number
  /** Content truncated to 25 chars (v2). */
  label: string
  /** Full content. */
  title: string
  content: string
  topics: string[]
  memory_type: 'long-term' | 'short-term' | string
  source: string
  created_at: ISODate
}

export interface MemoryGraph {
  nodes: MemoryGraphNode[]
  /** A link means cosine similarity >= memory.graph_link_similarity; `value` is that similarity. */
  links: Array<{ source: number; target: number; value: number }>
}

export interface MemoryWriteResult {
  /** `SKIP` = duplicate of an existing memory (its id is returned). */
  action: 'ADD' | 'UPDATE' | 'DELETE' | 'SKIP'
  id: number | null
  content: string
  status?: 'pending'
}

export interface MemoryImportResult {
  added: number
  updated: number
  skipped: number
  /** How many wait for review (older engines leave it out). */
  pending?: number
  source: string
}

export interface MemorySummary {
  id: string | number
  content: string
  start_at: ISODate
  end_at: ISODate
  session_id: string | null
  /** The app whose content that chat read; such summaries are kept out of other chats (ADR 0021). */
  untrusted?: string | null
}

export type WorkspaceFileId = 'soul' | 'user' | 'memory'

export interface WorkspaceSnapshot {
  soul: string
  user: string
  memory: string
  today: string
  yesterday: string
}

export interface Persona {
  id: string
  name: string
  description: string
  soul_md: string
}

// ============================================================================ §8 Skills & self-evolution
export type SkillAuthor = 'user' | 'assistant' | 'community'
export type SkillState = 'active' | 'pending_review' | 'stale' | 'archived' | 'rejected' | 'deleted'

export interface Skill {
  name: string
  description: string
  author: SkillAuthor | string
  state: SkillState
  tags: string[]
  requires_tools: string[]
  version: string | number
  use_count: number
  view_count: number
  patch_count: number
  last_used_at: ISODate | null
  created_by_review: boolean
  /** Browser profile named in the skill's frontmatter (§12). */
  browser_profile?: string | null
  /** Pending proposals only: why Sentient proposed it and where it came from. */
  reason?: string | null
  origin?: { session_id?: string; task_id?: string; run_id?: string; curator?: boolean; merged_from?: string } | null
  proposed_at?: string | null
  /** Active skills: the file differs from the version Sentient last wrote (edited in Skills or any editor).
   * Pending proposals: the skill it would replace was edited by the user. */
  user_edited?: boolean
  /** Active skills: `POST /api/skills/{name}/reset` can put the original back. */
  can_reset?: boolean
  /** Pending proposals: the active skill changed after this was drafted, so approving undoes those changes. */
  changed_since_proposal?: boolean
  /** What the skill linter found (#164). */
  lint?: SkillLintFinding[]
  /** Skills from outside (a link or Hermes): the safety check, run again on the files as they are now (#146). */
  vetting?: SkillVetting | null
  /** How to check the skill works, from its `validate` header (#386). */
  validate?: SkillCheck[]
  /** The last run of those checks, or null when it declares none or they haven't run. */
  check?: SkillCheckReport | null
}

/** One check in a skill's `validate` header: a file that must exist, or Python run in the code sandbox. */
export interface SkillCheck {
  file?: string
  run?: string
  /** Text the file or the output must contain. */
  expect?: string
  timeout_s?: number
}

/** `waiting`: a code check runs when the user approves. `not_run`: it can't run here (the detail says why). */
export interface SkillCheckReport {
  status: 'passed' | 'failed' | 'waiting' | 'not_run'
  results: { kind: 'file' | 'run'; target: string; ok: boolean | null; detail: string }[]
  checked_at: string
}

export interface SkillLintFinding {
  rule: 'frontmatter' | 'required' | 'size' | 'reference' | 'danger' | 'hidden' | 'outbound' | 'log_shaped' | 'sections' | 'validate' | string
  level: 'error' | 'warning'
  message: string
}

/** `high`: we suggest declining; approving needs "approve anyway". `check`: notes to read. `clear`: nothing found. */
export type VetVerdict = 'high' | 'check' | 'clear'

export interface SkillVetFinding extends SkillLintFinding {
  /** `SKILL.md` or another file in the skill's folder. */
  file?: string
}

export interface SkillVetting {
  verdict: VetVerdict
  /** One plain sentence for the top of the report. */
  summary: string
  findings: SkillVetFinding[]
  /** Every file the skill brings, relative to its folder. */
  files: string[]
  /** Scripts or programs among them. */
  programs: string[]
  hash: string
  scanner: string
  source: 'link' | 'hermes' | string | null
  /** The web address or Hermes folder it came from. */
  origin: string | null
  imported_at?: ISODate
  decision?: 'approved' | null
  changed_since_import?: boolean
  /** Approved, then its files changed. */
  changed_since_approval?: boolean
}

/** A SKILL.md in the skills folder that can't load. */
export interface BrokenSkill {
  folder: string
  state: 'active' | 'pending_review' | 'archived'
  path: string
  lint: SkillLintFinding[]
}

export type SkillDetail = Skill & { body: string }

export interface SkillsList {
  active: Skill[]
  pending: Skill[]
  archived: Skill[]
  broken?: BrokenSkill[]
}

export interface SkillCreate {
  name: string
  description: string
  body: string
  tags?: string[]
  requires_tools?: string[]
}

export interface SkillUpdate {
  description?: string
  body?: string
  tags?: string[]
  requires_tools?: string[]
  /** Which copy to edit: a proposal awaiting review is `pending`. Defaults to the active skill. */
  target?: 'active' | 'pending' | 'archived'
}

export interface SkillDiff {
  current: string
  proposed: string
  user_edited?: boolean
  changed_since_proposal?: boolean
}

export interface SkillReviewResult {
  reviewed: number | boolean
  proposed: string[]
}

export type EvolutionKind =
  | 'skill_created'
  | 'skill_patched'
  | 'skill_archived'
  | 'skill_reset'
  | 'profile_updated'
  | 'curator_run'
  | 'summary_created'

/**
 * `detail` examples: skill_created {name, pending, reason?, session_id?|task_id?},
 * curator_run {staled, archived, merge_proposals}, profile_updated {learned_appended, facts_considered,
 * summaries_considered, memory_chars}, summary_created {id, session_id, start_at, end_at}.
 */
export interface EvolutionLogEntry {
  ts: ISODate
  kind: EvolutionKind | string
  detail: Record<string, unknown>
}

/** Payload of a `skill` notification from the reviewer/curator. */
export interface SkillNotificationPayload {
  skill: string
  action: 'create' | 'patch'
  origin: string
}

// ============================================================================ §9 Voice
/** `standby` = wake mode, waiting for the wake word (§16). */
export type VoiceState = 'standby' | 'listening' | 'transcribing' | 'thinking' | 'speaking' | 'idle'

export interface WakeStatus {
  engine: string
  phrase: string
  ready: boolean
  error?: string
}

export interface VoiceStatus {
  sessions?: number
  stt: {
    provider: string
    model: string
    /** Model in memory (local) or key set (cloud). */
    ready: boolean
    device: string
    compute_type?: string
    /** Explains e.g. a CPU fallback. */
    note?: string
    error?: string
  }
  tts: {
    provider: string
    voice: string
    ready: boolean
    voices: Array<{ id: string; name: string; language: string }>
    backend?: string
    downloaded?: boolean
    variant?: string
    error?: string
  }
  /** Wake-word engine (§16). */
  wake?: WakeStatus
}

export interface TranscribeResult {
  text: string
}

/** `POST /api/voice/dictate`: `polished` is true only when the fast model's version was used. */
export interface DictateResult {
  text: string
  raw: string
  cleanup: DictationConfig['cleanup']
  polished: boolean
}

export type VoicePrepareTarget = 'all' | 'stt' | 'tts' | 'wake'

/** One NDJSON line of `POST /api/voice/prepare`; the stream ends with `{stage: "done", progress: 1, ok}`. */
export interface VoicePrepareProgress {
  stage: 'download' | 'loading' | 'ready' | 'error' | 'done' | string
  component?: 'stt' | 'tts'
  progress: number | null
  file?: string
  bytes?: number
  total?: number
  device?: string
  note?: string
  message?: string
  ok?: boolean
}

export type VoiceClientMessage =
  | { type: 'start'; session_id?: string; sample_rate: number }
  | { type: 'end_utterance' }
  | { type: 'interrupt' }
  | { type: 'text'; text: string }
  | { type: 'approval.respond'; approval_id: string; decision: ApprovalDecision }
  | { type: 'ping' }
  | { type: 'stop' }

export interface VoiceAudioMetrics {
  audio_ms?: number
  stt_ms?: number
  first_token_ms?: number
  first_tts_ms?: number
  first_audio_ms?: number
  total_ms: number
}

export type VoiceServerMessage =
  | { type: 'ready'; session_id: string; stt: string; tts: string; sample_rate: number }
  | { type: 'state'; state: VoiceState; session_id?: string }
  | { type: 'transcript'; text: string; final: boolean; session_id?: string; stt_ms?: number }
  /** Followed by one binary frame: a complete PCM16 mono WAV for this sentence. */
  | { type: 'audio'; format: 'wav'; sentence_index: number; text: string; session_id?: string }
  | { type: 'audio_end'; session_id?: string; sentences?: number; metrics?: VoiceAudioMetrics; interrupted?: boolean; reason?: 'client' | 'barge_in' | 'cancelled' }
  | { type: 'error'; message: string; recoverable?: boolean; session_id?: string }
  | ApprovalAckEvent
  | PongEvent
  | AgentEvent

// ============================================================================ §15 User model
export const INSIGHT_DIMENSIONS = [
  'preferences',
  'communication',
  'goals',
  'routines',
  'relationships',
  'values',
  'work_style',
  'dislikes',
  'context'
] as const
export type InsightDimension = (typeof INSIGHT_DIMENSIONS)[number]

export type InsightStatus = 'active' | 'confirmed' | 'disputed' | 'retired'

export interface InsightEvidence {
  kind: 'fact' | 'message' | 'summary' | 'feedback' | string
  ref: string | number | null
  quote: string
  at: ISODate | null
}

export interface Insight {
  id: string
  dimension: InsightDimension | string
  statement: string
  /** 0..1 */
  confidence: number
  status: InsightStatus
  source: 'inferred' | 'user' | 'import:hermes' | string
  evidence: InsightEvidence[]
  review?: ReviewNote | null
  created_at: ISODate
  updated_at: ISODate
}

export interface UserModelQuestion {
  id: string
  question: string
  insight_id: string | null
  created_at: ISODate
}

export interface UserModel {
  summary: string
  updated_at: ISODate | null
  insights: Insight[]
  questions: UserModelQuestion[]
}

export interface UserModelRefreshResult {
  added: number
  updated: number
  disputed: number
  questions: number
  /** New insights waiting for review. */
  held?: number
}

export interface UserModelUpdatedData {
  summary_changed: boolean
  insights: number | string[] | unknown
  questions: number | string[] | unknown
}

// ============================================================================ §15 Dreams
export interface DreamStats {
  facts_reviewed: number
  merged: number
  contradictions_resolved: number
  promoted: number
  expired: number
  insights_updated: number
  /** Memories kept once because they were saved in two languages (#353); also counted in `merged`. */
  languages_merged?: number
}

export interface Dream {
  id: string
  started_at: ISODate
  finished_at: ISODate | null
  status: 'running' | 'completed' | 'error'
  trigger: 'schedule' | 'manual'
  stats: Partial<DreamStats>
  journal_md: string
  error: string | null
}

// ============================================================================ §16 Webhooks and change feeds
export interface Hook {
  id: string
  name: string
  url: string
  created_at: ISODate
  last_called_at: ISODate | null
  calls: number
}

/** `POST /api/hooks` answer: the hook plus its secret, shown once. */
export type HookCreated = Hook & { secret: string }

export type WebhookSchedule = TriggeredSchedule & { source: 'webhook' }

export interface SourceItemsData {
  source: string
  event: string
  origin: 'poll' | 'feed' | 'webhook'
  items: unknown[]
}

/** `GET /api/integrations/feeds` row: a change feed (Gmail, Calendar) or IMAP push watcher. */
export interface FeedStatus {
  source: string
  display_name: string
  kind: 'gmail_history' | 'calendar_sync_token' | 'imap_idle' | string
  connected: boolean
  active: boolean
  status: 'disconnected' | 'off' | 'starting' | 'ok' | 'error' | string
  last_sync_at: ISODate | null
  last_success_at: ISODate | null
  last_error: string | null
  note: string | null
  failures: number
  next_attempt_at: ISODate | null
  emitted: number
}

// §19 moving from Hermes ------------------------------------------------------------------
export type HermesPart = 'skills' | 'memory' | 'persona' | 'jobs' | 'mcp'

/** One thing found in the Hermes folder, with what will happen to it (`note`, plain words). */
export interface HermesItem {
  key: string
  action: 'import' | 'skip'
  note: string
}
export interface HermesSkillItem extends HermesItem {
  name: string
  folder: string
  description?: string
  /** The name it gets in Sentient (made unique). */
  target?: string
  changed_builtin?: boolean
  /** What the skill linter found in it. */
  lint?: SkillLintFinding[]
  /** The safety check (#146). Older engines leave it out. */
  vetting?: SkillVetting
}
export interface HermesMemoryItem extends HermesItem {
  kind: 'fact' | 'insight'
  text: string
}
export interface HermesPersona extends HermesItem {
  current: string
  proposed: string
}
export interface HermesJobItem extends HermesItem {
  name: string
  prompt: string
  schedule_text: string
  schedule: TaskSchedule | null
  kind: 'task' | 'script'
  script: { path: string; code: string } | null
  then?: ScriptThen
  condition?: ScriptCondition
  delivery: 'desktop' | 'whatsapp' | 'telegram' | 'discord' | string
  deliver_to?: TaskDeliverTo
  hermes_deliver?: string | null
  skills: string[]
}
export interface HermesMcpItem extends HermesItem {
  name: string
  transport: 'stdio' | 'http'
  url: string | null
  command: string | null
  args: string[]
  auth: McpAuth
  header_keys: string[]
  env_keys: string[]
}
export interface HermesPreview {
  path: string
  counts: Record<HermesPart, number>
  skills: HermesSkillItem[]
  memory: HermesMemoryItem[]
  persona: HermesPersona | null
  jobs: HermesJobItem[]
  mcp: HermesMcpItem[]
  suggestions: { wake_word: string | null; tts_provider: string | null; tts_voice: string | null }
  never_read: string[]
}
export interface HermesSkipped {
  key: string
  name: string
  note: string
}
export interface HermesResult {
  path: string
  skills?: { imported: string[]; skipped: HermesSkipped[] }
  memory?: { facts: number; insights: number; skipped: HermesSkipped[] }
  persona?: { updated: boolean }
  jobs?: { created: Array<{ task_id: string; name: string }>; skipped: HermesSkipped[] }
  mcp?: { added: string[]; skipped: HermesSkipped[] }
}

/** POST /api/memories/export: the file written in Sentient's files folder (§7). */
export interface MemoryExport {
  path: string
  folder: string
  name: string
  format: 'markdown' | 'sqlite'
  memories: number
  about_you: number
  waiting: number
}

// §19 memory from ChatGPT, Claude and OpenClaw ---------------------------------------------
export type MemoryImportApp = 'chatgpt' | 'claude' | 'openclaw'

export interface MemoryImportItem {
  key: string
  action: 'import' | 'skip'
  note: string
  kind: 'fact' | 'insight'
  text: string
  title: string
  dated: string | null
}

export interface MemoryImportSource {
  key: string
  title: string
  dated: string | null
  messages: number
  action: 'import' | 'skip'
  note: string
  snippet?: string
}

export interface MemoryImportPart {
  id: string
  label: string
  /** Memories plus conversations or notes to import. */
  count: number
  items: MemoryImportItem[]
  sources: MemoryImportSource[]
  sources_total: number
  batches: number
  messages: number
}

export interface MemoryImportPreview {
  app: MemoryImportApp
  label: string
  path: string
  export: string
  counts: Record<string, number>
  batches: number
  parts: MemoryImportPart[]
  stats: Record<string, number>
  never_read: string[]
}

export interface MemoryImportJob {
  id: string
  app: MemoryImportApp
  label: string
  export: string
  path: string
  parts: string[]
  status: 'running' | 'done' | 'cancelled' | 'error'
  total: number
  done: number
  facts: number
  insights: number
  merged: number
  skipped: number
  failed: number
  started_at: ISODate
  finished_at: ISODate | null
  message: string
}

// §20 backups, restore and snapshots ------------------------------------------------------
export type BackupKind = 'manual' | 'scheduled' | 'snapshot'
export type BackupSchedule = 'off' | 'daily' | 'weekly'

/** Something signed in through the keychain. Keychain secrets are never in a backup. */
export interface BackupSignIn {
  kind: 'model' | 'integration' | 'channel' | 'browser'
  id: string
  label: string
  /** After a restore: whether it works on this computer now (`null`: Sentient can't tell). */
  signed_in?: boolean | null
}

export interface BackupEntry {
  name: string
  path: string
  kind: BackupKind
  /** Why a snapshot was taken, e.g. "before importing from Hermes". */
  reason: string | null
  created_at: ISODate | null
  version: string | null
  schema_version: number | null
  size: number
  /** Snapshots leave out the files folder. */
  includes_files: boolean
  counts: { chats?: number; memories?: number; tasks?: number; skills?: number }
  sign_ins: BackupSignIn[]
}

export interface BackupInspect extends BackupEntry {
  /** Made by a newer Sentient than this one. */
  newer: boolean
  contents: string[]
}

export interface RestoreStaged {
  staged: true
  restart_required: true
  source: string
  created_at: ISODate | null
  contents: string[]
  sign_ins: BackupSignIn[]
}

export interface RestoreResult {
  restored_at: ISODate
  source: string | null
  kind: BackupKind | null
  backup_created_at: ISODate | null
  contents: string[]
  sign_ins: BackupSignIn[]
  /** The snapshot of the data the restore replaced (restore it to undo). */
  snapshot: string | null
  /** A plain sentence when the restore didn't happen; the data was left as it was. */
  error: string | null
}

export interface BackupOverview {
  folder: string
  default_folder: string
  /** Sentient's data folder. */
  home: string
  /** The backup folder is inside the data folder (so a backup would not survive losing it). */
  folder_in_home: boolean
  /** The sync service the backup folder is inside (its backups are copied to the cloud too). */
  folder_cloud: 'OneDrive' | 'Dropbox' | 'Google Drive' | 'iCloud' | null
  backups: BackupEntry[]
  snapshots: BackupEntry[]
  schedule: BackupSchedule
  keep: number
  snapshots_keep: number
  last_scheduled_at: ISODate | null
  pending_restore: { source: string | null; created_at: ISODate | null; staged_at: ISODate | null; kind: BackupKind | null } | null
  last_restore: RestoreResult | null
}

export interface BackupUpdated {
  what: 'backup' | 'snapshot' | 'restore_staged' | 'restore_cancelled' | 'restore_dismissed'
  name?: string
  source?: string
}

// ---------------------------------------------------------------------------- §21 activity timeline (#147)
export type TimelineKind =
  | 'chat'
  | 'tasks'
  | 'reminders'
  | 'briefs'
  | 'monitors'
  | 'approvals'
  | 'channels'
  | 'notifications'
  | 'memory'
  | 'skills'
  | 'backups'
  | 'cost'

export type TimelineStatus = 'done' | 'failed' | 'waiting' | 'declined' | 'running' | 'stopped' | 'scheduled' | 'paused' | 'restored'

/** Where an item came from, for the click-through (the window maps it to a screen). */
export interface TimelineRef {
  type: 'chat' | 'task' | 'memory' | 'dream' | 'skill' | 'about' | 'backups' | 'usage' | 'approvals' | 'notification'
  id?: string
  run_id?: string | null
  message_id?: string
  notification_id?: string
  review?: boolean
}

export interface TimelineItem {
  id: string
  kind: TimelineKind
  /** null only for paused routines (they have no time). */
  at: ISODate | null
  /** What happened, in plain words: "Ran your Daily Brief". */
  title: string
  /** Why: "It runs every weekday at 7:30 AM". */
  why: string | null
  detail: string | null
  status: TimelineStatus | null
  app: string | null
  app_label: string | null
  ref: TimelineRef | null
  upcoming: boolean
  /** A way back the window can offer: Restore for something moved to the trash (#142) or for files code changed
   *  (Sentient kept copies first), Cancel for a timer or reminder that has not gone off yet (#150). */
  undo?: { action: 'restore' | 'cancel'; label: string } | null
}

export interface TimelinePage {
  items: TimelineItem[]
  /** Cursor for the next (older) page; null at the end. */
  next: string | null
}

export interface TimelineUpcoming {
  items: TimelineItem[]
  paused: TimelineItem[]
}

export interface TimelineFilters {
  kinds?: TimelineKind[]
  app?: string
  q?: string
}

// §22 folders and Obsidian vaults as searchable notes (#176) ------------------------------------------------------
export type FolderStatus = 'waiting' | 'scanning' | 'ready' | 'missing'

export interface WatchedFolder {
  id: string
  path: string
  name: string
  /** `obsidian` when the folder is an Obsidian vault (it has a .obsidian folder). */
  kind: 'folder' | 'obsidian'
  /** This folder's own ignore patterns (Settings > Folders adds them to `folders.ignore`). */
  ignore: string[]
  /** The chats its notes are used in (#392): `always` (every chat, the default), `personal`, `work` or a project.
   * Missing on older engines. */
  scope?: string
  status: FolderStatus
  /** While a look through it runs: new or changed files read so far. */
  progress: { done: number; total: number } | null
  added_at: ISODate
  last_scan_at: ISODate | null
  last_scan_ms: number | null
  files: number
  /** Files that could not be read. */
  unreadable: number
  /** Searchable pieces, and how many of them can be searched by meaning. */
  chunks: number
  embedded: number
  /** Files left out at the last look: too big, or over the limit. */
  skipped: number
  /** Notes that are clipped web pages (treated as written by someone else). */
  clippings: number
  error: string | null
}

export interface FoldersOverview {
  folders: WatchedFolder[]
  searching_by_meaning: boolean
  embedding_error: string | null
}

export interface FoldersUpdated {
  id: string
  removed?: boolean
  progress?: { done: number; total: number }
}

// ---------------------------------------------------------------------------- §23 timers and reminders (#150)
export type ReminderKind = 'timer' | 'reminder' | 'alarm'
export type ReminderStatus = 'scheduled' | 'fired' | 'cancelled'

export interface Reminder {
  id: string
  kind: ReminderKind
  /** What to remind about ("call Ravi"); '' for a timer. */
  message: string
  connector: '' | 'to' | 'about' | 'that'
  /** A timer's or an alarm's name ("pasta", "gym"). */
  label: string
  /** A timer's length, or a reminder set "in 20 minutes". */
  duration_s: number | null
  fire_at: ISODate
  status: ReminderStatus
  session_id: string | null
  channel: string | null
  language: string
  source: 'quick' | 'chat' | 'api'
  created_at: ISODate
  fired_at: ISODate | null
  cancelled_at: ISODate | null
  notification_id: string | null
  /** An alarm: when it was stopped or snoozed while ringing (#390). */
  stopped_at?: ISODate | null
  /** An alarm made by a snooze: the id of the alarm that was ringing (#390). */
  snoozed_from?: string | null
}

export interface SnoozedAlarm {
  stopped: Reminder
  reminder: Reminder
}
