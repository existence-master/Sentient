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
}

export interface OkResponse {
  ok: boolean
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
  max_tool_rounds: number
  request_timeout_s: number
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
}

export type ApprovalMode = 'off' | 'ask' | 'always'

/** Lasting rule for a tool or a whole app (ADR 0016). No entry means the normal approvals setting. */
export type ApprovalRule = 'allow' | 'ask' | 'never'

export interface ApprovalsConfig {
  mode: ApprovalMode
  remember_session: boolean
  timeout_s: number
  /** Key: a tool name (`gmail_send_email`) or an app id (`gmail`). A tool's own rule beats its app's rule. */
  rules: Record<string, ApprovalRule>
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
  enabled?: boolean
}

export interface IntegrationsConfig {
  oauth_redirect_port: number
  search_provider: 'duckduckgo' | 'brave' | 'google_cse' | 'searxng'
  searxng_url: string
  weather_provider: 'open_meteo' | 'accuweather'
  mcp_servers: Record<string, McpServerConfig>
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
  [key: string]: unknown
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
  [key: string]: unknown
}

export interface ToolCallRef {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

export type MessageRole = 'user' | 'assistant' | 'tool' | 'system'

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
  created_at: ISODate
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
}

export type ApprovalDecision = 'allow' | 'allow_session' | 'deny'

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
  totals: TokenCounts
  by_model: Array<TokenCounts & { model: string; calls: number }>
  by_source: Array<TokenCounts & { source: string; calls: number }>
  by_day: Array<TokenCounts & { day: string; calls?: number }>
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

export interface ModelTestResult {
  ok: boolean
  latency_ms: number
  reply?: string
  error?: string
  supports_tools?: boolean
}

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
}
export interface UsageEvent extends TurnScoped {
  type: 'usage'
  model: string
  prompt_tokens: number
  completion_tokens: number
}
export interface ErrorEvent extends TurnScoped {
  type: 'error'
  message: string
  recoverable?: boolean
  /** §17: a queued message was dropped by Stop everything (echoes the `client_id` it was sent with). */
  dropped?: boolean
  client_id?: string | null
}
export interface DoneEvent extends TurnScoped {
  type: 'done'
  content: string
  message_id: string | null
  cancelled?: boolean
  memory_sources?: MemorySource[]
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
  | UsageEvent
  | ErrorEvent
  | DoneEvent
  | ApprovalAckEvent
  | PongEvent
  | SteerAckEvent
  | UserInterjectionEvent

export type ChatEventType = ChatEvent['type']

/** Turn-scoped events as produced by `POST /api/chat` and the voice socket. */
export type AgentEvent = Exclude<ChatEvent, HelloEvent | ApprovalAckEvent | PongEvent | SteerAckEvent>

export interface MemoryUpdatedData {
  action: 'ADD' | 'UPDATE' | 'DELETE'
  /** null for bulk changes (import, forget-by-source, expiry purge) */
  id: number | null
  content?: string
  source?: string
  reason?: string
  count?: number
}

export interface DomainEventMap {
  'task.updated': Task
  'task.deleted': { task_id: string }
  'task.run_progress': { task_id: string; run_id: string; update: ProgressUpdate }
  'notification.new': Notification
  'notification.updated': Notification
  'notification.read': { id: string | null }
  'notification.deleted': { id: string | null }
  'integration.updated': Integration
  'memory.updated': MemoryUpdatedData
  'skill.updated': { name: string; state: SkillState }
  'session.updated': { session_id: string; title: string }
  'config.updated': { sections: string[] }
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
  // §15-16
  'user_model.updated': UserModelUpdatedData
  'dream.updated': Dream
  'source.items': SourceItemsData
  // §17
  'stop.updated': StopState
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
    }
  | { type: 'chat.cancel'; session_id: string }
  | { type: 'chat.steer'; session_id: string; text: string }
  | { type: 'approval.respond'; approval_id: string; decision: ApprovalDecision }
  | { type: 'ping' }

// ============================================================================ §10 Subagents
export type SubagentStatus = 'running' | 'completed' | 'error' | 'cancelled'

export interface Subagent {
  subagent_id: string
  session_id: string | null
  parent_call_id: string | null
  goal: string
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
}

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
export type ChannelId = 'telegram' | 'discord'
export type ChannelStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export interface PairedChat {
  chat_id: string
  label: string
  paired_at: ISODate
  deliver: boolean
  session_id: string | null
}

export interface Channel {
  id: ChannelId | string
  display_name: string
  status: ChannelStatus
  account_label: string | null
  error: string | null
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

export interface OnceSchedule {
  type: 'once'
  run_at: string | null
  timezone?: string
}
export interface RecurringSchedule {
  type: 'recurring'
  /** `interval` runs every `interval_minutes` (minimum 5; the engine normalizes `hourly` to 60) and ignores `days`/`time`. */
  frequency: 'daily' | 'weekly' | 'interval'
  days?: Weekday[] | string[]
  time: string
  interval_minutes?: number
  timezone?: string
}

// §16 script jobs
export type ScriptCondition = 'changed' | 'alert'
export type ScriptThen = 'notify' | 'run'

export interface TaskScript {
  code: string
  condition: ScriptCondition
  then: ScriptThen
  last_result: unknown
  last_run_at: ISODate | null
  last_error: string | null
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

export type RunStatus = 'processing' | 'waiting_for_user' | 'completed' | 'completed_with_errors' | 'error' | 'cancelled'

/** The question a `waiting_for_user` run asked (§4). Answer with `POST /api/tasks/{id}/runs/{run_id}/answer`. */
export interface RunQuestion {
  question: string
  /** Suggested answers (0 to 6). A free-text answer is always allowed. */
  options: string[]
  asked_at: ISODate | null
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
  model: string | null
  original_context: { source: 'manual_creation' | 'chat' | 'proactive' | 'trigger' | string; [k: string]: unknown }
  /** Last planning/run failure message (v2 `task.error`). */
  error: string | null
  next_execution_at: ISODate | null
  last_execution_at: ISODate | null
  created_at: ISODate
  updated_at: ISODate
}

export interface TaskCreateRequest {
  prompt: string
  is_swarm?: boolean
  assignee?: 'ai'
  model?: string
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

export type TaskPatch = Partial<Pick<Task, 'name' | 'description' | 'priority' | 'schedule' | 'plan' | 'enabled' | 'status' | 'model'>>

export interface ClarificationAnswer {
  question_id: string
  answer_text: string
}

// ============================================================================ §5 Integrations
export type IntegrationCategory = 'productivity' | 'communication' | 'knowledge' | 'development' | 'utilities' | 'core'
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
  tools: ToolInfo[]
}

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
}

export type McpServerStatus = 'connecting' | 'connected' | 'error' | 'disconnected' | 'disabled'

export interface McpServer {
  name: string
  transport: 'stdio' | 'http'
  command: string | null
  args: string[]
  url: string | null
  /** Env values live in the keychain and are never returned. */
  env_keys: string[]
  enabled: boolean
  status: McpServerStatus | string
  tools: McpToolInfo[]
  error: string | null
}

export interface McpServerCreate {
  name: string
  transport: 'stdio' | 'http'
  command?: string
  args?: string[]
  url?: string
  env?: Record<string, string>
  enabled?: boolean
}

export interface McpTestResult {
  ok: boolean
  /** MCP tool names. */
  tools: string[]
  error?: string
}

// ============================================================================ §6 Notifications & proactivity
export type NotificationKind = 'info' | 'task' | 'approval' | 'proactive' | 'skill' | 'error'

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
}

export interface MemoryQuery {
  topic?: string
  q?: string
  source?: string
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
}

export interface MemoryImportResult {
  added: number
  updated: number
  skipped: number
  source: string
}

export interface MemorySummary {
  id: string | number
  content: string
  start_at: ISODate
  end_at: ISODate
  session_id: string | null
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
  /** Pending proposals only: why Sentient proposed it and where it came from. */
  reason?: string | null
  origin?: { session_id?: string; task_id?: string; run_id?: string; curator?: boolean; merged_from?: string } | null
  proposed_at?: string | null
}

export type SkillDetail = Skill & { body: string }

export interface SkillsList {
  active: Skill[]
  pending: Skill[]
  archived: Skill[]
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
}

export interface SkillReviewResult {
  reviewed: number | boolean
  proposed: string[]
}

export type EvolutionKind =
  | 'skill_created'
  | 'skill_patched'
  | 'skill_archived'
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
  source: 'inferred' | 'user'
  evidence: InsightEvidence[]
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
