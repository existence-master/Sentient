/**
 * Typed REST client for every endpoint in docs/API.md.
 *
 *   import { api } from '@/lib/api'
 *   const sessions = await api.sessions.list()
 *
 * - The connection (base URL + token) is set once by the connection store.
 * - Non-2xx responses throw `ApiError` with the backend's `detail`.
 * - NDJSON endpoints use `streamNdjson` (async iterator) and accept an AbortSignal.
 * - Uploads use XHR so callers get progress events.
 * - In dev demo mode (`?demo=1`, never in a packaged build) a few endpoints answer from lib/demo.ts when the
 *   engine does not have them yet; whenever the engine answers, its data is used.
 */
import type {
  BrowserStatus,
  Channel,
  ChannelPairing,
  DeviceInvokeResult,
  DeviceLanInfo,
  DeviceNode,
  DevicePairing,
  Dream,
  FeedStatus,
  HermesPart,
  HermesPreview,
  HermesResult,
  Hook,
  HookCreated,
  Insight,
  InsightDimension,
  InsightStatus,
  SandboxResult,
  SandboxStatus,
  Subagent,
  TerminalStatus,
  ApprovalDecision,
  Bootstrap,
  StopResult,
  StopState,
  ChatRequest,
  AgentEvent,
  CheckupEvent,
  ClarificationAnswer,
  ConfigPatchResponse,
  ConnectResponse,
  DeepPartial,
  EmbeddingTestResult,
  EvolutionLogEntry,
  FallbacksResponse,
  FileEntry,
  Health,
  Integration,
  IntegrationTestResult,
  JsonSchema,
  LocalModels,
  McpServer,
  McpServerCreate,
  McpSignInStart,
  McpTestResult,
  Memory,
  MemoryGraph,
  MemoryImportResult,
  MemoryQuery,
  MemorySummary,
  MemoryTopic,
  MemoryWriteResult,
  MessageSearchHit,
  ModelPreset,
  ModelPresetList,
  ModelRoles,
  ModelTestResult,
  CatalogModel,
  ProviderKeyCheck,
  ProviderSignIn,
  ProviderSignInStatus,
  NotificationList,
  OkResponse,
  PresetApplyResult,
  OllamaPullProgress,
  OnboardingRequest,
  Persona,
  PrivacyFilters,
  ProactivityPreference,
  Brief,
  BriefFeedback,
  BriefKind,
  BriefSectionId,
  BriefSetup,
  BriefState,
  ProactivityStatus,
  ProgressUpdate,
  Provider,
  RoleName,
  SecretStatus,
  SentientConfig,
  Session,
  Skill,
  SkillCreate,
  SkillDetail,
  SkillDiff,
  SkillReviewResult,
  SkillsList,
  SkillUpdate,
  SuggestionActionResponse,
  Task,
  TaskCreateRequest,
  TaskPatch,
  TaskPreview,
  TaskScript,
  ToolPlugin,
  TranscribeResult,
  TranscriptMessage,
  UploadedFile,
  UsageReport,
  UserModel,
  UserModelRefreshResult,
  VoicePrepareProgress,
  VoicePrepareTarget,
  VoiceStatus,
  WorkspaceFileId,
  WorkspaceSnapshot
} from './types'
import type { Connection } from '@/types/bridge'
import { DEMO_FEEDS, demo, isDemoMode } from './demo'

// ---------------------------------------------------------------------------- connection
let connection: Connection | null = null
const waiters: Array<(c: Connection) => void> = []

export function setConnection(conn: Connection): void {
  connection = { baseUrl: conn.baseUrl.replace(/\/+$/, ''), token: conn.token }
  waiters.splice(0).forEach((w) => w(connection as Connection))
}

export function getConnection(): Connection | null {
  return connection
}

/** Resolves once the connection store has a connection. */
export function whenConnected(): Promise<Connection> {
  return connection ? Promise.resolve(connection) : new Promise((r) => waiters.push(r))
}

// ---------------------------------------------------------------------------- errors
export class ApiError extends Error {
  readonly status: number
  readonly detail: string
  readonly body: unknown
  readonly path: string

  constructor(status: number, detail: string, body: unknown, path: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.body = body
    this.path = path
  }

  get isNotFound() {
    return this.status === 404 || this.status === 405
  }
  get isNetwork() {
    return this.status === 0
  }
}

export function isApiError(e: unknown): e is ApiError {
  return e instanceof ApiError
}

/** True when an endpoint isn't implemented yet (404/405/501) - render an empty/"coming soon" state. */
export function isNotImplemented(e: unknown): boolean {
  return isApiError(e) && [404, 405, 501].includes(e.status)
}

/** React Query `retry`: give up at once when the endpoint is missing, otherwise retry twice. */
export function noRetryWhenMissing(count: number, err: unknown): boolean {
  return !isNotImplemented(err) && count < 2
}

export function errorMessage(e: unknown): string {
  if (isApiError(e)) return e.detail
  if (e instanceof Error) return e.message
  return String(e)
}

function detailFrom(body: unknown, fallback: string): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const d = (body as { detail: unknown }).detail
    if (typeof d === 'string') return d
    if (Array.isArray(d)) {
      return d
        .map((x: { loc?: unknown[]; msg?: string }) =>
          [Array.isArray(x.loc) ? x.loc.filter((p) => p !== 'body').join('.') : '', x.msg].filter(Boolean).join(': ')
        )
        .join('; ')
    }
  }
  return fallback
}

// ---------------------------------------------------------------------------- core request helpers
type Query = Record<string, string | number | boolean | null | undefined>

export interface RequestOptions {
  query?: Query
  body?: unknown
  signal?: AbortSignal
  headers?: Record<string, string>
}

export function buildUrl(path: string, query?: Query): string {
  const conn = connection
  if (!conn) throw new ApiError(0, 'Not connected to the Sentient engine yet.', null, path)
  const url = new URL(conn.baseUrl + path)
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v))
    }
  }
  return url.toString()
}

async function send(method: string, path: string, opts: RequestOptions = {}): Promise<Response> {
  const conn = connection ?? (await whenConnected())
  const isForm = typeof FormData !== 'undefined' && opts.body instanceof FormData
  let res: Response
  try {
    res = await fetch(buildUrl(path, opts.query), {
      method,
      signal: opts.signal,
      headers: {
        Authorization: `Bearer ${conn.token}`,
        ...(opts.body !== undefined && !isForm ? { 'Content-Type': 'application/json' } : {}),
        ...opts.headers
      },
      body: opts.body === undefined ? undefined : isForm ? (opts.body as FormData) : JSON.stringify(opts.body)
    })
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err
    throw new ApiError(0, "Can't reach the Sentient engine.", null, path)
  }
  if (!res.ok) {
    let body: unknown = null
    const text = await res.text().catch(() => '')
    try {
      body = text ? JSON.parse(text) : null
    } catch {
      body = text
    }
    throw new ApiError(res.status, detailFrom(body, text || `${res.status} ${res.statusText}`), body, path)
  }
  return res
}

export async function request<T>(method: string, path: string, opts?: RequestOptions): Promise<T> {
  const res = await send(method, path, opts)
  if (res.status === 204) return undefined as T
  const type = res.headers.get('content-type') ?? ''
  if (type.includes('application/json')) return (await res.json()) as T
  return (await res.text()) as unknown as T
}

export const http = {
  get: <T>(path: string, opts?: RequestOptions) => request<T>('GET', path, opts),
  post: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>('POST', path, { ...opts, body }),
  put: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>('PUT', path, { ...opts, body }),
  patch: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>('PATCH', path, { ...opts, body }),
  delete: <T>(path: string, opts?: RequestOptions) => request<T>('DELETE', path, opts),
  blob: async (method: string, path: string, opts?: RequestOptions) => (await send(method, path, opts)).blob()
}

/**
 * Stream an NDJSON endpoint line by line.
 *
 *   for await (const line of streamNdjson<OllamaPullProgress>('POST', '/api/models/ollama/pull', { body: { name } })) ...
 */
export async function* streamNdjson<T>(method: string, path: string, opts: RequestOptions = {}): AsyncGenerator<T> {
  const res = await send(method, path, opts)
  if (!res.body) return
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += value
      let nl: number
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, nl).trim()
        buffer = buffer.slice(nl + 1)
        if (line) {
          try {
            yield JSON.parse(line) as T
          } catch {
            /* skip malformed line */
          }
        }
      }
    }
    const tail = buffer.trim()
    if (tail) {
      try {
        yield JSON.parse(tail) as T
      } catch {
        /* ignore */
      }
    }
  } finally {
    reader.releaseLock()
  }
}

export interface UploadOptions {
  field?: string
  signal?: AbortSignal
  onProgress?: (fraction: number, loaded: number, total: number) => void
  extra?: Record<string, string>
}

/** Multipart upload with progress (XHR). */
export function upload<T>(path: string, file: Blob, filename: string, opts: UploadOptions = {}): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const conn = connection
    if (!conn) {
      reject(new ApiError(0, 'Not connected to the Sentient engine yet.', null, path))
      return
    }
    const xhr = new XMLHttpRequest()
    xhr.open('POST', buildUrl(path))
    xhr.setRequestHeader('Authorization', `Bearer ${conn.token}`)
    xhr.responseType = 'text'
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) opts.onProgress?.(e.loaded / e.total, e.loaded, e.total)
    }
    xhr.onload = () => {
      let body: unknown = null
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null
      } catch {
        body = xhr.responseText
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        opts.onProgress?.(1, file.size, file.size)
        resolve(body as T)
      } else {
        reject(new ApiError(xhr.status, detailFrom(body, `Upload failed (${xhr.status})`), body, path))
      }
    }
    xhr.onerror = () => reject(new ApiError(0, 'Upload failed: engine unreachable.', null, path))
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'))
    opts.signal?.addEventListener('abort', () => xhr.abort())
    const form = new FormData()
    form.append(opts.field ?? 'file', file, filename)
    for (const [k, v] of Object.entries(opts.extra ?? {})) form.append(k, v)
    xhr.send(form)
  })
}

/** URL usable in <img src> / <audio src>; the token travels as a query parameter. */
export function authedUrl(path: string, query?: Query): string {
  const conn = connection
  if (!conn) return ''
  return buildUrl(path, { ...query, token: conn.token })
}

const enc = encodeURIComponent
const encPath = (name: string) => name.split('/').map(enc).join('/')

/** Webhook URLs may come back relative (`/hooks/abc`); make them absolute against the engine. */
export function absoluteHookUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) return url
  const base = getConnection()?.baseUrl ?? 'http://127.0.0.1:7777'
  return `${base}${url.startsWith('/') ? '' : '/'}${url}`
}

/**
 * Real engine first, always. Only in dev demo mode, and only when the endpoint is missing (404/405/501) or the
 * engine is unreachable, answer from the demo fixtures in lib/demo.ts. Real engine data is never replaced.
 */
async function withDemoFallback<T>(real: () => Promise<T>, fake: () => T | Promise<T>): Promise<T> {
  if (!isDemoMode()) return real()
  try {
    return await real()
  } catch (err) {
    if (isNotImplemented(err) || (isApiError(err) && err.status === 0)) return fake()
    throw err
  }
}

// ---------------------------------------------------------------------------- endpoints
export const api = {
  // §2 core ------------------------------------------------------------------
  health: () => http.get<Health>('/api/health'),
  bootstrap: () => http.get<Bootstrap>('/api/bootstrap'),
  onboarding: (body: OnboardingRequest) => http.post<OkResponse>('/api/onboarding', body),

  // §17 stop everything
  stop: {
    get: () => http.get<StopState>('/api/stop'),
    all: (source = 'desktop') => http.post<StopResult>('/api/stop-all', { source }),
    resume: (source = 'desktop') => http.post<StopState>('/api/resume', { source })
  },

  config: {
    get: () => http.get<SentientConfig>('/api/config'),
    schema: () => http.get<JsonSchema>('/api/config/schema'),
    put: (config: SentientConfig) => http.put<{ saved: boolean }>('/api/config', config),
    patch: (patch: DeepPartial<SentientConfig> | Record<string, unknown>) =>
      http.patch<ConfigPatchResponse>('/api/config', patch)
  },

  sessions: {
    list: (limit = 100) => http.get<Session[]>('/api/sessions', { query: { limit } }),
    create: () => http.post<{ session_id: string }>('/api/sessions'),
    rename: (id: string, title: string) => http.patch<OkResponse>(`/api/sessions/${enc(id)}`, { title }),
    delete: (id: string) => http.delete<OkResponse>(`/api/sessions/${enc(id)}`),
    messages: (id: string, limit = 500) =>
      http.get<TranscriptMessage[]>(`/api/sessions/${enc(id)}/messages`, { query: { limit } }),
    search: (q: string) => http.get<MessageSearchHit[]>('/api/sessions/search', { query: { q } })
  },

  chat: {
    /** NDJSON fallback of a WebSocket turn. First line is `{type:"session", session_id}`. */
    stream: (body: ChatRequest, signal?: AbortSignal) =>
      streamNdjson<AgentEvent>('POST', '/api/chat', { body, signal })
  },

  approvals: {
    respond: (approval_id: string, decision: ApprovalDecision) =>
      http.post<{ resolved: boolean }>('/api/approvals', { approval_id, decision })
  },

  files: {
    upload: (file: File | Blob, filename: string, opts?: UploadOptions) =>
      upload<UploadedFile>('/api/files', file, filename, opts),
    list: () => http.get<FileEntry[]>('/api/files'),
    contentUrl: (name: string) => authedUrl(`/api/files/content/${encPath(name)}`),
    download: (name: string) => http.blob('GET', `/api/files/content/${encPath(name)}`),
    delete: (name: string) => http.delete<OkResponse>(`/api/files/${encPath(name)}`)
  },

  tools: {
    list: () => http.get<ToolPlugin[]>('/api/tools')
  },

  usage: {
    get: (days = 30) => http.get<UsageReport>('/api/usage', { query: { days } })
  },

  // §3 models & secrets ------------------------------------------------------
  models: {
    providers: () => http.get<Provider[]>('/api/models/providers'),
    local: () => http.get<LocalModels>('/api/models/local'),
    test: (model: string, role?: RoleName) => http.post<ModelTestResult>('/api/models/test', { model, role }),
    testEmbedding: (model: string) => http.post<EmbeddingTestResult>('/api/models/test-embedding', { model }),
    setRoles: (roles: Partial<Record<RoleName, string | null>>) => http.put<ModelRoles>('/api/models/roles', roles),
    setFallbacks: (fallbacks: Partial<Record<RoleName, string[]>>) =>
      http.put<FallbacksResponse>('/api/models/fallbacks', fallbacks),
    pullOllama: (name: string, signal?: AbortSignal) =>
      streamNdjson<OllamaPullProgress>('POST', '/api/models/ollama/pull', { body: { name }, signal }),
    /** Check each role's model (or only `roles`); never changes config. */
    checkup: (roles?: Partial<Record<RoleName, string | null>>, signal?: AbortSignal) =>
      streamNdjson<CheckupEvent>('POST', '/api/models/checkup', { body: roles ? { roles } : {}, signal }),
    /** Start OpenRouter's browser sign-in; open `auth_url`, then poll `signInStatus(state)`. */
    connectOpenRouter: () => http.post<ProviderSignIn>('/api/models/connect/openrouter'),
    signInStatus: (state: string) => http.get<ProviderSignInStatus>(`/api/models/connect/openrouter/${enc(state)}`),
    checkKey: (provider: string) => http.post<ProviderKeyCheck>(`/api/models/connect/${enc(provider)}/check`),
    catalog: (provider: string) => http.get<CatalogModel[]>(`/api/models/catalog/${enc(provider)}`),
    /** Model presets: switch every role at once, save your own, undo the last switch. */
    presets: {
      list: () => http.get<ModelPresetList>('/api/models/presets'),
      apply: (name: string) => http.post<PresetApplyResult>(`/api/models/presets/${enc(name)}/apply`),
      undo: () => http.post<PresetApplyResult>('/api/models/presets/undo'),
      save: (name: string, overwrite = false) => http.post<ModelPreset>('/api/models/presets', { name, overwrite }),
      rename: (name: string, to: string) => http.patch<ModelPreset>(`/api/models/presets/${enc(name)}`, { name: to }),
      delete: (name: string) => http.delete<OkResponse>(`/api/models/presets/${enc(name)}`)
    }
  },

  secrets: {
    list: () => http.get<SecretStatus[]>('/api/secrets'),
    set: (name: string, value: string) => http.put<OkResponse>(`/api/secrets/${enc(name)}`, { value }),
    delete: (name: string) => http.delete<OkResponse>(`/api/secrets/${enc(name)}`)
  },

  // §4 tasks -------------------------------------------------------------------
  tasks: {
    list: () => http.get<Task[]>('/api/tasks'),
    get: (id: string) => http.get<Task>(`/api/tasks/${enc(id)}`),
    create: (body: TaskCreateRequest) => http.post<Task>('/api/tasks', body),
    preview: (prompt: string) => http.post<TaskPreview>('/api/tasks/preview', { prompt }),
    update: (id: string, patch: TaskPatch) => http.patch<Task>(`/api/tasks/${enc(id)}`, patch),
    delete: (id: string) => http.delete<OkResponse>(`/api/tasks/${enc(id)}`),
    approve: (id: string) => http.post<Task>(`/api/tasks/${enc(id)}/approve`),
    decline: (id: string) => http.post<Task>(`/api/tasks/${enc(id)}/decline`),
    rerun: (id: string) => http.post<Task>(`/api/tasks/${enc(id)}/rerun`),
    runNow: (id: string) => http.post<Task>(`/api/tasks/${enc(id)}/run-now`),
    archive: (id: string) => http.post<Task>(`/api/tasks/${enc(id)}/archive`),
    chat: (id: string, message: string) => http.post<Task>(`/api/tasks/${enc(id)}/chat`, { message }),
    answerClarifications: (id: string, answers: ClarificationAnswer[]) =>
      http.post<Task>(`/api/tasks/${enc(id)}/clarifications`, { answers }),
    cancelRun: (id: string, runId: string) => http.post<Task>(`/api/tasks/${enc(id)}/runs/${enc(runId)}/cancel`),
    /** Answer the question a `waiting_for_user` run asked; the run carries on. 409 when it is not waiting, 400 when empty. */
    answerQuestion: (id: string, runId: string, answer: string) =>
      http.post<Task>(`/api/tasks/${enc(id)}/runs/${enc(runId)}/answer`, { answer }),
    runEvents: (id: string, runId: string) =>
      http.get<ProgressUpdate[]>(`/api/tasks/${enc(id)}/runs/${enc(runId)}/events`),
    /** Failed or cancelled run of a non-swarm task: a new run with `retry_of`. 409 otherwise. */
    retryRun: (id: string, runId: string) => http.post<Task>(`/api/tasks/${enc(id)}/runs/${enc(runId)}/retry`),
    /** §16 script jobs: runs the stored script, or `code` (unsaved edits), once. 400 when the code does not compile. */
    testScript: (id: string, code?: string) =>
      withDemoFallback(() => http.post<SandboxResult>(`/api/tasks/${enc(id)}/script/test`, code !== undefined ? { code } : undefined), () => demo.scriptTest(id)),
    /** `PATCH /api/tasks/{id}` with a partial `script`; 400 when the code does not compile. */
    updateScript: (id: string, script: Partial<Pick<TaskScript, 'condition' | 'then' | 'code'>>) =>
      withDemoFallback(() => http.patch<Task>(`/api/tasks/${enc(id)}`, { script }), () => demo.patchScript(id, script))
  },

  // §5 integrations ------------------------------------------------------------
  integrations: {
    list: () => http.get<Integration[]>('/api/integrations'),
    get: (id: string) => http.get<Integration>(`/api/integrations/${enc(id)}`),
    connect: (id: string, fields: Record<string, string> = {}) =>
      http.post<ConnectResponse>(`/api/integrations/${enc(id)}/connect`, { fields }),
    disconnect: (id: string) => http.post<Integration>(`/api/integrations/${enc(id)}/disconnect`),
    /** Abandon a pending browser sign-in so the integration stops showing "connecting". */
    cancel: (id: string) => http.post<Integration>(`/api/integrations/${enc(id)}/cancel`),
    test: (id: string) => http.post<IntegrationTestResult>(`/api/integrations/${enc(id)}/test`),
    getPrivacyFilters: (id: string) => http.get<PrivacyFilters>(`/api/integrations/${enc(id)}/privacy-filters`),
    setPrivacyFilters: (id: string, filters: PrivacyFilters) =>
      http.put<OkResponse>(`/api/integrations/${enc(id)}/privacy-filters`, filters),
    mcp: {
      list: () => http.get<McpServer[]>('/api/integrations/mcp'),
      add: (body: McpServerCreate) => http.post<McpServer>('/api/integrations/mcp', body),
      remove: (name: string) => http.delete<OkResponse>(`/api/integrations/mcp/${enc(name)}`),
      test: (name: string) => http.post<McpTestResult>(`/api/integrations/mcp/${enc(name)}/test`),
      signIn: (name: string) => http.post<McpSignInStart>(`/api/integrations/mcp/${enc(name)}/sign-in`),
      signOut: (name: string) => http.post<McpServer>(`/api/integrations/mcp/${enc(name)}/sign-out`),
      setEnabled: (name: string, enabled: boolean) => http.post<McpServer>(`/api/integrations/mcp/${enc(name)}/enabled`, { enabled })
    },
    /** §16 change feeds (Gmail, Calendar) and IMAP push watchers. */
    feeds: {
      list: () => withDemoFallback(() => http.get<FeedStatus[]>('/api/integrations/feeds'), () => DEMO_FEEDS),
      sync: (source: string) => http.post<{ ok: boolean; emitted?: number; error?: string }>(`/api/integrations/feeds/${enc(source)}/sync`)
    }
  },

  // §6 notifications & proactivity ---------------------------------------------
  notifications: {
    list: (opts: { limit?: number; unread_only?: boolean } = {}) =>
      http.get<NotificationList>('/api/notifications', { query: opts }),
    markRead: (id: string) => http.post<OkResponse>(`/api/notifications/${enc(id)}/read`),
    markAllRead: () => http.post<OkResponse>('/api/notifications/read-all'),
    delete: (id: string) => http.delete<OkResponse>(`/api/notifications/${enc(id)}`),
    clear: () => http.delete<OkResponse>('/api/notifications')
  },

  proactivity: {
    respond: (notificationId: string, action: 'approve' | 'dismiss') =>
      http.post<SuggestionActionResponse>(`/api/proactivity/suggestions/${enc(notificationId)}`, { action }),
    status: () => http.get<ProactivityStatus>('/api/proactivity/status'),
    pollNow: () => http.post<{ ok: boolean; events: number }>('/api/proactivity/poll-now'),
    preferences: () => http.get<ProactivityPreference[]>('/api/proactivity/preferences'),
    resetPreference: (suggestionType: string) =>
      http.delete<OkResponse>(`/api/proactivity/preferences/${enc(suggestionType)}`),
    /** Daily Brief: a recurring task the user can edit, pause or delete in Tasks. */
    brief: {
      get: () => http.get<BriefState>('/api/proactivity/brief'),
      /** Set it up (creates the task once) or change time, days, sections and topics. */
      setup: (body: BriefSetup = {}) => http.post<BriefState>('/api/proactivity/brief', body),
      runNow: (kind: BriefKind = 'morning') => http.post<{ ok: boolean; task_id: string }>('/api/proactivity/brief/run', { kind }),
      feedback: (body: { brief_id: string; value: BriefFeedback; item_id?: string; section?: BriefSectionId }) =>
        http.post<Brief>('/api/proactivity/brief/feedback', body)
    }
  },

  // §7 memory ------------------------------------------------------------------
  memories: {
    list: (query: MemoryQuery = {}) => http.get<Memory[]>('/api/memories', { query: { ...query } }),
    topics: () => http.get<MemoryTopic[]>('/api/memories/topics'),
    graph: () => http.get<MemoryGraph>('/api/memories/graph'),
    create: (content: string, source?: string) => http.post<MemoryWriteResult>('/api/memories', { content, source }),
    update: (id: number, content: string) => http.put<Memory>(`/api/memories/${id}`, { content }),
    delete: (id: number) => http.delete<{ deleted: boolean }>(`/api/memories/${id}`),
    deleteBySource: (source: string) => http.delete<{ deleted: number }>(`/api/memories/source/${enc(source)}`),
    import: (file: File, opts?: UploadOptions) =>
      upload<MemoryImportResult>('/api/memories/import', file, file.name, opts),
    summaries: (limit?: number) => http.get<MemorySummary[]>('/api/memories/summaries', { query: { limit } }),
    workspace: () => http.get<WorkspaceSnapshot>('/api/memories/workspace'),
    writeWorkspace: (which: WorkspaceFileId, content: string) =>
      http.put<{ saved: boolean }>(`/api/memories/workspace/${which}`, { content }),
    personas: () => http.get<Persona[]>('/api/memories/personas'),
    /** §15 dreams: overnight memory consolidation. */
    dreams: {
      list: (limit = 30) => withDemoFallback(() => http.get<Dream[]>('/api/memories/dreams', { query: { limit } }), () => demo.dreams()),
      run: () => withDemoFallback(() => http.post<Dream>('/api/memories/dreams/run'), () => demo.runDream())
    }
  },

  // §8 skills & self-evolution ---------------------------------------------------
  skills: {
    list: () => http.get<SkillsList>('/api/skills'),
    get: (name: string) => http.get<SkillDetail>(`/api/skills/${enc(name)}`),
    create: (body: SkillCreate) => http.post<Skill>('/api/skills', body),
    update: (name: string, body: SkillUpdate) => http.put<Skill>(`/api/skills/${enc(name)}`, body),
    approve: (name: string) => http.post<Skill>(`/api/skills/${enc(name)}/approve`),
    reject: (name: string) => http.post<OkResponse>(`/api/skills/${enc(name)}/reject`),
    archive: (name: string) => http.post<Skill>(`/api/skills/${enc(name)}/archive`),
    restore: (name: string) => http.post<Skill>(`/api/skills/${enc(name)}/restore`),
    delete: (name: string) => http.delete<OkResponse>(`/api/skills/${enc(name)}`),
    diff: (name: string) => http.get<SkillDiff>(`/api/skills/${enc(name)}/diff`),
    reviewNow: (session_id?: string) => http.post<SkillReviewResult>('/api/skills/review-now', { session_id }),
    evolutionLog: (limit?: number) =>
      http.get<EvolutionLogEntry[]>('/api/skills/evolution-log', { query: { limit } })
  },

  // §9 voice ---------------------------------------------------------------------
  voice: {
    status: () => http.get<VoiceStatus>('/api/voice/status'),
    transcribe: (audio: Blob, filename = 'dictation.webm', opts?: UploadOptions) =>
      upload<TranscribeResult>('/api/voice/transcribe', audio, filename, opts),
    speak: (text: string, voice?: string) => http.blob('POST', '/api/voice/speak', { body: { text, voice } }),
    prepare: (target: VoicePrepareTarget = 'all', signal?: AbortSignal) =>
      streamNdjson<VoicePrepareProgress>('POST', '/api/voice/prepare', { body: { target }, signal }),
    /** `WS /ws/voice?token=` URL. */
    socketUrl: () => {
      const conn = connection
      return conn ? `${conn.baseUrl.replace(/^http/, 'ws')}/ws/voice?token=${enc(conn.token)}` : ''
    }
  },

  // §10 subagents ------------------------------------------------------------------
  subagents: {
    forSession: (sessionId: string) => http.get<Subagent[]>(`/api/sessions/${enc(sessionId)}/subagents`),
    get: (id: string) => http.get<Subagent>(`/api/subagents/${enc(id)}`),
    cancel: (id: string) => http.post<Subagent>(`/api/subagents/${enc(id)}/cancel`)
  },

  // §11 code execution --------------------------------------------------------------
  sandbox: {
    status: () =>
      withDemoFallback(
        () => http.get<SandboxStatus>('/api/sandbox/status'),
        () => ({ enabled: true, backend: 'process', docker_available: false, python_version: '3.12.7' })
      ),
    run: (code: string) => http.post<SandboxResult>('/api/sandbox/run', { code })
  },

  // §18 terminal ------------------------------------------------------------------------
  terminal: {
    status: () => http.get<TerminalStatus>('/api/terminal/status'),
    /** Kills one running command; `id` is the tool call id. */
    stop: (id: string) => http.post<{ stopped: boolean }>('/api/terminal/stop', { id })
  },

  // §12 browser -------------------------------------------------------------------------
  browser: {
    status: () => http.get<BrowserStatus>('/api/browser/status'),
    /** Shows a visible window on Sentient's browser profile so the user can sign in themselves. */
    open: (url?: string) => http.post<BrowserStatus>('/api/browser/open', url ? { url } : {}),
    close: () => http.post<BrowserStatus>('/api/browser/close'),
    screenshotUrl: (bust?: number) => authedUrl('/api/browser/screenshot', { t: bust })
  },

  // §13 devices ---------------------------------------------------------------------------
  nodes: {
    list: () => http.get<DeviceNode[]>('/api/nodes'),
    get: (id: string) => http.get<DeviceNode>(`/api/nodes/${enc(id)}`),
    rename: (id: string, name: string) => http.patch<DeviceNode>(`/api/nodes/${enc(id)}`, { name }),
    remove: (id: string) => http.delete<OkResponse>(`/api/nodes/${enc(id)}`),
    invoke: (id: string, capability: string, params: Record<string, unknown> = {}, timeout_ms?: number) =>
      http.post<DeviceInvokeResult>(`/api/nodes/${enc(id)}/invoke`, { capability, params, timeout_ms }),
    pairing: () => http.post<DevicePairing>('/api/nodes/pairing'),
    lan: () => http.get<DeviceLanInfo>('/api/nodes/lan')
  },

  // §14 messaging channels ------------------------------------------------------------------
  channels: {
    list: () => http.get<Channel[]>('/api/channels'),
    connect: (id: string, fields: Record<string, string>) => http.post<Channel>(`/api/channels/${enc(id)}/connect`, { fields }),
    disconnect: (id: string) => http.post<Channel>(`/api/channels/${enc(id)}/disconnect`),
    pairing: (id: string) => http.post<ChannelPairing>(`/api/channels/${enc(id)}/pairing`),
    setDeliver: (id: string, chatId: string, deliver: boolean) =>
      http.patch<Channel>(`/api/channels/${enc(id)}/paired/${enc(chatId)}`, { deliver }),
    removePaired: (id: string, chatId: string) => http.delete<Channel>(`/api/channels/${enc(id)}/paired/${enc(chatId)}`),
    test: (id: string, chatId?: string) => http.post<{ ok: boolean; error?: string }>(`/api/channels/${enc(id)}/test`, chatId ? { chat_id: chatId } : {})
  },

  // §15 user model ------------------------------------------------------------------------
  userModel: {
    get: () => withDemoFallback(() => http.get<UserModel>('/api/user-model'), () => demo.userModel()),
    addInsight: (statement: string, dimension: InsightDimension | string) =>
      withDemoFallback(() => http.post<Insight>('/api/user-model/insights', { statement, dimension }), () => demo.addInsight(statement, dimension)),
    patchInsight: (id: string, patch: { statement?: string; status?: InsightStatus }) =>
      withDemoFallback(() => http.patch<Insight>(`/api/user-model/insights/${enc(id)}`, patch), () => demo.patchInsight(id, patch)),
    deleteInsight: (id: string) =>
      withDemoFallback(() => http.delete<{ ok: boolean }>(`/api/user-model/insights/${enc(id)}`), () => demo.deleteInsight(id)),
    refresh: () =>
      withDemoFallback(() => http.post<UserModelRefreshResult>('/api/user-model/refresh'), () => ({ added: 1, updated: 2, disputed: 0, questions: 1 })),
    answerQuestion: (id: string, answer: string) =>
      withDemoFallback(() => http.post<{ ok: boolean }>(`/api/user-model/questions/${enc(id)}`, { answer }), () => demo.dropQuestion(id)),
    dismissQuestion: (id: string) =>
      withDemoFallback(() => http.delete<{ ok: boolean }>(`/api/user-model/questions/${enc(id)}`), () => demo.dropQuestion(id))
  },

  // §19 moving from Hermes ------------------------------------------------------------------
  imports: {
    hermes: {
      info: () => http.get<{ path: string; exists: boolean }>('/api/import/hermes'),
      preview: (path?: string) => http.post<HermesPreview>('/api/import/hermes/preview', { path: path || null }),
      apply: (body: { path?: string; parts: HermesPart[]; skip?: string[] }) => http.post<HermesResult>('/api/import/hermes/apply', body),
      removeMemories: () => http.delete<{ facts: number; insights: number }>('/api/import/hermes/memories')
    }
  },

  // §16 webhooks --------------------------------------------------------------------------
  hooks: {
    list: () => withDemoFallback(() => http.get<Hook[]>('/api/hooks'), () => demo.hooks()),
    create: (name: string) => withDemoFallback(() => http.post<HookCreated>('/api/hooks', { name }), () => demo.createHook(name)),
    delete: (id: string) => withDemoFallback(() => http.delete<{ ok: boolean }>(`/api/hooks/${enc(id)}`), () => demo.deleteHook(id))
  }
} as const

export type Api = typeof api
