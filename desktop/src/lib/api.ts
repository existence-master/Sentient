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
 */
import type {
  BrowserStatus,
  Channel,
  ChannelPairing,
  DeviceInvokeResult,
  DeviceLanInfo,
  DeviceNode,
  DevicePairing,
  SandboxResult,
  SandboxStatus,
  Subagent,
  ApprovalDecision,
  Bootstrap,
  ChatRequest,
  AgentEvent,
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
  McpTestResult,
  Memory,
  MemoryGraph,
  MemoryImportResult,
  MemoryQuery,
  MemorySummary,
  MemoryTopic,
  MemoryWriteResult,
  MessageSearchHit,
  ModelRoles,
  ModelTestResult,
  NotificationList,
  OkResponse,
  OllamaPullProgress,
  OnboardingRequest,
  Persona,
  PrivacyFilters,
  ProactivityPreference,
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
  ToolPlugin,
  TranscribeResult,
  TranscriptMessage,
  UploadedFile,
  UsageReport,
  VoicePrepareProgress,
  VoicePrepareTarget,
  VoiceStatus,
  WorkspaceFileId,
  WorkspaceSnapshot
} from './types'
import type { Connection } from '@/types/bridge'

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

// ---------------------------------------------------------------------------- endpoints
export const api = {
  // §2 core ------------------------------------------------------------------
  health: () => http.get<Health>('/api/health'),
  bootstrap: () => http.get<Bootstrap>('/api/bootstrap'),
  onboarding: (body: OnboardingRequest) => http.post<OkResponse>('/api/onboarding', body),

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
      streamNdjson<OllamaPullProgress>('POST', '/api/models/ollama/pull', { body: { name }, signal })
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
    runEvents: (id: string, runId: string) =>
      http.get<ProgressUpdate[]>(`/api/tasks/${enc(id)}/runs/${enc(runId)}/events`)
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
      test: (name: string) => http.post<McpTestResult>(`/api/integrations/mcp/${enc(name)}/test`)
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
      http.delete<OkResponse>(`/api/proactivity/preferences/${enc(suggestionType)}`)
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
    personas: () => http.get<Persona[]>('/api/memories/personas')
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
    status: () => http.get<SandboxStatus>('/api/sandbox/status'),
    run: (code: string) => http.post<SandboxResult>('/api/sandbox/run', { code })
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
  }
} as const

export type Api = typeof api
