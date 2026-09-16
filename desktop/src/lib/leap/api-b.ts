/**
 * REST calls for desktop agent B screens (docs/API.md sections 11, 15 and 16).
 *
 *   import { apiB, qkB } from '@/lib/leap/api-b'
 *   const model = await apiB.userModel.get()
 *
 * Uses the shared request helper from lib/api.ts. In dev demo mode (`?demo=1`, never in a packaged build)
 * calls whose endpoint is not built yet answer from lib/leap/demo-b.ts so screens can be designed before the
 * engine lands; whenever the engine answers, its data is used.
 */
import { getConnection, http, isApiError, isNotImplemented } from '@/lib/api'
import type { Task } from '@/lib/types'
import { DEMO_FEEDS, demo, isLeapDemo } from './demo-b'
import type {
  Dream,
  FeedStatus,
  Hook,
  HookCreated,
  Insight,
  InsightDimension,
  InsightStatus,
  SandboxResult,
  SandboxStatus,
  ScriptJob,
  UserModel,
  UserModelRefreshResult
} from './types-b'

const enc = encodeURIComponent

export const qkB = {
  userModel: ['user-model'] as const,
  dreams: ['memories', 'dreams'] as const,
  hooks: ['hooks'] as const,
  feeds: ['integrations', 'feeds'] as const,
  sandboxStatus: ['sandbox', 'status'] as const
}

/** Webhook URLs may come back relative (`/hooks/abc`); make them absolute against the engine. */
export function absoluteHookUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) return url
  const base = getConnection()?.baseUrl ?? 'http://127.0.0.1:7777'
  return `${base}${url.startsWith('/') ? '' : '/'}${url}`
}

/**
 * Real engine first, always. Only in dev demo mode, and only when the endpoint is missing (404/405/501) or the
 * engine is unreachable, answer from the demo fixtures. Real engine data is never replaced.
 */
async function pick<T>(real: () => Promise<T>, fake: () => T | Promise<T>): Promise<T> {
  if (!isLeapDemo()) return real()
  try {
    return await real()
  } catch (err) {
    if (isNotImplemented(err) || (isApiError(err) && err.status === 0)) return fake()
    throw err
  }
}

export const apiB = {
  // §15 user model ------------------------------------------------------------
  userModel: {
    get: () => pick(() => http.get<UserModel>('/api/user-model'), () => demo.userModel()),
    addInsight: (statement: string, dimension: InsightDimension | string) =>
      pick(() => http.post<Insight>('/api/user-model/insights', { statement, dimension }), () => demo.addInsight(statement, dimension)),
    patchInsight: (id: string, patch: { statement?: string; status?: InsightStatus }) =>
      pick(() => http.patch<Insight>(`/api/user-model/insights/${enc(id)}`, patch), () => demo.patchInsight(id, patch)),
    deleteInsight: (id: string) => pick(() => http.delete<{ ok: boolean }>(`/api/user-model/insights/${enc(id)}`), () => demo.deleteInsight(id)),
    refresh: () => pick(() => http.post<UserModelRefreshResult>('/api/user-model/refresh'), () => ({ added: 1, updated: 2, disputed: 0, questions: 1 })),
    answerQuestion: (id: string, answer: string) =>
      pick(() => http.post<{ ok: boolean }>(`/api/user-model/questions/${enc(id)}`, { answer }), () => demo.dropQuestion(id)),
    dismissQuestion: (id: string) => pick(() => http.delete<{ ok: boolean }>(`/api/user-model/questions/${enc(id)}`), () => demo.dropQuestion(id))
  },

  // §15 dreams ------------------------------------------------------------------
  dreams: {
    list: (limit = 30) => pick(() => http.get<Dream[]>('/api/memories/dreams', { query: { limit } }), () => demo.dreams()),
    run: () => pick(() => http.post<Dream>('/api/memories/dreams/run'), () => demo.runDream())
  },

  // §16 webhooks ----------------------------------------------------------------
  hooks: {
    list: () => pick(() => http.get<Hook[]>('/api/hooks'), () => demo.hooks()),
    create: (name: string) => pick(() => http.post<HookCreated>('/api/hooks', { name }), () => demo.createHook(name)),
    delete: (id: string) => pick(() => http.delete<{ ok: boolean }>(`/api/hooks/${enc(id)}`), () => demo.deleteHook(id))
  },

  // §16 script jobs ----------------------------------------------------------------
  scripts: {
    /** Runs the stored script, or `code` (unsaved edits), once. 400 when the code does not compile. */
    test: (taskId: string, code?: string) =>
      pick(() => http.post<SandboxResult>(`/api/tasks/${enc(taskId)}/script/test`, code !== undefined ? { code } : undefined), () => demo.scriptTest(taskId)),
    /** `PATCH /api/tasks/{id}` with a partial `script`; 400 when the code does not compile. */
    update: (taskId: string, script: Partial<Pick<ScriptJob, 'condition' | 'then' | 'code'>>) =>
      pick(() => http.patch<Task>(`/api/tasks/${enc(taskId)}`, { script }), () => demo.patchScript(taskId, script))
  },

  // §5/§16 change feeds ------------------------------------------------------------
  feeds: {
    list: () => pick(() => http.get<FeedStatus[]>('/api/integrations/feeds'), () => DEMO_FEEDS),
    sync: (source: string) => http.post<{ ok: boolean; emitted?: number; error?: string }>(`/api/integrations/feeds/${enc(source)}/sync`)
  },

  // §4 runs -----------------------------------------------------------------------
  runs: {
    /** Failed or cancelled run of a non-swarm task: a new run with `retry_of`. 409 otherwise. */
    retry: (taskId: string, runId: string) => http.post<Task>(`/api/tasks/${enc(taskId)}/runs/${enc(runId)}/retry`)
  },

  // §11 sandbox -------------------------------------------------------------------
  sandbox: {
    status: () => pick(() => http.get<SandboxStatus>('/api/sandbox/status'), () => ({ enabled: true, backend: 'process', docker_available: false, python_version: '3.12.7' }))
  }
} as const
