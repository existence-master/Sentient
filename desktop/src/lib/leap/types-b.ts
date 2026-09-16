/**
 * Types for desktop agent B screens (docs/API.md sections 4, 11, 15 and 16).
 * Field names match the JSON exactly. Optional fields are ones the contract does not pin down yet.
 */
import type { ISODate, Notification, Skill, Task, TaskPreview, TriggeredSchedule, VoiceState, VoiceStatus } from '@/lib/types'

// ============================================================================ §15 user model
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

// ============================================================================ §15 dreams
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

// ============================================================================ §11 sandbox
export interface SandboxResult {
  ok: boolean
  backend: 'process' | 'docker' | string
  stdout: string
  stderr: string
  result: unknown
  files_created: string[]
  tool_calls: number | unknown[]
  duration_ms: number
  error: string | null
}

export interface SandboxStatus {
  enabled: boolean
  backend: string
  docker_available: boolean
  python_version: string
}

// ============================================================================ §16 script jobs
export type ScriptCondition = 'changed' | 'alert'
export type ScriptThen = 'notify' | 'run'

export interface ScriptJob {
  code: string
  condition: ScriptCondition
  then: ScriptThen
  last_result: unknown
  last_run_at: ISODate | null
  last_error: string | null
}

/** `{type: "recurring", frequency: "interval", interval_minutes}` (minimum 5). Not in lib/types RecurringSchedule yet. */
export interface IntervalSchedule {
  type: 'recurring'
  frequency: 'interval'
  interval_minutes: number
  timezone?: string
}

export function isIntervalSchedule(s: unknown): s is IntervalSchedule {
  const f = (s as { type?: string; frequency?: string } | null)?.frequency
  return (s as { type?: string } | null)?.type === 'recurring' && (f === 'interval' || f === 'hourly')
}

/** Every N minutes, never below 5 (`hourly` is 60). */
export function intervalMinutes(s: unknown): number {
  const raw = s as { frequency?: string; interval_minutes?: unknown }
  if (raw?.frequency === 'hourly') return 60
  const n = Math.round(Number(raw?.interval_minutes))
  return Number.isFinite(n) && n > 0 ? Math.max(5, n) : 60
}

/** A run that retries an earlier one (`POST /runs/{id}/retry`). */
export function retryOf(run: unknown): string | null {
  const r = (run as { retry_of?: unknown } | null)?.retry_of
  return typeof r === 'string' && r ? r : null
}

/** A task that may be a script job (`task_type: "script"` plus `script`). */
export type TaskWithScript = Task & { script?: ScriptJob | null }

/** `POST /api/tasks/preview` may also describe a script job when the planner proposes one. */
export type TaskPreviewB = TaskPreview & { task_type?: string; script?: Partial<ScriptJob> | null }

export function isScriptJob(task: Pick<Task, 'task_type'> | null | undefined): boolean {
  return (task as { task_type?: string } | null | undefined)?.task_type === 'script'
}

export function scriptOf(task: Task | null | undefined): ScriptJob | null {
  const s = (task as TaskWithScript | null | undefined)?.script
  return s && typeof s === 'object' && typeof s.code === 'string' ? s : null
}

// ============================================================================ §16 webhooks
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

// ============================================================================ §16 wake word
export type VoiceStateB = VoiceState | 'standby'

export interface WakeStatus {
  engine: string
  phrase: string
  ready: boolean
  error?: string
}

export type VoiceStatusB = VoiceStatus & { wake?: WakeStatus }

// ============================================================================ §10 subagents (read-only here)
export interface SubagentLite {
  subagent_id: string
  session_id: string
  goal: string
  status: 'running' | 'completed' | 'error' | 'cancelled'
  summary: string | null
  error: string | null
}

// ============================================================================ skills (repair proposals)
export type PendingSkillB = Skill & { origin?: (Skill['origin'] & { repair?: boolean }) | 'repair' | null }

export function isRepairProposal(skill: Skill): boolean {
  const o = (skill as PendingSkillB).origin
  return o === 'repair' || (!!o && typeof o === 'object' && o.repair === true)
}

// ============================================================================ notifications
export type NotificationVariant = 'script_alert' | 'script_failed' | 'script_recovered' | 'run_failed' | 'subagent' | 'dream' | 'skill_repair' | null

/** Recognises the leap notification flavours from `kind` or `payload.event` (docs/API.md §4, §15, §16). */
export function notificationVariant(n: Notification): NotificationVariant {
  const kind = String(n.kind)
  const p = (n.payload ?? {}) as Record<string, unknown>
  const event = typeof p.event === 'string' ? p.event : typeof p.type === 'string' ? p.type : ''
  if (kind === 'script_alert' || event === 'script_alert') return 'script_alert'
  if (event === 'script_failed') return 'script_failed'
  if (event === 'script_recovered') return 'script_recovered'
  if (event === 'run_failed') return 'run_failed'
  if (kind === 'subagent' || event.startsWith('subagent') || typeof p.subagent_id === 'string') return 'subagent'
  if (kind === 'dream' || event.startsWith('dream') || typeof p.dream_id === 'string') return 'dream'
  const origin = p.origin as unknown
  if (kind === 'skill' && (origin === 'repair' || (!!origin && typeof origin === 'object' && (origin as { repair?: unknown }).repair === true) || p.action === 'repair')) return 'skill_repair'
  return null
}
