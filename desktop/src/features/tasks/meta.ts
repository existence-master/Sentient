/** Presentation metadata for tasks: statuses, priorities, kinds, sources and list grouping (v2 constants.js). */
import {
  IconAlertCircle,
  IconAlertTriangle,
  IconArchive,
  IconBan,
  IconBolt,
  IconCalendarRepeat,
  IconCircleCheck,
  IconClock,
  IconLoader2,
  IconMessageQuestion,
  IconPlayerPlay,
  IconPlayerPause,
  IconRepeat,
  IconShieldCheck,
  IconSparkles,
  IconUsersGroup,
  IconX,
  IconCalendarTime,
  IconUser,
  IconMessageCircle,
  IconBulb,
  IconRadar,
  type Icon
} from '@tabler/icons-react'
import type { Tone } from '@/components/ui'
import type { RunStatus, Task, TaskScript, TaskStatus } from '@/lib/types'
import { parseDate } from '@/lib/utils'

export interface StatusMeta {
  label: string
  tone: Tone
  icon: Icon
  /** Show a pulsing indicator (work in progress). */
  live?: boolean
}

export const STATUS_META: Record<TaskStatus, StatusMeta> = {
  planning: { label: 'Planning', tone: 'info', icon: IconSparkles, live: true },
  clarification_pending: { label: 'Needs answers', tone: 'warning', icon: IconMessageQuestion },
  approval_pending: { label: 'Needs approval', tone: 'accent', icon: IconShieldCheck },
  pending: { label: 'Scheduled', tone: 'neutral', icon: IconClock },
  active: { label: 'Active', tone: 'success', icon: IconRepeat },
  processing: { label: 'Running', tone: 'info', icon: IconLoader2, live: true },
  completed: { label: 'Completed', tone: 'success', icon: IconCircleCheck },
  completed_with_errors: { label: 'Completed with errors', tone: 'warning', icon: IconAlertTriangle },
  error: { label: 'Failed', tone: 'danger', icon: IconAlertCircle },
  declined: { label: 'Declined', tone: 'neutral', icon: IconBan },
  cancelled: { label: 'Cancelled', tone: 'neutral', icon: IconX },
  archived: { label: 'Archived', tone: 'neutral', icon: IconArchive }
}

export const RUN_STATUS_META: Record<RunStatus, StatusMeta> = {
  processing: STATUS_META.processing,
  completed: STATUS_META.completed,
  completed_with_errors: STATUS_META.completed_with_errors,
  error: STATUS_META.error,
  cancelled: STATUS_META.cancelled
}

const UNKNOWN: StatusMeta = { label: 'Unknown', tone: 'neutral', icon: IconClock }

export function statusMeta(status: string | undefined, task?: Pick<Task, 'enabled'>): StatusMeta {
  const meta = (STATUS_META as Record<string, StatusMeta>)[status ?? ''] ?? UNKNOWN
  if (task && !task.enabled && (status === 'active' || status === 'pending')) {
    return { label: 'Paused', tone: 'neutral', icon: IconPlayerPause }
  }
  return meta
}

export function runStatusMeta(status: string | undefined): StatusMeta {
  return (RUN_STATUS_META as Record<string, StatusMeta>)[status ?? ''] ?? UNKNOWN
}

export const PRIORITY_META: Record<number, { label: string; tone: Tone; dot: string }> = {
  0: { label: 'High', tone: 'danger', dot: 'bg-danger' },
  1: { label: 'Medium', tone: 'warning', dot: 'bg-warning' },
  2: { label: 'Low', tone: 'neutral', dot: 'bg-fg-faint' }
}

export const priorityMeta = (p: number | undefined) => PRIORITY_META[p ?? 1] ?? PRIORITY_META[1]

// ---------------------------------------------------------------------------- kinds
export type TaskKind = 'one-off' | 'scheduled' | 'recurring' | 'triggered' | 'swarm' | 'script'

export const KIND_META: Record<TaskKind, { label: string; icon: Icon; description: string }> = {
  'one-off': { label: 'One-off', icon: IconPlayerPlay, description: 'Runs once, right after you approve it' },
  scheduled: { label: 'Scheduled', icon: IconCalendarTime, description: 'Runs once at a set time' },
  recurring: { label: 'Recurring', icon: IconCalendarRepeat, description: 'Runs on a repeating schedule' },
  triggered: { label: 'Triggered', icon: IconBolt, description: 'Runs when something happens in an app' },
  swarm: { label: 'Swarm', icon: IconUsersGroup, description: 'Runs in parallel with multiple agents' },
  script: { label: 'Watcher', icon: IconRadar, description: 'A small script checks something on a schedule' }
}

export function taskKind(task: Pick<Task, 'task_type' | 'schedule'>): TaskKind {
  if ((task.task_type as string) === 'script') return 'script'
  if (task.task_type === 'swarm') return 'swarm'
  const s = task.schedule
  if (s?.type === 'recurring') return 'recurring'
  if (s?.type === 'triggered') return 'triggered'
  if (s?.type === 'once' && s.run_at) return 'scheduled'
  return 'one-off'
}

// ---------------------------------------------------------------------------- sources
export type TaskSource = 'you' | 'chat' | 'proactive'

export const SOURCE_META: Record<TaskSource, { label: string; icon: Icon }> = {
  you: { label: 'You', icon: IconUser },
  chat: { label: 'Chat', icon: IconMessageCircle },
  proactive: { label: 'Proactive', icon: IconBulb }
}

export function taskSource(task: Pick<Task, 'original_context'>): TaskSource {
  const s = task.original_context?.source
  if (s === 'chat') return 'chat'
  if (s === 'proactive' || s === 'trigger') return 'proactive'
  return 'you'
}

// ---------------------------------------------------------------------------- grouping
export const ATTENTION_STATUSES: TaskStatus[] = ['approval_pending', 'clarification_pending', 'error']

export const needsAttention = (t: Pick<Task, 'status'>) => ATTENTION_STATUSES.includes(t.status)

export type GroupId = 'attention' | 'running' | 'scheduled' | 'triggered' | 'completed' | 'archived'

export const GROUPS: Array<{ id: GroupId; label: string; hint: string; defaultOpen: boolean }> = [
  { id: 'attention', label: 'Needs your attention', hint: 'Approve plans, answer questions and fix failures', defaultOpen: true },
  { id: 'running', label: 'Running', hint: 'Sentient is planning or working on these now', defaultOpen: true },
  { id: 'scheduled', label: 'Scheduled & recurring', hint: 'Waiting for their next run', defaultOpen: true },
  { id: 'triggered', label: 'Triggered', hint: 'Waiting for something to happen', defaultOpen: true },
  { id: 'completed', label: 'Completed', hint: 'Finished, declined or cancelled', defaultOpen: true },
  { id: 'archived', label: 'Archived', hint: 'Out of the way, still searchable', defaultOpen: false }
]

export function taskGroup(task: Task): GroupId {
  switch (task.status) {
    case 'approval_pending':
    case 'clarification_pending':
    case 'error':
      return 'attention'
    case 'planning':
    case 'processing':
      return 'running'
    case 'archived':
      return 'archived'
    case 'completed':
    case 'completed_with_errors':
    case 'declined':
    case 'cancelled':
      return 'completed'
    default:
      return taskKind(task) === 'triggered' ? 'triggered' : 'scheduled'
  }
}

/** Board columns (kanban by status). */
export const BOARD_COLUMNS: Array<{ id: string; label: string; statuses: TaskStatus[]; tone: Tone }> = [
  { id: 'planning', label: 'Planning', statuses: ['planning'], tone: 'info' },
  { id: 'you', label: 'Needs you', statuses: ['approval_pending', 'clarification_pending', 'error'], tone: 'accent' },
  { id: 'scheduled', label: 'Scheduled', statuses: ['pending', 'active'], tone: 'neutral' },
  { id: 'running', label: 'Running', statuses: ['processing'], tone: 'info' },
  { id: 'done', label: 'Done', statuses: ['completed', 'completed_with_errors', 'declined', 'cancelled'], tone: 'success' }
]

// ---------------------------------------------------------------------------- misc helpers
export function latestRun(task: Pick<Task, 'runs'>) {
  const runs = task.runs ?? []
  return runs.length ? runs[runs.length - 1] : undefined
}

export function processingRun(task: Pick<Task, 'runs'>) {
  return [...(task.runs ?? [])].reverse().find((r) => r.status === 'processing')
}

/** v2 getDisplayName: generic proactive names fall back to the description. */
export function displayName(task: Pick<Task, 'name' | 'description'>): string {
  if (task.name === 'Proactively generated plan' && task.description) return task.description
  return task.name?.trim() || 'Untitled task'
}

export function sortTasks(tasks: Task[]): Task[] {
  return [...tasks].sort((a, b) => {
    const pa = a.priority ?? 1
    const pb = b.priority ?? 1
    if (pa !== pb) return pa - pb
    const ua = parseDate(a.updated_at)?.getTime() ?? 0
    const ub = parseDate(b.updated_at)?.getTime() ?? 0
    return ub - ua
  })
}

export function runDurationMs(run: { execution_start_time?: string | null; created_at?: string; finished_at?: string | null }, now = Date.now()): number | null {
  const start = parseDate(run.execution_start_time ?? run.created_at)?.getTime()
  if (!start) return null
  const end = parseDate(run.finished_at)?.getTime() ?? now
  return Math.max(0, end - start)
}

// ---------------------------------------------------------------------------- script jobs and retries (§16, §4)
export function isScriptJob(task: Pick<Task, 'task_type'> | null | undefined): boolean {
  return (task as { task_type?: string } | null | undefined)?.task_type === 'script'
}

export function scriptOf(task: Task | null | undefined): TaskScript | null {
  const s = task?.script
  return s && typeof s === 'object' && typeof s.code === 'string' ? s : null
}

/** A run that retries an earlier one (`POST /runs/{id}/retry`). */
export function retryOf(run: unknown): string | null {
  const r = (run as { retry_of?: unknown } | null)?.retry_of
  return typeof r === 'string' && r ? r : null
}
