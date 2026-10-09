/** Filtering, grouping and click-through helpers for notifications. */
import type { Notification } from '@/lib/types'
import { parseDate } from '@/lib/utils'

export type FeedFilter = 'all' | 'suggestions' | 'tasks' | 'approvals' | 'skills'

export const FILTERS: Array<{ id: FeedFilter; label: string }> = [
  { id: 'all', label: 'All' },
  { id: 'suggestions', label: 'Suggestions' },
  { id: 'tasks', label: 'Tasks' },
  { id: 'approvals', label: 'Approvals' },
  { id: 'skills', label: 'Skills' }
]

export const SOURCE_LABEL: Record<string, string> = { gmail: 'Gmail', gcalendar: 'Calendar', email_imap: 'Email', heartbeat: 'Check-in' }

export function isApprovalLike(n: Notification): boolean {
  return n.kind === 'approval' || (n.kind === 'task' && n.payload?.event === 'approval_needed')
}

export function matchesFilter(n: Notification, f: FeedFilter): boolean {
  switch (f) {
    case 'all':
      return true
    case 'suggestions':
      return n.kind === 'proactive'
    case 'tasks': {
      const v = notificationVariant(n)
      return (n.kind === 'task' && !isApprovalLike(n)) || v === 'script_alert' || v === 'script_failed' || v === 'script_recovered' || v === 'run_failed'
    }
    case 'approvals':
      return isApprovalLike(n)
    case 'skills':
      return n.kind === 'skill'
  }
}

/** Waiting on the user (pending suggestion or approval). */
export function needsAction(n: Notification): boolean {
  if (n.kind === 'proactive') return (n.payload?.status ?? 'pending') === 'pending'
  if (n.kind === 'approval') return !n.payload?.status
  return false
}

function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

export function dayLabel(iso: string, now = new Date()): string {
  const d = parseDate(iso)
  if (!d) return 'Earlier'
  const days = Math.round((startOfDay(now) - startOfDay(d)) / 86_400_000)
  if (days <= 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 7) return d.toLocaleDateString(undefined, { weekday: 'long' })
  return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: days > 300 ? 'numeric' : undefined })
}

export function groupByDay(list: Notification[]): Array<{ label: string; items: Notification[] }> {
  const out: Array<{ label: string; items: Notification[] }> = []
  for (const n of list) {
    const label = dayLabel(n.created_at)
    const g = out[out.length - 1]
    if (g && g.label === label) g.items.push(n)
    else out.push({ label, items: [n] })
  }
  return out
}

export function taskRoute(taskId: string): string {
  return `/tasks/${encodeURIComponent(taskId)}`
}

/** Where clicking a notification goes (null: nowhere to go). */
export function clickRoute(n: Notification): string | null {
  const p = n.payload ?? {}
  const variant = notificationVariant(n)
  if (variant === 'dream') return '/about/dreams'
  if (variant === 'skill_repair') return `/skills?tab=pending${typeof p.skill === 'string' ? `&focus=${encodeURIComponent(p.skill)}` : ''}`
  if (n.kind === 'skill') return '/skills'
  const sessionId = typeof p.session_id === 'string' ? p.session_id : null
  if (sessionId) return `/chat/${encodeURIComponent(sessionId)}`
  const taskId = n.task_id ?? (typeof p.task_id === 'string' ? p.task_id : null)
  if (taskId) return taskRoute(taskId)
  if (typeof p.integration === 'string') return `/integrations?open=${encodeURIComponent(p.integration)}`
  return null
}

export type NotificationVariant = 'script_alert' | 'script_failed' | 'script_recovered' | 'run_failed' | 'subagent' | 'dream' | 'skill_repair' | null

/** Recognises the richer notification flavours from `kind` or `payload.event` (docs/API.md §4, §15, §16). */
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
