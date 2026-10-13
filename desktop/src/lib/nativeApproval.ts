/** Which notifications become an approval notification with Allow once and Deny (#323). Pure, for `npm test`. */
import type { NativeApproval } from '../types/bridge'
import type { Notification } from './types'

/** The waiting run a "needs your OK" task notification asks about, or undefined for every other notification
 *  (plain questions, stuck runs, limits, and requests already answered keep the plain notification). */
export function approvalTarget(n: Pick<Notification, 'kind' | 'task_id' | 'payload'>): NativeApproval | undefined {
  const p = (n.payload ?? {}) as Record<string, unknown>
  if (n.kind !== 'task' || p.event !== 'question' || p.status) return undefined
  const approval = p.approval
  if (!approval || typeof approval !== 'object') return undefined
  const taskId = n.task_id ?? (typeof p.task_id === 'string' ? p.task_id : null)
  const runId = typeof p.run_id === 'string' ? p.run_id : null
  if (!taskId || !runId) return undefined
  const expiresAt = typeof p.expires_at === 'string' ? p.expires_at : undefined
  return { taskId, runId, untrusted: (approval as { untrusted?: unknown }).untrusted === true, ...(expiresAt ? { expiresAt } : {}) }
}

/** The run whose approval notification should close: its request was answered, cancelled or expired. */
export function settledApprovalRun(n: Pick<Notification, 'kind' | 'payload'>): string | undefined {
  const p = (n.payload ?? {}) as Record<string, unknown>
  if (n.kind !== 'task' || p.event !== 'question' || !p.status || !p.approval) return undefined
  return typeof p.run_id === 'string' ? p.run_id : undefined
}
