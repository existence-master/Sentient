/**
 * Pure helpers behind the approval notification (#323): a task that needs your OK while the window is in the
 * background shows a native notification with **Allow once** and **Deny**. No Electron imports, so `npm test`
 * covers them.
 *
 * The buttons answer through the same route as the approval card (`POST /api/tasks/{id}/runs/{run}/answer`) and
 * never offer more than one call: wider answers ("Allow for this task", "Always allow"...) stay in the app, and
 * purchases, running code and calls after outside content only ever offer Allow once anyway (#318).
 */

/** The button labels, in order. Index 0 allows this one call, index 1 denies it. */
export const APPROVAL_BUTTONS = ['Allow once', 'Deny'] as const

/** Which waiting run a notification answers. `untrusted`: a call held after outside content (ADR 0018). */
export interface ApprovalTarget {
  taskId: string
  runId: string
  untrusted?: boolean
}

const ID = /^[\w-]{1,128}$/

/** The approval target from the renderer, or null when it is missing or malformed (the click still opens the app). */
export function cleanApproval(raw: unknown): ApprovalTarget | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const taskId = typeof r.taskId === 'string' ? r.taskId : ''
  const runId = typeof r.runId === 'string' ? r.runId : ''
  if (!ID.test(taskId) || !ID.test(runId)) return null
  return { taskId, runId, untrusted: r.untrusted === true }
}

/** The answer a button sends, worded the way the engine reads it for this kind of request; null for other indexes. */
export function approvalAnswer(actionIndex: number, untrusted = false): string | null {
  if (actionIndex === 0) return untrusted ? 'Yes, go ahead' : 'Allow once'
  if (actionIndex === 1) return untrusted ? 'No, stop the task' : 'Deny'
  return null
}

/** The REST path that answers the run. */
export function answerPath(t: ApprovalTarget): string {
  return `/api/tasks/${encodeURIComponent(t.taskId)}/runs/${encodeURIComponent(t.runId)}/answer`
}

/** One notification per waiting run, so a repeat replaces it and an answer in the app can close it. */
export function approvalNotificationId(runId: string): string {
  return `approval-${runId}`
}

/** The task's page, where the approval card waits. */
export function approvalRoute(t: ApprovalTarget): string {
  return `/tasks/${encodeURIComponent(t.taskId)}`
}

/** A plain sentence for a refused answer (`{detail}` from the engine), e.g. an expired or already answered request. */
export function answerProblem(status: number, body: unknown): string {
  const detail = body && typeof body === 'object' ? (body as { detail?: unknown }).detail : undefined
  if (typeof detail === 'string' && detail.trim()) return detail.trim().slice(0, 240)
  if (status === 404) return 'This task is gone.'
  if (status === 409) return 'This request was already answered.'
  return "Sentient couldn't take the answer. Open the task to answer there."
}
