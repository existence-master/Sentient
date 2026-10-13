// Unit tests for the approval notification (#323): which notifications get Allow once and Deny, and what each
// button answers. Run with `npm test` (no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  answerPath,
  answerProblem,
  APPROVAL_BUTTONS,
  approvalAnswer,
  approvalNotificationId,
  approvalRoute,
  cleanApproval,
  stillWaiting,
  taskPath
} from '../electron/main/approval-notify.ts'
import { approvalTarget, settledApprovalRun } from '../src/lib/nativeApproval.ts'

const waiting = {
  kind: 'task' as const,
  task_id: 't1',
  payload: { event: 'question', task_id: 't1', run_id: 'r1', options: ['Allow once', 'Deny'], approval: { tool: 'file_write', choices: ['allow', 'deny'] } }
}

test('a task waiting for an OK becomes an approval notification', () => {
  assert.deepEqual(approvalTarget(waiting), { taskId: 't1', runId: 'r1', untrusted: false })
})

test('the request carries its expiry, which tells it apart from a later one', () => {
  const n = { ...waiting, payload: { ...waiting.payload, expires_at: '2026-10-13T12:00:00+00:00' } }
  assert.deepEqual(approvalTarget(n), { taskId: 't1', runId: 'r1', untrusted: false, expiresAt: '2026-10-13T12:00:00+00:00' })
  assert.equal(cleanApproval({ taskId: 'a', runId: 'b', expiresAt: 'x'.repeat(65) })?.expiresAt, undefined)
})

test('an answer goes only to the very request the notification showed', () => {
  const t = { taskId: 't1', runId: 'r1', expiresAt: 'E1' }
  const task = (status: string, kind: string, expires: string) => ({
    runs: [{ run_id: 'r0', status: 'completed' }, { run_id: 'r1', status, pending_question: { kind, expires_at: expires } }]
  })
  assert.equal(stillWaiting(task('waiting_for_user', 'approval', 'E1'), t), true)
  assert.equal(stillWaiting(task('waiting_for_user', 'approval', 'E2'), t), false) // a later request of the run
  assert.equal(stillWaiting(task('processing', 'approval', 'E1'), t), false) // answered meanwhile
  assert.equal(stillWaiting(task('waiting_for_user', 'question', 'E1'), t), false) // now a plain question
  assert.equal(stillWaiting(task('waiting_for_user', 'approval', 'E9'), { taskId: 't1', runId: 'r1' }), true)
  assert.equal(stillWaiting({ runs: [] }, t), false)
  assert.equal(stillWaiting(null, t), false)
  assert.equal(taskPath(t), '/api/tasks/t1')
})

test('a call held after outside content is marked untrusted', () => {
  const n = { ...waiting, payload: { ...waiting.payload, approval: { choices: ['allow', 'deny'], untrusted: true } } }
  assert.equal(approvalTarget(n)?.untrusted, true)
})

test('plain questions, answered requests and other notifications stay plain', () => {
  assert.equal(approvalTarget({ ...waiting, payload: { ...waiting.payload, approval: undefined } }), undefined)
  assert.equal(approvalTarget({ ...waiting, payload: { ...waiting.payload, status: 'answered' } }), undefined)
  assert.equal(approvalTarget({ ...waiting, kind: 'info' as never }), undefined)
  assert.equal(approvalTarget({ ...waiting, payload: { ...waiting.payload, run_id: undefined } }), undefined)
})

test('the task id falls back to the payload', () => {
  assert.equal(approvalTarget({ ...waiting, task_id: null })?.taskId, 't1')
})

test('an answered, cancelled or expired request closes its notification', () => {
  assert.equal(settledApprovalRun({ ...waiting, payload: { ...waiting.payload, status: 'expired' } }), 'r1')
  assert.equal(settledApprovalRun(waiting), undefined)
  assert.equal(settledApprovalRun({ ...waiting, payload: { event: 'question', run_id: 'r1', status: 'answered' } }), undefined)
})

test('the buttons only ever allow one call or deny', () => {
  assert.deepEqual([...APPROVAL_BUTTONS], ['Allow once', 'Deny'])
  assert.equal(approvalAnswer(0), 'Allow once')
  assert.equal(approvalAnswer(1), 'Deny')
  assert.equal(approvalAnswer(0, true), 'Yes, go ahead')
  assert.equal(approvalAnswer(1, true), 'No, stop the task')
  assert.equal(approvalAnswer(2), null)
  assert.equal(approvalAnswer(-1), null)
})

test('the target from the renderer is checked before it reaches a URL', () => {
  assert.deepEqual(cleanApproval({ taskId: 'abc', runId: 'def' }), { taskId: 'abc', runId: 'def', untrusted: false })
  assert.equal(cleanApproval({ taskId: '../x', runId: 'def' }), null)
  assert.equal(cleanApproval({ taskId: 'abc' }), null)
  assert.equal(cleanApproval(null), null)
  assert.equal(cleanApproval('abc'), null)
})

test('paths, ids and routes', () => {
  const t = { taskId: 'abc', runId: 'def' }
  assert.equal(answerPath(t), '/api/tasks/abc/runs/def/answer')
  assert.equal(approvalRoute(t), '/tasks/abc')
  assert.equal(approvalNotificationId('def'), 'approval-def')
})

test('a refused answer says why in plain words', () => {
  assert.equal(answerProblem(409, { detail: 'This approval request expired.' }), 'This approval request expired.')
  assert.equal(answerProblem(409, null), 'This request was already answered.')
  assert.equal(answerProblem(404, {}), 'This task is gone.')
  assert.match(answerProblem(500, null), /Open the task/)
})
