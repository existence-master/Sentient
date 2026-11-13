// Unit tests for the tray's running tasks (#182). Run with `npm test` (no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { parseRunningTasks, runningTaskLabel, runningTasksTooltip } from '../electron/main/tasks-logic.ts'

test('the engine list is read defensively', () => {
  assert.deepEqual(parseRunningTasks(null), [])
  assert.deepEqual(parseRunningTasks({ type: 'tasks', running: 'nope' }), [])
  const tasks = parseRunningTasks({
    type: 'tasks',
    running: [
      { id: 't1', name: 'Book a flight', status: 'processing', activity: 'Using Gmail' },
      { id: 't2', name: '  ', status: 'waiting_for_user', activity: null },
      { name: 'no id' },
      null
    ]
  })
  assert.deepEqual(tasks, [
    { id: 't1', name: 'Book a flight', status: 'processing', activity: 'Using Gmail' },
    { id: 't2', name: 'Task', status: 'waiting_for_user', activity: null }
  ])
})

test('a menu line says what the task is doing, or that it waits for an answer', () => {
  assert.equal(runningTaskLabel({ id: 'a', name: 'Book a flight', status: 'processing', activity: 'Using Gmail' }), 'Book a flight: Using Gmail')
  assert.equal(runningTaskLabel({ id: 'a', name: 'Book a flight', status: 'processing', activity: null }), 'Book a flight: Running')
  assert.equal(
    runningTaskLabel({ id: 'a', name: 'Book a flight', status: 'waiting_for_user', activity: 'Using Gmail' }),
    'Book a flight: Waiting for your answer'
  )
  const long = runningTaskLabel({ id: 'a', name: 'x'.repeat(80), status: 'processing', activity: null })
  assert.equal(long.length, 60)
  assert.ok(long.endsWith('…'))
})

test('the tooltip counts running tasks', () => {
  assert.equal(runningTasksTooltip([]), '')
  assert.equal(runningTasksTooltip(parseRunningTasks({ running: [{ id: 'a' }] })), '1 task running')
  assert.equal(runningTasksTooltip(parseRunningTasks({ running: [{ id: 'a' }, { id: 'b' }] })), '2 tasks running')
})
