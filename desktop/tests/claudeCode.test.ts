// Claude through your own Claude Code (#246): the status asked for right after the switch changes.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { retryWhileSaving, SAVE_RETRIES, statusOnceSaved, SwitchNotSavedYet } from '../src/lib/claudeCode.ts'

test('a status from before the switch was saved is asked for again', async () => {
  const answers = [
    { enabled: false, version: null },
    { enabled: true, version: '2.1.269 (Claude Code)' }
  ]
  const get = async () => answers.shift()!
  await assert.rejects(statusOnceSaved(get, true), SwitchNotSavedYet)
  assert.deepEqual(await statusOnceSaved(get, true), { enabled: true, version: '2.1.269 (Claude Code)' })
})

test('only waiting for the save is retried, and not forever', () => {
  assert.equal(retryWhileSaving(0, new SwitchNotSavedYet()), true)
  assert.equal(retryWhileSaving(SAVE_RETRIES - 1, new SwitchNotSavedYet()), true)
  assert.equal(retryWhileSaving(SAVE_RETRIES, new SwitchNotSavedYet()), false)
  assert.equal(retryWhileSaving(0, new Error('engine is down')), false)
})
