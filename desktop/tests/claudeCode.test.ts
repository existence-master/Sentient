// Claude through your own Claude Code (#246): the status asked for right after the switch changes.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { allowedForRole, claudeCodeState, claudeWayFor, isClaudeCode, otherModel, refusingRoles, retryWhileSaving, SAVE_RETRIES, statusOnceSaved, SwitchNotSavedYet } from '../src/lib/claudeCode.ts'

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

// #442: Claude Code only answers chats, so background roles never get it.
const ACCEPTING = ['primary', 'voice', 'vision']

test('only the roles the engine names may use Claude Code', () => {
  assert.equal(allowedForRole('claude-code/sonnet', 'primary', ACCEPTING), true)
  assert.equal(allowedForRole('claude-code/sonnet', 'voice', ACCEPTING), true)
  for (const role of ['fast', 'planner', 'executor', 'embedding']) assert.equal(allowedForRole('claude-code/opus', role, ACCEPTING), false, role)
  assert.equal(allowedForRole('claude-code/sonnet', 'primary', undefined), false) // the engine hasn't said yet
  assert.equal(allowedForRole('claude-code/sonnet', undefined, ACCEPTING), false)
  assert.equal(allowedForRole('ollama_chat/qwen3:8b', 'fast', ACCEPTING), true)
  assert.equal(allowedForRole('', 'fast', undefined), true)
})

test('roles that use Claude Code but refuse it, their own or through the main model', () => {
  const roles = { primary: 'claude-code/sonnet', fast: 'claude-code/sonnet', voice: null, planner: null, executor: 'ollama_chat/qwen3:8b', embedding: 'ollama/nomic-embed-text', vision: null }
  assert.deepEqual(refusingRoles(roles, ACCEPTING), [
    { role: 'fast', model: 'claude-code/sonnet', inherited: false },
    { role: 'planner', model: 'claude-code/sonnet', inherited: true }
  ])
  assert.deepEqual(refusingRoles({ ...roles, primary: 'ollama_chat/qwen3:8b', fast: 'ollama_chat/qwen3:8b' }, ACCEPTING), [])
  assert.deepEqual(refusingRoles(roles, undefined), [])
})

test('the background model is the first one that is not Claude Code', () => {
  assert.equal(otherModel(['claude-code/sonnet', '', null, 'ollama_chat/qwen3:8b', 'openai/gpt-5']), 'ollama_chat/qwen3:8b')
  assert.equal(otherModel(['claude-code/sonnet', undefined]), '')
  assert.equal(isClaudeCode('claude-code/opus'), true)
  assert.equal(isClaudeCode('anthropic/claude-sonnet-5-5'), false)
})

test('Connect Claude starts on the choice that is set up now (#444)', () => {
  assert.equal(claudeWayFor(undefined, ''), 'key')
  assert.equal(claudeWayFor({ experimental_claude_code: true }, 'anthropic/claude-sonnet-5-5'), 'key')
  assert.equal(claudeWayFor({ experimental_claude_code: true }, 'claude-code/sonnet'), 'code_chats')
  assert.equal(claudeWayFor({ experimental_claude_code: true, claude_code_everywhere: true }, 'claude-code/sonnet'), 'code_everything')
  // "everything" only counts while Claude Code itself is on
  assert.equal(claudeWayFor({ experimental_claude_code: false, claude_code_everywhere: true }, 'ollama_chat/qwen3:8b'), 'key')
})

test('Settings says plainly whether Claude Code does everything', () => {
  assert.deepEqual(claudeCodeState(false, true), { label: 'Off', tone: 'neutral' })
  assert.deepEqual(claudeCodeState(true, false), { label: 'Chats only', tone: 'warning' })
  assert.deepEqual(claudeCodeState(true, true), { label: 'Does everything', tone: 'danger' })
  assert.equal(isClaudeCode('claude-code/haiku'), true)
  assert.equal(isClaudeCode('anthropic/claude-haiku-5-5'), false)
  assert.equal(isClaudeCode(null), false)
})
