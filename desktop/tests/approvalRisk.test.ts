// Unit tests for the approval card wording (#344): an email call is named from its tool name
// and declared risk, never from words in the arguments' content.
// Run with `npm test` (no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { approvalRisk } from '../src/features/chat/approvalRisk.ts'

const email = (name: string, body: string, reason = 'mcp_mailer: Send an email to one address') => ({
  risk: 'send' as const,
  name,
  reason,
  arguments: { to: 'someone@example.com', subject: 'Hello', body }
})

test('an email whose body says "post" still reads as sending an email', () => {
  const w = approvalRisk(email('send_email', 'I want to post about our launch today'))
  assert.equal(w.label, 'Sends an email')
  assert.equal(w.verb, 'send an email')
})

test('an email whose body says "publish" still reads as sending an email', () => {
  const w = approvalRisk(email('gmail_send', 'We will publish the results next week', 'gmail: Send a message'))
  assert.equal(w.label, 'Sends an email')
  assert.equal(w.verb, 'send an email')
})

test('an email whose body says "tweet" still reads as sending an email', () => {
  const w = approvalRisk(email('send_email', 'Did you see that tweet about the outage?'))
  assert.equal(w.label, 'Sends an email')
  assert.equal(w.verb, 'send an email')
})

test('a mail tool whose name only deletes keeps the deletion wording', () => {
  const w = approvalRisk(email('gmail_delete', 'post it', 'gmail: Move a message to trash'))
  assert.equal(w.label, 'Deletes')
})

test('a real post tool is still named from its name and description', () => {
  const w = approvalRisk({
    risk: 'send',
    name: 'social_post',
    reason: 'Post an update to your timeline',
    arguments: { text: 'hello' }
  })
  assert.equal(w.label, 'Posts publicly')
  assert.equal(w.verb, 'post something publicly')
})

test('arguments content never steers the wording of other tools either', () => {
  const w = approvalRisk({
    risk: 'send',
    name: 'browser_click',
    reason: 'Click "Confirm"',
    arguments: { text: 'please post this form' }
  })
  assert.equal(w.label, 'Submits')
})

test('purchases, commands, code and reads keep their wording', () => {
  assert.equal(approvalRisk({ risk: 'send', name: 'browser_click', reason: 'Click "Place order"' }).label, 'Purchase')
  assert.equal(approvalRisk({ risk: 'exec', name: 'terminal_run', reason: 'run a command' }).label, 'Runs a command')
  assert.equal(approvalRisk({ risk: 'exec', name: 'execute_code', reason: 'run code' }).label, 'Runs code')
  assert.equal(approvalRisk({ risk: 'write', name: 'file_write', reason: 'write a file' }).label, 'Changes things')
  assert.equal(approvalRisk({ risk: 'read', name: 'file_read', reason: 'read a file' }).label, 'Reads')
})
