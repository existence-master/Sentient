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

test('a voicemail tool is not misread as sending an email', () => {
  const w = approvalRisk({ risk: 'send', name: 'send_voicemail', reason: 'Send a voice message', arguments: {} })
  assert.equal(w.label, 'Sends')
})

test('an unrecognized tool whose description says it sends email reads as sending an email', () => {
  const w = approvalRisk({
    risk: 'send',
    name: 'send_message',
    reason: 'Send an email to one address',
    arguments: {}
  })
  assert.equal(w.label, 'Sends an email')
  assert.equal(w.verb, 'send an email')
})

test('"Discard draft" is not misread as a purchase', () => {
  const w = approvalRisk({ risk: 'send', name: 'browser_click', reason: 'Click "Discard draft"', arguments: {} })
  assert.equal(w.label, 'Sends')
})

test('a write-risk MCP email tool that sends reads as sending an email', () => {
  const w = approvalRisk({
    risk: 'write',
    name: 'mcp_mailer_send_email',
    reason: 'Send an email',
    arguments: {}
  })
  assert.equal(w.label, 'Sends an email')
  assert.equal(w.verb, 'send an email')
})

test('a write-risk MCP draft tool keeps the generic change wording', () => {
  const w = approvalRisk({ risk: 'write', name: 'gmail_draft_create', reason: 'Create a draft', arguments: {} })
  assert.equal(w.label, 'Changes things')
})

test('a namespaced mailer draft tool is not misread as sending an email', () => {
  const w = approvalRisk({
    risk: 'write',
    name: 'mcp_mailer_create_draft',
    reason: 'Create a draft email',
    arguments: {}
  })
  assert.equal(w.label, 'Changes things')
})

test('a reply-draft tool is not misread as sending an email', () => {
  const w = approvalRisk({
    risk: 'write',
    name: 'mcp_mailer_create_reply_draft',
    reason: 'Create a reply draft',
    arguments: {}
  })
  assert.equal(w.label, 'Changes things')
})

test('a Composio multi-execute delete names the slug action, not the wrapper', () => {
  const w = approvalRisk({
    risk: 'send',
    name: 'COMPOSIO_MULTI_EXECUTE_TOOL',
    reason: 'Run tools',
    arguments: { tools: [{ tool_slug: 'GMAIL_DELETE_MESSAGE', arguments: { id: 'abc123' } }] }
  })
  assert.equal(w.label, 'Deletes')
})

test('a Composio multi-execute email whose body says "post" still reads as sending an email', () => {
  const w = approvalRisk({
    risk: 'send',
    name: 'COMPOSIO_MULTI_EXECUTE_TOOL',
    reason: 'Run tools',
    arguments: {
      tools: [
        {
          tool_slug: 'GMAIL_SEND_EMAIL',
          arguments: { to: 'arjun@example.com', body: 'please post this on the blog and tweet it' }
        }
      ]
    }
  })
  assert.equal(w.label, 'Sends an email')
  assert.equal(w.verb, 'send an email')
})

test('a write-risk MCP mail delete reads as deletion', () => {
  const w = approvalRisk({
    risk: 'write',
    name: 'mcp_gmail_gmail_delete',
    reason: 'Delete a message',
    arguments: {}
  })
  assert.equal(w.label, 'Deletes')
})

test('Home Assistant locks, alarms and garage doors read as home security, with a plain caution (#175)', () => {
  const unlock = approvalRisk({
    risk: 'send',
    name: 'home_assistant_unlock',
    reason: 'home_assistant: Unlock a door lock in Home Assistant. Always asks first.',
    arguments: { entity_id: 'lock.front_door' }
  })
  assert.equal(unlock.label, 'Home security')
  assert.equal(unlock.verb, 'unlock a door')
  assert.equal(unlock.tone, 'danger')
  assert.match(unlock.caution ?? '', /walk in/)
  const garage = approvalRisk({ risk: 'send', name: 'home_assistant_open_cover', arguments: { entity_id: 'cover.garage_door' } })
  assert.equal(garage.label, 'Home security')
  // blinds are only a change: the engine leaves their risk at write
  const blinds = approvalRisk({ risk: 'write', name: 'home_assistant_open_cover', arguments: { entity_id: 'cover.bedroom_blinds' } })
  assert.equal(blinds.label, 'Changes things')
  assert.equal(blinds.caution, undefined)
})

// #378: a free subscribe the engine labels "Signs up" reads as a sign-up, not a purchase;
// a priced one keeps the purchase wording.
const click = (riskLabel: string) => ({
  risk: 'send' as const,
  name: 'browser_click',
  reason: 'browser: Click the element with this ref. Buttons that buy, pay, send, post, delete or confirm need approval.',
  arguments: { ref: 'e3' },
  riskLabel
})

test('a free newsletter button reads as signing up', () => {
  const w = approvalRisk(click('Signs up'))
  assert.equal(w.label, 'Signs up')
  assert.equal(w.verb, 'sign you up for something')
  assert.equal(w.tone, 'danger')
})

test('a priced subscribe button still reads as a purchase', () => {
  const w = approvalRisk(click('Purchase'))
  assert.equal(w.label, 'Purchase')
  assert.equal(w.verb, 'make a purchase')
})
