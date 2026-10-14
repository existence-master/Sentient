// Unit tests for the approval card wording (#347): the engine's `risk_label` names the
// action first, and the tool-name/reason rules are only the fallback.
// Run with `npm test` (no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { effectiveRisk } from '../src/features/chat/approvalRisk.ts'

const BROWSER_DESC =
  'Click a button or link in Sentient\'s own browser. Use to buy, pay, submit forms, and interact with pages.'

function approval(over: Record<string, unknown> = {}) {
  return {
    risk: 'send' as const,
    reason: BROWSER_DESC,
    name: 'browser_click',
    arguments: { ref: 'e12' },
    riskLabel: null,
    ...over
  }
}

test('engine label "Clicks" beats the buy/pay wording in the tool description (#347)', () => {
  const r = effectiveRisk(approval({ riskLabel: 'Clicks' }))
  assert.equal(r.label, 'Clicks')
  assert.equal(r.verb, 'click something on the page')
  assert.equal(r.tone, 'warning')
})

test('engine label "Purchase" still shows a purchase for "Place order"', () => {
  const r = effectiveRisk(approval({ riskLabel: 'Purchase', arguments: { ref: 'e44' } }))
  assert.equal(r.label, 'Purchase')
  assert.equal(r.verb, 'make a purchase')
  assert.equal(r.tone, 'danger')
})

test('engine labels are matched case-insensitively', () => {
  const r = effectiveRisk(approval({ riskLabel: 'purchase' }))
  assert.equal(r.label, 'Purchase')
  assert.equal(r.tone, 'danger')
})

test('a blank engine label falls back to the tool-name rules', () => {
  const r = effectiveRisk(approval({ riskLabel: '   ' }))
  // no risk_label: the description's "buy, pay" still reads as a purchase
  assert.equal(r.label, 'Purchase')
})

test('no engine label keeps the old fallback behavior', () => {
  const r = effectiveRisk(approval())
  assert.equal(r.label, 'Purchase')
  const deletes = effectiveRisk(approval({ riskLabel: 'Deletes' }))
  assert.equal(deletes.label, 'Deletes')
  assert.equal(deletes.tone, 'danger')
})

test('engine labels for typing, pressing and commands map to plain wording', () => {
  assert.equal(effectiveRisk(approval({ riskLabel: 'Types into the page' })).label, 'Types into the page')
  assert.equal(effectiveRisk(approval({ riskLabel: 'Presses a key' })).label, 'Presses a key')
  const cmd = effectiveRisk(approval({ name: 'terminal_run', risk: 'exec', riskLabel: 'Runs a command' }))
  assert.deepEqual([cmd.label, cmd.verb, cmd.tone], ['Runs a command', 'run a command on this computer', 'danger'])
})

test('an unknown engine label shows verbatim in the badge, with fallback verb and tone', () => {
  const r = effectiveRisk(approval({ riskLabel: 'Archives' }))
  assert.equal(r.label, 'Archives')
  // the description still steers the rest: buy/pay reads as a purchase
  assert.equal(r.tone, 'danger')
  assert.equal(r.verb, 'make a purchase')
})

test('a trash preview still reads as moving to the trash, even with an engine label', () => {
  const r = effectiveRisk(
    approval({
      riskLabel: 'Deletes',
      preview: { kind: 'trash', title: 'Move to trash', fields: [] },
      risk: 'write',
      name: 'gmail_trash'
    })
  )
  assert.equal(r.label, 'Moves to the trash')
})
