import assert from 'node:assert/strict'
import { after, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'
import type { AssistantTurnView } from '../src/lib/chatFold.ts'
import type { TranscriptMessage } from '../src/lib/types.ts'

// Load the real chat cards with the renderer's JSX and path alias, without starting Electron.
const vite = await createServer({
  root: fileURLToPath(new URL('../', import.meta.url)),
  configFile: false,
  resolve: { alias: { '@': fileURLToPath(new URL('../src', import.meta.url)) } },
  esbuild: { jsx: 'automatic' },
  server: { middlewareMode: true, ws: false, watch: null },
  appType: 'custom',
  logLevel: 'error'
})
after(() => vite.close())

const { applyAgentEvent, emptyTurn, foldTranscript } = await vite.ssrLoadModule('/src/lib/chatFold.ts')
const { ToolCallCard } = await vite.ssrLoadModule('/src/features/chat/ToolCallCard.tsx')
const { BrowserCard } = await vite.ssrLoadModule('/src/features/chat/cards/BrowserCard.tsx')

const call = { call_id: 'step', session_id: 'chat', turn_id: 'turn' }
const fileArgs = { name: 'balcony-plants.txt', content: 'Water on Thursday' }
const clickArgs = { ref: 'e4' }
const savedFileCall: TranscriptMessage = {
  id: 'assistant',
  role: 'assistant',
  content: null,
  created_at: '2026-11-24T00:00:00Z',
  tool_calls: [{ id: call.call_id, type: 'function', function: { name: 'file_write', arguments: JSON.stringify(fileArgs) } }]
}

function waitingTurn(name: string, args: Record<string, unknown>, target?: string): AssistantTurnView {
  const started = applyAgentEvent(emptyTurn(call.turn_id), { ...call, type: 'tool_call', name, arguments: args })
  return applyAgentEvent(started, {
    ...call,
    type: 'approval_request',
    approval_id: 'approval',
    name,
    arguments: args,
    risk: 'write',
    reason: '',
    target
  })
}

function finishStep(turn: AssistantTurnView, result: unknown, isError = false): AssistantTurnView {
  return applyAgentEvent(turn, {
    ...call,
    type: 'tool_result',
    name: turn.tools.step.name,
    result,
    is_error: isError,
    duration_ms: 2
  })
}

function fileRow(turn: AssistantTurnView): string {
  return renderToStaticMarkup(createElement(ToolCallCard, { tool: turn.tools.step, approval: turn.approvals[0] }))
}

function browserRow(turn: AssistantTurnView): string {
  return renderToStaticMarkup(createElement(BrowserCard, { tools: [turn.tools.step], approvals: turn.approvals }))
}

test('a waiting file write names the file and reports success on its result', () => {
  const turn = waitingTurn('file_write', fileArgs)
  const waiting = fileRow(turn)
  assert.match(waiting, />Would write</)
  assert.match(waiting, />balcony-plants\.txt</)
  assert.doesNotMatch(waiting, /Wrote file/)
  assert.match(fileRow(finishStep(turn, { saved: fileArgs.name })), /Wrote file/)
})

test('a waiting browser click names the target and reports success on its result', () => {
  const turn = waitingTurn('browser_click', clickArgs, 'Save preferences')
  const waiting = browserRow(turn)
  assert.match(waiting, /Would click “Save preferences”/)
  assert.doesNotMatch(waiting, /Clicked/)
  assert.match(browserRow(finishStep(turn, { clicked: 'Save preferences' })), /Clicked “Save preferences”/)
})

test('declined and failed actions have their own wording', () => {
  const file = waitingTurn('file_write', fileArgs)
  const click = waitingTurn('browser_click', clickArgs, 'Save preferences')
  const declined = { error: 'The user declined this action.', declined: true }
  const failed = { error: 'The action failed.' }
  for (const [turn, render, done] of [[file, fileRow, /Wrote file/], [click, browserRow, /Clicked/]] as const) {
    const declinedRow = render(finishStep(turn, declined, true))
    assert.match(declinedRow, /You declined:/)
    assert.doesNotMatch(declinedRow, done)
    const failedRow = render(finishStep(turn, failed, true))
    assert.match(failedRow, /Step failed/)
    assert.doesNotMatch(failedRow, done)
  }
})

test('a chat ending without a tool result leaves the step interrupted in the live chat and history', () => {
  const turn = waitingTurn('file_write', fileArgs)
  const click = waitingTurn('browser_click', clickArgs, 'Save preferences')
  for (const cancelled of [true, false]) {
    const event = { ...call, type: 'done', cancelled, content: '', message_id: null } as const
    const ended = applyAgentEvent(turn, event)
    assert.match(fileRow(ended), /Step interrupted/)
    assert.equal(ended.approvals[0].status, 'expired')
    const browser = browserRow(applyAgentEvent(click, event))
    assert.match(browser, /Step interrupted/)
    assert.doesNotMatch(browser, /Clicked|Used the browser/)
  }

  const [reloaded] = foldTranscript([savedFileCall, {
    id: 'stopped', role: 'assistant', content: 'Saving your note.\n\n_(stopped)_', created_at: savedFileCall.created_at
  }])
  assert.match(fileRow(reloaded), /Step interrupted/)
})

test('a saved file write stays pending in history until its result arrives', () => {
  const helperNote: TranscriptMessage = {
    id: 'helper', role: 'assistant', content: 'Your helper finished checking the forecast.', created_at: savedFileCall.created_at
  }
  const [pending] = foldTranscript([savedFileCall, helperNote])
  const waiting = fileRow(pending)
  assert.match(waiting, /Waiting for result/)
  assert.doesNotMatch(waiting, /Wrote file|Step interrupted/)

  const [done] = foldTranscript([savedFileCall, helperNote, {
    id: 'result', role: 'tool', tool_call_id: call.call_id,
    content: JSON.stringify({ saved: fileArgs.name }), created_at: savedFileCall.created_at
  }])
  assert.match(fileRow(done), /Wrote file/)
})

test('a saved interruption marks its named tool as interrupted in history', () => {
  const failure: TranscriptMessage = {
    id: 'failed', role: 'event', name: 'turn_interrupted',
    content: JSON.stringify([call.call_id]), created_at: savedFileCall.created_at
  }
  const [failed] = foldTranscript([savedFileCall, failure])
  const row = fileRow(failed)
  assert.match(row, /Step interrupted/)
  assert.doesNotMatch(row, /Waiting for result|Wrote file/)
})

test('a group message does not hide which earlier tool was interrupted', () => {
  const createdAt = savedFileCall.created_at
  const history: TranscriptMessage[] = [{
    ...savedFileCall,
    tool_calls: [
      { id: 'lookup', type: 'function', function: { name: 'current_datetime', arguments: '{}' } },
      ...(savedFileCall.tool_calls ?? [])
    ]
  }, {
    id: 'time', role: 'tool', tool_call_id: 'lookup', content: '"9:00 AM"', created_at: createdAt
  }, {
    id: 'group-message', role: 'other', name: 'Priya', content: 'I watered the plants.', created_at: createdAt
  }]
  const newerCall: TranscriptMessage = {
    ...savedFileCall, id: 'new-activity',
    tool_calls: [{ id: 'next', type: 'function', function: { name: 'file_read', arguments: '{"name":"weather.txt"}' } }]
  }
  const interruption: TranscriptMessage = {
    id: 'interrupted', role: 'event', name: 'turn_interrupted',
    content: JSON.stringify([call.call_id]), created_at: createdAt
  }

  for (const newerRows of [[], [newerCall]]) {
    const [earlier, groupMessage, newer] = foldTranscript([...history, ...newerRows, interruption])
    assert.match(fileRow(earlier), /Step interrupted/)
    assert.equal(earlier.tools.lookup.status, 'done')
    assert.equal(groupMessage.kind, 'other')
    if (newer) assert.equal(newer.tools.next.status, 'awaiting_result')
  }
})
