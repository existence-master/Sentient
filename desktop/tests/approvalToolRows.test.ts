import assert from 'node:assert/strict'
import { after, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'
import type { AssistantTurnView } from '../src/lib/chatFold.ts'

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

  const [reloaded] = foldTranscript([{
    id: 'assistant',
    role: 'assistant',
    content: null,
    created_at: '2026-11-24T00:00:00Z',
    tool_calls: [{ id: call.call_id, function: { name: 'file_write', arguments: JSON.stringify(fileArgs) } }]
  }])
  assert.match(fileRow(reloaded), /Step interrupted/)
})
