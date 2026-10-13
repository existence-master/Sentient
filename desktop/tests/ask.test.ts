// Ask about selected text (#173): the pure helpers of the shell and the ask window.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  acceleratorKeyCode,
  appName,
  askBounds,
  captureCommand,
  cleanSelection,
  MAX_SELECTION_CHARS,
  parseCapture,
  parsePaste,
  pasteBackCommand,
  resizedBounds,
  WIN_HELPER
} from '../electron/main/ask-logic.ts'
import { actionPrompt, emptyMessage, languageName, selectionFileName } from '../src/ask/prompts.ts'

const screen = { x: 0, y: 0, width: 1920, height: 1040 }

test('the ask window opens just below and right of the pointer', () => {
  assert.deepEqual(askBounds({ x: 500, y: 300 }, screen, 460, 240), { x: 514, y: 314, width: 460, height: 240 })
})

test('near the right or bottom edge it opens on the other side of the pointer, and stays on the screen', () => {
  assert.deepEqual(askBounds({ x: 1800, y: 1000 }, screen, 460, 240), { x: 1326, y: 746, width: 460, height: 240 })
  // a second screen to the left, with negative coordinates
  const left = { x: -1280, y: 0, width: 1280, height: 1000 }
  const b = askBounds({ x: -1270, y: 5 }, left, 460, 240)
  assert.ok(b.x >= left.x && b.x + b.width <= left.x + left.width)
  assert.ok(b.y >= left.y && b.y + b.height <= left.y + left.height)
  // a window taller than the screen is cut to fit
  assert.equal(askBounds({ x: 10, y: 10 }, { x: 0, y: 0, width: 800, height: 400 }, 460, 640).height, 400)
})

test('resizing keeps the top edge, but never runs off the bottom of the screen', () => {
  const at = { x: 100, y: 200, width: 460, height: 240 }
  assert.deepEqual(resizedBounds(at, screen, 400), { x: 100, y: 200, width: 460, height: 400 })
  assert.deepEqual(resizedBounds({ ...at, y: 900 }, screen, 400), { x: 100, y: 640, width: 460, height: 400 })
  assert.equal(resizedBounds(at, screen, 20).height, 180) // a minimum
})

test('the selection: plain line ends, blank is nothing, long text is cut and says so', () => {
  assert.deepEqual(cleanSelection('Hi Maya,\r\nsee you at 3.\r\n'), { text: 'Hi Maya,\nsee you at 3.\n', truncated: false })
  assert.deepEqual(cleanSelection('  \r\n\t'), { text: '', truncated: false })
  const long = cleanSelection('x'.repeat(MAX_SELECTION_CHARS + 5))
  assert.equal(long.text.length, MAX_SELECTION_CHARS)
  assert.equal(long.truncated, true)
})

test('app names and the file the selection is uploaded as', () => {
  assert.equal(appName('Notepad', 'Notepad'), 'Notepad')
  assert.equal(appName('', 'chrome.exe'), 'Chrome')
  assert.equal(appName(undefined, undefined), 'another app')
  assert.equal(selectionFileName('Notepad'), 'Notepad.txt')
  assert.equal(selectionFileName('Mail: Inbox / Maya'), 'Mail Inbox Maya.txt')
  assert.equal(selectionFileName('another app'), 'Selected text.txt')
  assert.equal(selectionFileName(''), 'Selected text.txt')
})

test('the shortcut key the helper waits for to be let go', () => {
  assert.equal(acceleratorKeyCode('CommandOrControl+Alt+Shift+A'), 65)
  assert.equal(acceleratorKeyCode('Ctrl+Shift+7'), 55)
  assert.equal(acceleratorKeyCode('Alt+F9'), 0x78)
  assert.equal(acceleratorKeyCode('Ctrl+Shift+Space'), 0x20)
  assert.equal(acceleratorKeyCode(''), 0)
})

test('capture reports are read tolerantly', () => {
  assert.deepEqual(
    parseCapture('{"result":"copied","window":"722396","pid":4120,"process":"Notepad","description":"Notepad"}'),
    { result: 'copied', window: '722396', app: 'Notepad', pid: 4120 }
  )
  assert.equal(parseCapture('{"result":"password","window":"1","pid":1,"process":"KeePass"}').result, 'password')
  assert.equal(parseCapture('{"result":"own","window":"1","pid":1,"process":"electron"}').result, 'own')
  assert.equal(parseCapture('').result, 'failed')
  assert.equal(parseCapture('garbage').result, 'failed')
  // a window id that isn't a number is never used to paste
  assert.equal(parseCapture('{"result":"copied","window":"1; rm -rf /"}').window, '')
  assert.equal(parsePaste('pasted\n'), 'pasted')
  assert.equal(parsePaste('password'), 'password')
  assert.equal(parsePaste(''), 'failed')
  assert.equal(parsePaste('{"result":"failed","error":"x"}'), 'failed')
})

test('the Windows helper waits for the keys, never copies from a password box, and answers one line per command', () => {
  assert.match(WIN_HELPER, /KeysUp\(1500, \$key\)/)
  assert.match(WIN_HELPER, /Test-Password/)
  assert.match(WIN_HELPER, /\[Console\]::In\.ReadLine\(\)/)
  assert.ok(!WIN_HELPER.includes('${'), 'no template placeholders in the PowerShell script')
})

test('macOS and Linux commands, and pasting back only into a numeric target', () => {
  assert.equal(captureCommand('win32'), null)
  assert.equal(captureCommand('darwin')?.file, 'osascript')
  assert.equal(captureCommand('linux')?.file, 'sh')
  assert.equal(pasteBackCommand('linux', '1234')?.args[1], 'xdotool windowactivate --sync 1234 key --clearmodifiers ctrl+v && echo pasted')
  assert.equal(pasteBackCommand('linux', '12; reboot'), null)
  assert.ok(pasteBackCommand('darwin', '4120')?.args.some((a) => a.includes('unix id is 4120')))
})

test('quick action prompts are short and plain', () => {
  assert.equal(actionPrompt('explain'), 'Explain the selected text in plain words.')
  assert.match(actionPrompt('translate', 'de'), /into German\. If it is already in German, translate it into English\./)
  assert.match(actionPrompt('translate', undefined), /into English/)
  assert.match(actionPrompt('reply'), /Give only the reply text\./)
  assert.equal(languageName('hi-IN'), 'Hindi')
  assert.equal(languageName('not a language!'), 'not a language!')
})

test('empty states say what happened, without em-dashes', () => {
  for (const kind of [undefined, 'nothing', 'password', 'failed', 'own-window']) {
    const text = emptyMessage(kind, 'Ctrl+Alt+Shift+A', 'win32')
    assert.ok(text.length > 10)
    assert.ok(!text.includes('—'))
  }
  assert.match(emptyMessage('nothing', 'Ctrl+Alt+Shift+A', 'win32'), /press Ctrl\+Alt\+Shift\+A/)
  assert.match(emptyMessage('failed', 'x', 'linux'), /xdotool/)
})
