// Push to talk and dictation into any app (#169): the pure helpers of the shell, the pill and Settings.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  encodePowerShell,
  FIRST_REPEAT_MS,
  pasteCommands,
  pasteOutcome,
  pillBounds,
  RELEASE_GAP_MS,
  ShortcutPresses
} from '../electron/main/dictation-logic.ts'
import { acceleratorProblem, normalizeAccelerator } from '../src/lib/accelerator.ts'
import { meterLevel, rms, SilenceDetector } from '../src/pill/silence.ts'

// ------------------------------------------------------------------ holding a shortcut
test('a held shortcut repeats, and is let go when the repeats stop', () => {
  const p = new ShortcutPresses()
  assert.equal(p.trigger(0), 'press')
  assert.equal(p.held, false)
  assert.equal(p.trigger(500), 'repeat') // the keyboard delay before repeating
  let last = 500
  for (let t = 533; t < 2000; t += 33) {
    assert.equal(p.trigger(t), 'repeat')
    last = t
  }
  assert.equal(p.held, true)
  assert.equal(p.released(last + 40), false)
  assert.equal(p.released(last + RELEASE_GAP_MS + 1), true)
})

test('a slow keyboard delay still counts as holding', () => {
  const p = new ShortcutPresses()
  p.trigger(0)
  assert.equal(p.trigger(FIRST_REPEAT_MS - 50), 'repeat')
})

test('a second press later is a new press', () => {
  const p = new ShortcutPresses()
  assert.equal(p.trigger(0), 'press')
  assert.equal(p.released(5000), false) // never repeated: a tap, not a hold
  assert.equal(p.trigger(5000), 'press')
  // after a hold was let go, the next press is new too
  p.trigger(5400)
  assert.equal(p.trigger(6000 + RELEASE_GAP_MS + 1), 'press')
})

// ------------------------------------------------------------------ shortcut choice
test('push to talk and dictate defaults differ from every other shortcut', () => {
  const ours = ['CommandOrControl+Alt+Shift+T', 'CommandOrControl+Alt+Shift+D']
  const others = ['CommandOrControl+Shift+Space', 'CommandOrControl+Alt+Shift+S', 'CommandOrControl+Alt+Shift+W', 'CommandOrControl+Alt+Shift+R']
  for (const platform of ['win32', 'darwin', 'linux'] as const) {
    const all = [...ours, ...others].map((a) => normalizeAccelerator(a, platform))
    assert.equal(new Set(all).size, all.length)
    for (const acc of ours) assert.equal(acceleratorProblem(acc, platform, others), null)
  }
})

// ------------------------------------------------------------------ typing into another app
test('the paste command per platform', () => {
  const [win] = pasteCommands('win32', {})
  assert.equal(win.file, 'powershell.exe')
  const script = Buffer.from(win.args[win.args.indexOf('-EncodedCommand') + 1], 'base64').toString('utf16le')
  assert.match(script, /IsPassword/)
  assert.match(script, /SendWait\('\^v'\)/)
  const [mac] = pasteCommands('darwin', {})
  assert.equal(mac.file, 'osascript')
  assert.ok(mac.args.some((a) => a.includes('AXSecureTextField')))
  assert.deepEqual(pasteCommands('linux', {}).map((c) => c.file), ['xdotool'])
  assert.deepEqual(pasteCommands('linux', { WAYLAND_DISPLAY: 'wayland-0' }).map((c) => c.file), ['wtype', 'xdotool'])
})

test('PowerShell encoding round-trips', () => {
  assert.equal(Buffer.from(encodePowerShell("Write-Output 'é ok'"), 'base64').toString('utf16le'), "Write-Output 'é ok'")
})

test('a password box is never typed into', () => {
  assert.equal(pasteOutcome('password\r\n'), 'password')
  assert.equal(pasteOutcome('pasted\r\n'), 'pasted')
  assert.equal(pasteOutcome(''), 'pasted') // xdotool and wtype print nothing
})

test('the pill sits centered near the bottom of the screen', () => {
  assert.deepEqual(pillBounds({ x: 1920, y: 0, width: 1920, height: 1040 }, 300, 56), { x: 2730, y: 912, width: 300, height: 56 })
})

// ------------------------------------------------------------------ pause detection
const speech = 0.08
const quiet = 0.002

test('stops after speech and then a pause', () => {
  const d = new SilenceDetector(1000)
  for (let t = 0; t < 600; t += 50) assert.equal(d.feed(speech, 50), null)
  assert.equal(d.heardSpeech, true)
  let verdict = null
  let waited = 0
  while (!verdict && waited < 3000) {
    verdict = d.feed(quiet, 50)
    waited += 50
  }
  assert.equal(verdict, 'stop')
  assert.equal(waited, 1000)
})

test('a short pause mid-sentence does not stop it', () => {
  const d = new SilenceDetector(1500)
  for (let t = 0; t < 500; t += 50) d.feed(speech, 50)
  for (let t = 0; t < 1000; t += 50) assert.equal(d.feed(quiet, 50), null)
  for (let t = 0; t < 500; t += 50) assert.equal(d.feed(speech, 50), null)
})

test('nobody speaking ends with nothing; 0 waits for the shortcut', () => {
  const d = new SilenceDetector(2000)
  let verdict = null
  for (let t = 0; t < 9000 && !verdict; t += 50) verdict = d.feed(quiet, 50)
  assert.equal(verdict, 'nothing')
  const manual = new SilenceDetector(0)
  for (let t = 0; t < 600; t += 50) manual.feed(speech, 50)
  for (let t = 0; t < 20000; t += 50) assert.equal(manual.feed(quiet, 50), null)
  manual.setSilence(500) // push to talk when the shell can't tell the key is held
  let later = null
  for (let t = 0; t < 1000 && !later; t += 50) later = manual.feed(quiet, 50)
  assert.equal(later, 'stop')
})

test('a cough is not speech', () => {
  const d = new SilenceDetector(500)
  d.feed(speech, 50)
  d.feed(speech, 50)
  for (let t = 0; t < 2000; t += 50) assert.equal(d.feed(quiet, 50), null)
  assert.equal(d.heardSpeech, false)
})

test('loudness and the level meter', () => {
  assert.equal(rms([]), 0)
  assert.ok(Math.abs(rms([0.5, -0.5, 0.5, -0.5]) - 0.5) < 1e-9)
  assert.equal(meterLevel(0), 0)
  assert.equal(meterLevel(0.0005), 0)
  assert.equal(meterLevel(1), 1)
  assert.ok(meterLevel(0.05) > 0.5 && meterLevel(0.05) < 1)
})
