// Unit tests for the pure helpers behind the share-a-window and share-a-region shortcuts (#172).
// Run with `npm test` (Node's test runner with TypeScript type stripping; no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  captureFileName,
  cleanRegion,
  fitWithin,
  MAX_SIDE,
  pickActiveWindow,
  rectFromDrag,
  regionInImage
} from '../electron/main/share-helpers.ts'
import { acceleratorFromKey, acceleratorProblem, formatAccelerator, normalizeAccelerator } from '../src/lib/accelerator.ts'

const display = { width: 1440, height: 900 }

test('a drag in any direction gives the same rectangle', () => {
  const want = { x: 100, y: 50, width: 200, height: 120 }
  assert.deepEqual(rectFromDrag({ x: 100, y: 50 }, { x: 300, y: 170 }), want)
  assert.deepEqual(rectFromDrag({ x: 300, y: 170 }, { x: 100, y: 50 }), want)
  assert.deepEqual(rectFromDrag({ x: 300, y: 50 }, { x: 100, y: 170 }), want)
})

test('a region is kept inside the display, rounded, and tiny ones are dropped', () => {
  assert.deepEqual(cleanRegion({ x: -20, y: 10.4, width: 120, height: 50.2 }, display), { x: 0, y: 10, width: 100, height: 51 })
  assert.deepEqual(cleanRegion({ x: 1400, y: 880, width: 500, height: 500 }, display), { x: 1400, y: 880, width: 40, height: 20 })
  assert.equal(cleanRegion({ x: 10, y: 10, width: 4, height: 300 }, display), null)
  assert.equal(cleanRegion({ x: 2000, y: 10, width: 100, height: 100 }, display), null)
  assert.equal(cleanRegion(null, display), null)
  assert.equal(cleanRegion({ x: 1, y: 2, width: Number.NaN, height: 3 }, display), null)
  assert.equal(cleanRegion({ x: '1', y: 2, width: 30, height: 30 }, display), null)
})

test('a region maps onto a high-density screenshot', () => {
  const image = { width: 2880, height: 1800 } // scale factor 2
  assert.deepEqual(regionInImage({ x: 100, y: 50, width: 200, height: 120 }, display, image), { x: 200, y: 100, width: 400, height: 240 })
  // fractional scale (1.25) rounds outwards so nothing the user selected is cut off
  const scaled = regionInImage({ x: 101, y: 33, width: 9, height: 9 }, { width: 1536, height: 864 }, { width: 1920, height: 1080 })
  assert.deepEqual(scaled, { x: 126, y: 41, width: 12, height: 12 })
  // the whole display is the whole image, never past its edge
  assert.deepEqual(regionInImage({ x: 0, y: 0, width: 1440, height: 900 }, display, image), { x: 0, y: 0, ...image })
})

test('images are scaled down to the longest side, never up', () => {
  assert.deepEqual(fitWithin({ width: 3840, height: 2160 }), { width: MAX_SIDE, height: 1080 })
  assert.deepEqual(fitWithin({ width: 1000, height: 4000 }, 2000), { width: 500, height: 2000 })
  assert.deepEqual(fitWithin({ width: 640, height: 480 }), { width: 640, height: 480 })
})

test("the window shared is the front-most one that isn't Sentient's", () => {
  const sources = [
    { id: 'window:111:0', name: 'Sentient' },
    { id: 'window:222:0', name: 'Inbox - Mail' },
    { id: 'window:333:0', name: 'Notes' }
  ]
  assert.equal(pickActiveWindow(sources, ['window:111:0'])?.name, 'Inbox - Mail')
  // Sentient's own window, whatever the suffix, and windows of other Sentient windows (overlays) are skipped
  assert.equal(pickActiveWindow(sources, ['window:111:1', 'window:222:0'])?.name, 'Notes')
  // with Sentient in the tray the first window is the one in front
  assert.equal(pickActiveWindow(sources.slice(1), [])?.name, 'Inbox - Mail')
})

test('untitled, minimized and desktop surfaces are never picked', () => {
  const sources = [
    { id: 'window:1:0', name: '' },
    { id: 'window:2:0', name: 'Program Manager' },
    { id: 'window:3:0', name: 'Paint', empty: true },
    { id: 'window:4:0', name: 'Calculator' }
  ]
  assert.equal(pickActiveWindow(sources, [])?.name, 'Calculator')
  assert.equal(pickActiveWindow(sources.slice(0, 3), []), null)
  assert.equal(pickActiveWindow([], []), null)
})

test('capture file names come from the window title and are safe on every system', () => {
  assert.equal(captureFileName('window', 'Inbox - Mail'), 'Inbox - Mail.png')
  assert.equal(captureFileName('window', 'C:\\Users\\maya\\report.docx: Word  <draft>'), 'C Users maya report.docx Word draft.png')
  assert.equal(captureFileName('window', '...hidden'), 'hidden.png')
  assert.equal(captureFileName('window', '   '), 'Shared window.png')
  assert.equal(captureFileName('window', 'x'.repeat(200)).length, 84)
  assert.equal(captureFileName('region', 'Inbox - Mail'), 'Screen region.png')
})

test('shortcuts are compared in one spelling per platform', () => {
  assert.equal(normalizeAccelerator('CommandOrControl+Shift+Alt+w', 'win32'), 'Control+Alt+Shift+W')
  assert.equal(normalizeAccelerator('Ctrl+Alt+Shift+W', 'win32'), 'Control+Alt+Shift+W')
  assert.equal(normalizeAccelerator('CmdOrCtrl+Option+Shift+W', 'darwin'), 'Command+Alt+Shift+W')
  assert.equal(normalizeAccelerator('W', 'win32'), null)
  assert.equal(normalizeAccelerator('Ctrl+Hyper+W', 'win32'), null)
})

test('a shortcut needs a real modifier and must not clash with another one', () => {
  const others = ['CommandOrControl+Shift+Space', 'CommandOrControl+Alt+Shift+S']
  assert.equal(acceleratorProblem('CommandOrControl+Alt+Shift+W', 'win32', others), null)
  assert.equal(acceleratorProblem('Super+F9', 'linux', others), null)
  assert.match(acceleratorProblem('Shift+W', 'win32', others) ?? '', /Ctrl, Alt or the Windows key/)
  assert.match(acceleratorProblem('Ctrl+Shift+Space', 'win32', others) ?? '', /already uses/)
  assert.match(acceleratorProblem('Ctrl+', 'win32', others) ?? '', /letter, number or function key/)
})

const key = (code: string, k: string, mods: Partial<{ ctrl: boolean; meta: boolean; alt: boolean; shift: boolean }> = {}) => ({
  code,
  key: k,
  ctrlKey: !!mods.ctrl,
  metaKey: !!mods.meta,
  altKey: !!mods.alt,
  shiftKey: !!mods.shift
})

test('a recorded key press becomes a portable shortcut', () => {
  assert.equal(acceleratorFromKey(key('KeyW', 'W', { ctrl: true, alt: true, shift: true }), 'win32'), 'CommandOrControl+Alt+Shift+W')
  // Option changes the character on a Mac; the physical key still counts
  assert.equal(acceleratorFromKey(key('KeyW', '∑', { meta: true, alt: true }), 'darwin'), 'CommandOrControl+Alt+W')
  assert.equal(acceleratorFromKey(key('Digit2', '@', { ctrl: true, shift: true }), 'linux'), 'CommandOrControl+Shift+2')
  assert.equal(acceleratorFromKey(key('Space', ' ', { ctrl: true, meta: true }), 'darwin'), 'CommandOrControl+Control+Space')
  assert.equal(acceleratorFromKey(key('F9', 'F9', { meta: true }), 'win32'), 'Super+F9')
  // only modifiers held so far
  assert.equal(acceleratorFromKey(key('ShiftLeft', 'Shift', { shift: true }), 'win32'), null)
})

test('shortcuts read the way each system writes them', () => {
  assert.equal(formatAccelerator('CommandOrControl+Alt+Shift+W', 'win32'), 'Ctrl+Alt+Shift+W')
  assert.equal(formatAccelerator('CommandOrControl+Alt+Shift+W', 'darwin'), '⌘+⌥+⇧+W')
})
