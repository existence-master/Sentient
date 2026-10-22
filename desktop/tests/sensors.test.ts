// Unit tests for the tray's microphone and camera indicator (#170). Run with `npm test` (no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  drawLiveDot,
  liveSensors,
  LOCAL_MIC_ID,
  parseSensors,
  sensorMenuLabel,
  sensorTooltip,
  type LiveSensor
} from '../electron/main/sensors-logic.ts'

const glasses: LiveSensor = {
  id: 'voice:1', sensor: 'mic', device_id: 'g1', device: "Maya's glasses", device_kind: 'glasses',
  purpose: 'Voice chat', since: '2026-10-18T10:00:00+00:00', can_stop: true
} // prettier-ignore

test('the engine list is read defensively', () => {
  assert.deepEqual(parseSensors(null), { mic: false, camera: false, active: [], camera_off: [] })
  const s = parseSensors({ type: 'sensors', active: [glasses, { id: 1 }, { id: 'x', sensor: 'radio' }], camera_off: [{ device_id: 'g1', device: 'G' }] })
  assert.deepEqual(s.active, [glasses])
  assert.equal(s.mic, true)
  assert.equal(s.camera, false)
  assert.equal(s.camera_off.length, 1)
})

test('push to talk shows before the engine has said so, and only once', () => {
  const now = new Date('2026-10-18T10:00:00Z')
  const local = liveSensors(null, 'Push to talk', now)
  assert.equal(local.length, 1)
  assert.equal(local[0].id, LOCAL_MIC_ID)
  assert.equal(local[0].device, 'This computer')
  const fromEngine = { ...local[0], since: 'earlier' }
  assert.deepEqual(liveSensors({ mic: true, camera: false, active: [fromEngine], camera_off: [] }, 'Push to talk', now), [fromEngine])
  assert.deepEqual(liveSensors(null, null), [])
})

test('tooltip and menu say which device and what for', () => {
  assert.equal(sensorTooltip([]), null)
  assert.equal(sensorTooltip([glasses]), "Sentient: microphone on (Maya's glasses, Voice chat)")
  const photo: LiveSensor = { ...glasses, id: 'p', sensor: 'camera', purpose: 'Taking a photo' }
  assert.equal(sensorTooltip([glasses, photo]), "Sentient: microphone and camera on (Maya's glasses)")
  const here = liveSensors(null, 'Dictation')[0]
  assert.equal(sensorTooltip([glasses, here]), 'Sentient: microphone on (2 devices)')
  assert.equal(sensorMenuLabel(here), 'Turn off microphone: this computer (Dictation)')
  assert.equal(sensorMenuLabel(photo), "Turn off camera: Maya's glasses (Taking a photo)")
})

test('the live dot is drawn in the bottom right corner', () => {
  const w = 16
  const blank = new Uint8Array(w * w * 4)
  const dotted = drawLiveDot(blank, w, w)
  assert.equal(blank.every((v) => v === 0), true) // the original is left alone
  const px = (x: number, y: number) => Array.from(dotted.slice((y * w + x) * 4, (y * w + x) * 4 + 4))
  assert.deepEqual(px(0, 0), [0, 0, 0, 0])
  const r = Math.max(3, Math.round(w * 0.22))
  assert.deepEqual(px(w - r - 1, w - r - 1), [38, 38, 220, 255]) // red centre (BGRA)
})
