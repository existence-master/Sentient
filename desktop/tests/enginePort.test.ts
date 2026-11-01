// Unit tests for which port the engine starts on (#544). Run with `npm test` (no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { choosePort, type PortProbe, rememberedPort } from '../electron/main/port-logic.ts'

/** A fake probe: `taken` ports are busy, and fresh picks come from `next`. Records what was asked. */
function probe(taken: number[] = [], next = 51234): PortProbe & { checked: number[]; picks: number } {
  const p = {
    checked: [] as number[],
    picks: 0,
    isFree: async (port: number) => {
      p.checked.push(port)
      return !taken.includes(port)
    },
    pickFree: async () => {
      p.picks++
      return next
    }
  }
  return p
}

test('a remembered port that is free is used again', async () => {
  const p = probe()
  assert.equal(await choosePort(49800, p), 49800)
  assert.deepEqual(p.checked, [49800])
  assert.equal(p.picks, 0)
})

test('a remembered port that is taken falls back to a fresh free port', async () => {
  const p = probe([49800], 51234)
  assert.equal(await choosePort(49800, p), 51234)
  assert.equal(p.picks, 1)
})

test('no remembered port picks a free one, as before', async () => {
  for (const missing of [undefined, null, 0]) {
    const p = probe()
    assert.equal(await choosePort(missing, p), 51234)
    assert.deepEqual(p.checked, [])
    assert.equal(p.picks, 1)
  }
})

test('a broken stored value is ignored and never checked', async () => {
  for (const bad of ['49800', 49800.5, -1, 80, 1023, 65536, Number.NaN, Infinity, true, {}, [49800]]) {
    const p = probe()
    assert.equal(await choosePort(bad, p), 51234, `stored ${String(bad)}`)
    assert.deepEqual(p.checked, [])
  }
})

test('a probe that fails counts as taken, so the engine still starts', async () => {
  const p = probe()
  p.isFree = async () => {
    throw new Error('EACCES')
  }
  assert.equal(await choosePort(49800, p), 51234)
})

test('rememberedPort accepts only whole numbers from 1024 to 65535', () => {
  assert.equal(rememberedPort(1024), 1024)
  assert.equal(rememberedPort(65535), 65535)
  assert.equal(rememberedPort(1023), undefined)
  assert.equal(rememberedPort('7777'), undefined)
  assert.equal(rememberedPort(undefined), undefined)
})
