// Unit tests for which port the engine starts on (#544). Run with `npm test` (no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  choosePort,
  loadRememberedPort,
  type PortMemory,
  type PortProbe,
  rememberedPort,
  rememberPort
} from '../electron/main/port-logic.ts'

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

/** A fake settings file: one stored value, with a count of writes. */
function memory(initial?: unknown): PortMemory & { stored: unknown; saves: number } {
  const m = {
    stored: initial,
    saves: 0,
    load: () => m.stored,
    save: (port: number) => {
      m.saves++
      m.stored = port
    }
  }
  return m
}

test('a port the engine started on is remembered and used on the next launch', async () => {
  const m = memory()
  const first = await choosePort(loadRememberedPort(m), probe([], 49800))
  rememberPort(m, first)
  assert.equal(m.stored, 49800)
  const p = probe([], 51234)
  assert.equal(await choosePort(loadRememberedPort(m), p), 49800)
  assert.equal(p.picks, 0)
})

test('a taken port is replaced in memory by the new one', async () => {
  const m = memory(49800)
  const port = await choosePort(loadRememberedPort(m), probe([49800], 51234))
  rememberPort(m, port)
  assert.equal(m.stored, 51234)
  assert.equal(m.saves, 1)
})

test('the same port is not written again', () => {
  const m = memory(49800)
  rememberPort(m, 49800)
  assert.equal(m.saves, 0)
})

test('a settings file that cannot be read or written never throws', async () => {
  const broken: PortMemory = {
    load: () => {
      throw new Error('corrupt')
    },
    save: () => {
      throw new Error('read-only')
    }
  }
  assert.equal(loadRememberedPort(broken), undefined)
  assert.doesNotThrow(() => rememberPort(broken, 49800))
  assert.equal(loadRememberedPort(undefined), undefined)
  assert.doesNotThrow(() => rememberPort(undefined, 49800))
  assert.equal(await choosePort(loadRememberedPort(broken), probe()), 51234)
})
