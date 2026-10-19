// Unit tests for the Add a device dialog's pairing code (#477): when the Wi-Fi listener starts, stops or moves,
// the code on screen points at the wrong address and a new one is made. Run with `npm test`.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { lanPollMs, pairingFreshness } from '../src/features/devices/pairingFreshness.ts'
import type { DeviceLanInfo, DevicePairing } from '../src/lib/types.ts'

const LAN = 'https://192.168.1.24:7443/node/'
const off: DeviceLanInfo = { enabled: false, running: false, port: 7443, urls: [], web_urls: [], fingerprint: null, error: null }
const starting: DeviceLanInfo = { ...off, enabled: true }
const on: DeviceLanInfo = { ...starting, running: true, urls: ['wss://192.168.1.24:7443/ws/node'], web_urls: [LAN], fingerprint: 'AB:CD' }
const failed: DeviceLanInfo = { ...starting, error: 'Could not listen on port 7443. Another program may be using it.' }

const code = (web_url: string, fingerprint: string | null = null): DevicePairing => ({
  code: '482913',
  expires_at: '2026-10-19T12:10:00+00:00',
  lan_enabled: !!fingerprint,
  urls: [],
  web_url: `${web_url}#code=482913`,
  fingerprint,
  qr: 'sentient://pair?url=x&code=482913'
})
const localCode = code('http://127.0.0.1:8784/node/')
const lanCode = code(LAN, 'AB:CD')

test('a local code is fine while Wi-Fi devices are off', () => {
  assert.deepEqual(pairingFreshness(localCode, off), { starting: false, stale: false, target: 'local' })
})

test('turning Wi-Fi on waits for the listener instead of showing the local QR', () => {
  assert.deepEqual(pairingFreshness(localCode, starting), { starting: true, stale: false, target: 'local' })
})

test('once the listener runs, the local code is stale and the new one must point at the LAN address', () => {
  assert.deepEqual(pairingFreshness(localCode, on), { starting: false, stale: true, target: LAN })
  assert.equal(pairingFreshness(lanCode, on).stale, false)
})

test('turning Wi-Fi off makes the LAN code stale, back to this computer', () => {
  assert.deepEqual(pairingFreshness(lanCode, off), { starting: false, stale: true, target: 'local' })
})

test('a new network address or port makes the code stale', () => {
  const moved: DeviceLanInfo = { ...on, web_urls: ['https://10.0.0.5:7443/node/'] }
  assert.deepEqual(pairingFreshness(lanCode, moved), { starting: false, stale: true, target: 'https://10.0.0.5:7443/node/' })
})

test('a listener that could not start keeps the local code (the error is shown instead)', () => {
  assert.deepEqual(pairingFreshness(localCode, failed), { starting: false, stale: false, target: 'local' })
})

test('no network address: the engine falls back to the local code, which is not stale', () => {
  assert.equal(pairingFreshness(localCode, { ...on, web_urls: [], urls: [] }).stale, false)
})

test('nothing is stale before the code or the listener status has loaded', () => {
  assert.equal(pairingFreshness(null, on).stale, false)
  assert.equal(pairingFreshness(localCode, undefined).stale, false)
})

test('engines without web_url fall back to the fingerprint', () => {
  const { web_url: _ignored, ...old } = lanCode
  assert.equal(pairingFreshness(old, off).stale, true)
  assert.equal(pairingFreshness(old, on).stale, false)
  assert.equal(pairingFreshness({ ...old, fingerprint: null }, off).stale, false)
  assert.equal(pairingFreshness({ ...old, fingerprint: null }, on).stale, true)
  // With device addresses on both sides, a moved listener is noticed too.
  const withUrls = { ...old, urls: ['wss://192.168.1.24:7443/ws/node'] }
  assert.equal(pairingFreshness(withUrls, on).stale, false)
  assert.equal(pairingFreshness(withUrls, { ...on, urls: ['wss://10.0.0.5:7443/ws/node'], web_urls: ['https://10.0.0.5:7443/node/'] }).stale, true)
})

test('the dialog polls quickly only while the listener starts', () => {
  assert.equal(lanPollMs(starting), 500)
  assert.equal(lanPollMs(on), 5000)
  assert.equal(lanPollMs(off), 5000)
  assert.equal(lanPollMs(failed), 5000)
  assert.equal(lanPollMs(undefined), 5000)
})
