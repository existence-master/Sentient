// Unit tests for what closing the main window does (#439). Run with `npm test` (no Electron needed).
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { closeAction, type CloseState } from '../electron/main/close-logic.ts'

const base: CloseState = { quitting: false, smoke: false, hasTray: true, minimizeToTray: true, platform: 'win32' }

test('minimize to tray on hides the window and keeps running', () => {
  assert.equal(closeAction(base), 'hide')
  assert.equal(closeAction({ ...base, minimizeToTray: undefined }), 'hide') // the config default is on
  assert.equal(closeAction({ ...base, platform: 'darwin' }), 'hide')
  assert.equal(closeAction({ ...base, platform: 'linux' }), 'hide')
})

test('minimize to tray off quits the app, even with hidden helper windows open', () => {
  assert.equal(closeAction({ ...base, minimizeToTray: false }), 'quit')
  assert.equal(closeAction({ ...base, minimizeToTray: false, platform: 'linux' }), 'quit')
})

test('no tray icon means nothing could bring a hidden window back, so it quits', () => {
  assert.equal(closeAction({ ...base, hasTray: false }), 'quit')
})

test('macOS keeps the app in the Dock when the window closes', () => {
  assert.equal(closeAction({ ...base, minimizeToTray: false, platform: 'darwin' }), 'close')
  assert.equal(closeAction({ ...base, hasTray: false, platform: 'darwin' }), 'close')
})

test('a quit already under way, or a smoke run, just lets the window close', () => {
  assert.equal(closeAction({ ...base, quitting: true }), 'close')
  assert.equal(closeAction({ ...base, quitting: true, minimizeToTray: false }), 'close')
  assert.equal(closeAction({ ...base, smoke: true, hasTray: false }), 'close')
})
