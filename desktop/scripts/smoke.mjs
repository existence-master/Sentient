#!/usr/bin/env node
/**
 * Visual smoke test: launch the built app on a route, capture a PNG, exit.
 *
 *   npm run build
 *   node scripts/smoke.mjs <route> <out.png> [WIDTHxHEIGHT] [--keep-live]
 *
 * Honors SENTIENT_HOME (use a throwaway profile). See electron/main/smoke.ts.
 * Before it starts the app on a SENTIENT_HOME it pauses that home's tasks and turns off its messaging deliveries
 * (scripts/seed_safety.py, issue #302), so a seeded demo never runs a task or messages anyone. --keep-live skips that.
 */
import { spawn, spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const argv = process.argv.slice(2)
const keepLive = argv.includes('--keep-live')
const [rawRoute = '/', out, size = '1280x820'] = argv.filter((a) => a !== '--keep-live')
// Accept "settings/models" or "#/settings/models". (Git Bash rewrites a leading "/" into a
// Windows path unless MSYS_NO_PATHCONV=1, so prefer the slash-less form there.)
const route = `/${rawRoute.replace(/^#?\/*/, '')}`
if (!out) {
  console.error('usage: node scripts/smoke.mjs <route> <out.png> [WIDTHxHEIGHT] [--keep-live]')
  process.exit(64)
}
const main = join(root, 'out', 'main', 'index.js')
if (!existsSync(main)) {
  console.error('Build first: npm run build')
  process.exit(1)
}

// The same Python the app would start the engine with (electron/main/paths.ts): SENTIENT_PYTHON, else the repo's .venv.
function enginePython() {
  if (process.env.SENTIENT_PYTHON?.trim()) return process.env.SENTIENT_PYTHON.trim()
  const venv = join(root, '..', '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')
  if (existsSync(venv)) return venv
  return process.platform === 'win32' ? 'python' : 'python3'
}

const home = process.env.SENTIENT_HOME?.trim()
if (home && !keepLive) {
  const quiet = spawnSync(enginePython(), [join(here, 'seed_safety.py'), home], { stdio: 'inherit' })
  if (quiet.status !== 0) {
    console.error('[smoke] could not pause the tasks and messaging deliveries in SENTIENT_HOME (pass --keep-live to skip)')
    process.exit(1)
  }
} else if (!home) {
  console.warn('[smoke] SENTIENT_HOME is not set: the app uses your real data folder')
}

const require = createRequire(import.meta.url)
const electron = require('electron')
const env = { ...process.env, SENTIENT_SMOKE_SCREENSHOT: resolve(out), SENTIENT_SMOKE_ROUTE: route, SENTIENT_SMOKE_SIZE: size }
delete env.ELECTRON_RUN_AS_NODE

const started = Date.now()
const child = spawn(electron, [main], { env, stdio: 'inherit', windowsHide: false })
const timeout = setTimeout(() => {
  console.error('[smoke] timed out; killing')
  child.kill()
}, Number(process.env.SENTIENT_SMOKE_HARD_TIMEOUT_MS) || 240_000)
child.on('exit', (code) => {
  clearTimeout(timeout)
  console.log(`[smoke] exit ${code} after ${((Date.now() - started) / 1000).toFixed(1)}s -> ${resolve(out)}`)
  process.exit(code ?? 1)
})
