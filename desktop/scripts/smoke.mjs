#!/usr/bin/env node
/**
 * Visual smoke test: launch the built app on a route, capture a PNG, exit.
 *
 *   npm run build
 *   node scripts/smoke.mjs <route> <out.png> [WIDTHxHEIGHT]
 *
 * Honors SENTIENT_HOME (use a throwaway profile). See electron/main/smoke.ts.
 */
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const [rawRoute = '/', out, size = '1280x820'] = process.argv.slice(2)
// Accept "settings/models" or "#/settings/models". (Git Bash rewrites a leading "/" into a
// Windows path unless MSYS_NO_PATHCONV=1, so prefer the slash-less form there.)
const route = `/${rawRoute.replace(/^#?\/*/, '')}`
if (!out) {
  console.error('usage: node scripts/smoke.mjs <route> <out.png> [WIDTHxHEIGHT]')
  process.exit(64)
}
const main = join(root, 'out', 'main', 'index.js')
if (!existsSync(main)) {
  console.error('Build first: npm run build')
  process.exit(1)
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
