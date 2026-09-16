#!/usr/bin/env node
/**
 * npm-side wrapper for `packaging/build_engine.py`: finds the project's Python
 * and freezes the engine into `desktop/build/engine`, which electron-builder
 * ships as `resources/engine`.
 *
 *   npm run package:engine              # freeze (reuses the bundle if nothing changed)
 *   npm run package:engine -- --clean   # from scratch
 *
 * Override the interpreter with SENTIENT_BUILD_PYTHON.
 */
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const repo = resolve(here, '..', '..')
const win = process.platform === 'win32'

const candidates = [
  process.env.SENTIENT_BUILD_PYTHON,
  process.env.SENTIENT_PYTHON,
  join(repo, '.venv', win ? 'Scripts/python.exe' : 'bin/python')
].filter(Boolean)

const python = candidates.find((p) => existsSync(p)) ?? (win ? 'python' : 'python3')
const args = [join(repo, 'packaging', 'build_engine.py'), ...process.argv.slice(2)]

console.log(`[engine] ${python} ${args.join(' ')}`)
const started = Date.now()
const r = spawnSync(python, args, { cwd: repo, stdio: 'inherit' })
if (r.error) {
  console.error(`[engine] could not run ${python}: ${r.error.message}`)
  process.exit(1)
}
console.log(`[engine] finished in ${((Date.now() - started) / 1000).toFixed(0)}s`)
process.exit(r.status ?? 1)
