import { app } from 'electron'
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'

/** Sentient's data folder: SENTIENT_HOME or ~/.sentient (mirrors sentient/paths.py). */
export function sentientHome(): string {
  const raw = process.env.SENTIENT_HOME?.trim()
  if (!raw) return join(homedir(), '.sentient')
  const expanded = raw === '~' || raw.startsWith('~/') || raw.startsWith('~\\') ? join(homedir(), raw.slice(1)) : raw
  return resolve(expanded)
}

export const homePaths = {
  home: () => sentientHome(),
  logs: () => join(sentientHome(), 'logs'),
  files: () => join(sentientHome(), 'files'),
  workspace: () => join(sentientHome(), 'workspace'),
  skills: () => join(sentientHome(), 'skills')
}

/**
 * Find the repository root in development: the nearest ancestor containing
 * `sentient/__main__.py`. Starts from the app path and this file's folder so both
 * `electron-vite dev` and `npx electron desktop/out/main/index.js` work.
 */
export function findRepoRoot(): string | null {
  const starts = [app.getAppPath(), __dirname, process.cwd()]
  for (const start of starts) {
    let dir = resolve(start)
    for (let i = 0; i < 8; i++) {
      if (existsSync(join(dir, 'sentient', '__main__.py'))) return dir
      const parent = dirname(dir)
      if (parent === dir) break
      dir = parent
    }
  }
  return null
}

/** Frozen engine shipped by electron-builder: resources/engine/sentient-engine[.exe]. */
export const ENGINE_EXE = process.platform === 'win32' ? 'sentient-engine.exe' : 'sentient-engine'

export function bundledEngine(): string | null {
  const candidates = [
    join(process.resourcesPath, 'engine', ENGINE_EXE),
    // `npm run build` + `npx electron out/main/index.js` (unpackaged, but staged)
    join(app.getAppPath(), 'build', 'engine', ENGINE_EXE)
  ]
  return candidates.find((p) => existsSync(p)) ?? null
}

export interface EngineChoice {
  /** Executable to spawn. */
  command: string
  /** Arguments before the host/port flags. */
  args: string[]
  source: 'env-engine' | 'frozen' | 'env-python' | 'venv' | 'python-path'
  cwd: string
  /** True when this is the self-contained engine (no Python on the machine). */
  frozen: boolean
}

/**
 * How to start the engine, in both worlds:
 *
 *  - development: `.venv/Scripts/python.exe -m sentient serve` from the repo root
 *  - packaged:    `resources/engine/sentient-engine.exe serve`
 *
 * `SENTIENT_ENGINE` (a frozen build) and `SENTIENT_PYTHON` (an interpreter) override
 * both, in that order, so either mode can be tested from the other.
 */
export function resolveEngine(): EngineChoice {
  const repo = findRepoRoot()
  const home = sentientHome()

  const envEngine = process.env.SENTIENT_ENGINE?.trim()
  if (envEngine) return { command: envEngine, args: ['serve'], source: 'env-engine', cwd: home, frozen: true }

  const envPython = process.env.SENTIENT_PYTHON?.trim()
  if (envPython) {
    return {
      command: envPython,
      args: ['-m', 'sentient', 'serve'],
      source: 'env-python',
      cwd: repo ?? home,
      frozen: false
    }
  }

  // A packaged app ships its own engine; prefer it over anything found on the machine.
  if (app.isPackaged) {
    const bundled = bundledEngine()
    if (bundled) return { command: bundled, args: ['serve'], source: 'frozen', cwd: home, frozen: true }
  }

  if (repo) {
    const venv =
      process.platform === 'win32'
        ? join(repo, '.venv', 'Scripts', 'python.exe')
        : join(repo, '.venv', 'bin', 'python')
    if (existsSync(venv)) {
      return { command: venv, args: ['-m', 'sentient', 'serve'], source: 'venv', cwd: repo, frozen: false }
    }
  }

  // Unpackaged but staged (scripts/smoke.mjs against a built-but-not-installed app).
  const staged = bundledEngine()
  if (staged) return { command: staged, args: ['serve'], source: 'frozen', cwd: home, frozen: true }

  return {
    command: process.platform === 'win32' ? 'python' : 'python3',
    args: ['-m', 'sentient', 'serve'],
    source: 'python-path',
    cwd: repo ?? home,
    frozen: false
  }
}
