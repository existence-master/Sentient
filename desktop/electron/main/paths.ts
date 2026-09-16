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

export interface PythonChoice {
  command: string
  source: 'env' | 'venv' | 'bundled' | 'path'
  cwd: string
}

/** SENTIENT_PYTHON -> <repo>/.venv -> bundled runtime (packaged) -> `python` on PATH. */
export function resolvePython(): PythonChoice {
  const repo = findRepoRoot()
  const cwd = repo ?? sentientHome()
  const env = process.env.SENTIENT_PYTHON?.trim()
  if (env) return { command: env, source: 'env', cwd }
  if (repo) {
    const venv =
      process.platform === 'win32'
        ? join(repo, '.venv', 'Scripts', 'python.exe')
        : join(repo, '.venv', 'bin', 'python')
    if (existsSync(venv)) return { command: venv, source: 'venv', cwd }
  }
  if (app.isPackaged) {
    const bundled =
      process.platform === 'win32'
        ? join(process.resourcesPath, 'python', 'python.exe')
        : join(process.resourcesPath, 'python', 'bin', 'python3')
    if (existsSync(bundled)) return { command: bundled, source: 'bundled', cwd }
  }
  return { command: process.platform === 'win32' ? 'python' : 'python3', source: 'path', cwd }
}
