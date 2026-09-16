import { app } from 'electron'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import type { ShellPrefs } from '../../src/types/bridge'

export interface WindowState {
  x?: number
  y?: number
  width: number
  height: number
  maximized: boolean
}

interface Stored {
  prefs: ShellPrefs
  window?: WindowState
}

const file = () => join(app.getPath('userData'), 'shell-state.json')
let cache: Stored | null = null

function load(): Stored {
  if (cache) return cache
  try {
    cache = existsSync(file()) ? (JSON.parse(readFileSync(file(), 'utf8')) as Stored) : { prefs: {} }
  } catch {
    cache = { prefs: {} }
  }
  cache.prefs ??= {}
  return cache
}

function save(): void {
  try {
    mkdirSync(dirname(file()), { recursive: true })
    writeFileSync(file(), JSON.stringify(load(), null, 2))
  } catch {
    /* non-fatal */
  }
}

/** Tiny persisted state for the shell (window bounds, theme, tray prefs). */
export const shellState = {
  prefs: (): ShellPrefs => load().prefs,
  setPrefs(patch: ShellPrefs) {
    const s = load()
    s.prefs = { ...s.prefs, ...patch }
    save()
  },
  window: (): WindowState | undefined => load().window,
  setWindow(w: WindowState) {
    load().window = w
    save()
  }
}

export const THEME_BG = { dark: '#0b0b0f', light: '#f7f7f8' } as const
export const THEME_TITLEBAR = {
  dark: { color: '#0b0b0f', symbolColor: '#a1a1aa' },
  light: { color: '#f7f7f8', symbolColor: '#52525b' }
} as const

export function resolvedTheme(prefs: ShellPrefs, systemDark: boolean): 'dark' | 'light' {
  const t = prefs.theme ?? 'dark'
  return t === 'system' ? (systemDark ? 'dark' : 'light') : t
}
