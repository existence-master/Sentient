import { getBridge } from './bridge'
import { ACCENTS, type AccentName, type ThemePreference } from './types'

export const ACCENT_META: Record<AccentName, { label: string; color: string }> = {
  sentient: { label: 'Sentient', color: '#F1A21D' },
  violet: { label: 'Violet', color: '#8b5cf6' },
  blue: { label: 'Blue', color: '#4a9eff' },
  emerald: { label: 'Emerald', color: '#10b981' },
  rose: { label: 'Rose', color: '#f43f5e' }
}

export function normalizeAccent(accent: string | null | undefined): AccentName {
  return (ACCENTS as readonly string[]).includes(accent ?? '') ? (accent as AccentName) : 'sentient'
}

const TITLEBAR = {
  dark: { color: '#0b0b0f', symbolColor: '#a1a1aa' },
  light: { color: '#f6f6f7', symbolColor: '#52525c' }
} as const

const media = typeof window !== 'undefined' ? window.matchMedia('(prefers-color-scheme: dark)') : null

export function resolveTheme(pref: ThemePreference): 'dark' | 'light' {
  if (pref === 'system') return media?.matches === false ? 'light' : 'dark'
  return pref
}

let current: { theme: ThemePreference; accent: AccentName } = { theme: 'dark', accent: 'sentient' }
let lastSynced = ''

/** Apply theme + accent to <html> and tell the shell (title bar overlay + window background). */
export function applyTheme(theme: ThemePreference, accent: string): 'dark' | 'light' {
  current = { theme, accent: normalizeAccent(accent) }
  const resolved = resolveTheme(theme)
  const root = document.documentElement
  root.classList.toggle('dark', resolved === 'dark')
  root.classList.toggle('light', resolved === 'light')
  root.dataset.accent = current.accent
  const key = `${theme}:${resolved}`
  if (key !== lastSynced) {
    lastSynced = key
    void getBridge().syncPrefs({ theme, titleBar: TITLEBAR[resolved] })
  }
  return resolved
}

media?.addEventListener('change', () => {
  if (current.theme === 'system') applyTheme(current.theme, current.accent)
})
