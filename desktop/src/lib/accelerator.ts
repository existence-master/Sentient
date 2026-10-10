/**
 * Global shortcut strings in Electron's accelerator format ("CommandOrControl+Alt+Shift+W").
 * Pure, with no imports: shared by the main process (validation before registering) and Settings
 * (recording and showing a shortcut), and unit-tested with `npm test`.
 */

export type Platform = 'win32' | 'darwin' | 'linux' | 'web'

const MODIFIER_ORDER = ['CommandOrControl', 'Command', 'Control', 'Alt', 'Shift', 'Super'] as const
type Modifier = (typeof MODIFIER_ORDER)[number]

const MODIFIER_ALIASES: Record<string, Modifier> = {
  commandorcontrol: 'CommandOrControl',
  cmdorctrl: 'CommandOrControl',
  command: 'Command',
  cmd: 'Command',
  control: 'Control',
  ctrl: 'Control',
  alt: 'Alt',
  option: 'Alt',
  altgr: 'Alt',
  shift: 'Shift',
  super: 'Super',
  meta: 'Super'
}

const NAMED_KEYS = ['Space', 'Tab', 'Backspace', 'Delete', 'Insert', 'Home', 'End', 'PageUp', 'PageDown', 'Up', 'Down', 'Left', 'Right']

/** The key part of an accelerator, normalised ("w" -> "W"), or null when Sentient doesn't accept it. */
function normalizeKey(raw: string): string | null {
  const k = raw.trim()
  if (/^[a-z0-9]$/i.test(k)) return k.toUpperCase()
  if (/^f([1-9]|1[0-9]|2[0-4])$/i.test(k)) return k.toUpperCase()
  const named = NAMED_KEYS.find((n) => n.toLowerCase() === k.toLowerCase())
  if (named) return named
  if (/^[`\-=[\]\\;',./]$/.test(k)) return k
  return null
}

interface Parsed {
  modifiers: Modifier[]
  key: string
}

function parse(accelerator: string, platform: Platform): Parsed | null {
  const parts = accelerator.split('+').map((p) => p.trim())
  // "Ctrl++" style (a literal plus key) is not supported; every part must be non-empty
  if (parts.length < 2 || parts.some((p) => !p)) return null
  const key = normalizeKey(parts[parts.length - 1])
  if (!key) return null
  const mods = new Set<Modifier>()
  for (const p of parts.slice(0, -1)) {
    const m = MODIFIER_ALIASES[p.toLowerCase()]
    if (!m) return null
    // CommandOrControl is Command on macOS and Control elsewhere
    mods.add(m === 'CommandOrControl' ? (platform === 'darwin' ? 'Command' : 'Control') : m)
  }
  return { modifiers: MODIFIER_ORDER.filter((m) => mods.has(m)), key }
}

/** One canonical spelling for this platform, for comparing two shortcuts ("Ctrl+Shift+w" == "CommandOrControl+Shift+W" on Windows). */
export function normalizeAccelerator(accelerator: string, platform: Platform): string | null {
  const p = parse(accelerator, platform)
  return p ? [...p.modifiers, p.key].join('+') : null
}

/**
 * Why Sentient won't use this shortcut, in plain words, or null when it is fine. A global shortcut
 * takes the keys away from every other app, so it needs Ctrl, Alt, Command or the Windows key
 * (Shift alone would swallow capital letters).
 */
export function acceleratorProblem(accelerator: string, platform: Platform, others: string[] = []): string | null {
  const p = parse(accelerator, platform)
  if (!p) return 'Use a letter, number or function key together with Ctrl, Alt or Shift.'
  if (!p.modifiers.some((m) => m !== 'Shift')) return 'Add Ctrl, Alt or the Windows key (Command or Option on a Mac), so typing in other apps keeps working.'
  const mine = [...p.modifiers, p.key].join('+')
  if (others.some((o) => normalizeAccelerator(o, platform) === mine)) return 'Sentient already uses this shortcut for something else.'
  return null
}

/** Minimal keyboard event shape, so this stays testable without a DOM. */
export interface KeyLike {
  key: string
  code: string
  ctrlKey: boolean
  metaKey: boolean
  altKey: boolean
  shiftKey: boolean
}

/**
 * The accelerator for a key press while recording a shortcut, or null while only modifiers are held.
 * Uses `code` for letters and digits so Shift and Option don't change the key ("Alt+Shift+2", not "@").
 * Ctrl on Windows and Linux, and Command on macOS, become CommandOrControl so the setting reads the same everywhere.
 */
export function acceleratorFromKey(e: KeyLike, platform: Platform): string | null {
  let key: string | null = null
  const letter = /^Key([A-Z])$/.exec(e.code)
  const digit = /^(?:Digit|Numpad)([0-9])$/.exec(e.code)
  if (letter) key = letter[1]
  else if (digit) key = digit[1]
  else if (e.key === ' ' || e.code === 'Space') key = 'Space'
  else if (e.key.startsWith('Arrow')) key = e.key.slice(5)
  else key = normalizeKey(e.key)
  if (!key) return null
  const mac = platform === 'darwin'
  const mods: string[] = []
  if (mac ? e.metaKey : e.ctrlKey) mods.push('CommandOrControl')
  if (mac && e.ctrlKey) mods.push('Control')
  if (e.altKey) mods.push('Alt')
  if (e.shiftKey) mods.push('Shift')
  if (!mac && e.metaKey) mods.push('Super')
  return [...mods, key].join('+')
}

const MAC_SYMBOLS: Record<Modifier, string> = { CommandOrControl: '⌘', Command: '⌘', Control: '⌃', Alt: '⌥', Shift: '⇧', Super: '⌘' }
const PC_NAMES: Record<Modifier, string> = { CommandOrControl: 'Ctrl', Command: 'Win', Control: 'Ctrl', Alt: 'Alt', Shift: 'Shift', Super: 'Win' }

/** "Ctrl+Alt+Shift+W" on Windows and Linux, "⌘+⌥+⇧+W" on macOS (split on "+" to draw the keys). */
export function formatAccelerator(accelerator: string, platform: Platform): string {
  const p = parse(accelerator, platform)
  if (!p) return accelerator
  const names = platform === 'darwin' ? MAC_SYMBOLS : PC_NAMES
  return [...p.modifiers.map((m) => names[m]), p.key].join('+')
}
