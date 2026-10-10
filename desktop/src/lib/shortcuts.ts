/**
 * Global shortcuts as Electron accelerators ("CommandOrControl+Alt+Shift+D"): recording one from a key press in
 * Settings and showing it in plain words. Pure, with no imports, so `npm test` runs it in Node.
 */

export interface KeyPress {
  key: string
  code: string
  ctrlKey: boolean
  metaKey: boolean
  altKey: boolean
  shiftKey: boolean
}

const CODE_KEYS: Record<string, string> = {
  Space: 'Space',
  Enter: 'Enter',
  Tab: 'Tab',
  Backspace: 'Backspace',
  Delete: 'Delete',
  Insert: 'Insert',
  Home: 'Home',
  End: 'End',
  PageUp: 'PageUp',
  PageDown: 'PageDown',
  ArrowUp: 'Up',
  ArrowDown: 'Down',
  ArrowLeft: 'Left',
  ArrowRight: 'Right',
  Minus: '-',
  Equal: '=',
  BracketLeft: '[',
  BracketRight: ']',
  Backslash: '\\',
  Semicolon: ';',
  Quote: "'",
  Comma: ',',
  Period: '.',
  Slash: '/',
  Backquote: '`'
}

/** The accelerator key for a physical key (`KeyD` -> `D`), or null for keys a shortcut can't use. */
function keyFromCode(code: string): string | null {
  let m = /^Key([A-Z])$/.exec(code)
  if (m) return m[1]
  m = /^Digit([0-9])$/.exec(code)
  if (m) return m[1]
  m = /^F([1-9]|1[0-9]|2[0-4])$/.exec(code)
  if (m) return `F${m[1]}`
  return CODE_KEYS[code] ?? null
}

/**
 * A key press as an accelerator. `null` while only modifiers are held, `''` for a press that can't be a shortcut
 * (no Ctrl, Alt or Cmd, or an unsupported key). Cmd on a Mac and Ctrl elsewhere become `CommandOrControl`.
 */
export function acceleratorFromKeyPress(e: KeyPress, platform: string): string | null {
  if (['Control', 'Meta', 'Alt', 'Shift', 'AltGraph', 'OS'].includes(e.key)) return null
  const key = keyFromCode(e.code)
  if (!key) return ''
  const mac = platform === 'darwin'
  const mods: string[] = []
  if (mac ? e.metaKey : e.ctrlKey) mods.push('CommandOrControl')
  if (mac && e.ctrlKey) mods.push('Control')
  if (!mac && e.metaKey) mods.push('Super')
  if (e.altKey) mods.push('Alt')
  if (e.shiftKey) mods.push('Shift')
  const fKey = /^F\d+$/.test(key)
  const strong = mods.some((m) => m !== 'Shift')
  if (!strong && !fKey) return '' // Shift+letter is just typing
  return [...mods, key].join('+')
}

const NAMES: Record<string, { mac: string; other: string }> = {
  commandorcontrol: { mac: 'Cmd', other: 'Ctrl' },
  cmdorctrl: { mac: 'Cmd', other: 'Ctrl' },
  command: { mac: 'Cmd', other: 'Cmd' },
  cmd: { mac: 'Cmd', other: 'Cmd' },
  control: { mac: 'Control', other: 'Ctrl' },
  ctrl: { mac: 'Control', other: 'Ctrl' },
  alt: { mac: 'Option', other: 'Alt' },
  option: { mac: 'Option', other: 'Alt' },
  shift: { mac: 'Shift', other: 'Shift' },
  super: { mac: 'Cmd', other: 'Win' },
  meta: { mac: 'Cmd', other: 'Win' }
}

/** The keys of an accelerator as people say them: `Ctrl+Alt+Shift+D`, or `Cmd+Option+Shift+D` on a Mac. */
export function formatAccelerator(accelerator: string, platform: string): string {
  if (!accelerator.trim()) return 'Not set'
  const mac = platform === 'darwin'
  return accelerator
    .split('+')
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const name = NAMES[part.toLowerCase()]
      return name ? (mac ? name.mac : name.other) : part.length === 1 ? part.toUpperCase() : part
    })
    .join('+')
}
