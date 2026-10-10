/**
 * Every global shortcut Sentient registers, in one place. Fixed ones (New chat, Stop everything) and
 * the ones people can change or turn off in Settings > General (Share this window, Share a region, Push to talk,
 * Dictate into any app).
 * Global shortcuts work while Sentient is hidden in the tray, and take the keys away from other apps,
 * so defaults use Ctrl+Alt+Shift (Command+Option+Shift on a Mac), a combination apps rarely use.
 */
import { globalShortcut } from 'electron'
import type { ShellPrefs, ShortcutId, ShortcutInfo, ShortcutResult } from '../../src/types/bridge'
import { acceleratorProblem, type Platform } from '../../src/lib/accelerator'

export const NEW_CHAT_ACCELERATOR = 'CommandOrControl+Shift+Space'
export const STOP_ACCELERATOR = 'CommandOrControl+Alt+Shift+S'

export const SHORTCUTS: Record<ShortcutId, { label: string; description: string; default: string }> = {
  shareWindow: {
    label: 'Share this window',
    description: 'Adds a picture of the window you are using to a new chat.',
    default: 'CommandOrControl+Alt+Shift+W'
  },
  shareRegion: {
    label: 'Share a region',
    description: 'Drag over part of your screen to add it to a new chat.',
    default: 'CommandOrControl+Alt+Shift+R'
  },
  pushToTalk: {
    label: 'Push to talk',
    description: 'Hold it, speak, and let go. What you said goes to Sentient as a chat message.',
    default: 'CommandOrControl+Alt+Shift+T'
  },
  dictate: {
    label: 'Dictate into any app',
    description: 'Press it, speak, then press it again or pause. Your words are typed where your cursor is.',
    default: 'CommandOrControl+Alt+Shift+D'
  }
}

const IDS = Object.keys(SHORTCUTS) as ShortcutId[]
const FIXED = [NEW_CHAT_ACCELERATOR, STOP_ACCELERATOR]
const platform: Platform = process.platform === 'darwin' || process.platform === 'win32' ? process.platform : 'linux'

function tryRegister(accelerator: string, handler: () => void): boolean {
  try {
    return globalShortcut.register(accelerator, handler)
  } catch {
    return false // an accelerator Electron can't parse
  }
}

export class Shortcuts {
  private handlers: Partial<Record<ShortcutId, () => void>> = {}
  private active: Partial<Record<ShortcutId, string>> = {}
  private taken = new Set<ShortcutId>()

  constructor(
    private prefs: () => ShellPrefs,
    private setPrefs: (patch: ShellPrefs) => void,
    private onChange: () => void = () => undefined
  ) {}

  /** New chat and Stop everything: always on. */
  registerFixed(accelerator: string, handler: () => void): void {
    if (!tryRegister(accelerator, handler)) console.warn(`[shell] global shortcut ${accelerator} is taken by another app`)
  }

  /** Register the configurable shortcuts from the saved settings. */
  start(handlers: Record<ShortcutId, () => void>): void {
    this.handlers = handlers
    for (const id of IDS) {
      const accelerator = this.accelerator(id)
      if (!accelerator) continue
      if (tryRegister(accelerator, handlers[id])) this.active[id] = accelerator
      else {
        this.taken.add(id)
        console.warn(`[shell] global shortcut ${accelerator} (${id}) is taken by another app`)
      }
    }
  }

  /** The shortcut for `id`: what the user set, "" when turned off, else the default. */
  accelerator(id: ShortcutId): string {
    const saved = this.prefs().shortcuts?.[id]
    return typeof saved === 'string' ? saved : SHORTCUTS[id].default
  }

  list(): ShortcutInfo[] {
    return IDS.map((id) => ({
      id,
      label: SHORTCUTS[id].label,
      description: SHORTCUTS[id].description,
      accelerator: this.accelerator(id),
      defaultAccelerator: SHORTCUTS[id].default,
      taken: this.taken.has(id)
    }))
  }

  /** Change a shortcut: an accelerator, "" to turn it off, or null for the default. Keeps the old one when the new one can't be used. */
  set(id: ShortcutId, accelerator: string | null): ShortcutResult {
    if (!Object.hasOwn(SHORTCUTS, id)) return { ok: false, error: 'Unknown shortcut.', shortcuts: this.list() }
    const next = accelerator === null ? SHORTCUTS[id].default : String(accelerator).trim()
    if (next) {
      const others = [...FIXED, ...IDS.filter((o) => o !== id).map((o) => this.accelerator(o)).filter(Boolean)]
      const problem = acceleratorProblem(next, platform, others)
      if (problem) return { ok: false, error: problem, shortcuts: this.list() }
    }
    const previous = this.active[id]
    if (previous) globalShortcut.unregister(previous)
    delete this.active[id]
    const handler = this.handlers[id]
    if (next && handler) {
      if (!tryRegister(next, handler)) {
        if (previous && tryRegister(previous, handler)) this.active[id] = previous
        return { ok: false, error: 'Another app is already using this shortcut. Try a different one.', shortcuts: this.list() }
      }
      this.active[id] = next
    }
    this.taken.delete(id)
    const saved = { ...this.prefs().shortcuts }
    if (accelerator === null) delete saved[id]
    else saved[id] = next
    this.setPrefs({ shortcuts: saved })
    this.onChange()
    return { ok: true, shortcuts: this.list() }
  }

  stopAll(): void {
    globalShortcut.unregisterAll()
    this.active = {}
  }
}
