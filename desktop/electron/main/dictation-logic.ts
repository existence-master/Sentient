/**
 * Pure helpers for push to talk and dictation into any app (#169). No imports, so `npm test` runs them in Node.
 */

export type DictationMode = 'talk' | 'dictate'

/** A held key repeats after the keyboard delay (up to 1 s on Windows), so a trigger sooner than this is the same press. */
export const FIRST_REPEAT_MS = 1100
/** Once a held shortcut repeats, a pause this long means it was let go. */
export const RELEASE_GAP_MS = 450
/** The longest single dictation, so a forgotten microphone never stays on. */
export const MAX_RECORDING_MS = 120_000
/** After the paste keystroke, wait this long before putting the old clipboard back. */
export const RESTORE_CLIPBOARD_MS = 600

/**
 * Tells a new press of a global shortcut from the autorepeat of one that is held down, and notices when a held
 * shortcut is let go (its repeats stop). Electron reports only key presses for global shortcuts, never releases.
 */
export class ShortcutPresses {
  private last = Number.NEGATIVE_INFINITY
  private repeats = 0

  trigger(now: number): 'press' | 'repeat' {
    const gap = now - this.last
    this.last = now
    if (gap < (this.repeats > 0 ? RELEASE_GAP_MS : FIRST_REPEAT_MS)) {
      this.repeats++
      return 'repeat'
    }
    this.repeats = 0
    return 'press'
  }

  /** True while the shortcut is known to be held (it has repeated). */
  get held(): boolean {
    return this.repeats > 0
  }

  /** True once a shortcut that was repeating has stopped: it was let go. */
  released(now: number): boolean {
    return this.repeats > 0 && now - this.last > RELEASE_GAP_MS
  }
}

export interface PasteCommand {
  file: string
  args: string[]
}

const WIN_PASTE = [
  "$ErrorActionPreference = 'Stop'",
  'try {',
  '  Add-Type -AssemblyName UIAutomationClient',
  '  $f = [System.Windows.Automation.AutomationElement]::FocusedElement',
  "  if ($f -and $f.Current.IsPassword) { Write-Output 'password'; exit 0 }",
  '} catch {}',
  'Add-Type -AssemblyName System.Windows.Forms',
  "[System.Windows.Forms.SendKeys]::SendWait('^v')",
  "Write-Output 'pasted'"
].join('\n')

const MAC_PASTE = [
  'tell application "System Events"',
  'try',
  'set f to value of attribute "AXFocusedUIElement" of (first application process whose frontmost is true)',
  'if (value of attribute "AXSubrole" of f) is "AXSecureTextField" then return "password"',
  'end try',
  'keystroke "v" using command down',
  'end tell',
  'return "pasted"'
]

/** UTF-16LE base64 for PowerShell's -EncodedCommand, so the script needs no quoting. */
export function encodePowerShell(script: string): string {
  const bytes: number[] = []
  for (let i = 0; i < script.length; i++) {
    const code = script.charCodeAt(i)
    bytes.push(code & 0xff, code >> 8)
  }
  let binary = ''
  for (const b of bytes) binary += String.fromCharCode(b)
  return btoa(binary)
}

/**
 * Commands that press the paste keys in the app that has focus, tried in order. Each prints `password` instead of
 * pasting when it can tell a password box has focus (Windows and macOS), else `pasted`.
 */
export function pasteCommands(platform: string, env: Record<string, string | undefined>): PasteCommand[] {
  if (platform === 'win32') {
    return [{ file: 'powershell.exe', args: ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encodePowerShell(WIN_PASTE)] }]
  }
  if (platform === 'darwin') return [{ file: 'osascript', args: MAC_PASTE.flatMap((line) => ['-e', line]) }]
  const xdotool = { file: 'xdotool', args: ['key', '--clearmodifiers', 'ctrl+v'] }
  const wtype = { file: 'wtype', args: ['-M', 'ctrl', 'v', '-m', 'ctrl'] }
  return env.WAYLAND_DISPLAY ? [wtype, xdotool] : [xdotool]
}

/** What a paste command that exited cleanly reported. Linux tools print nothing when they work. */
export function pasteOutcome(stdout: string): 'pasted' | 'password' {
  return stdout.trim().toLowerCase().endsWith('password') ? 'password' : 'pasted'
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

/** Where the listening pill goes: centered near the bottom of the screen the pointer is on. */
export function pillBounds(workArea: Rect, width: number, height: number, margin = 72): Rect {
  return {
    x: Math.round(workArea.x + (workArea.width - width) / 2),
    y: Math.round(workArea.y + workArea.height - height - margin),
    width,
    height
  }
}
