/**
 * Push to talk and dictation into any app (#169).
 *
 * Two global shortcuts from `voice.dictation`. Push to talk: hold it, speak, let go; the words go to Sentient as a
 * chat message. Dictate: press it, speak, press it again (or pause); the cleaned-up words are typed into the app
 * that has focus.
 *
 * A small always-on-top "pill" window owns the microphone. It records only while it shows (with a level meter and
 * Esc to cancel), and sends the audio to the engine, where speech is recognized on this computer. The shell then
 * types the text with the clipboard and the paste keys, and puts the old clipboard back.
 */
import { BrowserWindow, clipboard, ClipboardItem, globalShortcut, screen, shell, systemPreferences } from 'electron'
import { execFile } from 'node:child_process'
import { join } from 'node:path'
import type {
  DictationCommand,
  DictationEvent,
  DictationShellSettings,
  DictationShortcutStatus,
  DictationStatus
} from '../../src/types/bridge'
import { CH } from './channels'
import {
  FIRST_REPEAT_MS,
  MAX_RECORDING_MS,
  pasteCommands,
  pasteOutcome,
  pillBounds,
  RESTORE_CLIPBOARD_MS,
  ShortcutPresses,
  shortcutProblem,
  type DictationMode
} from './dictation-logic'

export const DICTATION_DEFAULTS: DictationShellSettings = {
  pushToTalk: true,
  pushToTalkShortcut: 'CommandOrControl+Alt+Shift+T',
  dictate: true,
  dictateShortcut: 'CommandOrControl+Alt+Shift+D',
  stopAfterSilenceS: 2.5
}

/** `voice.dictation` from `GET /api/config` (snake_case) to the shell's settings. */
export function dictationSettingsFromConfig(d: Record<string, unknown> | undefined): DictationShellSettings {
  const str = (v: unknown, fallback: string) => (typeof v === 'string' ? v : fallback)
  const bool = (v: unknown, fallback: boolean) => (typeof v === 'boolean' ? v : fallback)
  return {
    pushToTalk: bool(d?.push_to_talk, DICTATION_DEFAULTS.pushToTalk),
    pushToTalkShortcut: str(d?.push_to_talk_shortcut, DICTATION_DEFAULTS.pushToTalkShortcut),
    dictate: bool(d?.dictate, DICTATION_DEFAULTS.dictate),
    dictateShortcut: str(d?.dictate_shortcut, DICTATION_DEFAULTS.dictateShortcut),
    stopAfterSilenceS: typeof d?.stop_after_silence_s === 'number' ? d.stop_after_silence_s : DICTATION_DEFAULTS.stopAfterSilenceS
  }
}

const MODES: DictationMode[] = ['talk', 'dictate']
const LABEL: Record<DictationMode, string> = { talk: 'Push to talk', dictate: 'Dictate' }
const CANCEL_KEY = 'Escape'
const PILL_WIDTH = 300
const PILL_HEIGHT = 56
const PASTE_KEYS = process.platform === 'darwin' ? 'Cmd+V' : 'Ctrl+V'
const MAC_PANES = {
  microphone: 'x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone',
  accessibility: 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'
} as const

export interface DictationHost {
  /** Shortcuts Sentient already uses: accelerator -> plain name. */
  reserved: Record<string, string>
  /** Push to talk heard something: send it to Sentient. */
  talk(text: string): void
  notify(title: string, body: string): void
  /** The microphone turned on or off (tray tooltip). */
  micChanged(on: boolean): void
}

type Phase = 'idle' | 'starting' | 'listening' | 'working'

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

export class DictationController {
  private settings: DictationShellSettings = DICTATION_DEFAULTS
  private paused = false
  private registered: Partial<Record<DictationMode, string>> = {}
  private problems: Partial<Record<DictationMode, string>> = {}
  private pill: BrowserWindow | null = null
  private pillLoaded: Promise<void> | null = null
  private phase: Phase = 'idle'
  private mode: DictationMode | null = null
  private presses: Record<DictationMode, ShortcutPresses> = { talk: new ShortcutPresses(), dictate: new ShortcutPresses() }
  private startSent = false
  private startedAt = 0
  private lastTrigger = 0
  private autoStopSent = false
  private watch: NodeJS.Timeout | null = null
  private hideTimer: NodeJS.Timeout | null = null
  private mic = false

  constructor(private host: DictationHost) {}

  get micOn(): boolean {
    return this.mic
  }

  // ---------------------------------------------------------------- shortcuts
  /** Register the shortcuts from `voice.dictation` (called on start and whenever the config changes). */
  apply(next: DictationShellSettings): DictationStatus {
    this.settings = { ...DICTATION_DEFAULTS, ...next }
    this.unregisterAll()
    if (this.paused) return this.status()
    const taken = { ...this.host.reserved }
    for (const mode of MODES) {
      const { enabled, accelerator } = this.shortcut(mode)
      if (!enabled) continue
      const problem = shortcutProblem(accelerator, process.platform, taken)
      if (problem) {
        this.problems[mode] = problem
        continue
      }
      let ok = false
      try {
        ok = globalShortcut.register(accelerator, () => this.trigger(mode))
      } catch {
        ok = false // Electron couldn't read the accelerator
      }
      if (ok) {
        this.registered[mode] = accelerator
        taken[accelerator] = LABEL[mode]
      } else {
        this.problems[mode] = 'Another app is using this shortcut. Pick a different one.'
        console.warn(`[shell] global shortcut ${accelerator} (${mode}) is taken by another app`)
      }
    }
    if (this.mode && !this.registered[this.mode]) this.cancel()
    if (MODES.some((m) => this.registered[m])) void this.ensurePill().catch(() => undefined) // ready before the first press
    return this.status()
  }

  /** While Settings records a new shortcut, the current ones must not start the microphone. */
  pause(paused: boolean): void {
    if (paused === this.paused) return
    this.paused = paused
    if (paused) {
      this.cancel()
      this.unregisterAll()
    } else this.apply(this.settings)
  }

  private unregisterAll(): void {
    for (const mode of MODES) {
      const acc = this.registered[mode]
      if (acc) globalShortcut.unregister(acc)
      delete this.registered[mode]
      delete this.problems[mode]
    }
  }

  status(): DictationStatus {
    const mac = process.platform === 'darwin'
    return {
      talk: this.shortcutStatus('talk'),
      dictate: this.shortcutStatus('dictate'),
      accessibility: mac ? systemPreferences.isTrustedAccessibilityClient(false) : null,
      microphone: mac ? systemPreferences.getMediaAccessStatus('microphone') : null
    }
  }

  openPermissionSettings(kind: 'microphone' | 'accessibility'): void {
    if (process.platform !== 'darwin' || !(kind in MAC_PANES)) return
    if (kind === 'accessibility') systemPreferences.isTrustedAccessibilityClient(true) // macOS's own prompt
    void shell.openExternal(MAC_PANES[kind])
  }

  private shortcut(mode: DictationMode): { enabled: boolean; accelerator: string } {
    const s = this.settings
    return mode === 'talk'
      ? { enabled: s.pushToTalk, accelerator: s.pushToTalkShortcut.trim() }
      : { enabled: s.dictate, accelerator: s.dictateShortcut.trim() }
  }

  private shortcutStatus(mode: DictationMode): DictationShortcutStatus {
    const { enabled, accelerator } = this.shortcut(mode)
    return { accelerator, enabled, registered: !!this.registered[mode], ...(this.problems[mode] ? { problem: this.problems[mode] } : {}) }
  }

  private trigger(mode: DictationMode): void {
    const now = Date.now()
    const kind = this.presses[mode].trigger(now)
    this.lastTrigger = now
    if (this.phase === 'idle') {
      if (kind === 'press') void this.start(mode)
      return
    }
    if (this.mode !== mode || kind === 'repeat') return // a held key repeating, or the other shortcut
    if (this.phase === 'starting' || this.phase === 'listening') this.stop()
  }

  // ---------------------------------------------------------------- one recording
  private async start(mode: DictationMode): Promise<void> {
    if (this.hideTimer) clearTimeout(this.hideTimer)
    this.phase = 'starting'
    this.mode = mode
    this.startSent = false
    this.autoStopSent = false
    this.startedAt = Date.now()
    try {
      globalShortcut.register(CANCEL_KEY, () => this.cancel())
    } catch {
      /* Esc stays with the other app; the pill's own button and Esc still cancel */
    }
    if (process.platform === 'darwin' && !(await this.macMicrophone())) {
      this.finish()
      this.host.notify(
        'Sentient can’t use the microphone',
        'Allow Sentient in System Settings > Privacy & Security > Microphone, then try again.'
      )
      return
    }
    let pill: BrowserWindow
    try {
      pill = await this.ensurePill()
    } catch {
      this.finish()
      return
    }
    if (this.phase !== 'starting' || this.mode !== mode) return // cancelled while getting ready
    const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
    pill.setBounds(pillBounds(display.workArea, PILL_WIDTH, PILL_HEIGHT))
    if (mode === 'talk' && process.platform === 'win32') {
      // With focus the pill sees the shortcut keys being let go; dictation keeps focus in the app it types into.
      pill.setFocusable(true)
      pill.show()
      pill.focus()
    } else {
      if (process.platform !== 'linux') pill.setFocusable(false)
      pill.showInactive()
    }
    const silenceMs = mode === 'dictate' ? Math.round(this.settings.stopAfterSilenceS * 1000) : 0
    this.send({ type: 'start', mode, silenceMs, maxMs: MAX_RECORDING_MS })
    this.startSent = true
    if (mode === 'talk') this.watchHold()
  }

  /** Push to talk: stop when the held shortcut is let go. When the shell can't tell it is held, a pause stops it. */
  private watchHold(): void {
    if (this.watch) clearInterval(this.watch)
    this.watch = setInterval(() => {
      if (this.mode !== 'talk' || (this.phase !== 'listening' && this.phase !== 'starting')) return
      const now = Date.now()
      const presses = this.presses.talk
      if (presses.released(now)) this.stop()
      else if (!presses.held && !this.autoStopSent && now - this.startedAt > FIRST_REPEAT_MS) {
        this.autoStopSent = true
        const silenceMs = Math.round(this.settings.stopAfterSilenceS * 1000)
        if (silenceMs > 0) this.send({ type: 'auto-stop', silenceMs })
      }
    }, 80)
  }

  /** Stop listening and use what was heard. */
  stop(): void {
    if (this.phase !== 'starting' && this.phase !== 'listening') return
    if (!this.startSent) {
      this.cancel() // nothing was recorded yet
      return
    }
    this.phase = 'working'
    this.send({ type: 'stop' })
  }

  /** Turn the microphone off and drop what was heard (Esc, Stop everything, the shortcut turned off). */
  cancel(): void {
    if (this.phase === 'idle') return
    this.send({ type: 'cancel' })
    this.finish()
  }

  private finish(lingerMs = 0): void {
    this.phase = 'idle'
    this.mode = null
    this.startSent = false
    if (this.watch) clearInterval(this.watch)
    this.watch = null
    globalShortcut.unregister(CANCEL_KEY)
    this.setMic(false)
    const pill = this.pill
    if (!pill || pill.isDestroyed()) return
    if (this.hideTimer) clearTimeout(this.hideTimer)
    const hide = () => {
      if (pill.isDestroyed() || this.phase !== 'idle') return
      pill.hide()
      if (process.platform !== 'linux') pill.setFocusable(false)
    }
    if (lingerMs > 0) this.hideTimer = setTimeout(hide, lingerMs)
    else hide()
  }

  private setMic(on: boolean): void {
    if (this.mic === on) return
    this.mic = on
    this.host.micChanged(on)
  }

  /** Events from the pill window; anything from another window is ignored. */
  onPillEvent(senderId: number, ev: DictationEvent): void {
    if (!this.pill || this.pill.isDestroyed() || this.pill.webContents.id !== senderId || !ev) return
    switch (ev.type) {
      case 'listening':
        if (this.phase === 'starting') this.phase = 'listening'
        this.setMic(this.phase !== 'idle')
        break
      case 'working':
        if (this.phase !== 'idle') this.phase = 'working'
        this.setMic(false)
        break
      case 'result': {
        const mode = this.mode
        if (this.phase === 'idle' || !mode) return
        this.finish()
        const text = String(ev.text ?? '').trim()
        if (!text) return
        if (mode === 'talk') this.host.talk(text)
        else void this.insert(text)
        break
      }
      case 'empty':
        this.finish(1400) // the pill says it didn't catch anything
        break
      case 'error':
        this.finish(3200) // the pill shows the error
        break
      case 'cancelled':
        this.finish()
        break
      case 'keyup':
        if (this.mode === 'talk' && Date.now() - this.startedAt > 250) this.stop()
        break
      case 'escape':
        this.cancel()
        break
    }
  }

  private send(cmd: DictationCommand): void {
    const pill = this.pill
    if (pill && !pill.isDestroyed()) pill.webContents.send(CH.dictationCommand, cmd)
  }

  private async macMicrophone(): Promise<boolean> {
    const state = systemPreferences.getMediaAccessStatus('microphone')
    if (state === 'granted') return true
    if (state === 'not-determined') return systemPreferences.askForMediaAccess('microphone')
    return false
  }

  // ---------------------------------------------------------------- the pill window
  ownsWebContents(id: number): boolean {
    return !!this.pill && !this.pill.isDestroyed() && this.pill.webContents.id === id
  }

  private ensurePill(): Promise<BrowserWindow> {
    const existing = this.pill
    if (existing && !existing.isDestroyed() && this.pillLoaded) return this.pillLoaded.then(() => existing)
    const pill = new BrowserWindow({
      width: PILL_WIDTH,
      height: PILL_HEIGHT,
      show: false,
      frame: false,
      transparent: true,
      resizable: false,
      movable: false,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      skipTaskbar: true,
      alwaysOnTop: true,
      focusable: false,
      hasShadow: false,
      backgroundColor: '#00000000',
      title: 'Sentient is listening',
      webPreferences: {
        preload: join(__dirname, '../preload/index.js'),
        contextIsolation: true,
        sandbox: true,
        nodeIntegration: false,
        webviewTag: false,
        spellcheck: false,
        backgroundThrottling: false
      }
    })
    pill.setAlwaysOnTop(true, 'screen-saver')
    if (process.platform === 'darwin') pill.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true, skipTransformProcessType: true })
    pill.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
    pill.webContents.on('will-navigate', (event) => event.preventDefault())
    pill.on('closed', () => {
      if (this.pill !== pill) return
      this.pill = null
      this.pillLoaded = null
      if (this.phase !== 'idle') this.finish()
    })
    this.pill = pill
    this.pillLoaded = new Promise<void>((resolve, reject) => {
      pill.webContents.once('did-finish-load', () => resolve())
      pill.webContents.once('did-fail-load', (_e, code, description) => reject(new Error(`${code} ${description}`)))
    })
    if (process.env.ELECTRON_RENDERER_URL) void pill.loadURL(`${process.env.ELECTRON_RENDERER_URL}/pill.html`)
    else void pill.loadFile(join(__dirname, '../renderer/pill.html'))
    return this.pillLoaded.then(
      () => pill,
      (error: unknown) => {
        // A pill that failed to load is dropped, so the next press builds a new one.
        if (this.pill === pill) {
          this.pill = null
          this.pillLoaded = null
          if (!pill.isDestroyed()) pill.destroy()
        }
        throw error
      }
    )
  }

  dispose(): void {
    this.cancel()
    this.unregisterAll()
    if (this.pill && !this.pill.isDestroyed()) this.pill.destroy()
    this.pill = null
  }

  // ---------------------------------------------------------------- typing into the focused app
  /** Type `text` where the cursor is: clipboard, paste keys, then the old clipboard back. Never into a password box. */
  private async insert(text: string): Promise<void> {
    const saved = await this.saveClipboard()
    try {
      await clipboard.writeText(text)
    } catch (err) {
      console.warn(`[shell] dictation: could not use the clipboard: ${String(err)}`)
      this.host.notify('Couldn’t type into that app', 'Sentient couldn’t use the clipboard. Please try again.')
      return
    }
    await delay(Math.max(0, this.lastTrigger + 350 - Date.now())) // the shortcut keys are let go first
    let outcome: 'pasted' | 'password' | 'failed'
    if (process.platform === 'darwin' && !systemPreferences.isTrustedAccessibilityClient(false)) {
      systemPreferences.isTrustedAccessibilityClient(true) // macOS's own "allow" prompt
      outcome = 'failed'
    } else {
      outcome = await this.pressPaste()
    }
    if (outcome === 'pasted') {
      setTimeout(() => void this.restoreClipboard(saved, text), RESTORE_CLIPBOARD_MS)
      return
    }
    if (outcome === 'password') {
      await this.restoreClipboard(saved, text)
      this.host.notify('Nothing was typed', 'A password box had focus, so Sentient left it alone.')
      return
    }
    // Typing failed: keep the words on the clipboard so they aren't lost.
    const how =
      process.platform === 'darwin'
        ? ' To let Sentient type for you, allow it in System Settings > Privacy & Security > Accessibility.'
        : process.platform === 'linux'
          ? ' Installing xdotool (or wtype on Wayland) lets Sentient type for you.'
          : ''
    this.host.notify('Couldn’t type into that app', `Your words are on the clipboard. Press ${PASTE_KEYS} to paste them.${how}`)
  }

  private pressPaste(): Promise<'pasted' | 'password' | 'failed'> {
    const commands = pasteCommands(process.platform, process.env)
    const run = (i: number): Promise<'pasted' | 'password' | 'failed'> => {
      const cmd = commands[i]
      if (!cmd) return Promise.resolve('failed')
      return new Promise((resolve) => {
        execFile(cmd.file, cmd.args, { timeout: 8000, windowsHide: true }, (err, stdout) => {
          if (!err) resolve(pasteOutcome(String(stdout)))
          else if ((err as NodeJS.ErrnoException).code === 'ENOENT') resolve(run(i + 1)) // not installed: try the next
          else {
            console.warn(`[shell] dictation: ${cmd.file} could not paste: ${err.message}`)
            resolve('failed')
          }
        })
      })
    }
    return run(0)
  }

  /** Everything on the clipboard (every format of every item), or null when it can't be read. */
  private async saveClipboard(): Promise<ClipboardItem[] | null> {
    try {
      const saved: ClipboardItem[] = []
      for (const item of await clipboard.read()) {
        const entries: Record<string, Blob | string | Electron.ClipboardBookmark> = {}
        for (const type of item.types) {
          try {
            entries[type] = await item.getType(type)
          } catch {
            /* a format that can't be read back */
          }
        }
        if (Object.keys(entries).length) saved.push(new ClipboardItem(entries))
      }
      return saved
    } catch {
      return null
    }
  }

  /** Put back what was copied before, unless something new was copied meanwhile. */
  private async restoreClipboard(saved: ClipboardItem[] | null, ours: string): Promise<void> {
    try {
      if (saved === null || (await clipboard.readText()) !== ours) return
      if (saved.length) await clipboard.write(saved)
      else clipboard.clear()
    } catch (err) {
      console.warn(`[shell] dictation: could not restore the clipboard: ${String(err)}`)
    }
  }
}
