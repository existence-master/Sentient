/**
 * Ask about selected text, in any app (#173).
 *
 * Only runs when the user presses its shortcut (Settings > General). The shell presses the app's own copy keys,
 * reads the text, and puts the old clipboard back right away. A small window opens by the pointer with the text it
 * read and quick actions; the window sends the text to the engine only when the user picks one, as outside content
 * that marks the chat (ADR 0018, ADR 0023). Copy puts an answer on the clipboard; Replace selection pastes it over the
 * selection in the app it came from and then puts the clipboard back again.
 */
import { BrowserWindow, clipboard, nativeTheme, screen } from 'electron'
import { execFile, spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { join } from 'node:path'
import { createInterface } from 'node:readline'
import type { AskState } from '../../src/types/bridge'
import {
  acceleratorKeyCode,
  askBounds,
  captureCommand,
  cleanSelection,
  COPY_WAIT_MS,
  HELPER_IDLE_MS,
  parseCapture,
  parsePaste,
  pasteBackCommand,
  resizedBounds,
  WIN_HELPER,
  type CaptureReport
} from './ask-logic'
import { CH } from './channels'
import { restoreClipboard, saveClipboard } from './clipboard-keep'
import { encodePowerShell, RESTORE_CLIPBOARD_MS } from './dictation-logic'
import { resolvedTheme, shellState, THEME_BG } from './prefs'

const WIDTH = 460
const START_HEIGHT = 236
const MAX_HEIGHT = 640

export interface AskHost {
  /** The shortcut as it is set now. */
  shortcut(): string
  /** Open a chat in the main window. */
  openChat(sessionId: string): void
  /** Say what happened in the tray tooltip. */
  notice(text: string): void
}

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

/** The Windows helper: one PowerShell that stays open for a while, because starting PowerShell takes seconds. */
class WindowsHelper {
  private proc: ChildProcessWithoutNullStreams | null = null
  private ready: Promise<boolean> | null = null
  private waiting: Array<(line: string) => void> = []
  private chain: Promise<unknown> = Promise.resolve()
  private idle: NodeJS.Timeout | null = null

  private start(): Promise<boolean> {
    if (this.proc && this.ready) return this.ready
    const proc = spawn(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encodePowerShell(WIN_HELPER)],
      { windowsHide: true }
    )
    this.proc = proc
    this.ready = new Promise<boolean>((resolve) => {
      let started = false
      const lines = createInterface({ input: proc.stdout })
      lines.on('line', (line) => {
        if (!started) {
          if (line.trim() === 'ready') {
            started = true
            resolve(true)
          }
          return
        }
        this.waiting.shift()?.(line)
      })
      proc.stderr.on('data', (chunk: Buffer) => console.warn(`[shell] ask helper: ${String(chunk).slice(0, 300)}`))
      proc.on('error', (err) => console.warn(`[shell] ask helper could not start: ${err.message}`))
      proc.on('exit', () => {
        if (this.proc === proc) {
          this.proc = null
          this.ready = null
        }
        for (const answer of this.waiting.splice(0)) answer('')
        resolve(false)
      })
    })
    return this.ready
  }

  /** Send one command and wait for its one-line answer; "" when the helper failed or took too long. */
  request(command: string, timeoutMs = 8000): Promise<string> {
    const run = async (): Promise<string> => {
      if (this.idle) clearTimeout(this.idle)
      if (!(await this.start()) || !this.proc) return ''
      const proc = this.proc
      const answer = await new Promise<string>((resolve) => {
        const timer = setTimeout(() => {
          this.stop() // a stuck helper is replaced next time
          resolve('')
        }, timeoutMs)
        this.waiting.push((line) => {
          clearTimeout(timer)
          resolve(line)
        })
        proc.stdin.write(`${command}\n`)
      })
      this.idle = setTimeout(() => this.stop(), HELPER_IDLE_MS)
      return answer
    }
    const next = this.chain.then(run, run)
    this.chain = next.catch(() => undefined)
    return next
  }

  stop(): void {
    if (this.idle) clearTimeout(this.idle)
    this.idle = null
    const proc = this.proc
    this.proc = null
    this.ready = null
    for (const answer of this.waiting.splice(0)) answer('')
    if (proc && proc.exitCode === null) {
      try {
        proc.stdin.end()
        proc.kill()
      } catch {
        /* already gone */
      }
    }
  }
}

function runOnce(file: string, args: string[]): Promise<string> {
  return new Promise((resolve) => {
    execFile(file, args, { timeout: 8000, windowsHide: true }, (err, stdout) => {
      if (err) console.warn(`[shell] ask: ${file} failed: ${err.message}`)
      resolve(err ? '' : String(stdout))
    })
  })
}

interface Captured {
  text: string
  app: string
  target: string
  empty?: 'nothing' | 'password' | 'failed' | 'own-window'
}

export class AskController {
  private win: BrowserWindow | null = null
  private loaded: Promise<void> | null = null
  private state: AskState | null = null
  private target = ''
  private busy = false
  private disposed = false
  private helper = process.platform === 'win32' ? new WindowsHelper() : null

  constructor(private host: AskHost) {}

  /** The shortcut was pressed: copy the selection and show it. Esc or the close button hides the window. */
  async trigger(): Promise<void> {
    if (this.busy || this.disposed) return
    this.busy = true
    try {
      // When a Sentient window has focus there is nothing to copy from another app. On Windows the helper checks
      // the window in front itself, which is more reliable than Electron's focus state.
      const own = !this.helper && BrowserWindow.getFocusedWindow() !== null
      const win = await this.ensureWindow()
      const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
      win.setBounds(askBounds(screen.getCursorScreenPoint(), display.workArea, WIDTH, START_HEIGHT))
      this.send({ phase: 'reading' })
      if (!win.isVisible()) win.showInactive() // shown at once, without taking focus from the app being copied from
      const captured: Captured = own ? { text: '', app: 'Sentient', target: '', empty: 'own-window' } : await this.capture()
      const clean = cleanSelection(captured.text)
      this.target = clean.text ? captured.target : ''
      this.send({
        phase: 'ready',
        text: clean.text,
        app: captured.app,
        truncated: clean.truncated,
        ...(clean.text ? {} : { empty: captured.empty ?? 'nothing' }),
        canReplace: !!this.target && (process.platform === 'win32' || pasteBackCommand(process.platform, this.target) !== null),
        shortcut: this.host.shortcut(),
        capturedAt: new Date().toISOString()
      })
      if (clean.text) this.host.notice(`read the text you selected in ${captured.app}`)
      if (win.isDestroyed()) return
      win.show()
      win.focus()
    } catch (err) {
      console.warn(`[shell] ask: ${String(err)}`)
    } finally {
      this.busy = false
    }
  }

  /** Copy the selection with the app's own copy keys and put the clipboard back. Nothing else is read. */
  private async capture(): Promise<Captured> {
    const saved = await saveClipboard()
    if (saved !== null) clipboard.clear() // so a copy that does nothing (no selection) reads as nothing
    let report: CaptureReport
    if (this.helper) {
      report = parseCapture(await this.helper.request(`capture ${acceleratorKeyCode(this.host.shortcut())} ${process.pid}`))
    } else {
      const cmd = captureCommand(process.platform)
      report = parseCapture(cmd ? await runOnce(cmd.file, cmd.args) : '')
    }
    let text = ''
    if (report.result === 'copied') {
      const until = Date.now() + COPY_WAIT_MS
      for (;;) {
        try {
          text = await clipboard.readText()
        } catch {
          text = ''
        }
        if (text || Date.now() > until) break
        await delay(40)
      }
    }
    await restoreClipboard(saved, text)
    const empty = text ? undefined : report.result === 'copied' ? 'nothing' : report.result === 'own' ? 'own-window' : report.result
    return { text, app: report.app, target: report.window, ...(empty ? { empty } : {}) }
  }

  /** Paste `text` over the selection in the app it came from; the clipboard is put back afterwards. */
  async replace(text: string): Promise<'pasted' | 'password' | 'failed'> {
    const target = this.target
    if (!target || !text) return 'failed'
    const saved = await saveClipboard()
    try {
      await clipboard.writeText(text)
    } catch {
      return 'failed'
    }
    this.hide()
    let outcome: 'pasted' | 'password' | 'failed' = 'failed'
    if (this.helper) outcome = parsePaste(await this.helper.request(`paste ${target}`))
    else {
      const cmd = pasteBackCommand(process.platform, target)
      if (cmd) outcome = parsePaste(await runOnce(cmd.file, cmd.args))
    }
    if (outcome === 'pasted') {
      setTimeout(() => void restoreClipboard(saved, text), RESTORE_CLIPBOARD_MS)
    } else {
      // A password box: nothing typed, the clipboard goes back. Failed: the answer stays on the clipboard to paste.
      if (outcome === 'password') await restoreClipboard(saved, text)
      const win = this.win
      if (win && !win.isDestroyed()) {
        win.show()
        win.focus()
      }
    }
    return outcome
  }

  async copy(text: string): Promise<boolean> {
    try {
      await clipboard.writeText(String(text ?? ''))
      return true
    } catch {
      return false
    }
  }

  openChat(sessionId: string): void {
    this.hide()
    this.host.openChat(sessionId)
  }

  currentState(): AskState | null {
    return this.state
  }

  resize(height: number): void {
    const win = this.win
    if (!win || win.isDestroyed() || !Number.isFinite(height)) return
    // The width is always set, never read back: on Windows a frameless window's bounds can come back a little
    // wider than they were set, and feeding them back on every resize made the window grow.
    const bounds = { ...win.getBounds(), width: WIDTH }
    const display = screen.getDisplayMatching(bounds)
    const next = resizedBounds(bounds, display.workArea, Math.min(MAX_HEIGHT, Math.ceil(height)))
    const now = win.getBounds()
    if (Math.abs(next.height - now.height) > 1 || next.y !== now.y || now.width !== WIDTH) win.setBounds(next)
  }

  hide(): void {
    const win = this.win
    if (win && !win.isDestroyed() && win.isVisible()) win.hide()
  }

  ownsWebContents(id: number): boolean {
    return !!this.win && !this.win.isDestroyed() && this.win.webContents.id === id
  }

  /** The shortcut was turned off or Stop everything: close the window and the helper. */
  close(): void {
    this.hide()
    this.helper?.stop()
  }

  dispose(): void {
    this.disposed = true
    this.helper?.stop()
    if (this.win && !this.win.isDestroyed()) this.win.destroy()
    this.win = null
  }

  private send(state: AskState): void {
    this.state = state
    const win = this.win
    if (win && !win.isDestroyed()) win.webContents.send(CH.askState, state)
  }

  private ensureWindow(): Promise<BrowserWindow> {
    const existing = this.win
    if (existing && !existing.isDestroyed() && this.loaded) return this.loaded.then(() => existing)
    const theme = resolvedTheme(shellState.prefs(), nativeTheme.shouldUseDarkColors)
    const win = new BrowserWindow({
      width: WIDTH,
      height: START_HEIGHT,
      show: false,
      frame: false,
      resizable: false,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      skipTaskbar: true,
      alwaysOnTop: true,
      backgroundColor: THEME_BG[theme],
      title: 'Ask Sentient',
      webPreferences: {
        preload: join(__dirname, '../preload/index.js'),
        contextIsolation: true,
        sandbox: true,
        nodeIntegration: false,
        webviewTag: false,
        spellcheck: true
      }
    })
    win.setMenu(null)
    if (process.platform === 'darwin') win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true, skipTransformProcessType: true })
    win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
    win.webContents.on('will-navigate', (event) => event.preventDefault())
    win.on('close', (event) => {
      if (this.disposed) return
      event.preventDefault() // kept for the next press; hidden instead
      win.hide()
    })
    win.on('closed', () => {
      if (this.win === win) {
        this.win = null
        this.loaded = null
      }
    })
    this.win = win
    this.loaded = new Promise<void>((resolve, reject) => {
      win.webContents.once('did-finish-load', () => resolve())
      win.webContents.once('did-fail-load', (_e, code, description) => reject(new Error(`${code} ${description}`)))
    })
    if (process.env.ELECTRON_RENDERER_URL) void win.loadURL(`${process.env.ELECTRON_RENDERER_URL}/ask.html`)
    else void win.loadFile(join(__dirname, '../renderer/ask.html'))
    return this.loaded.then(
      () => win,
      (error: unknown) => {
        if (this.win === win) {
          this.win = null
          this.loaded = null
          if (!win.isDestroyed()) win.destroy()
        }
        throw error
      }
    )
  }
}
