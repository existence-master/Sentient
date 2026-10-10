/**
 * The desktop app as a Sentient device (docs/API.md §13).
 *
 * Connects to `WS /ws/node?token=<gateway token>`; the engine treats that as the built-in
 * node `desktop`. Capabilities: screen.capture, camera.photo, clipboard.read,
 * clipboard.write, notify.show.
 *
 * Privacy: the screen and camera are only used after the user said yes (asked once, the first
 * time, then remembered; switches on the Devices page). Every capture triggers a visible notice
 * in the app, the tray tooltip and a system notification when the window isn't focused.
 * Reconnects with backoff when the engine restarts.
 */
import { app, BrowserWindow, clipboard, desktopCapturer, dialog, Notification, screen } from 'electron'
import { writeFileSync } from 'node:fs'
import { hostname } from 'node:os'
import { join } from 'node:path'
import type { DesktopNodeState, ShellPrefs } from '../../src/types/bridge'

const CAPABILITIES = ['screen.capture', 'camera.photo', 'clipboard.read', 'clipboard.write', 'notify.show'] as const
const PING_MS = 25_000
const MAX_BACKOFF_MS = 30_000
const CAMERA_TIMEOUT_MS = 15_000

const CAMERA_HTML = `<!doctype html><meta charset="utf-8"><title>camera</title><script>
window.takePhoto = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
  try {
    const video = document.createElement('video')
    video.muted = true
    video.srcObject = stream
    await video.play()
    await new Promise((r) => setTimeout(r, 700))
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 1280
    canvas.height = video.videoHeight || 720
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
    return canvas.toDataURL('image/jpeg', 0.85)
  } finally {
    stream.getTracks().forEach((t) => t.stop())
  }
}
</script>`

type Privacy = 'screen' | 'camera'

export interface DesktopNodeHost {
  prefs(): ShellPrefs
  setPrefs(patch: ShellPrefs): void
  window(): BrowserWindow | null
  /** Show that the screen, camera or clipboard was just used. */
  notice(kind: Privacy | 'clipboard'): void
  onState(state: DesktopNodeState): void
  /** §17: the engine's Stop everything state (welcome and `stop_state`). */
  onStopState?(stopped: boolean): void
  icon: string
}

interface InvokeMessage {
  type: 'invoke'
  id: string
  capability: string
  params?: Record<string, unknown>
  timeout_ms?: number
}

function withTimeout<T>(p: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(message)), ms)
    p.then(
      (v) => {
        clearTimeout(t)
        resolve(v)
      },
      (e) => {
        clearTimeout(t)
        reject(e)
      }
    )
  })
}

export class DesktopNode {
  state: DesktopNodeState = 'off'
  private ws: WebSocket | null = null
  private url = ''
  private stopped = true
  private attempt = 0
  private failuresWithoutWelcome = 0
  private reconnectTimer: NodeJS.Timeout | null = null
  private pingTimer: NodeJS.Timeout | null = null
  private cameraWebContents = new Set<number>()
  private asking: Partial<Record<Privacy, Promise<boolean>>> = {}

  constructor(private host: DesktopNodeHost) {}

  /** (Re)connect for this engine. Safe to call on every "ready" status. */
  start(baseUrl: string, token: string): void {
    const url = `${baseUrl.replace(/^http/, 'ws')}/ws/node?token=${encodeURIComponent(token)}`
    if (url === this.url && !this.stopped && this.ws) return
    this.teardown()
    this.url = url
    this.stopped = false
    this.attempt = 0
    this.failuresWithoutWelcome = 0
    this.open()
  }

  stop(): void {
    this.stopped = true
    this.teardown()
    this.setState('off')
  }

  /** The permission handler allows camera access only for the hidden capture window. */
  ownsWebContents(id: number): boolean {
    return this.cameraWebContents.has(id)
  }

  // ------------------------------------------------------------------ socket
  private setState(s: DesktopNodeState): void {
    if (this.state === s) return
    this.state = s
    this.host.onState(s)
  }

  private open(): void {
    if (this.stopped) return
    this.setState(this.state === 'online' || this.state === 'off' ? 'connecting' : this.state)
    let ws: WebSocket
    try {
      ws = new WebSocket(this.url)
    } catch {
      this.scheduleReconnect()
      return
    }
    this.ws = ws
    let welcomed = false
    ws.onopen = () => {
      this.send({
        type: 'hello',
        protocol: 1,
        name: hostname() || 'This computer',
        kind: 'desktop',
        platform: process.platform,
        app_version: app.getVersion(),
        capabilities: [...CAPABILITIES]
      })
      this.startPing()
    }
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return
      let msg: { type?: string; [k: string]: unknown }
      try {
        msg = JSON.parse(ev.data)
      } catch {
        return
      }
      switch (msg.type) {
        case 'welcome': {
          welcomed = true
          this.attempt = 0
          this.failuresWithoutWelcome = 0
          // The engine never pings devices and drops silent ones: keep alive at its requested interval.
          const keepalive = Number(msg.keepalive_s)
          if (keepalive > 0) this.startPing(Math.max(5_000, keepalive * 1000))
          this.setState('online')
          if (typeof msg.stopped === 'boolean') this.host.onStopState?.(msg.stopped)
          break
        }
        case 'stop_state':
          if (typeof msg.stopped === 'boolean') this.host.onStopState?.(msg.stopped)
          break
        case 'error':
          console.warn(`[desktop-device] engine refused the connection: ${String(msg.code)} ${String(msg.message ?? '')}`)
          if (msg.code === 'disabled') this.failuresWithoutWelcome = 99
          break
        case 'invoke':
          void this.handleInvoke(msg as unknown as InvokeMessage)
          break
        case 'ping':
          this.send({ type: 'pong' })
          break
      }
    }
    ws.onclose = () => {
      if (this.ws !== ws) return
      this.stopPing()
      this.ws = null
      if (!welcomed) this.failuresWithoutWelcome++
      if (this.stopped) return
      // An engine without /ws/node closes before welcoming: say so, keep retrying slowly.
      this.setState(this.failuresWithoutWelcome >= 3 ? 'unsupported' : 'offline')
      this.scheduleReconnect()
    }
    ws.onerror = () => {
      /* onclose follows */
    }
  }

  private scheduleReconnect(): void {
    if (this.stopped) return
    const cap = this.failuresWithoutWelcome >= 3 ? 60_000 : MAX_BACKOFF_MS
    const delay = Math.min(cap, 1000 * 2 ** this.attempt) * (0.8 + Math.random() * 0.4)
    this.attempt++
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = setTimeout(() => this.open(), delay)
  }

  private send(msg: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg))
  }

  private startPing(intervalMs = PING_MS): void {
    this.stopPing()
    this.pingTimer = setInterval(() => this.send({ type: 'ping' }), intervalMs)
  }

  private stopPing(): void {
    if (this.pingTimer) clearInterval(this.pingTimer)
    this.pingTimer = null
  }

  private teardown(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = null
    this.stopPing()
    const ws = this.ws
    this.ws = null
    if (ws) {
      ws.onclose = null
      ws.onmessage = null
      try {
        ws.close()
      } catch {
        /* ignore */
      }
    }
  }

  // ------------------------------------------------------------------ capabilities
  private async handleInvoke(msg: InvokeMessage): Promise<void> {
    const timeout = Math.max(3000, Math.min(120_000, msg.timeout_ms ?? 30_000))
    try {
      const data = await withTimeout(this.run(msg.capability, msg.params ?? {}), timeout, 'Timed out on this computer.')
      this.send({ type: 'result', id: msg.id, ok: true, data })
    } catch (err) {
      this.send({ type: 'result', id: msg.id, ok: false, error: err instanceof Error ? err.message : String(err) })
    }
  }

  private async run(capability: string, params: Record<string, unknown>): Promise<Record<string, unknown>> {
    switch (capability) {
      case 'screen.capture': {
        await this.allowed('screen')
        const shot = await this.captureScreen()
        this.host.notice('screen')
        return shot
      }
      case 'camera.photo': {
        await this.allowed('camera')
        const photo = await this.takePhoto()
        this.host.notice('camera')
        return photo
      }
      // Electron 44's clipboard is async: without the await the answer would carry a Promise, sent as {}
      case 'clipboard.read': {
        const text = await clipboard.readText()
        this.host.notice('clipboard')
        return { text }
      }
      case 'clipboard.write':
        await clipboard.writeText(String(params.text ?? ''))
        return {}
      case 'notify.show': {
        if (!Notification.isSupported()) throw new Error('Notifications are not supported on this computer.')
        const body = String(params.text ?? params.body ?? '')
        new Notification({ title: String(params.title ?? 'Sentient'), body, icon: this.host.icon }).show()
        return {}
      }
      default:
        throw new Error(`This computer can't do "${capability}".`)
    }
  }

  /** Resolves when the user allows it; asks once (first use), then remembers the answer. */
  private async allowed(kind: Privacy): Promise<void> {
    const key = kind === 'screen' ? 'screenCapture' : 'cameraCapture'
    const pref = this.host.prefs()[key]
    const what = kind === 'screen' ? 'see your screen' : 'use your camera'
    if (pref === true) return
    if (pref === false) throw new Error(`The user turned off "Let Sentient ${what} when I ask" in Sentient's Devices page.`)
    this.asking[kind] ??= this.ask(kind).finally(() => {
      delete this.asking[kind]
    })
    const yes = await this.asking[kind]
    this.host.setPrefs({ [key]: yes })
    if (!yes) throw new Error(`The user chose not to let Sentient ${what}.`)
  }

  private async ask(kind: Privacy): Promise<boolean> {
    const win = this.host.window()
    const options: Electron.MessageBoxOptions = {
      type: 'question',
      buttons: ['Allow', "Don't allow"],
      defaultId: 0,
      cancelId: 1,
      noLink: true,
      title: 'Sentient',
      message: kind === 'screen' ? 'Let Sentient see your screen when you ask?' : 'Let Sentient use your camera when you ask?',
      detail:
        kind === 'screen'
          ? 'Sentient takes a single screenshot only when you ask about something on your screen. You will see a notice every time. You can change this later on the Devices page.'
          : 'Sentient takes a single photo only when you ask it to. The camera light turns on for a moment and you will see a notice every time. You can change this later on the Devices page.'
    }
    if (win && !win.isDestroyed()) {
      if (!win.isVisible()) win.show()
      return (await dialog.showMessageBox(win, options)).response === 0
    }
    return (await dialog.showMessageBox(options)).response === 0
  }

  private async captureScreen(): Promise<Record<string, unknown>> {
    const primary = screen.getPrimaryDisplay()
    const w = Math.round(primary.size.width * primary.scaleFactor)
    const h = Math.round(primary.size.height * primary.scaleFactor)
    const scale = Math.min(1, 1920 / w)
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width: Math.round(w * scale), height: Math.round(h * scale) }
    })
    const source = sources.find((s) => s.display_id === String(primary.id)) ?? sources[0]
    if (!source || source.thumbnail.isEmpty()) throw new Error("Couldn't capture the screen.")
    const size = source.thumbnail.getSize()
    return { mime: 'image/jpeg', base64: source.thumbnail.toJPEG(82).toString('base64'), width: size.width, height: size.height }
  }

  private async takePhoto(): Promise<Record<string, unknown>> {
    const file = join(app.getPath('temp'), 'sentient-camera.html')
    writeFileSync(file, CAMERA_HTML)
    const win = new BrowserWindow({
      show: false,
      width: 640,
      height: 480,
      skipTaskbar: true,
      webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false, backgroundThrottling: false }
    })
    const id = win.webContents.id
    this.cameraWebContents.add(id)
    try {
      await win.loadFile(file)
      const dataUrl = await withTimeout(
        win.webContents.executeJavaScript('window.takePhoto()', true) as Promise<string>,
        CAMERA_TIMEOUT_MS,
        "The camera didn't respond."
      )
      const m = /^data:(image\/[a-z]+);base64,(.+)$/.exec(String(dataUrl))
      if (!m) throw new Error("Couldn't read a photo from the camera.")
      return { mime: m[1], base64: m[2] }
    } catch (err) {
      const text = err instanceof Error ? err.message : String(err)
      if (/NotFound|DevicesNotFound|Requested device not found/i.test(text)) throw new Error('No camera was found on this computer.')
      if (/NotAllowed|Permission/i.test(text)) throw new Error('Camera access was blocked by the system.')
      throw err
    } finally {
      this.cameraWebContents.delete(id)
      if (!win.isDestroyed()) win.destroy()
    }
  }
}
