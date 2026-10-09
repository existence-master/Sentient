/**
 * Sentient desktop shell: window, tray, global shortcut, native notifications,
 * and supervision of the local Python engine. The renderer talks to the engine
 * directly over HTTP/WebSocket using the connection handed out by the preload bridge.
 */
import {
  app,
  BrowserWindow,
  dialog,
  globalShortcut,
  ipcMain,
  nativeTheme,
  Notification,
  session,
  shell
} from 'electron'
import { existsSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve, sep } from 'node:path'
import type { AppCommand, CaptureNotice, DevicePrivacy, NativeNotification, OpenPathTarget, ShellPrefs } from '../../src/types/bridge'
import notificationIcon from '../../resources/icon.png?asset'
import { BackendManager } from './backend'
import { CH } from './channels'
import { DesktopNode } from './node'
import { homePaths, sentientHome } from './paths'
import { shellState, THEME_BG } from './prefs'
import { SmokeRunner, smokeConfig } from './smoke'
import { AppTray, STOP_ACCELERATOR } from './tray'
import { createMainWindow } from './window'

const smoke = smokeConfig()
const startHidden = process.argv.includes('--hidden')
const GLOBAL_SHORTCUT = 'CommandOrControl+Shift+Space'

if (smoke) {
  // Isolate smoke runs from the real profile's window state and single-instance lock.
  app.setPath('userData', join(process.env.SENTIENT_HOME ? sentientHome() : tmpdir(), 'electron-smoke'))
}

const backend = new BackendManager()
let win: BrowserWindow | null = null
let tray: AppTray | null = null
let quitting = false
let smokeRunner: SmokeRunner | null = null
let engineStopped: boolean | undefined // §17 Stop everything, as the engine last reported it
const liveNotifications = new Set<Notification>()

function showWindow(): void {
  if (!win || win.isDestroyed()) {
    win = createWindow('/')
    return
  }
  if (win.isMinimized()) win.restore()
  if (!win.isVisible()) win.show()
  win.focus()
}

function sendCommand(cmd: AppCommand): void {
  showWindow()
  const target = win
  if (!target) return
  const deliver = () => target.webContents.send(CH.command, cmd)
  if (target.webContents.isLoading()) target.webContents.once('did-finish-load', deliver)
  else deliver()
}

function createWindow(route: string): BrowserWindow {
  const w = createMainWindow({
    route,
    smoke: !!smoke,
    smokeSize: smoke?.size,
    startHidden
  })
  w.on('close', (event) => {
    const toTray = shellState.prefs().minimizeToTray !== false
    if (!quitting && !smoke && tray && toTray) {
      event.preventDefault()
      w.hide()
    }
  })
  w.on('closed', () => {
    if (win === w) win = null
  })
  // Always listening needs the renderer (mic + voice socket) to keep running while hidden in the tray.
  if (!smoke && shellState.prefs().alwaysListening) w.webContents.setBackgroundThrottling(false)
  const sendMax = () => w.webContents.send(CH.winMaximizedChanged, w.isMaximized())
  w.on('maximize', sendMax)
  w.on('unmaximize', sendMax)
  return w
}

// ------------------------------------------------------------------ backend <-> shell
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${backend.baseUrl}${path}`, {
    ...init,
    headers: { Authorization: `Bearer ${backend.token}`, 'Content-Type': 'application/json', ...init?.headers },
    signal: AbortSignal.timeout(10_000)
  })
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`)
  return (await r.json()) as T
}

interface ConfigSubset {
  ui?: { minimize_to_tray?: boolean; theme?: 'dark' | 'light' | 'system' }
  proactivity?: { enabled?: boolean }
}

async function syncFromBackend(): Promise<void> {
  try {
    engineStopped = (await api<{ stopped: boolean }>('/api/stop')).stopped
  } catch {
    /* an older engine without Stop everything */
  }
  try {
    const cfg = await api<ConfigSubset>('/api/config')
    shellState.setPrefs({
      minimizeToTray: cfg.ui?.minimize_to_tray ?? true,
      proactivityEnabled: cfg.proactivity?.enabled,
      theme: cfg.ui?.theme ?? shellState.prefs().theme
    })
  } catch {
    /* engine may not expose config yet */
  }
  tray?.refresh()
}

async function toggleProactivity(): Promise<void> {
  const next = !(shellState.prefs().proactivityEnabled ?? true)
  try {
    await api('/api/config', { method: 'PATCH', body: JSON.stringify({ proactivity: { enabled: next } }) })
    shellState.setPrefs({ proactivityEnabled: next })
  } catch (err) {
    dialog.showErrorBox('Sentient', `Couldn't change proactivity: ${String(err)}`)
  }
  tray?.refresh()
}

// ------------------------------------------------------------------ stop everything (§17)
/** Stop or resume from the tray or the global shortcut. Works with the window hidden; never asks the model. */
async function setStopped(stop: boolean, source: 'tray' | 'hotkey'): Promise<void> {
  try {
    const state = await api<{ stopped: boolean; cancelled?: number }>(stop ? '/api/stop-all' : '/api/resume', {
      method: 'POST',
      body: JSON.stringify({ source })
    })
    engineStopped = state.stopped
    if (Notification.isSupported() && (!win || !win.isVisible() || !win.isFocused())) {
      new Notification({
        title: stop ? 'Sentient stopped everything' : 'Sentient resumed',
        body: stop ? 'Nothing new will start until you resume from the tray or the app.' : 'Scheduled tasks and suggestions are back on.',
        silent: true,
        icon: notificationIcon
      }).show()
    }
  } catch (err) {
    dialog.showErrorBox('Sentient', `Couldn't ${stop ? 'stop' : 'resume'} Sentient: ${String(err)}`)
  }
  tray?.refresh()
}

// ------------------------------------------------------------------ this computer as a device (§13)
const CAPTURE_TEXT: Record<CaptureNotice['kind'], { tray: string; title: string; body: string }> = {
  screen: { tray: 'looked at your screen', title: 'Sentient looked at your screen', body: 'A single screenshot was taken because you asked.' },
  camera: { tray: 'took a photo with your camera', title: 'Sentient used your camera', body: 'A single photo was taken because you asked.' },
  clipboard: { tray: 'read your clipboard', title: 'Sentient read your clipboard', body: 'It read the text you last copied because you asked.' }
}

const desktopNode = new DesktopNode({
  prefs: () => shellState.prefs(),
  setPrefs: (patch) => shellState.setPrefs(patch),
  window: () => win,
  icon: notificationIcon,
  onState: (state) => {
    if (win && !win.isDestroyed()) win.webContents.send(CH.desktopNodeState, state)
  },
  onStopState: (stopped) => {
    engineStopped = stopped
    tray?.refresh()
  },
  notice: (kind) => {
    const text = CAPTURE_TEXT[kind]
    tray?.flash(text.tray)
    const target = win && !win.isDestroyed() ? win : null
    target?.webContents.send(CH.captureNotice, { kind, at: new Date().toISOString() } satisfies CaptureNotice)
    if (kind !== 'clipboard' && (!target || !target.isVisible() || !target.isFocused()) && Notification.isSupported()) {
      new Notification({ title: text.title, body: text.body, silent: true, icon: notificationIcon }).show()
    }
  }
})

// ------------------------------------------------------------------ always listening for the wake word
function setAlwaysListening(enabled: boolean, notifyRenderer: boolean): void {
  shellState.setPrefs({ alwaysListening: enabled })
  if (win && !win.isDestroyed()) {
    win.webContents.setBackgroundThrottling(!enabled)
    if (notifyRenderer) win.webContents.send(CH.alwaysListeningChanged, enabled)
  }
  tray?.refresh()
}

function devicePrivacy(): DevicePrivacy {
  const p = shellState.prefs()
  return { screen: p.screenCapture ?? null, camera: p.cameraCapture ?? null }
}

backend.on('status', (status) => {
  if (win && !win.isDestroyed()) win.webContents.send(CH.backendStatus, status)
  if (status.state === 'ready') {
    void syncFromBackend()
    desktopNode.start(backend.baseUrl, backend.token)
  } else {
    if (status.state === 'failed' || status.state === 'stopped') desktopNode.stop()
    tray?.refresh()
  }
})

// ------------------------------------------------------------------ security
function installSecurity(): void {
  const ses = session.defaultSession
  ses.setPermissionRequestHandler((wc, permission, callback, details) => {
    if (permission === 'media') {
      const types = (details as { mediaTypes?: string[] }).mediaTypes ?? []
      // Video only for the hidden capture window of the desktop device, after the user allowed the camera.
      if (desktopNode.ownsWebContents(wc.id) && shellState.prefs().cameraCapture === true) {
        callback(types.every((t) => t === 'video'))
        return
      }
      callback(types.every((t) => t === 'audio'))
      return
    }
    callback(['notifications', 'clipboard-sanitized-write', 'fullscreen'].includes(permission))
  })
  ses.setPermissionCheckHandler((_wc, permission) =>
    ['media', 'notifications', 'clipboard-sanitized-write', 'fullscreen'].includes(permission)
  )
  // Dev server responses get an exact-origin CSP (the production build ships a meta CSP).
  const devUrl = process.env.ELECTRON_RENDERER_URL
  if (devUrl) {
    ses.webRequest.onHeadersReceived((details, callback) => {
      if (details.resourceType !== 'mainFrame' || !details.url.startsWith(devUrl)) {
        callback({ responseHeaders: details.responseHeaders })
        return
      }
      const origin = backend.baseUrl.replace('http://', '')
      const csp = [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline'",
        "style-src 'self' 'unsafe-inline'",
        `img-src 'self' data: blob: http://${origin}`,
        `media-src 'self' data: blob: http://${origin}`,
        "font-src 'self' data:",
        `connect-src 'self' ws://localhost:* http://localhost:* http://${origin} ws://${origin}`,
        "object-src 'none'"
      ].join('; ')
      callback({ responseHeaders: { ...details.responseHeaders, 'Content-Security-Policy': [csp] } })
    })
  }
  app.on('web-contents-created', (_e, contents) => {
    contents.on('will-attach-webview', (event) => event.preventDefault())
  })
}

// ------------------------------------------------------------------ IPC
function registerIpc(): void {
  ipcMain.handle(CH.getConnection, () => ({ baseUrl: backend.baseUrl, token: backend.token }))
  ipcMain.handle(CH.getBackendStatus, () => backend.status)
  ipcMain.handle(CH.restartBackend, () => backend.restart())
  ipcMain.handle(CH.openExternal, (_e, url: string) => {
    if (typeof url === 'string' && /^(https?:|mailto:)/i.test(url)) return shell.openExternal(url)
    return undefined
  })
  ipcMain.handle(CH.openPath, async (_e, target: OpenPathTarget) => {
    const resolver = homePaths[target as keyof typeof homePaths]
    if (!resolver) return
    const dir = resolver()
    mkdirSync(dir, { recursive: true })
    await shell.openPath(dir)
  })
  ipcMain.handle(CH.showNotification, (_e, n: NativeNotification) => {
    if (!Notification.isSupported()) return
    const note = new Notification({
      title: String(n.title ?? 'Sentient'),
      body: String(n.body ?? ''),
      silent: !!n.silent,
      icon: notificationIcon
    })
    liveNotifications.add(note)
    note.on('click', () => {
      liveNotifications.delete(note)
      if (n.route) sendCommand({ type: 'navigate', route: n.route })
      else showWindow()
    })
    note.on('close', () => liveNotifications.delete(note))
    note.show()
  })
  ipcMain.handle(CH.setLaunchAtLogin, (_e, enabled: boolean) => {
    app.setLoginItemSettings({ openAtLogin: !!enabled, args: ['--hidden'] })
  })
  ipcMain.handle(CH.getLaunchAtLogin, () => app.getLoginItemSettings({ args: ['--hidden'] }).openAtLogin)
  ipcMain.handle(CH.syncPrefs, (_e, prefs: ShellPrefs) => {
    shellState.setPrefs(prefs)
    if (win && !win.isDestroyed()) {
      if (prefs.titleBar && process.platform !== 'darwin') {
        try {
          win.setTitleBarOverlay({ color: prefs.titleBar.color, symbolColor: prefs.titleBar.symbolColor })
        } catch {
          /* overlay unsupported */
        }
      }
      if (prefs.theme) {
        const dark = prefs.theme === 'dark' || (prefs.theme === 'system' && nativeTheme.shouldUseDarkColors)
        win.setBackgroundColor(dark ? THEME_BG.dark : THEME_BG.light)
        nativeTheme.themeSource = prefs.theme
      }
    }
    tray?.refresh()
  })
  ipcMain.handle(CH.winMinimize, () => win?.minimize())
  ipcMain.handle(CH.winMaximize, () => {
    if (!win) return
    if (win.isMaximized()) win.unmaximize()
    else win.maximize()
  })
  ipcMain.handle(CH.winClose, () => win?.close())
  ipcMain.handle(CH.winIsMaximized, () => win?.isMaximized() ?? false)
  ipcMain.handle(CH.isFocused, () => win?.isFocused() ?? false)
  ipcMain.handle(CH.getVersion, () => ({
    app: app.getVersion(),
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node,
    platform: process.platform,
    arch: process.arch
  }))
  ipcMain.handle(CH.pickFiles, async (_e, options?: { multiple?: boolean }) => {
    const parent = win ?? undefined
    const props: Array<'openFile' | 'multiSelections'> = ['openFile']
    if (options?.multiple !== false) props.push('multiSelections')
    const result = parent
      ? await dialog.showOpenDialog(parent, { properties: props })
      : await dialog.showOpenDialog({ properties: props })
    return result.canceled ? [] : result.filePaths
  })
  ipcMain.on(CH.readyForScreenshot, () => smokeRunner?.ready())
  ipcMain.handle(CH.openFile, async (_e, name: string) => {
    if (typeof name !== 'string' || !name) return false
    const base = resolve(homePaths.files())
    const target = resolve(base, name)
    if (!target.startsWith(base + sep) || !existsSync(target)) return false
    return (await shell.openPath(target)) === ''
  })
  ipcMain.handle(CH.getDevicePrivacy, () => devicePrivacy())
  ipcMain.handle(CH.setDevicePrivacy, (_e, patch: Partial<DevicePrivacy>) => {
    const next: ShellPrefs = {}
    if (patch && 'screen' in patch) next.screenCapture = patch.screen === null ? null : !!patch.screen
    if (patch && 'camera' in patch) next.cameraCapture = patch.camera === null ? null : !!patch.camera
    shellState.setPrefs(next)
    return devicePrivacy()
  })
  ipcMain.handle(CH.getDesktopNodeState, () => desktopNode.state)
  ipcMain.handle(CH.setAlwaysListening, (_e, enabled: boolean) => setAlwaysListening(!!enabled, false))
  ipcMain.handle(CH.getAlwaysListening, () => shellState.prefs().alwaysListening ?? null)
  // The renderer heard the wake word and already navigated to Voice mode: bring the window forward.
  ipcMain.handle(CH.wakeDetected, () => {
    if (!smoke) showWindow()
  })
}

// ------------------------------------------------------------------ lifecycle
function boot(): void {
  app.on('second-instance', () => showWindow())

  app.whenReady().then(() => {
    app.setAppUserModelId('technology.existence.sentient')
    const prefs = shellState.prefs()
    if (prefs.theme) nativeTheme.themeSource = prefs.theme
    installSecurity()
    registerIpc()

    win = createWindow(smoke?.route ?? '/')
    void backend.start()

    if (smoke) {
      smokeRunner = new SmokeRunner(smoke, () => win, (code) => {
        quitting = true
        backend.stop()
        app.exit(code)
      })
      return
    }

    tray = new AppTray({
      open: showWindow,
      newChat: () => sendCommand({ type: 'new-chat' }),
      voiceMode: () => sendCommand({ type: 'voice-mode' }),
      toggleProactivity: () => void toggleProactivity(),
      quit: () => {
        quitting = true
        app.quit()
      },
      proactivityEnabled: () => shellState.prefs().proactivityEnabled,
      backendReady: () => backend.status.state === 'ready',
      alwaysListening: () => shellState.prefs().alwaysListening === true,
      toggleAlwaysListening: () => setAlwaysListening(shellState.prefs().alwaysListening !== true, true),
      stopped: () => engineStopped,
      toggleStopped: () => void setStopped(engineStopped !== true, 'tray')
    })

    if (!globalShortcut.register(GLOBAL_SHORTCUT, () => sendCommand({ type: 'new-chat' }))) {
      console.warn(`[shell] global shortcut ${GLOBAL_SHORTCUT} is taken by another app`)
    }
    // Stop only: resuming is always a deliberate click.
    if (!globalShortcut.register(STOP_ACCELERATOR, () => {
      if (backend.status.state === 'ready') void setStopped(true, 'hotkey')
    })) {
      console.warn(`[shell] global shortcut ${STOP_ACCELERATOR} is taken by another app`)
    }
  })

  app.on('activate', () => showWindow())

  app.on('before-quit', () => {
    quitting = true
    globalShortcut.unregisterAll()
    desktopNode.stop()
    backend.stop()
  })

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin' || smoke) app.quit()
  })

  // Last line of defence so the engine never outlives the shell.
  process.on('exit', () => backend.stop())
}

if (smoke || app.requestSingleInstanceLock()) {
  boot()
} else {
  app.quit()
}
