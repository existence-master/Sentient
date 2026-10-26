/**
 * Sentient desktop shell: window, tray, global shortcuts, native notifications,
 * and supervision of the local Python engine. The renderer talks to the engine
 * directly over HTTP/WebSocket using the connection handed out by the preload bridge.
 */
import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  nativeTheme,
  Notification,
  session,
  shell
} from 'electron'
import { existsSync, mkdirSync, realpathSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, isAbsolute, join, relative, resolve, sep } from 'node:path'
import type {
  AppCommand,
  CaptureNotice,
  DevicePrivacy,
  DictationEvent,
  DictationShellSettings,
  DictationStatus,
  NativeNotification,
  OpenPathTarget,
  PickFilesOptions,
  ShellPrefs,
  ShortcutId
} from '../../src/types/bridge'
import notificationIcon from '../../resources/icon.png?asset'
import {
  answerPath,
  answerProblem,
  APPROVAL_BUTTONS,
  approvalAnswer,
  approvalNotificationId,
  approvalRoute,
  type ApprovalTarget,
  answerBody,
  cleanApproval
} from './approval-notify'
import { AskController } from './ask'
import { BackendManager } from './backend'
import { CH } from './channels'
import { closeAction } from './close-logic'
import { DictationController, dictationSettingsFromConfig } from './dictation'
import { DesktopNode } from './node'
import { liveSensors, LOCAL_MIC_ID, type LiveSensors } from './sensors-logic'
import { homePaths, sentientHome } from './paths'
import { shellState, THEME_BG } from './prefs'
import { ScreenSharer } from './share'
import { NEW_CHAT_ACCELERATOR, Shortcuts, STOP_ACCELERATOR } from './shortcuts'
import { SmokeRunner, smokeConfig } from './smoke'
import { AppTray } from './tray'
import { createMainWindow } from './window'

const smoke = smokeConfig()
const startHidden = process.argv.includes('--hidden')

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
let dictation: DictationController | null = null // #169 push to talk and dictation into any app
let ask: AskController | null = null // #173 ask about selected text
let sensorState: LiveSensors | null = null // #170 live microphones and cameras, as the engine last reported them
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

/** The window is on screen and has focus. Windows can report a minimized window as focused when nothing else took
 *  focus, so minimized or hidden never counts (#323: a task's approval then shows as a notification). */
function inFront(): boolean {
  return !!win && !win.isDestroyed() && win.isVisible() && !win.isMinimized() && win.isFocused()
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
  let quitWhenClosed = false
  w.on('close', (event) => {
    const action = closeAction({
      quitting,
      smoke: !!smoke,
      hasTray: !!tray,
      minimizeToTray: shellState.prefs().minimizeToTray,
      platform: process.platform
    })
    if (action === 'hide') {
      event.preventDefault()
      w.hide()
    }
    // #439 The hidden pill and ask windows would keep the app (and its engine) running, so quit on purpose.
    quitWhenClosed = action === 'quit'
  })
  w.on('closed', () => {
    if (win === w) win = null
    if (quitWhenClosed) app.quit()
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
  voice?: { dictation?: Record<string, unknown> }
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
    dictation?.apply(dictationSettingsFromConfig(cfg.voice?.dictation))
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
  if (stop) {
    dictation?.cancel() // the microphone goes off at once, before the engine answers
    ask?.close()
  }
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

// ------------------------------------------------------------------ microphones and cameras (#170)
/** Every live microphone and camera: the engine's list, plus push to talk or dictation the shell knows about first. */
function currentSensors() {
  return liveSensors(sensorState, dictation?.micPurpose ?? null)
}

/** Turn off one (`id`) or every live microphone and camera, from the tray. */
async function stopSensor(id?: string): Promise<void> {
  if (!id || id === LOCAL_MIC_ID) dictation?.cancel() // this computer's microphone goes off at once
  if (backend.status.state !== 'ready') return
  try {
    await api('/api/sensors/stop', { method: 'POST', body: JSON.stringify(id ? { id } : {}) })
  } catch (err) {
    dialog.showErrorBox('Sentient', `Couldn't turn it off: ${String(err)}`)
  }
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
  onSensors: (state) => {
    sensorState = state
    tray?.refresh()
  },
  onStopSensor: (sensor) => {
    if (sensor === 'mic') dictation?.cancel() // Turn off in the title bar or on another device
  },
  onStopState: (stopped) => {
    engineStopped = stopped
    if (stopped) {
      dictation?.cancel()
      ask?.close()
    }
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

// ------------------------------------------------------------------ share this window / a region (#172)
const shortcuts = new Shortcuts(
  () => shellState.prefs(),
  (patch) => shellState.setPrefs(patch),
  () => tray?.refresh()
)

const sharer = new ScreenSharer({
  deliver: (share) => sendCommand({ type: 'share-screen', share }),
  notice: (text) => tray?.flash(text)
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

// ------------------------------------------------------------------ approvals from a notification (#323)
/** The approval notification of each waiting run. Kept after the toast closes, because Windows keeps it in the
 *  notification center and its buttons still work from there; dropped once answered here or in the app. */
const approvalNotes = new Map<string, Notification>()

function showApprovalNotification(n: NativeNotification, target: ApprovalTarget): void {
  approvalNotes.get(target.runId)?.close()
  const note = new Notification({
    id: approvalNotificationId(target.runId),
    title: String(n.title || 'Sentient needs your OK'),
    body: String(n.body ?? ''),
    icon: notificationIcon,
    timeoutType: 'never', // it waits with the task until someone answers
    actions: APPROVAL_BUTTONS.map((text) => ({ type: 'button' as const, text }))
  })
  approvalNotes.set(target.runId, note)
  const forget = () => {
    if (approvalNotes.get(target.runId) === note) approvalNotes.delete(target.runId)
  }
  // a click on the notification itself (or a system without buttons): open the task, where the approval card waits
  note.on('click', () => {
    forget()
    sendCommand({ type: 'navigate', route: approvalRoute(target) })
  })
  note.on('action', (details) => {
    forget()
    const answer = approvalAnswer(details.actionIndex, target.untrusted)
    if (answer) void answerApproval(target, answer)
    else sendCommand({ type: 'navigate', route: approvalRoute(target) })
  })
  // notifications turned off in the system settings: flash the taskbar button instead, until the window is focused
  note.on('failed', () => {
    forget()
    if (win && !win.isDestroyed() && !inFront()) win.flashFrame(true)
  })
  note.show()
}

/** Answer the run the way the approval card does. A refused answer (expired, already answered) says why. */
async function answerApproval(target: ApprovalTarget, answer: string): Promise<void> {
  let problem: string | null = null
  try {
    // scoped to the request this notification showed: the engine refuses it when the run now waits on another one
    const r = await fetch(`${backend.baseUrl}${answerPath(target)}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${backend.token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(answerBody(target, answer)),
      signal: AbortSignal.timeout(10_000)
    })
    if (!r.ok) problem = answerProblem(r.status, await r.json().catch(() => null))
  } catch {
    problem = "Sentient isn't reachable right now. Open the task to answer there."
  }
  if (!problem || !Notification.isSupported()) return
  const note = new Notification({ title: "Your answer didn't go through", body: problem, silent: true, icon: notificationIcon })
  liveNotifications.add(note)
  note.on('click', () => {
    liveNotifications.delete(note)
    sendCommand({ type: 'navigate', route: approvalRoute(target) })
  })
  note.on('close', () => liveNotifications.delete(note))
  note.show()
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
    const approval = cleanApproval(n.approval)
    if (approval) {
      showApprovalNotification(n, approval)
      return
    }
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
  ipcMain.handle(CH.closeApprovalNotification, (_e, runId: string) => {
    if (typeof runId !== 'string') return
    const note = approvalNotes.get(runId)
    approvalNotes.delete(runId)
    note?.close()
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
  ipcMain.handle(CH.isFocused, () => inFront())
  ipcMain.handle(CH.getVersion, () => ({
    app: app.getVersion(),
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node,
    platform: process.platform,
    arch: process.arch
  }))
  ipcMain.handle(CH.pickFiles, async (_e, options?: PickFilesOptions) => {
    const parent = win ?? undefined
    const props: Array<'openFile' | 'openDirectory' | 'multiSelections' | 'showHiddenFiles'> = options?.directory
      ? ['openDirectory', 'showHiddenFiles']
      : ['openFile']
    if (options?.multiple !== false) props.push('multiSelections')
    const filters = Array.isArray(options?.filters)
      ? options.filters
          .filter((f) => typeof f?.name === 'string' && Array.isArray(f.extensions))
          .map((f) => ({ name: f.name, extensions: f.extensions.filter((x) => typeof x === 'string' && /^[a-z0-9]{1,10}$/i.test(x)) }))
      : undefined
    const result = parent
      ? await dialog.showOpenDialog(parent, { properties: props, filters })
      : await dialog.showOpenDialog({ properties: props, filters })
    return result.canceled ? [] : result.filePaths
  })
  ipcMain.on(CH.readyForScreenshot, () => smokeRunner?.ready())
  ipcMain.handle(CH.showInFolder, async (_e, path: string) => {
    if (typeof path !== 'string' || !path) return false
    const target = resolve(path)
    let isDir: boolean
    try {
      isDir = statSync(target).isDirectory()
    } catch {
      return false
    }
    // a folder opens in the file manager; only a Sentient backup file is selected in its folder; nothing is opened or run
    if (isDir) return (await shell.openPath(target)) === ''
    if (!/^sentient-(backup|snapshot)-[\w-]+\.zip$/i.test(basename(target))) return false
    shell.showItemInFolder(target)
    return true
  })
  ipcMain.handle(CH.openFile, async (_e, name: string) => {
    if (typeof name !== 'string' || !name) return false
    const base = resolve(homePaths.files())
    const target = resolve(base, name)
    if (!target.startsWith(base + sep) || !existsSync(target)) return false
    return (await shell.openPath(target)) === ''
  })
  ipcMain.handle(CH.openFolderFile, async (_e, target: { path?: unknown; url?: unknown }) => {
    // a note from a folder the user added (#176): only files really inside one of those folders (links followed),
    // only note and document types, so a reply can never make the window open or run anything else
    const path = typeof target?.path === 'string' ? target.path : ''
    if (!path) return false
    let folders: Array<{ path: string; name: string; kind: string }> = []
    try {
      folders = (await api<{ folders: Array<{ path: string; name: string; kind: string }> }>('/api/folders')).folders
    } catch {
      return false
    }
    let file: string
    try {
      file = realpathSync(resolve(path))
    } catch {
      return false
    }
    if (!/\.(md|markdown|txt|text|pdf|docx)$/i.test(file) || !statSync(file).isFile()) return false
    let folder: (typeof folders)[number] | undefined
    let rel = ''
    for (const f of folders) {
      let root: string
      try {
        root = realpathSync(resolve(f.path))
      } catch {
        continue
      }
      const r = relative(root, file)
      if (r && !r.startsWith('..') && !isAbsolute(r)) {
        folder = f
        rel = r
        break
      }
    }
    if (!folder) return false
    // in a vault, Obsidian opens it when installed; the link is built here from the checked file, never taken as given
    const wantsObsidian = typeof target?.url === 'string' && target.url.startsWith('obsidian://')
    if (wantsObsidian && folder.kind === 'obsidian' && app.getApplicationNameForProtocol('obsidian://')) {
      const note = rel.split(sep).join('/').replace(/\.md$/i, '')
      await shell.openExternal(`obsidian://open?vault=${encodeURIComponent(folder.name)}&file=${encodeURIComponent(note)}`)
      return true
    }
    if ((await shell.openPath(file)) === '') return true
    shell.showItemInFolder(file) // no app opens this kind of file here: show it in its folder instead
    return true
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
  ipcMain.handle(CH.getShortcuts, () => shortcuts.list())
  ipcMain.handle(CH.setShortcut, (_e, id: ShortcutId, accelerator: string | null) =>
    shortcuts.set(id, accelerator === null ? null : String(accelerator ?? ''))
  )
  // #169 push to talk and dictation: settings from voice.dictation, events from the listening pill.
  ipcMain.handle(CH.dictationApply, (_e, settings: DictationShellSettings) =>
    dictation ? dictation.apply(settings) : DICTATION_OFF
  )
  ipcMain.handle(CH.dictationStatus, () => (dictation ? dictation.status() : DICTATION_OFF))
  ipcMain.handle(CH.dictationCancel, () => dictation?.cancel())
  ipcMain.handle(CH.dictationPermission, (_e, kind: 'microphone' | 'accessibility') => dictation?.openPermissionSettings(kind))
  ipcMain.on(CH.dictationEvent, (e, ev: DictationEvent) => dictation?.onPillEvent(e.sender.id, ev))
  // #173 ask about selected text: only the ask window may use these
  const fromAsk = (e: Electron.IpcMainInvokeEvent | Electron.IpcMainEvent) => !!ask && ask.ownsWebContents(e.sender.id)
  ipcMain.handle(CH.askGetState, (e) => (fromAsk(e) ? ask?.currentState() ?? null : null))
  ipcMain.handle(CH.askCopy, (e, text: string) => (fromAsk(e) && typeof text === 'string' ? ask?.copy(text) : false))
  ipcMain.handle(CH.askReplace, (e, text: string) =>
    fromAsk(e) && typeof text === 'string' ? ask?.replace(text) ?? 'failed' : 'failed'
  )
  ipcMain.handle(CH.askClose, (e) => {
    if (fromAsk(e)) ask?.hide()
  })
  ipcMain.handle(CH.askOpenChat, (e, sessionId: string) => {
    if (fromAsk(e) && typeof sessionId === 'string' && /^[\w-]{1,80}$/.test(sessionId)) ask?.openChat(sessionId)
  })
  ipcMain.on(CH.askResize, (e, height: number) => {
    if (fromAsk(e) && typeof height === 'number') ask?.resize(height)
  })
}

const DICTATION_OFF: DictationStatus = { accessibility: null, microphone: null }

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
      toggleStopped: () => void setStopped(engineStopped !== true, 'tray'),
      shareWindow: () => void sharer.shareWindow(),
      shareRegion: () => void sharer.shareRegion(),
      shareAccelerators: () => ({ window: shortcuts.accelerator('shareWindow'), region: shortcuts.accelerator('shareRegion') }),
      micOn: () => dictation?.micOn === true,
      sensors: currentSensors,
      stopSensor: (id) => void stopSensor(id),
      showSensors: () => sendCommand({ type: 'show-sensors' })
    })

    dictation = new DictationController({
      talk: (text) => sendCommand({ type: 'push-to-talk', text }),
      notify: (title, body) => {
        if (Notification.isSupported()) new Notification({ title, body, silent: true, icon: notificationIcon }).show()
      },
      micChanged: () => {
        desktopNode.reportMic(dictation?.micPurpose ?? null) // the engine's indicator, in the title bar too
        tray?.refresh()
      }
    })

    ask = new AskController({
      shortcut: () => shortcuts.accelerator('askSelection'),
      openChat: (sessionId) => sendCommand({ type: 'navigate', route: `/chat/${sessionId}` }),
      notice: (text) => tray?.flash(text)
    })

    shortcuts.registerFixed(NEW_CHAT_ACCELERATOR, () => sendCommand({ type: 'new-chat' }))
    // Stop only: resuming is always a deliberate click.
    shortcuts.registerFixed(STOP_ACCELERATOR, () => {
      if (backend.status.state === 'ready') void setStopped(true, 'hotkey')
    })
    shortcuts.start({
      shareWindow: () => void sharer.shareWindow(),
      shareRegion: () => void sharer.shareRegion(),
      // Called on every press and every autorepeat of a held key: the controller tells holding from pressing.
      pushToTalk: () => dictation?.trigger('talk'),
      dictate: () => dictation?.trigger('dictate'),
      askSelection: () => void ask?.trigger()
    })
  })

  app.on('activate', () => showWindow())

  app.on('before-quit', () => {
    quitting = true
    dictation?.dispose()
    ask?.dispose()
    shortcuts.stopAll()
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
