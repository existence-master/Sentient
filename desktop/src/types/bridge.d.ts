/**
 * The preload bridge exposed to the renderer as `window.sentient`.
 *
 * Implemented in `electron/preload/index.ts` (IPC channel names in `electron/main/channels.ts`).
 * When the renderer runs in a normal browser the bridge is missing and
 * `src/lib/bridge.ts` provides a stub (connection from `?api=&token=`).
 * Always go through `getBridge()` from `@/lib/bridge` instead of touching `window.sentient`.
 */

/** Lifecycle of the Python engine the shell spawns. */
export type BackendState =
  | 'starting' // spawned, polling /api/health
  | 'ready' // health check passed; connection is valid
  | 'crashed' // exited unexpectedly; `willRestart` says whether a restart is scheduled
  | 'restarting' // waiting for backoff before respawning
  | 'failed' // gave up (start timeout or too many crashes); user must press Retry
  | 'stopped' // app is quitting

export interface BackendStatus {
  state: BackendState
  /** Human readable detail ("Python not found", "exited with code 1"...). */
  message?: string
  /** Last lines of backend.log, for the error screen. */
  logTail?: string
  /** Base URL, present when state is `ready`. */
  baseUrl?: string
  attempt?: number
  willRestart?: boolean
  /** Milliseconds until the next restart attempt when `restarting`. */
  retryInMs?: number
  /** Path of the engine that was launched: the bundled `sentient-engine` exe, or Python in a dev checkout. */
  pythonPath?: string
  logPath?: string
}

export interface Connection {
  /** e.g. `http://127.0.0.1:53412` (no trailing slash) */
  baseUrl: string
  token: string
}

export interface NativeNotification {
  title: string
  body: string
  /** Hash route to open when clicked, e.g. `/tasks?task=abc`. */
  route?: string
  silent?: boolean
}

/** Commands pushed from the shell (tray, global shortcut, notification click). */
export type AppCommand =
  | { type: 'navigate'; route: string }
  | { type: 'new-chat' }
  /** `wake: true` when opened by the wake word, so Voice mode starts listening right away. */
  | { type: 'voice-mode'; wake?: boolean }
  | { type: 'open-settings'; section?: string }

export type OpenPathTarget = 'home' | 'logs' | 'files' | 'workspace' | 'skills'

export interface VersionInfo {
  app: string
  electron: string
  chrome: string
  node: string
  platform: string
  arch: string
}

/** Preferences the shell needs before the renderer and backend are up. */
export interface ShellPrefs {
  theme?: 'dark' | 'light' | 'system'
  /** Resolved colors for the native title bar overlay (Windows). */
  titleBar?: { color: string; symbolColor: string }
  minimizeToTray?: boolean
  proactivityEnabled?: boolean
  /** Desktop device: may Sentient capture the screen / camera when asked? Unset = ask once. */
  screenCapture?: boolean | null
  cameraCapture?: boolean | null
  /** Listen for "Hey Sentient" in the background, also while the window is hidden in the tray. */
  alwaysListening?: boolean
}

/** Desktop-as-a-device privacy switches. `null` = never asked (Sentient asks once, the first time). */
export interface DevicePrivacy {
  screen: boolean | null
  camera: boolean | null
}

/** Pushed to the renderer whenever the desktop device captured the screen or camera, or used the clipboard. */
export interface CaptureNotice {
  kind: 'screen' | 'camera' | 'clipboard'
  at: string
}

/** Connection state of the built-in desktop device (`/ws/node`). */
export type DesktopNodeState = 'off' | 'connecting' | 'online' | 'offline' | 'unsupported'

export interface WindowControls {
  minimize(): Promise<void>
  /** Toggles maximize / restore. */
  maximize(): Promise<void>
  close(): Promise<void>
  isMaximized(): Promise<boolean>
  onMaximizedChange(cb: (maximized: boolean) => void): () => void
}

export interface SentientBridge {
  /** True inside Electron. The browser stub sets this to false. */
  readonly isDesktop: boolean
  readonly platform: 'win32' | 'darwin' | 'linux' | 'web'
  getConnection(): Promise<Connection>
  getBackendStatus(): Promise<BackendStatus>
  /** Subscribe to backend lifecycle changes. Returns an unsubscribe function. */
  onBackendStatus(cb: (status: BackendStatus) => void): () => void
  restartBackend(): Promise<void>
  openExternal(url: string): Promise<void>
  /** Open one of Sentient's folders in the OS file manager. */
  openPath(target: OpenPathTarget): Promise<void>
  showNotification(n: NativeNotification): Promise<void>
  setLaunchAtLogin(enabled: boolean): Promise<void>
  getLaunchAtLogin(): Promise<boolean>
  /** Keep the shell's copy of theme / tray / proactivity prefs in sync. */
  syncPrefs(prefs: ShellPrefs): Promise<void>
  onCommand(cb: (cmd: AppCommand) => void): () => void
  window: WindowControls
  getVersion(): Promise<VersionInfo>
  /** Smoke-test hook: the current route has rendered and its data has loaded. */
  readyForScreenshot(): void
  /** Native open dialog. Returns absolute paths (drag and drop is the primary path). `directory` picks a folder
   *  (hidden folders such as ~/.hermes are shown). */
  pickFiles(options?: { multiple?: boolean; directory?: boolean }): Promise<string[]>
  /** Whether the window currently has OS focus. */
  isFocused(): Promise<boolean>
  /** Open a file from Sentient's files folder (`outputs/chart.png`) with the default app. */
  openFile(name: string): Promise<boolean>
  getDevicePrivacy(): Promise<DevicePrivacy>
  setDevicePrivacy(patch: Partial<DevicePrivacy>): Promise<DevicePrivacy>
  onCaptureNotice(cb: (notice: CaptureNotice) => void): () => void
  getDesktopNodeState(): Promise<DesktopNodeState>
  onDesktopNodeState(cb: (state: DesktopNodeState) => void): () => void
  /** Keep the renderer (and the mic) running while hidden, and show it in the tray. */
  setAlwaysListening(enabled: boolean): Promise<void>
  /** `null` = never set in the shell. */
  getAlwaysListening(): Promise<boolean | null>
  /** Tray checkbox changes. */
  onAlwaysListeningChange(cb: (enabled: boolean) => void): () => void
  /** The wake word was heard: show and focus the window. */
  notifyWake(): Promise<void>
}

declare global {
  interface Window {
    sentient?: SentientBridge
  }
}
