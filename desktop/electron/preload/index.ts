/**
 * Preload (sandboxed): exposes the typed `window.sentient` bridge.
 * Contract and docs: src/types/bridge.d.ts
 */
import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'
import type {
  AppCommand,
  BackendStatus,
  CaptureNotice,
  Connection,
  DesktopNodeState,
  DevicePrivacy,
  DictationCommand,
  DictationEvent,
  DictationShellSettings,
  DictationStatus,
  NativeNotification,
  OpenPathTarget,
  SentientBridge,
  ShellPrefs,
  ShortcutInfo,
  ShortcutResult,
  VersionInfo
} from '../../src/types/bridge'
import { CH } from '../main/channels'

function subscribe<T>(channel: string, cb: (payload: T) => void): () => void {
  const listener = (_e: IpcRendererEvent, payload: T) => cb(payload)
  ipcRenderer.on(channel, listener)
  return () => {
    ipcRenderer.removeListener(channel, listener)
  }
}

const platform = (['win32', 'darwin', 'linux'] as const).find((p) => p === process.platform) ?? 'linux'

const bridge: SentientBridge = {
  isDesktop: true,
  platform,
  getConnection: () => ipcRenderer.invoke(CH.getConnection) as Promise<Connection>,
  getBackendStatus: () => ipcRenderer.invoke(CH.getBackendStatus) as Promise<BackendStatus>,
  onBackendStatus: (cb) => subscribe<BackendStatus>(CH.backendStatus, cb),
  restartBackend: () => ipcRenderer.invoke(CH.restartBackend),
  openExternal: (url: string) => ipcRenderer.invoke(CH.openExternal, url),
  openPath: (target: OpenPathTarget) => ipcRenderer.invoke(CH.openPath, target),
  showNotification: (n: NativeNotification) => ipcRenderer.invoke(CH.showNotification, n),
  setLaunchAtLogin: (enabled: boolean) => ipcRenderer.invoke(CH.setLaunchAtLogin, enabled),
  getLaunchAtLogin: () => ipcRenderer.invoke(CH.getLaunchAtLogin) as Promise<boolean>,
  syncPrefs: (prefs: ShellPrefs) => ipcRenderer.invoke(CH.syncPrefs, prefs),
  onCommand: (cb) => subscribe<AppCommand>(CH.command, cb),
  window: {
    minimize: () => ipcRenderer.invoke(CH.winMinimize),
    maximize: () => ipcRenderer.invoke(CH.winMaximize),
    close: () => ipcRenderer.invoke(CH.winClose),
    isMaximized: () => ipcRenderer.invoke(CH.winIsMaximized) as Promise<boolean>,
    onMaximizedChange: (cb) => subscribe<boolean>(CH.winMaximizedChanged, cb)
  },
  getVersion: () => ipcRenderer.invoke(CH.getVersion) as Promise<VersionInfo>,
  readyForScreenshot: () => ipcRenderer.send(CH.readyForScreenshot),
  pickFiles: (options) => ipcRenderer.invoke(CH.pickFiles, options) as Promise<string[]>,
  isFocused: () => ipcRenderer.invoke(CH.isFocused) as Promise<boolean>,
  openFile: (name: string) => ipcRenderer.invoke(CH.openFile, name) as Promise<boolean>,
  getDevicePrivacy: () => ipcRenderer.invoke(CH.getDevicePrivacy) as Promise<DevicePrivacy>,
  setDevicePrivacy: (patch) => ipcRenderer.invoke(CH.setDevicePrivacy, patch) as Promise<DevicePrivacy>,
  onCaptureNotice: (cb) => subscribe<CaptureNotice>(CH.captureNotice, cb),
  getDesktopNodeState: () => ipcRenderer.invoke(CH.getDesktopNodeState) as Promise<DesktopNodeState>,
  onDesktopNodeState: (cb) => subscribe<DesktopNodeState>(CH.desktopNodeState, cb),
  setAlwaysListening: (enabled: boolean) => ipcRenderer.invoke(CH.setAlwaysListening, enabled),
  getAlwaysListening: () => ipcRenderer.invoke(CH.getAlwaysListening) as Promise<boolean | null>,
  onAlwaysListeningChange: (cb) => subscribe<boolean>(CH.alwaysListeningChanged, cb),
  notifyWake: () => ipcRenderer.invoke(CH.wakeDetected),
  dictation: {
    apply: (settings: DictationShellSettings) => ipcRenderer.invoke(CH.dictationApply, settings) as Promise<DictationStatus>,
    status: () => ipcRenderer.invoke(CH.dictationStatus) as Promise<DictationStatus>,
    cancel: () => ipcRenderer.invoke(CH.dictationCancel),
    openPermissionSettings: (kind) => ipcRenderer.invoke(CH.dictationPermission, kind),
    onCommand: (cb) => subscribe<DictationCommand>(CH.dictationCommand, cb),
    report: (event: DictationEvent) => ipcRenderer.send(CH.dictationEvent, event)
  },
  getShortcuts: () => ipcRenderer.invoke(CH.getShortcuts) as Promise<ShortcutInfo[]>,
  setShortcut: (id, accelerator) => ipcRenderer.invoke(CH.setShortcut, id, accelerator) as Promise<ShortcutResult>
}

contextBridge.exposeInMainWorld('sentient', bridge)
