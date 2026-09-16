/**
 * Access to the Electron preload bridge, with a browser fallback.
 *
 * In a normal browser (`npm run dev:web`, or the UI served by `sentient serve`)
 * `window.sentient` is missing. The stub reads the connection from
 * `?api=http://127.0.0.1:7777&token=...` (persisted to localStorage), or uses the page
 * origin when served by the backend, and polls /api/health for status.
 */
import type {
  AppCommand,
  BackendStatus,
  Connection,
  SentientBridge,
  VersionInfo
} from '@/types/bridge'

const STORAGE_KEY = 'sentient.connection'

function readBrowserConnection(): Connection {
  const params = new URLSearchParams(window.location.search)
  let stored: Partial<Connection> = {}
  try {
    stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') as Partial<Connection>
  } catch {
    stored = {}
  }
  const apiParam = params.get('api')
  const tokenParam = params.get('token')
  const servedByBackend = window.location.protocol.startsWith('http') && !import.meta.env.DEV
  const baseUrl = (apiParam ?? stored.baseUrl ?? (servedByBackend ? window.location.origin : 'http://127.0.0.1:7777')).replace(
    /\/+$/,
    ''
  )
  const token = tokenParam ?? stored.token ?? ''
  if (apiParam || tokenParam) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ baseUrl, token }))
    } catch {
      /* private mode */
    }
  }
  return { baseUrl, token }
}

function createBrowserBridge(): SentientBridge {
  const conn = readBrowserConnection()
  let last: BackendStatus = { state: 'starting' }
  const listeners = new Set<(s: BackendStatus) => void>()

  async function probe(): Promise<BackendStatus> {
    try {
      const r = await fetch(`${conn.baseUrl}/api/health`, { signal: AbortSignal.timeout(2500) })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      if (!conn.token) {
        return {
          state: 'failed',
          message: 'No token. Open this page with ?api=<backend url>&token=<gateway token>.'
        }
      }
      return { state: 'ready', baseUrl: conn.baseUrl }
    } catch (err) {
      return {
        state: 'failed',
        message: `Can't reach the engine at ${conn.baseUrl} (${String(err)}). Run: sentient serve`
      }
    }
  }

  let timer: number | undefined
  const poll = async () => {
    const next = await probe()
    if (next.state !== last.state) {
      last = next
      listeners.forEach((l) => l(next))
    }
    timer = window.setTimeout(poll, next.state === 'ready' ? 10_000 : 3_000)
  }

  const noopUnsub = () => () => undefined
  const version: VersionInfo = {
    app: 'web',
    electron: '-',
    chrome: navigator.userAgent,
    node: '-',
    platform: 'web',
    arch: '-'
  }

  return {
    isDesktop: false,
    platform: 'web',
    getConnection: async () => conn,
    getBackendStatus: async () => {
      last = await probe()
      return last
    },
    onBackendStatus: (cb) => {
      listeners.add(cb)
      if (timer === undefined) timer = window.setTimeout(poll, 3000)
      return () => {
        listeners.delete(cb)
        if (!listeners.size && timer !== undefined) {
          clearTimeout(timer)
          timer = undefined
        }
      }
    },
    restartBackend: async () => {
      window.location.reload()
    },
    openExternal: async (url) => {
      window.open(url, '_blank', 'noopener,noreferrer')
    },
    openPath: async () => undefined,
    showNotification: async (n) => {
      if (!('Notification' in window)) return
      if (Notification.permission === 'default') await Notification.requestPermission()
      if (Notification.permission === 'granted') {
        const note = new Notification(n.title, { body: n.body, silent: n.silent })
        note.onclick = () => {
          window.focus()
          if (n.route) window.location.hash = n.route
        }
      }
    },
    setLaunchAtLogin: async () => undefined,
    getLaunchAtLogin: async () => false,
    syncPrefs: async () => undefined,
    onCommand: noopUnsub as unknown as (cb: (cmd: AppCommand) => void) => () => void,
    window: {
      minimize: async () => undefined,
      maximize: async () => undefined,
      close: async () => undefined,
      isMaximized: async () => false,
      onMaximizedChange: () => () => undefined
    },
    getVersion: async () => version,
    readyForScreenshot: () => undefined,
    pickFiles: async () => [],
    isFocused: async () => document.hasFocus(),
    openFile: async () => false,
    getDevicePrivacy: async () => ({ screen: null, camera: null }),
    setDevicePrivacy: async (patch) => ({ screen: null, camera: null, ...patch }),
    onCaptureNotice: () => () => undefined,
    getDesktopNodeState: async () => 'unsupported' as const,
    onDesktopNodeState: () => () => undefined,
    setAlwaysListening: async () => undefined,
    getAlwaysListening: async () => null,
    onAlwaysListeningChange: () => () => undefined,
    notifyWake: async () => {
      window.focus()
    }
  }
}

let cached: SentientBridge | null = null

export function getBridge(): SentientBridge {
  if (cached) return cached
  cached = window.sentient ?? createBrowserBridge()
  return cached
}

export const isDesktop = () => getBridge().isDesktop
