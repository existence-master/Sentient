/**
 * "Always listening for 'Hey Sentient'": the renderer side of background wake-word mode (docs/API.md §16).
 *
 * While enabled (and not already in Voice mode) this keeps a `/ws/voice` socket in `mode: "wake"`, streams the mic,
 * and opens Voice mode when the engine sends `{type: "wake"}`. It only runs while the renderer is alive.
 *
 * TODO(desktop-a): keep listening when the window is hidden in the tray. Needed from electron main + preload:
 *   - `bridge.setAlwaysListening(enabled: boolean): Promise<void>` over IPC channel `sentient:set-always-listening`
 *     (persist it like other ShellPrefs; add a tray menu checkbox "Always listening for 'Hey Sentient'").
 *   - keep the renderer running while hidden: `webPreferences.backgroundThrottling = false` when enabled, and allow
 *     `media` in `session.setPermissionRequestHandler` so the mic keeps working without focus.
 *   - on wake, show and focus the window and send AppCommand `{type: 'voice-mode'}` (already routed to /voice);
 *     optionally a new `{type: 'voice-mode', wake: true}` so Voice mode skips its setup card.
 *   - `bridge.onAlwaysListeningChange(cb)` so the tray checkbox and Settings stay in sync.
 */
import { create } from 'zustand'
import { api, whenConnected } from '@/lib/api'
import { MicError, startMic, TARGET_RATE, type MicCapture } from './audio/capture'

const PREF_KEY = 'sentient.voice.alwaysListening'
const CHANGE_EVENT = 'sentient:always-listening'

export type WakeListenerStatus = 'off' | 'starting' | 'standby' | 'paused' | 'error'

interface WakeStore {
  enabled: boolean
  status: WakeListenerStatus
  error: string | null
  phrase: string | null
}

function readPref(): boolean {
  try {
    return localStorage.getItem(PREF_KEY) === '1'
  } catch {
    return false
  }
}

export const useWakeStore = create<WakeStore>(() => ({ enabled: readPref(), status: 'off', error: null, phrase: null }))

/** Turn background wake-word listening on or off (persisted per computer). */
export function setAlwaysListening(enabled: boolean): void {
  try {
    localStorage.setItem(PREF_KEY, enabled ? '1' : '0')
  } catch {
    /* private mode */
  }
  useWakeStore.setState({ enabled, error: null })
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT))
}

const inVoiceMode = () => window.location.hash.startsWith('#/voice')
const isSmoke = () => /[?&](demo|voicePreview)=/.test(window.location.hash)

class WakeListener {
  private ws: WebSocket | null = null
  private mic: MicCapture | null = null
  private retry: number | undefined
  private attempts = 0

  sync(): void {
    const { enabled } = useWakeStore.getState()
    if (!enabled) return this.stop('off')
    if (inVoiceMode() || isSmoke()) return this.stop('paused')
    if (!this.ws) void this.start()
  }

  private async start(): Promise<void> {
    useWakeStore.setState({ status: 'starting', error: null })
    await whenConnected()
    const url = api.voice.socketUrl()
    if (!url || !useWakeStore.getState().enabled || inVoiceMode()) return
    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    this.ws = ws
    ws.onopen = () => ws.send(JSON.stringify({ type: 'start', mode: 'wake', sample_rate: TARGET_RATE }))
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return
      let msg: { type?: string; state?: string; phrase?: string; message?: string; recoverable?: boolean }
      try {
        msg = JSON.parse(ev.data)
      } catch {
        return
      }
      if (msg.type === 'state' && msg.state === 'standby') {
        this.attempts = 0
        useWakeStore.setState({ status: 'standby', error: null })
      } else if (msg.type === 'wake') {
        useWakeStore.setState({ phrase: msg.phrase ?? null })
        this.stop('paused')
        window.location.hash = '#/voice?wake=1'
      } else if (msg.type === 'error' && msg.recoverable === false) {
        useWakeStore.setState({ status: 'error', error: msg.message ?? 'Wake word listening stopped.' })
      }
    }
    ws.onclose = () => {
      if (this.ws !== ws) return
      this.teardown()
      if (!useWakeStore.getState().enabled || inVoiceMode()) return
      useWakeStore.setState({ status: 'error', error: useWakeStore.getState().error ?? 'Lost the connection to the voice engine. Trying again…' })
      const delay = Math.min(60_000, 2_000 * 2 ** this.attempts++)
      window.clearTimeout(this.retry)
      this.retry = window.setTimeout(() => this.sync(), delay)
    }
    try {
      this.mic = await startMic((pcm) => {
        if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(pcm)
      })
      if (this.ws !== ws) {
        this.mic.stop()
        this.mic = null
      }
    } catch (err) {
      useWakeStore.setState({ status: 'error', error: err instanceof MicError ? err.message : String(err) })
      this.stop('error')
    }
  }

  private teardown(): void {
    this.mic?.stop()
    this.mic = null
    this.ws = null
  }

  stop(status: WakeListenerStatus): void {
    window.clearTimeout(this.retry)
    const ws = this.ws
    this.teardown()
    if (ws) {
      ws.onclose = null
      try {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'stop' }))
        ws.close()
      } catch {
        /* ignore */
      }
    }
    useWakeStore.setState((s) => ({ status, error: status === 'error' ? s.error : null }))
  }
}

const listener = new WakeListener()

/** Installed once from lib/leap/events-b.ts. Does nothing unless the user turned it on. */
export function installWakeListener(): () => void {
  const sync = () => listener.sync()
  window.addEventListener('hashchange', sync)
  window.addEventListener(CHANGE_EVENT, sync)
  const t = window.setTimeout(sync, 1500)
  return () => {
    window.clearTimeout(t)
    window.removeEventListener('hashchange', sync)
    window.removeEventListener(CHANGE_EVENT, sync)
    listener.stop('off')
  }
}
