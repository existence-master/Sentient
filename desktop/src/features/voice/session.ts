/**
 * The live voice conversation over `WS /ws/voice` (docs/API.md §9): socket, mic capture,
 * sentence playback and the transcript, exposed to React through a zustand store.
 */
import { toast } from 'sonner'
import { create } from 'zustand'
import { api } from '@/lib/api'
import { qk } from '@/hooks/queryKeys'
import type { VoiceStateB } from '@/lib/leap/types-b'
import { queryClient } from '@/lib/queryClient'
import type { ApprovalDecision, Risk, VoiceAudioMetrics, VoiceServerMessage } from '@/lib/types'
import { uid } from '@/lib/utils'
import { MicError, startMic, TARGET_RATE, type MicCapture } from './audio/capture'
import { SentencePlayer } from './audio/player'

export type VoicePhase = 'setup' | 'connecting' | 'live' | 'ended' | 'error'
export type InputMode = 'handsfree' | 'ptt'

export interface VoiceToolChip {
  callId: string
  name: string
  status: 'running' | 'done' | 'error' | 'awaiting'
}

export interface VoiceApproval {
  approvalId: string
  callId: string
  name: string
  risk: Risk
  reason: string
  arguments: Record<string, unknown>
  status: 'pending' | ApprovalDecision | 'expired'
}

export interface VoiceTurn {
  id: string
  role: 'user' | 'assistant'
  text: string
  /** Sentences already sent as audio (assistant). */
  spoken: string[]
  tools: VoiceToolChip[]
  approvals: VoiceApproval[]
  done: boolean
  cancelled?: boolean
  interrupted?: boolean
  error?: string
}

interface VoiceStoreState {
  phase: VoicePhase
  state: VoiceStateB
  sessionId: string | null
  mode: InputMode
  muted: boolean
  talking: boolean
  micError: string | null
  error: string | null
  engine: { stt: string; tts: string } | null
  turns: VoiceTurn[]
  metrics: VoiceAudioMetrics | null
  preview: boolean
  /** Session runs in `mode: "wake"`: standby until "Hey Sentient". */
  wakeMode: boolean
  /** Phrase the engine reported on the last wake. */
  wakePhrase: string | null
  /** When the last wake happened (ms), for the wake animation. */
  wokeAt: number | null
  /** Follow-ups are accepted without the wake word until this time (ms). */
  followUpUntil: number | null
  followUpMs: number | null
}

const initial: VoiceStoreState = {
  phase: 'setup',
  state: 'idle',
  sessionId: null,
  mode: 'handsfree',
  muted: false,
  talking: false,
  micError: null,
  error: null,
  engine: null,
  turns: [],
  metrics: null,
  preview: false,
  wakeMode: false,
  wakePhrase: null,
  wokeAt: null,
  followUpUntil: null,
  followUpMs: null
}

function followUpMs(): number {
  const cfg = queryClient.getQueryData(qk.config) as { voice?: { follow_up_seconds?: number } } | undefined
  const s = Number(cfg?.voice?.follow_up_seconds)
  return (Number.isFinite(s) && s > 0 ? s : 8) * 1000
}

export const useVoiceStore = create<VoiceStoreState>(() => ({ ...initial }))

/** Live levels (0..1) read by the visualizer every animation frame; not React state on purpose. */
export const levels = { mic: 0, out: 0 }

const BARGE_LEVEL = 0.09
const BARGE_MS = 340
const PING_MS = 20_000

type AudioHeader = Extract<VoiceServerMessage, { type: 'audio' }>

class VoiceController {
  private ws: WebSocket | null = null
  private mic: MicCapture | null = null
  private player = new SentencePlayer()
  private headers: AudioHeader[] = []
  private ping: number | undefined
  private levelTimer: number | undefined
  private stopping = false
  private bargeMs = 0

  private get s() {
    return useVoiceStore.getState()
  }
  private set(patch: Partial<VoiceStoreState>) {
    useVoiceStore.setState(patch)
  }

  // ------------------------------------------------------------------ lifecycle
  async start(opts: { sessionId?: string | null; keepTranscript?: boolean; wake?: boolean; woke?: boolean } = {}): Promise<void> {
    if (this.s.phase === 'connecting' || this.s.phase === 'live') return
    const url = api.voice.socketUrl()
    if (!url) {
      this.set({ phase: 'error', error: 'Not connected to the Sentient engine yet.' })
      return
    }
    this.stopping = false
    this.headers = []
    this.player.ensure()
    this.set({
      phase: 'connecting',
      state: 'idle',
      error: null,
      micError: null,
      metrics: null,
      talking: false,
      preview: false,
      sessionId: opts.sessionId ?? null,
      turns: opts.keepTranscript ? this.s.turns : [],
      wakeMode: opts.wake ?? this.s.wakeMode,
      wokeAt: opts.woke ? Date.now() : null,
      followUpUntil: null
    })

    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    this.ws = ws
    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: 'start',
          sample_rate: TARGET_RATE,
          ...(opts.sessionId ? { session_id: opts.sessionId } : {}),
          ...(this.s.wakeMode ? { mode: 'wake' } : {})
        })
      )
    }
    ws.onmessage = (ev) => this.onMessage(ev)
    ws.onclose = () => {
      if (this.ws !== ws) return
      this.teardown()
      if (this.stopping) return
      this.set({ phase: 'error', state: 'idle', error: 'The connection to the voice engine was lost.' })
    }
    ws.onerror = () => {
      /* onclose follows */
    }

    this.levelTimer = window.setInterval(() => {
      levels.out = this.player.level()
    }, 33)

    try {
      this.mic = await startMic((pcm, level) => this.onFrame(pcm, level))
      if (this.stopping || this.ws !== ws) {
        this.mic.stop()
        this.mic = null
        return
      }
      this.mic.setEnabled(!this.s.muted)
    } catch (err) {
      this.set({ micError: err instanceof MicError ? err.message : String(err) })
    }
  }

  stop(): void {
    if (this.s.phase !== 'live' && this.s.phase !== 'connecting') return
    this.stopping = true
    const ws = this.ws
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }))
      window.setTimeout(() => ws.readyState === WebSocket.OPEN && ws.close(), 1500)
    } else ws?.close()
    this.teardown()
    this.set({ phase: this.s.sessionId ? 'ended' : 'setup', state: 'idle', talking: false, followUpUntil: null })
  }

  /** Stop everything without changing the transcript (route left). */
  dispose(): void {
    if (this.s.preview) {
      useVoiceStore.setState({ ...initial })
      return
    }
    this.stop()
  }

  reset(): void {
    this.stop()
    useVoiceStore.setState({ ...initial, mode: this.s.mode })
  }

  private teardown(): void {
    window.clearInterval(this.ping)
    window.clearInterval(this.levelTimer)
    this.ping = undefined
    this.levelTimer = undefined
    this.mic?.stop()
    this.mic = null
    this.player.stop()
    this.headers = []
    levels.mic = 0
    levels.out = 0
    this.bargeMs = 0
  }

  private send(msg: Record<string, unknown>): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false
    this.ws.send(JSON.stringify(msg))
    return true
  }

  // ------------------------------------------------------------------ controls
  sendText(text: string): void {
    const t = text.trim()
    if (!t) return
    this.player.stop()
    if (!this.send({ type: 'text', text: t })) toast.error('Voice mode is not connected')
  }

  interrupt(): void {
    this.player.stop()
    levels.out = 0
    this.send({ type: 'interrupt' })
  }

  setMode(mode: InputMode): void {
    this.set({ mode, talking: false })
  }

  /** Switch between talking freely and waiting for "Hey Sentient" (keeps the conversation). */
  setWakeMode(on: boolean): void {
    this.set({ wakeMode: on, followUpUntil: null })
    if (this.s.phase !== 'live') return
    this.send({ type: 'start', sample_rate: TARGET_RATE, mode: on ? 'wake' : 'conversation', ...(this.s.sessionId ? { session_id: this.s.sessionId } : {}) })
  }

  /** Wake from standby without saying the phrase (orb click). */
  wakeNow(): void {
    if (this.s.state !== 'standby') return
    this.send({ type: 'wake' })
  }

  toggleMute(): void {
    const muted = !this.s.muted
    this.mic?.setEnabled(!muted)
    if (muted) levels.mic = 0
    this.set({ muted })
  }

  pttDown(): void {
    if (this.s.mode !== 'ptt' || this.s.phase !== 'live' || this.s.talking) return
    if (this.s.state === 'speaking' || this.player.playing) this.interrupt()
    if (this.s.muted) this.toggleMute()
    this.set({ talking: true })
  }

  pttUp(): void {
    if (!this.s.talking) return
    this.set({ talking: false })
    this.send({ type: 'end_utterance' })
  }

  respondApproval(approvalId: string, decision: ApprovalDecision): void {
    this.patchApproval(approvalId, { status: decision })
    if (!this.send({ type: 'approval.respond', approval_id: approvalId, decision })) {
      void api.approvals.respond(approvalId, decision).catch(() => this.patchApproval(approvalId, { status: 'expired' }))
    }
  }

  // ------------------------------------------------------------------ audio in
  private onFrame(pcm: ArrayBuffer, level: number): void {
    const s = this.s
    levels.mic = s.muted ? 0 : levels.mic * 0.5 + Math.min(1, level) * 0.5
    if (s.phase !== 'live' || s.muted || this.ws?.readyState !== WebSocket.OPEN) return
    if (s.mode === 'ptt' && !s.talking) return
    this.ws.send(pcm)
    if (s.mode === 'handsfree' && this.player.playing) {
      this.bargeMs = level > BARGE_LEVEL ? this.bargeMs + 20 : Math.max(0, this.bargeMs - 20)
      if (this.bargeMs >= BARGE_MS) {
        this.bargeMs = 0
        this.interrupt()
      }
    } else this.bargeMs = 0
  }

  // ------------------------------------------------------------------ server messages
  private onMessage(ev: MessageEvent): void {
    if (typeof ev.data !== 'string') {
      const header = this.headers.shift()
      if (header && ev.data instanceof ArrayBuffer) void this.player.enqueue(header.sentence_index, ev.data)
      return
    }
    let msg: VoiceServerMessage
    try {
      msg = JSON.parse(ev.data) as VoiceServerMessage
    } catch {
      return
    }
    const loose = msg as unknown as { type: string; phrase?: string; earcon?: boolean }
    if (loose.type === 'wake') {
      this.set({ wokeAt: Date.now(), wakePhrase: loose.phrase ?? this.s.wakePhrase, followUpUntil: null })
      return
    }
    switch (msg.type) {
      case 'ready':
        this.set({ phase: 'live', sessionId: msg.session_id, engine: { stt: msg.stt, tts: msg.tts } })
        window.clearInterval(this.ping)
        this.ping = window.setInterval(() => this.send({ type: 'ping' }), PING_MS)
        break
      case 'state':
        this.set((msg.state as string) === 'standby' ? { state: 'standby', followUpUntil: null } : { state: msg.state })
        break
      case 'transcript':
        this.player.stop()
        if (this.s.followUpUntil) this.set({ followUpUntil: null })
        this.pushTurn({ role: 'user', text: msg.text, done: true })
        break
      case 'text_delta':
        this.patchAssistant((t) => ({ text: t.text + msg.text }))
        break
      case 'tool_call':
        this.patchAssistant((t) => ({ tools: [...t.tools.filter((c) => c.callId !== msg.call_id), { callId: msg.call_id, name: msg.name, status: 'running' }] }))
        break
      case 'tool_result':
        this.patchAssistant((t) => ({
          tools: t.tools.map((c) => (c.callId === msg.call_id ? { ...c, status: msg.is_error ? 'error' : 'done' } : c)),
          approvals: t.approvals.map((a) => (a.callId === msg.call_id && a.status === 'pending' ? { ...a, status: 'allow' } : a))
        }))
        break
      case 'approval_request':
        this.patchAssistant((t) => ({
          tools: t.tools.map((c) => (c.callId === msg.call_id ? { ...c, status: 'awaiting' } : c)),
          approvals: [
            ...t.approvals,
            { approvalId: msg.approval_id, callId: msg.call_id, name: msg.name, risk: msg.risk, reason: msg.reason, arguments: msg.arguments ?? {}, status: 'pending' }
          ]
        }))
        break
      case 'approval.ack':
        if (!msg.resolved) this.patchApproval(msg.approval_id, { status: 'expired' })
        break
      case 'audio':
        this.headers.push(msg)
        // the wake chime (`earcon: true`) is played but is not part of the reply
        if (!loose.earcon) this.patchAssistant((t) => ({ spoken: [...t.spoken, msg.text] }))
        break
      case 'audio_end':
        if (msg.interrupted) {
          this.player.stop()
          if (msg.reason !== 'cancelled') this.patchLastAssistant({ interrupted: true })
        } else {
          if (msg.metrics) this.set({ metrics: msg.metrics })
          if (this.s.wakeMode) {
            const ms = followUpMs()
            this.set({ followUpUntil: Date.now() + ms, followUpMs: ms })
          }
        }
        break
      case 'done':
        this.patchLastAssistant((t) => ({
          done: true,
          cancelled: msg.cancelled || undefined,
          text: t.text || (msg.content ?? '')
        }))
        break
      case 'error':
        if (msg.recoverable === false) {
          this.patchLastAssistant({ error: msg.message, done: true })
          toast.error('Voice error', { description: msg.message })
        } else if (!/starting a new one/.test(msg.message)) {
          toast(msg.message)
        }
        break
      default:
        break
    }
  }

  // ------------------------------------------------------------------ transcript helpers
  private pushTurn(t: Pick<VoiceTurn, 'role' | 'text' | 'done'>): void {
    const turns = this.s.turns.map((x) => (x.role === 'assistant' && !x.done ? { ...x, done: true } : x))
    // drop empty cancelled assistant placeholders
    const cleaned = turns.filter((x) => !(x.role === 'assistant' && !x.text && !x.tools.length && !x.spoken.length && !x.approvals.length))
    this.set({ turns: [...cleaned, { id: uid(), spoken: [], tools: [], approvals: [], ...t }] })
  }

  private patchAssistant(fn: (t: VoiceTurn) => Partial<VoiceTurn>): void {
    const turns = this.s.turns.slice()
    let last = turns[turns.length - 1]
    if (!last || last.role !== 'assistant' || last.done) {
      last = { id: uid(), role: 'assistant', text: '', spoken: [], tools: [], approvals: [], done: false }
      turns.push(last)
    }
    turns[turns.length - 1] = { ...last, ...fn(last) }
    this.set({ turns })
  }

  private patchLastAssistant(patch: Partial<VoiceTurn> | ((t: VoiceTurn) => Partial<VoiceTurn>)): void {
    const turns = this.s.turns.slice()
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].role === 'assistant') {
        turns[i] = { ...turns[i], ...(typeof patch === 'function' ? patch(turns[i]) : patch) }
        const empty = turns[i].cancelled && !turns[i].text && !turns[i].tools.length && !turns[i].spoken.length
        if (empty) turns.splice(i, 1)
        this.set({ turns })
        return
      }
      if (turns[i].role === 'user') return
    }
  }

  private patchApproval(approvalId: string, patch: Partial<VoiceApproval>): void {
    this.set({
      turns: this.s.turns.map((t) =>
        t.approvals.some((a) => a.approvalId === approvalId)
          ? { ...t, approvals: t.approvals.map((a) => (a.approvalId === approvalId ? { ...a, ...patch } : a)) }
          : t
      )
    })
  }

  // ------------------------------------------------------------------ preview (dev only)
  loadPreview(data: Partial<VoiceStoreState>): () => void {
    useVoiceStore.setState({ ...initial, ...data, preview: true })
    let raf = 0
    const tick = (now: number) => {
      const t = now / 1000
      const st = this.s.state
      const syllables = Math.max(0, Math.sin(t * 7.3) * 0.5 + Math.sin(t * 3.1 + 1) * 0.35 + Math.sin(t * 13.7) * 0.15)
      levels.mic = st === 'listening' ? 0.02 + syllables * 0.16 : 0
      levels.out = st === 'speaking' ? 0.04 + syllables * 0.22 : 0
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      levels.mic = 0
      levels.out = 0
    }
  }
}

export const voice = new VoiceController()
