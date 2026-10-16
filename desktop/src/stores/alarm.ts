/**
 * A ringing alarm (#359). An alarm's `reminder` notification (`payload.event = "alarm"`) starts a chime made with
 * Web Audio, so no sound file ships with the app. It rings until Stop, or for `MAX_RING_MS` at most.
 */
import { create } from 'zustand'
import type { Notification } from '@/lib/types'

const MAX_RING_MS = 3 * 60_000
const PATTERN_MS = 1400

export interface RingingAlarm {
  id: string
  title: string
  message: string
  startedAt: number
}

interface AlarmState {
  ringing: RingingAlarm | null
  /** True while the sound is really playing (the audio output started). */
  playing: boolean
  ring: (n: Notification) => void
  stop: () => void
}

let ctx: AudioContext | null = null
let beat: number | undefined
let limit: number | undefined

/** One "beep beep" of two short tones. */
function chime(audio: AudioContext): void {
  const now = audio.currentTime
  for (const [offset, freq] of [[0, 880], [0.22, 1175]] as const) {
    const osc = audio.createOscillator()
    const gain = audio.createGain()
    osc.type = 'sine'
    osc.frequency.value = freq
    gain.gain.setValueAtTime(0.0001, now + offset)
    gain.gain.exponentialRampToValueAtTime(0.35, now + offset + 0.02)
    gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.18)
    osc.connect(gain).connect(audio.destination)
    osc.start(now + offset)
    osc.stop(now + offset + 0.2)
  }
}

function silence(): void {
  window.clearInterval(beat)
  window.clearTimeout(limit)
  beat = limit = undefined
  const audio = ctx
  ctx = null
  if (audio) void audio.close().catch(() => undefined)
}

export function isAlarm(n: Notification): boolean {
  return n.kind === 'reminder' && n.payload?.event === 'alarm'
}

export const useAlarm = create<AlarmState>((set, get) => ({
  ringing: null,
  playing: false,
  ring: (n) => {
    silence()
    set({ ringing: { id: n.id, title: n.title || 'Alarm', message: n.message, startedAt: Date.now() }, playing: false })
    try {
      const audio = new AudioContext()
      ctx = audio
      const start = () => {
        if (ctx !== audio) return
        chime(audio)
        beat = window.setInterval(() => chime(audio), PATTERN_MS)
        set({ playing: audio.state === 'running' })
        audio.onstatechange = () => {
          if (ctx === audio) set({ playing: audio.state === 'running' })
        }
      }
      void audio.resume().then(start, start)
    } catch {
      set({ playing: false }) // no audio output: the card and the notification still show
    }
    limit = window.setTimeout(() => {
      if (get().ringing?.id === n.id) get().stop()
    }, MAX_RING_MS)
  },
  stop: () => {
    silence()
    set({ ringing: null, playing: false })
  }
}))
