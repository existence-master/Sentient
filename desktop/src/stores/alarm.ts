/**
 * A ringing alarm (#359). An alarm's `reminder` notification (`payload.event = "alarm"`) starts a chime made with
 * Web Audio, so no sound file ships with the app. It rings until Stop or Snooze, or for `MAX_RING_MS` at most.
 *
 * The engine knows which alarm rings too (#390): Stop and Snooze here tell it, and saying "stop" or "snooze" in a chat,
 * by voice or in a messaging app stops or snoozes it there. Its `reminder.updated` events then update the banner:
 * a snooze turns it into "Snoozed until 7:39 AM", a stop hides it.
 */
import { create } from 'zustand'
import { api } from '@/lib/api'
import type { Notification, Reminder } from '@/lib/types'

const MAX_RING_MS = 3 * 60_000 // the engine's RING_S
const PATTERN_MS = 1400

export interface RingingAlarm {
  id: string
  /** The alarm in the engine (`payload.reminder_id`), when the notification has one. */
  reminderId?: string
  title: string
  message: string
  startedAt: number
}

export interface Snoozed {
  /** The new alarm the snooze set. */
  reminderId: string
  title: string
  until: string
}

interface AlarmState {
  ringing: RingingAlarm | null
  snoozed: Snoozed | null
  /** True while the sound is really playing (the audio output started). */
  playing: boolean
  ring: (n: Notification) => void
  /** Stop button: silence it here and tell the engine. */
  stop: () => void
  /** Snooze button: silence it here and ask the engine for a snooze. */
  snooze: () => void
  /** Turn off a snoozed alarm (cancels the alarm the snooze set). */
  turnOff: () => void
  /** A `reminder.updated` event: a stop or snooze from anywhere (#390). */
  updated: (r: Reminder) => void
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

function reminderOf(n: Notification): string | undefined {
  const id = n.payload?.reminder_id
  return typeof id === 'string' && id ? id : undefined
}

export const useAlarm = create<AlarmState>((set, get) => ({
  ringing: null,
  snoozed: null,
  playing: false,
  ring: (n) => {
    silence()
    set({
      ringing: { id: n.id, reminderId: reminderOf(n), title: n.title || 'Alarm', message: n.message, startedAt: Date.now() },
      snoozed: null,
      playing: false
    })
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
      if (get().ringing?.id === n.id) {
        silence() // the engine stops counting it as ringing at the same time
        set({ ringing: null, playing: false })
      }
    }, MAX_RING_MS)
  },
  stop: () => {
    const id = get().ringing?.reminderId
    silence()
    set({ ringing: null, playing: false })
    if (id) void api.reminders.stop(id).catch(() => undefined) // already stopped elsewhere: nothing to do
  },
  snooze: () => {
    const ringing = get().ringing
    if (!ringing?.reminderId) return get().stop()
    silence()
    set({ ringing: null, playing: false })
    void api.reminders.snooze(ringing.reminderId).then(
      ({ reminder }) => {
        if (!get().ringing) set({ snoozed: { reminderId: reminder.id, title: ringing.title, until: reminder.fire_at } })
      },
      () => undefined // it had stopped already
    )
  },
  turnOff: () => {
    const snoozed = get().snoozed
    set({ snoozed: null })
    if (snoozed) void api.reminders.cancel(snoozed.reminderId).catch(() => undefined)
  },
  updated: (r) => {
    const { ringing, snoozed } = get()
    if (ringing?.reminderId && r.snoozed_from === ringing.reminderId && r.status === 'scheduled') {
      silence()
      set({ ringing: null, playing: false, snoozed: { reminderId: r.id, title: ringing.title, until: r.fire_at } })
    } else if (ringing?.reminderId === r.id && r.stopped_at) {
      silence()
      set({ ringing: null, playing: false })
    } else if (snoozed?.reminderId === r.id && r.status !== 'scheduled') {
      set({ snoozed: null }) // it was turned off, or it is ringing again
    }
  }
}))
