/**
 * Pure helpers for the listening pill (#169): when has someone finished speaking, and how full is the level meter.
 * No imports, so `npm test` runs them in Node.
 */

/** Quieter than this never counts as speech, however quiet the room is. */
const MIN_SPEECH_RMS = 0.012
/** Speech is this many times louder than the room's background noise. */
const SPEECH_OVER_FLOOR = 3
/** Less speech than this (a cough, a click) doesn't count. */
const MIN_SPEECH_MS = 200
/** With pause detection on, give up when nobody speaks for this long. */
const NO_SPEECH_MS = 8000

/**
 * Pause detection. Feed it the microphone loudness (RMS, 0 to 1) every few milliseconds; it says `stop` once speech
 * was heard and `silenceMs` of quiet followed, and `nothing` when nobody spoke at all. `silenceMs` 0 turns both off,
 * so only the shortcut stops the recording.
 */
export class SilenceDetector {
  private silenceMs: number
  private floor = 0.004
  private speechMs = 0
  private quietMs = 0
  private elapsedMs = 0

  constructor(silenceMs: number) {
    this.silenceMs = Math.max(0, silenceMs)
  }

  setSilence(ms: number): void {
    this.silenceMs = Math.max(0, ms)
    this.quietMs = 0
  }

  get heardSpeech(): boolean {
    return this.speechMs >= MIN_SPEECH_MS
  }

  feed(rms: number, stepMs: number): 'stop' | 'nothing' | null {
    this.elapsedMs += stepMs
    const speaking = rms > Math.max(MIN_SPEECH_RMS, this.floor * SPEECH_OVER_FLOOR)
    if (speaking) {
      this.speechMs += stepMs
      this.quietMs = 0
    } else {
      this.quietMs += stepMs
      this.floor = Math.min(0.05, this.floor * 0.95 + rms * 0.05)
    }
    if (this.silenceMs <= 0) return null
    if (this.heardSpeech && this.quietMs >= this.silenceMs) return 'stop'
    if (!this.heardSpeech && this.elapsedMs >= Math.max(NO_SPEECH_MS, this.silenceMs)) return 'nothing'
    return null
  }
}

/** Loudness of a block of samples (-1..1). */
export function rms(samples: ArrayLike<number>): number {
  if (!samples.length) return 0
  let sum = 0
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i]
  return Math.sqrt(sum / samples.length)
}

/** Level meter fill (0 to 1): -60 dB is empty, -10 dB is full. */
export function meterLevel(value: number): number {
  if (value <= 0) return 0
  const db = 20 * Math.log10(value)
  return Math.min(1, Math.max(0, (db + 60) / 50))
}
