/**
 * Plays the per-sentence WAV frames from `/ws/voice` in `sentence_index` order, gaplessly,
 * and exposes the output level for the visualizer. `stop()` silences everything at once.
 */
export class SentencePlayer {
  private ctx: AudioContext | null = null
  private gain: GainNode | null = null
  private analyser: AnalyserNode | null = null
  private scratch: Float32Array<ArrayBuffer> = new Float32Array(512)
  private pending = new Map<number, AudioBuffer>()
  private sources = new Set<AudioBufferSourceNode>()
  private next = 0
  private playhead = 0
  private generation = 0
  onIdle?: () => void

  /** Create/resume the context. Call from a user gesture when possible. */
  ensure(): AudioContext {
    if (!this.ctx || this.ctx.state === 'closed') {
      this.ctx = new AudioContext({ latencyHint: 'interactive' })
      this.gain = this.ctx.createGain()
      this.analyser = this.ctx.createAnalyser()
      this.analyser.fftSize = 512
      this.scratch = new Float32Array(this.analyser.fftSize)
      this.gain.connect(this.analyser)
      this.analyser.connect(this.ctx.destination)
    }
    if (this.ctx.state === 'suspended') void this.ctx.resume().catch(() => undefined)
    return this.ctx
  }

  get playing(): boolean {
    return this.sources.size > 0
  }

  async enqueue(index: number, wav: ArrayBuffer): Promise<void> {
    const ctx = this.ensure()
    if (index === 0) this.resetTurn()
    const gen = this.generation
    let buffer: AudioBuffer
    try {
      buffer = await ctx.decodeAudioData(wav.slice(0))
    } catch (err) {
      console.warn('[voice] could not decode sentence', index, err)
      if (gen === this.generation && index === this.next) {
        this.next++
        this.pump()
      }
      return
    }
    if (gen !== this.generation || index < this.next) return
    this.pending.set(index, buffer)
    this.pump()
  }

  private pump(): void {
    const ctx = this.ctx
    if (!ctx || !this.gain) return
    while (this.pending.has(this.next)) {
      const buffer = this.pending.get(this.next) as AudioBuffer
      this.pending.delete(this.next)
      const src = ctx.createBufferSource()
      src.buffer = buffer
      src.connect(this.gain)
      const at = Math.max(ctx.currentTime + 0.03, this.playhead)
      src.start(at)
      this.playhead = at + buffer.duration
      this.sources.add(src)
      src.onended = () => {
        this.sources.delete(src)
        if (!this.sources.size && !this.pending.size) this.onIdle?.()
      }
      this.next++
    }
  }

  /** A new reply starts at sentence 0. */
  private resetTurn(): void {
    this.silence()
    this.next = 0
  }

  /** Stop sound now and drop anything queued or still decoding. */
  stop(): void {
    this.silence()
  }

  private silence(): void {
    this.generation++
    for (const s of this.sources) {
      s.onended = null
      try {
        s.stop()
      } catch {
        /* not started */
      }
    }
    this.sources.clear()
    this.pending.clear()
    this.playhead = 0
  }

  /** RMS of the current output, 0..1. */
  level(): number {
    if (!this.analyser || !this.sources.size) return 0
    this.analyser.getFloatTimeDomainData(this.scratch)
    let sum = 0
    for (let i = 0; i < this.scratch.length; i++) sum += this.scratch[i] * this.scratch[i]
    return Math.sqrt(sum / this.scratch.length)
  }

  close(): void {
    this.silence()
    void this.ctx?.close().catch(() => undefined)
    this.ctx = null
    this.gain = null
    this.analyser = null
  }
}
