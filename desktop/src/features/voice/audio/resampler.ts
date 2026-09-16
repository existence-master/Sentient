/**
 * Streaming downmix + low-pass + linear resampler to PCM16 little-endian mono frames.
 * Shared by the AudioWorklet (preferred) and the ScriptProcessor fallback.
 */
export class Pcm16Resampler {
  private readonly ratio: number
  private readonly alpha: number
  private frame: Int16Array
  private fill = 0
  /** Read position relative to the current block; may be in [-1, 0) (between last block's tail and this block). */
  private pos = 0
  private last = 0
  private lp = 0
  private energy = 0
  private count = 0

  constructor(
    inputRate: number,
    targetRate: number,
    private readonly frameSamples: number,
    private readonly emit: (pcm: ArrayBuffer, level: number) => void
  ) {
    this.ratio = inputRate / targetRate
    // one-pole low-pass at ~0.45 * target rate to tame aliasing before decimation
    const fc = Math.min(targetRate * 0.45, inputRate * 0.45)
    this.alpha = 1 - Math.exp((-2 * Math.PI * fc) / inputRate)
    this.frame = new Int16Array(frameSamples)
  }

  push(channels: Float32Array[]): void {
    const n = channels[0]?.length ?? 0
    if (!n) return
    const mono = new Float32Array(n)
    const chs = channels.length
    for (let c = 0; c < chs; c++) {
      const data = channels[c]
      for (let i = 0; i < n; i++) mono[i] += data[i] / chs
    }
    for (let i = 0; i < n; i++) {
      this.energy += mono[i] * mono[i]
      this.lp += this.alpha * (mono[i] - this.lp)
      mono[i] = this.lp
    }
    this.count += n

    while (this.pos < n - 1) {
      const i0 = Math.floor(this.pos)
      const frac = this.pos - i0
      const s0 = i0 < 0 ? this.last : mono[i0]
      const s1 = mono[i0 + 1]
      const s = s0 + (s1 - s0) * frac
      const v = Math.max(-1, Math.min(1, s))
      this.frame[this.fill++] = v < 0 ? v * 0x8000 : v * 0x7fff
      if (this.fill === this.frameSamples) this.flush()
      this.pos += this.ratio
    }
    this.pos -= n
    this.last = mono[n - 1]
  }

  private flush(): void {
    const level = this.count ? Math.sqrt(this.energy / this.count) : 0
    const buf = this.frame.buffer as ArrayBuffer
    this.frame = new Int16Array(this.frameSamples)
    this.fill = 0
    this.energy = 0
    this.count = 0
    this.emit(buf, level)
  }
}
