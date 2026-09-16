/**
 * AudioWorklet processor: microphone -> 16 kHz mono PCM16 LE frames (20 ms) posted to the main thread.
 * Bundled by Vite (`?worker&url`), so it loads from the app itself (no CDN, CSP 'self').
 */
import { Pcm16Resampler } from './resampler'

declare const sampleRate: number
declare function registerProcessor(name: string, ctor: unknown): void
declare class AudioWorkletProcessor {
  readonly port: MessagePort
  constructor(options?: unknown)
}

interface CaptureOptions {
  processorOptions?: { targetRate?: number; frameSamples?: number }
}

class Pcm16CaptureProcessor extends AudioWorkletProcessor {
  private readonly resampler: Pcm16Resampler

  constructor(options: CaptureOptions) {
    super(options)
    const target = options.processorOptions?.targetRate ?? 16000
    const frame = options.processorOptions?.frameSamples ?? Math.round(target * 0.02)
    this.resampler = new Pcm16Resampler(sampleRate, target, frame, (pcm, level) => this.port.postMessage({ pcm, level }, [pcm]))
  }

  process(inputs: Float32Array[][]): boolean {
    const input = inputs[0]
    if (input && input.length) this.resampler.push(input)
    return true
  }
}

registerProcessor('pcm16-capture', Pcm16CaptureProcessor)
