import workletUrl from './pcm-capture.worklet.ts?worker&url'
import { Pcm16Resampler } from './resampler'

export const TARGET_RATE = 16000
/** 20 ms frames. */
export const FRAME_SAMPLES = 320

export interface MicCapture {
  readonly inputRate: number
  readonly usingWorklet: boolean
  setEnabled(on: boolean): void
  stop(): void
}

export class MicError extends Error {
  constructor(
    message: string,
    readonly code: 'denied' | 'missing' | 'busy' | 'unsupported' | 'unknown'
  ) {
    super(message)
  }
}

function friendly(err: unknown): MicError {
  const name = (err as { name?: string })?.name ?? ''
  if (name === 'NotAllowedError' || name === 'SecurityError')
    return new MicError('Microphone access is blocked. Allow it in your system privacy settings, or type instead.', 'denied')
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return new MicError('No microphone was found. You can still type to Sentient.', 'missing')
  if (name === 'NotReadableError' || name === 'AbortError') return new MicError('The microphone is in use by another app.', 'busy')
  return new MicError(`Couldn't open the microphone: ${(err as Error)?.message ?? String(err)}`, 'unknown')
}

/**
 * Opens the mic with echo cancellation, noise suppression and AGC, and streams
 * 16 kHz PCM16 frames to `onFrame`. Uses an AudioWorklet; falls back to a ScriptProcessor.
 */
export async function startMic(onFrame: (pcm: ArrayBuffer, level: number) => void): Promise<MicCapture> {
  if (!navigator.mediaDevices?.getUserMedia) throw new MicError('This window has no microphone support.', 'unsupported')
  let stream: MediaStream
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }
    })
  } catch (err) {
    throw friendly(err)
  }

  const ctx = new AudioContext({ latencyHint: 'interactive' })
  if (ctx.state === 'suspended') await ctx.resume().catch(() => undefined)
  const source = ctx.createMediaStreamSource(stream)
  const sink = ctx.createGain()
  sink.gain.value = 0
  sink.connect(ctx.destination)

  let node: AudioNode
  let usingWorklet = true
  try {
    await ctx.audioWorklet.addModule(workletUrl)
    const w = new AudioWorkletNode(ctx, 'pcm16-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { targetRate: TARGET_RATE, frameSamples: FRAME_SAMPLES }
    })
    w.port.onmessage = (e: MessageEvent<{ pcm: ArrayBuffer; level: number }>) => onFrame(e.data.pcm, e.data.level)
    node = w
  } catch (err) {
    console.warn('[voice] AudioWorklet unavailable, using ScriptProcessor', err)
    usingWorklet = false
    const proc = ctx.createScriptProcessor(1024, 1, 1)
    const rs = new Pcm16Resampler(ctx.sampleRate, TARGET_RATE, FRAME_SAMPLES, onFrame)
    proc.onaudioprocess = (e) => rs.push([e.inputBuffer.getChannelData(0)])
    node = proc
  }
  source.connect(node)
  node.connect(sink)

  return {
    inputRate: ctx.sampleRate,
    usingWorklet,
    setEnabled(on) {
      stream.getAudioTracks().forEach((t) => (t.enabled = on))
    },
    stop() {
      try {
        source.disconnect()
        node.disconnect()
        if ('port' in node) (node as AudioWorkletNode).port.onmessage = null
      } catch {
        /* already gone */
      }
      stream.getTracks().forEach((t) => t.stop())
      void ctx.close().catch(() => undefined)
    }
  }
}
