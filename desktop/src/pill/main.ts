/**
 * The listening pill (#169): a small always-on-top window that owns the microphone for push to talk and dictation.
 *
 * The shell (electron/main/dictation.ts) sends `start`, `stop`, `auto-stop` and `cancel`. While listening the pill
 * records with MediaRecorder and shows a level meter; the microphone is released the moment recording stops. The
 * audio goes only to the local engine's `/api/voice/dictate` (speech recognition on this computer, then cleanup;
 * push to talk asks for the local tidy-up only). The text goes back to the shell, which types it or sends it to a chat.
 */
import type { DictationCommand, DictationEvent, DictationMode } from '@/types/bridge'
import { meterLevel, rms, SilenceDetector } from './silence'

const STEP_MS = 50
const BARS = 9

interface Recording {
  mode: DictationMode
  stream: MediaStream | null
  ctx: AudioContext | null
  recorder: MediaRecorder | null
  chunks: Blob[]
  timer: number | undefined
  detector: SilenceDetector
  startedAt: number
  maxMs: number
  cancelled: boolean
  abort: AbortController
}

const bridge = window.sentient
const root = document.getElementById('pill') as HTMLDivElement
const label = document.getElementById('label') as HTMLSpanElement
const meter = document.getElementById('meter') as HTMLDivElement
const closeButton = document.getElementById('close') as HTMLButtonElement
const bars: HTMLSpanElement[] = []
for (let i = 0; i < BARS; i++) {
  const bar = document.createElement('span')
  meter.appendChild(bar)
  bars.push(bar)
}

let current: Recording | null = null

function report(event: DictationEvent): void {
  bridge?.dictation.report(event)
}

function show(state: 'starting' | 'listening' | 'working' | 'empty' | 'error', text: string): void {
  root.dataset.state = state
  label.textContent = text
  if (state !== 'listening') drawLevel(0)
}

function drawLevel(level: number): void {
  // A small hump: the middle bars move most.
  bars.forEach((bar, i) => {
    const shape = 1 - Math.abs(i - (BARS - 1) / 2) / BARS
    bar.style.transform = `scaleY(${Math.max(0.15, Math.min(1, level * shape * 1.6))})`
  })
}

function release(rec: Recording): void {
  window.clearInterval(rec.timer)
  rec.stream?.getTracks().forEach((t) => t.stop())
  rec.stream = null
  void rec.ctx?.close().catch(() => undefined)
  rec.ctx = null
}

async function start(cmd: Extract<DictationCommand, { type: 'start' }>): Promise<void> {
  if (current) cancel(false)
  const rec: Recording = {
    mode: cmd.mode,
    stream: null,
    ctx: null,
    recorder: null,
    chunks: [],
    timer: undefined,
    detector: new SilenceDetector(cmd.silenceMs),
    startedAt: Date.now(),
    maxMs: cmd.maxMs,
    cancelled: false,
    abort: new AbortController()
  }
  current = rec
  root.dataset.mode = cmd.mode
  show('starting', 'Starting…')
  try {
    rec.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }
    })
  } catch {
    if (rec.cancelled) return
    current = null
    show('error', 'Sentient can’t use the microphone')
    report({ type: 'error', message: 'microphone unavailable' })
    return
  }
  if (rec.cancelled) {
    release(rec)
    return
  }
  const ctx = new AudioContext()
  rec.ctx = ctx
  const analyser = ctx.createAnalyser()
  analyser.fftSize = 1024
  ctx.createMediaStreamSource(rec.stream).connect(analyser)
  const samples = new Float32Array(analyser.fftSize)
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm'
  const recorder = new MediaRecorder(rec.stream, { mimeType: mime })
  rec.recorder = recorder
  recorder.ondataavailable = (e) => {
    if (e.data.size) rec.chunks.push(e.data)
  }
  recorder.onstop = () => void transcribe(rec)
  recorder.start(250)
  rec.timer = window.setInterval(() => {
    analyser.getFloatTimeDomainData(samples)
    const loudness = rms(samples)
    drawLevel(meterLevel(loudness))
    const verdict = rec.detector.feed(loudness, STEP_MS)
    if (verdict === 'nothing') {
      rec.cancelled = true
      stopRecording(rec)
      current = null
      show('empty', 'Didn’t catch that')
      report({ type: 'empty' })
    } else if (verdict === 'stop' || Date.now() - rec.startedAt >= rec.maxMs) {
      stopRecording(rec)
    }
  }, STEP_MS)
  show('listening', cmd.mode === 'talk' ? 'Listening' : 'Dictating')
  report({ type: 'listening' })
}

function stopRecording(rec: Recording): void {
  if (rec.recorder?.state === 'recording') rec.recorder.stop()
  else release(rec)
}

async function transcribe(rec: Recording): Promise<void> {
  release(rec)
  report({ type: 'working' })
  if (rec.cancelled) return
  if (!rec.chunks.length || !rec.detector.heardSpeech) {
    current = null
    show('empty', 'Didn’t catch that')
    report({ type: 'empty' })
    return
  }
  show('working', rec.mode === 'talk' ? 'Sending…' : 'Writing…')
  try {
    const conn = await bridge!.getConnection()
    const form = new FormData()
    form.append('file', new Blob(rec.chunks, { type: 'audio/webm' }), 'dictation.webm')
    if (rec.mode === 'talk') form.append('cleanup', 'tidy') // the chat's model reads it anyway: no polish
    const r = await fetch(`${conn.baseUrl}/api/voice/dictate`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${conn.token}` },
      body: form,
      signal: rec.abort.signal
    })
    if (rec.cancelled) return
    const body = (await r.json().catch(() => ({}))) as { text?: string; detail?: string }
    if (!r.ok) throw new Error(r.status === 404 ? 'Update Sentient to use this.' : body.detail || `Error ${r.status}`)
    const text = (body.text ?? '').trim()
    current = null
    if (!text) {
      show('empty', 'Didn’t catch that')
      report({ type: 'empty' })
    } else report({ type: 'result', text })
  } catch (err) {
    if (rec.cancelled || (err as Error)?.name === 'AbortError') return
    current = null
    const message = err instanceof TypeError ? 'Sentient isn’t running' : String((err as Error)?.message || err)
    show('error', message.length > 60 ? `${message.slice(0, 57)}…` : message)
    report({ type: 'error', message })
  }
}

/** Drop the recording. `fromShell` false means the pill was asked to start over or the user cancelled here. */
function cancel(fromShell: boolean): void {
  const rec = current
  current = null
  if (!rec) return
  rec.cancelled = true
  rec.abort.abort()
  stopRecording(rec)
  release(rec)
  if (!fromShell) report({ type: 'cancelled' })
}

bridge?.dictation.onCommand((cmd) => {
  switch (cmd.type) {
    case 'start':
      void start(cmd)
      break
    case 'stop':
      if (current) stopRecording(current)
      break
    case 'auto-stop':
      current?.detector.setSilence(cmd.silenceMs)
      break
    case 'cancel':
      cancel(true)
      break
  }
})

// Push to talk on Windows gives the pill focus, so it sees the held shortcut being let go.
window.addEventListener('keyup', (e) => {
  if (e.key !== 'Escape') report({ type: 'keyup' })
})
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') report({ type: 'escape' })
})
closeButton.addEventListener('click', () => report({ type: 'escape' }))
