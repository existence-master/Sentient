import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { api, errorMessage, isNotImplemented } from '@/lib/api'

export type DictationState = 'idle' | 'recording' | 'transcribing'

/** Mic button: MediaRecorder (webm/opus) -> POST /api/voice/transcribe -> text. */
export function useDictation(onText: (text: string) => void) {
  const [state, setState] = useState<DictationState>('idle')
  const [unavailable, setUnavailable] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const recorder = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])
  const cancelled = useRef(false)
  const timer = useRef<number | undefined>(undefined)
  const onTextRef = useRef(onText)
  onTextRef.current = onText

  const supported = typeof window !== 'undefined' && 'MediaRecorder' in window && !!navigator.mediaDevices?.getUserMedia

  const stopTracks = () => recorder.current?.stream.getTracks().forEach((t) => t.stop())

  const start = useCallback(async () => {
    if (!supported || state !== 'idle') return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }
      })
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm'
      const rec = new MediaRecorder(stream, { mimeType: mime })
      chunks.current = []
      cancelled.current = false
      rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data)
      rec.onstop = async () => {
        stopTracks()
        window.clearInterval(timer.current)
        if (cancelled.current || !chunks.current.length) {
          setState('idle')
          return
        }
        setState('transcribing')
        try {
          const blob = new Blob(chunks.current, { type: 'audio/webm' })
          const res = await api.voice.transcribe(blob, 'dictation.webm')
          if (res.text?.trim()) onTextRef.current(res.text.trim())
          else toast.message("Didn't catch that", { description: 'Try speaking a little closer to the microphone.' })
        } catch (err) {
          if (isNotImplemented(err)) {
            setUnavailable(true)
            toast.message('Dictation is coming soon', { description: "The voice engine isn't installed in this build yet." })
          } else toast.error("Couldn't transcribe", { description: errorMessage(err) })
        } finally {
          setState('idle')
        }
      }
      recorder.current = rec
      rec.start(250)
      setElapsed(0)
      const startedAt = Date.now()
      timer.current = window.setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 500)
      setState('recording')
    } catch (err) {
      toast.error('Microphone unavailable', { description: errorMessage(err) })
      setState('idle')
    }
  }, [supported, state])

  const stop = useCallback(() => {
    if (recorder.current?.state === 'recording') recorder.current.stop()
  }, [])

  const cancel = useCallback(() => {
    cancelled.current = true
    stop()
  }, [stop])

  useEffect(() => () => {
    cancelled.current = true
    if (recorder.current?.state === 'recording') recorder.current.stop()
    window.clearInterval(timer.current)
  }, [])

  return { state, elapsed, start, stop, cancel, supported, unavailable, setUnavailable }
}
