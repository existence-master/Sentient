/** React Query hooks for §9 voice (REST side). The `/ws/voice` socket belongs to features/voice. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useRef, useState } from 'react'
import { api, errorMessage } from '@/lib/api'
import type { VoicePrepareTarget } from '@/lib/types'
import { qk } from './queryKeys'

export function useVoiceStatus() {
  return useQuery({ queryKey: qk.voiceStatus, queryFn: api.voice.status, staleTime: 30_000 })
}

/** Plays `POST /api/voice/speak` output. */
export function useSpeak() {
  const audio = useRef<HTMLAudioElement | null>(null)
  return useMutation({
    mutationFn: async ({ text, voice }: { text: string; voice?: string }) => {
      const blob = await api.voice.speak(text, voice)
      const url = URL.createObjectURL(blob)
      audio.current?.pause()
      const el = new Audio(url)
      audio.current = el
      el.onended = () => URL.revokeObjectURL(url)
      await el.play()
    }
  })
}

export function useVoicePrepare() {
  const qc = useQueryClient()
  const [state, setState] = useState<{ running: boolean; stage: string; progress: number | null; error: string | null; done: boolean }>({
    running: false,
    stage: '',
    progress: null,
    error: null,
    done: false
  })
  const run = useCallback(async (target: VoicePrepareTarget = 'all') => {
    setState({ running: true, stage: 'Starting…', progress: null, error: null, done: false })
    try {
      for await (const line of api.voice.prepare(target)) {
        if (line.stage === 'error') throw new Error(line.message ?? 'Voice model preparation failed')
        if (line.stage === 'done' && line.ok === false) throw new Error(line.message ?? 'Voice models are not ready')
        const label =
          line.stage === 'download'
            ? `Downloading ${line.component === 'tts' ? 'voice' : 'speech recognition'} model${line.file ? ` · ${line.file}` : ''}`
            : line.stage === 'loading'
              ? `Loading ${line.component === 'tts' ? 'voice' : 'speech recognition'} model`
              : line.stage === 'ready'
                ? `${line.component === 'tts' ? 'Voice' : 'Speech recognition'} ready${line.note ? ` · ${line.note}` : ''}`
                : line.stage
        setState((s) => ({ ...s, stage: label, progress: typeof line.progress === 'number' ? line.progress : null }))
      }
      setState((s) => ({ ...s, running: false, done: true, progress: 1, stage: 'Ready' }))
      void qc.invalidateQueries({ queryKey: qk.voiceStatus })
    } catch (err) {
      setState((s) => ({ ...s, running: false, error: errorMessage(err) }))
    }
  }, [qc])
  return { ...state, run }
}
