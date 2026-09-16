/** React Query hooks for §3 models & secrets. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useRef, useState } from 'react'
import { api, errorMessage } from '@/lib/api'
import type { ModelRoles, OllamaPullProgress, RoleName, SentientConfig } from '@/lib/types'
import { qk } from './queryKeys'

export function useProviders() {
  return useQuery({ queryKey: qk.providers, queryFn: api.models.providers, staleTime: 60_000 })
}

export function useLocalModels() {
  return useQuery({ queryKey: qk.localModels, queryFn: api.models.local, staleTime: 15_000 })
}

export function useSecrets() {
  return useQuery({ queryKey: qk.secrets, queryFn: api.secrets.list })
}

export function useTestModel() {
  return useMutation({ mutationFn: ({ model, role }: { model: string; role?: RoleName }) => api.models.test(model, role) })
}

export function useTestEmbedding() {
  return useMutation({ mutationFn: (model: string) => api.models.testEmbedding(model) })
}

export function useSetRoles() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (roles: Partial<Record<RoleName, string | null>>) => api.models.setRoles(roles),
    onSuccess: (roles: ModelRoles) => {
      qc.setQueryData<SentientConfig>(qk.config, (old) => (old ? { ...old, models: { ...old.models, roles } } : old))
      void qc.invalidateQueries({ queryKey: qk.bootstrap })
    }
  })
}

export function useSetFallbacks() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (fallbacks: Partial<Record<RoleName, string[]>>) => api.models.setFallbacks(fallbacks),
    onSuccess: () => void qc.invalidateQueries({ queryKey: qk.config })
  })
}

export function useSetSecret() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, value }: { name: string; value: string }) => api.secrets.set(name, value),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.secrets })
      void qc.invalidateQueries({ queryKey: qk.providers })
    }
  })
}

export function useDeleteSecret() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => api.secrets.delete(name),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.secrets })
      void qc.invalidateQueries({ queryKey: qk.providers })
    }
  })
}

export interface PullState {
  name: string | null
  status: string
  completed: number
  total: number
  /** 0..1, or null while the size is unknown */
  progress: number | null
  running: boolean
  error: string | null
  done: boolean
}

const idle: PullState = { name: null, status: '', completed: 0, total: 0, progress: null, running: false, error: null, done: false }

/** `POST /api/models/ollama/pull` with live progress. */
export function useOllamaPull() {
  const qc = useQueryClient()
  const [state, setState] = useState<PullState>(idle)
  const abort = useRef<AbortController | null>(null)

  const pull = useCallback(
    async (name: string) => {
      abort.current?.abort()
      const ctrl = new AbortController()
      abort.current = ctrl
      setState({ ...idle, name, running: true, status: 'Starting…' })
      try {
        let last: OllamaPullProgress | null = null
        for await (const line of api.models.pullOllama(name, ctrl.signal)) {
          last = line
          if (line.error || line.status === 'error') throw new Error(line.error ?? 'Pull failed')
          setState((s) => ({
            ...s,
            status: line.status,
            completed: line.completed ?? s.completed,
            total: line.total ?? s.total,
            progress: line.total ? (line.completed ?? 0) / line.total : s.progress
          }))
        }
        const ok = last?.status === 'success'
        setState((s) => ({ ...s, running: false, done: ok, progress: ok ? 1 : s.progress, status: ok ? 'Installed' : (last?.status ?? 'Finished') }))
        void qc.invalidateQueries({ queryKey: qk.localModels })
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') setState(idle)
        else setState((s) => ({ ...s, running: false, error: errorMessage(err) }))
      }
    },
    [qc]
  )

  const cancel = useCallback(() => abort.current?.abort(), [])
  const reset = useCallback(() => setState(idle), [])
  return { ...state, pull, cancel, reset }
}
