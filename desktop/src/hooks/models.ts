/** React Query hooks for §3 models & secrets. */
import { useMutation, useMutationState, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, errorMessage } from '@/lib/api'
import { demo, isDemoMode } from '@/lib/demo'
import type { CheckupRole, CheckupStatus, ModelRoles, OllamaPullProgress, RoleName, SentientConfig } from '@/lib/types'
import { qk } from './queryKeys'

export function useProviders() {
  return useQuery({ queryKey: qk.providers, queryFn: api.models.providers, staleTime: 60_000 })
}

export function useLocalModels() {
  return useQuery({ queryKey: qk.localModels, queryFn: api.models.local, staleTime: 15_000 })
}

/** A cloud provider's live model list. Only fetched when `enabled`, so nothing is asked of providers you don't use. */
export function useModelCatalog(provider: string, enabled: boolean) {
  return useQuery({ queryKey: qk.catalog(provider), queryFn: () => api.models.catalog(provider), enabled, staleTime: 10 * 60_000, retry: false })
}

/** Polls an OpenRouter sign-in until it is connected or failed. */
export function useSignInStatus(state: string | null) {
  return useQuery({
    queryKey: qk.signIn(state ?? ''),
    queryFn: () => api.models.signInStatus(state ?? ''),
    enabled: !!state,
    refetchInterval: (q) => (q.state.data && ['connected', 'failed'].includes(q.state.data.status) ? false : 1500)
  })
}

export function useCheckKey() {
  return useMutation({ mutationFn: (provider: string) => api.models.checkKey(provider) })
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

export function useModelPresets() {
  return useQuery({ queryKey: qk.modelPresets, queryFn: api.models.presets.list, staleTime: 10_000 })
}

/** After a preset changes the models: refresh everything that shows them. */
function useInvalidateModels() {
  const qc = useQueryClient()
  return () => {
    void qc.invalidateQueries({ queryKey: qk.modelPresets })
    void qc.invalidateQueries({ queryKey: qk.config })
    void qc.invalidateQueries({ queryKey: qk.bootstrap })
  }
}

export function useApplyPreset() {
  const refresh = useInvalidateModels()
  return useMutation({ mutationFn: (name: string) => api.models.presets.apply(name), onSuccess: refresh })
}

export function useUndoPreset() {
  const refresh = useInvalidateModels()
  return useMutation({ mutationFn: () => api.models.presets.undo(), onSuccess: refresh })
}

export function useSavePreset() {
  const refresh = useInvalidateModels()
  return useMutation({ mutationFn: ({ name, overwrite }: { name: string; overwrite?: boolean }) => api.models.presets.save(name, overwrite), onSuccess: refresh })
}

export function useRenamePreset() {
  const refresh = useInvalidateModels()
  return useMutation({ mutationFn: ({ name, to }: { name: string; to: string }) => api.models.presets.rename(name, to), onSuccess: refresh })
}

export function useDeletePreset() {
  const refresh = useInvalidateModels()
  return useMutation({ mutationFn: (name: string) => api.models.presets.delete(name), onSuccess: refresh })
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
    mutationKey: qk.saveSecret,
    mutationFn: ({ name, value }: { name: string; value: string }) => api.secrets.set(name, value),
    onSuccess: (_res, { name }) => {
      void qc.invalidateQueries({ queryKey: qk.secrets })
      void qc.invalidateQueries({ queryKey: qk.providers })
      void qc.invalidateQueries({ queryKey: qk.catalog(name) }) // a new key can mean a different account's models
      void qc.invalidateQueries({ queryKey: qk.modelPresets })
    }
  })
}

/** How many times this secret was saved since the window opened. It changes when a key is replaced. */
export function useSecretSaves(name: string) {
  const saved = useMutationState({
    filters: { mutationKey: qk.saveSecret, status: 'success' },
    select: (m) => (m.state.variables as { name?: string } | undefined)?.name
  })
  return saved.filter((n) => n === name).length
}

export function useDeleteSecret() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => api.secrets.delete(name),
    onSuccess: (_res, name) => {
      void qc.invalidateQueries({ queryKey: qk.secrets })
      void qc.invalidateQueries({ queryKey: qk.providers })
      qc.removeQueries({ queryKey: qk.catalog(name) })
      void qc.invalidateQueries({ queryKey: qk.modelPresets })
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

export interface CheckupRow {
  role: RoleName
  model: string | null
  /** What is being tried right now, while this role is being checked. */
  step: string | null
  result: CheckupRole | null
}

export interface CheckupState {
  rows: CheckupRow[]
  running: boolean
  error: string | null
  /** Worst result once finished. */
  status: CheckupStatus | null
  /** Showing the dev-only example result. */
  example: boolean
}

const noCheckup: CheckupState = { rows: [], running: false, error: null, status: null, example: false }

function exampleCheckup(): CheckupState {
  const roles = demo.modelCheckup()
  const rank: Record<CheckupStatus, number> = { skip: 0, pass: 1, warn: 2, fail: 3 }
  const status = roles.reduce<CheckupStatus>((w, r) => (rank[r.status] > rank[w] ? r.status : w), 'skip')
  return { rows: roles.map((r) => ({ role: r.role, model: r.model, step: null, result: r })), running: false, error: null, status, example: true }
}

/** `POST /api/models/checkup` with live progress. `roles` limits the check to these models (onboarding). */
export function useModelCheckup() {
  const [state, setRunState] = useState<CheckupState>(() => (isDemoMode() ? exampleCheckup() : noCheckup))
  const abort = useRef<AbortController | null>(null)
  const current = useRef(0)

  const run = useCallback(async (roles?: Partial<Record<RoleName, string | null>>) => {
    abort.current?.abort()
    const ctrl = new AbortController()
    abort.current = ctrl
    // A stopped run settles later; only the latest run may touch the state.
    const id = ++current.current
    const setState = (next: CheckupState | ((s: CheckupState) => CheckupState)) => {
      if (current.current === id) setRunState(next)
    }
    setState({ ...noCheckup, running: true })
    const patchRow = (role: RoleName, p: Partial<CheckupRow>) =>
      setState((s) => ({ ...s, rows: s.rows.map((r) => (r.role === role ? { ...r, ...p } : r)) }))
    try {
      for await (const e of api.models.checkup(roles, ctrl.signal)) {
        if (e.type === 'start') setState((s) => ({ ...s, rows: e.roles.map((r) => ({ ...r, step: null, result: null })) }))
        else if (e.type === 'step') patchRow(e.role, { step: e.label })
        else if (e.type === 'role') {
          const { type: _type, ...result } = e
          void _type
          patchRow(e.role, { step: null, result })
        } else if (e.type === 'done') setState((s) => ({ ...s, status: e.status }))
      }
      setState((s) => ({ ...s, running: false, error: s.status ? null : "The check-up stopped before it finished. Try again." }))
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') setState((s) => ({ ...s, running: false }))
      else setState((s) => ({ ...s, running: false, error: errorMessage(err) }))
    }
  }, [])

  const cancel = useCallback(() => abort.current?.abort(), [])
  useEffect(() => () => abort.current?.abort(), [])
  return { ...state, run, cancel }
}
