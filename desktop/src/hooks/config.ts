/**
 * Config editing with debounced autosave (`PATCH /api/config`).
 *
 *   const { config, setValue, errors } = useConfigEditor()
 *   setValue('memory.facts_top_k', 10)          // optimistic, saved ~600ms later
 *   const status = useSaveStatus((s) => s.status) // 'idle' | 'saving' | 'saved' | 'error'
 */
import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useRef } from 'react'
import { create } from 'zustand'
import { api, errorMessage, isApiError } from '@/lib/api'
import type { SentientConfig } from '@/lib/types'
import { useConfig } from './core'
import { qk } from './queryKeys'

type Obj = Record<string, unknown>

export function getPath(obj: unknown, path: string): unknown {
  return path.split('.').reduce<unknown>((acc, k) => (acc && typeof acc === 'object' ? (acc as Obj)[k] : undefined), obj)
}

export function setPath<T extends Obj>(obj: T, path: string, value: unknown): T {
  const keys = path.split('.')
  const root: Obj = { ...obj }
  let cur = root
  keys.forEach((k, i) => {
    if (i === keys.length - 1) cur[k] = value
    else {
      const next = cur[k]
      cur[k] = next && typeof next === 'object' && !Array.isArray(next) ? { ...(next as Obj) } : {}
      cur = cur[k] as Obj
    }
  })
  return root as T
}

export function deepMerge<T extends Obj>(base: T, patch: Obj): T {
  const out: Obj = { ...base }
  for (const [k, v] of Object.entries(patch)) {
    const b = out[k]
    out[k] = v && typeof v === 'object' && !Array.isArray(v) && b && typeof b === 'object' && !Array.isArray(b) ? deepMerge(b as Obj, v as Obj) : v
  }
  return out as T
}

/** Parse pydantic's "1 validation error ... \n memory.facts_top_k\n  Input should be ..." into {path: message}. */
export function parseValidationErrors(detail: string): Record<string, string> {
  const out: Record<string, string> = {}
  const lines = detail.split('\n')
  for (let i = 0; i < lines.length - 1; i++) {
    const path = lines[i].trim()
    const msg = lines[i + 1]
    if (/^[\w.]+$/.test(path) && /^\s{2,}\S/.test(msg)) out[path] = msg.trim().replace(/\s*\[type=.*$/, '')
  }
  return out
}

interface SaveStatusState {
  status: 'idle' | 'saving' | 'saved' | 'error'
  error?: string
  errors: Record<string, string>
  savedAt?: number
  set: (s: Partial<SaveStatusState>) => void
}

export const useSaveStatus = create<SaveStatusState>((set) => ({ status: 'idle', errors: {}, set: (s) => set(s) }))

let pending: Obj = {}
let timer: number | undefined

export function useConfigEditor(delayMs = 600) {
  const qc = useQueryClient()
  const query = useConfig()
  const setStatus = useSaveStatus((s) => s.set)
  const errors = useSaveStatus((s) => s.errors)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const flush = useCallback(async () => {
    const patch = pending
    pending = {}
    if (!Object.keys(patch).length) return
    setStatus({ status: 'saving' })
    try {
      const res = await api.config.patch(patch)
      qc.setQueryData(qk.config, res.config)
      void qc.invalidateQueries({ queryKey: qk.bootstrap })
      setStatus({ status: 'saved', errors: {}, error: undefined, savedAt: Date.now() })
    } catch (err) {
      const detail = errorMessage(err)
      const fieldErrors = isApiError(err) && err.status === 422 ? parseValidationErrors(detail) : {}
      setStatus({ status: 'error', error: detail, errors: fieldErrors })
      void qc.invalidateQueries({ queryKey: qk.config })
    }
  }, [qc, setStatus])

  /** Merge a nested patch, update the cache optimistically and schedule a save. */
  const patch = useCallback(
    (p: Obj, opts: { immediate?: boolean } = {}) => {
      pending = deepMerge(pending, p)
      qc.setQueryData<SentientConfig>(qk.config, (old) => (old ? deepMerge(old as unknown as Obj, p) as unknown as SentientConfig : old))
      const errs = { ...useSaveStatus.getState().errors }
      for (const path of Object.keys(errs)) if (getPath(p, path) !== undefined) delete errs[path]
      setStatus({ status: 'saving', errors: errs })
      window.clearTimeout(timer)
      if (opts.immediate) void flush()
      else timer = window.setTimeout(() => void flush(), delayMs)
    },
    [qc, flush, delayMs, setStatus]
  )

  const setValue = useCallback((path: string, value: unknown, opts?: { immediate?: boolean }) => patch(setPath({}, path, value), opts), [patch])

  return {
    config: query.data,
    isLoading: query.isLoading,
    error: query.error,
    patch,
    setValue,
    get: (path: string) => getPath(query.data, path),
    errors,
    flush
  }
}
