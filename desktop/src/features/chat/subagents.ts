/** Hooks for §10 subagents ("helpers" in the UI). Live updates arrive via `subagent.updated` (lib/events.ts). */
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { api, isNotImplemented } from '@/lib/api'
import type { Subagent } from '@/lib/types'
import { previewOr, previewSubagent, previewSubagents } from '@/features/devices/preview'

export const subagentKeys = {
  all: ['subagents'] as const,
  detail: (id: string) => ['subagents', 'detail', id] as const,
  session: (sessionId: string) => ['subagents', 'session', sessionId] as const
}

/** Merge an update into every cache entry that holds this subagent. Returns the previous copy, if any. */
export function upsertSubagent(qc: QueryClient, sub: Subagent): Subagent | undefined {
  const prev =
    qc.getQueryData<Subagent>(subagentKeys.detail(sub.subagent_id)) ??
    (sub.session_id ? qc.getQueryData<Subagent[]>(subagentKeys.session(sub.session_id))?.find((s) => s.subagent_id === sub.subagent_id) : undefined)
  qc.setQueryData<Subagent>(subagentKeys.detail(sub.subagent_id), (old) => ({ ...old, ...sub, events: old?.events }))
  if (sub.session_id) {
    qc.setQueryData<Subagent[]>(subagentKeys.session(sub.session_id), (old) => {
      if (!old) return old
      const i = old.findIndex((s) => s.subagent_id === sub.subagent_id)
      if (i === -1) return [sub, ...old]
      const next = old.slice()
      next[i] = { ...old[i], ...sub }
      return next
    })
  }
  return prev
}

export function useSessionSubagents(sessionId: string | undefined) {
  return useQuery({
    queryKey: subagentKeys.session(sessionId ?? ''),
    queryFn: async () => {
      try {
        return await previewOr(() => api.subagents.forSession(sessionId as string), () => previewSubagents(sessionId as string))
      } catch (err) {
        if (isNotImplemented(err)) return [] as Subagent[]
        throw err
      }
    },
    enabled: !!sessionId,
    staleTime: 15_000
  })
}

export function useSubagent(id: string | undefined) {
  return useQuery({
    queryKey: subagentKeys.detail(id ?? ''),
    queryFn: async () => {
      try {
        return await previewOr(() => api.subagents.get(id as string), () => previewSubagent(id as string))
      } catch (err) {
        if (isNotImplemented(err)) return null
        throw err
      }
    },
    enabled: !!id,
    staleTime: 30_000
  })
}

export function useCancelSubagent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.subagents.cancel(id),
    onSuccess: (sub) => void upsertSubagent(qc, sub)
  })
}
