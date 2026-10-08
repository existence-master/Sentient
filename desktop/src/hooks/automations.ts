/** React Query hooks for §16 webhooks and change feeds. Live updates arrive via lib/events.ts. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, noRetryWhenMissing } from '@/lib/api'
import type { Hook } from '@/lib/types'
import { qk } from './queryKeys'

// ---------------------------------------------------------------------------- webhooks
export function useHooks() {
  return useQuery({ queryKey: qk.hooks, queryFn: api.hooks.list, retry: noRetryWhenMissing })
}

export function useHookActions() {
  const qc = useQueryClient()
  return {
    create: useMutation({
      mutationFn: (name: string) => api.hooks.create(name),
      onSuccess: (h) => {
        const { secret: _secret, ...hook } = h
        qc.setQueryData<Hook[]>(qk.hooks, (old) => [hook, ...(old ?? []).filter((x) => x.id !== hook.id)])
      }
    }),
    remove: useMutation({
      mutationFn: (id: string) => api.hooks.delete(id),
      onMutate: (id) => {
        const prev = qc.getQueryData<Hook[]>(qk.hooks)
        qc.setQueryData<Hook[]>(qk.hooks, (old) => old?.filter((h) => h.id !== id))
        return prev
      },
      onError: (_e, _v, prev) => prev && qc.setQueryData(qk.hooks, prev)
    })
  }
}

// ---------------------------------------------------------------------------- change feeds
export function useFeeds() {
  return useQuery({ queryKey: qk.integrations.feeds, queryFn: api.integrations.feeds.list, retry: noRetryWhenMissing, refetchInterval: 60_000 })
}

export function useFeedSync() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (source: string) => api.integrations.feeds.sync(source),
    onSettled: () => void qc.invalidateQueries({ queryKey: qk.integrations.feeds })
  })
}
