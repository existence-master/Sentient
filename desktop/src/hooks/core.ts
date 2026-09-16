/** React Query hooks for §2 core resources. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { OnboardingRequest, Session } from '@/lib/types'
import { qk } from './queryKeys'

export function useBootstrap() {
  return useQuery({ queryKey: qk.bootstrap, queryFn: api.bootstrap, staleTime: 60_000 })
}

export function useConfig() {
  return useQuery({ queryKey: qk.config, queryFn: api.config.get })
}

export function useConfigSchema() {
  return useQuery({ queryKey: qk.configSchema, queryFn: api.config.schema, staleTime: 5 * 60_000 })
}

export function useOnboarding() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: OnboardingRequest) => api.onboarding(body),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: qk.bootstrap })
      void qc.invalidateQueries({ queryKey: qk.config })
      void qc.invalidateQueries({ queryKey: qk.memories.workspace })
    }
  })
}

// ---------------------------------------------------------------------------- sessions
export function useSessions(limit = 100) {
  return useQuery({ queryKey: qk.sessions, queryFn: () => api.sessions.list(limit) })
}

export function useMessages(sessionId: string | undefined) {
  return useQuery({
    queryKey: qk.messages(sessionId ?? ''),
    queryFn: () => api.sessions.messages(sessionId as string),
    enabled: !!sessionId,
    staleTime: Infinity
  })
}

export function useSessionSearch(q: string) {
  const query = q.trim()
  return useQuery({
    queryKey: qk.sessionSearch(query),
    queryFn: () => api.sessions.search(query),
    enabled: query.length >= 2,
    staleTime: 10_000
  })
}

export function useRenameSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => api.sessions.rename(id, title),
    onMutate: ({ id, title }) => {
      qc.setQueryData<Session[]>(qk.sessions, (old) => old?.map((s) => (s.id === id ? { ...s, title } : s)))
    },
    onError: () => void qc.invalidateQueries({ queryKey: qk.sessions })
  })
}

export function useDeleteSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.sessions.delete(id),
    onMutate: (id) => {
      qc.setQueryData<Session[]>(qk.sessions, (old) => old?.filter((s) => s.id !== id))
    },
    onSettled: (_d, _e, id) => {
      qc.removeQueries({ queryKey: qk.messages(id) })
      void qc.invalidateQueries({ queryKey: qk.sessions })
    }
  })
}

// ---------------------------------------------------------------------------- files, tools, usage
export function useFiles() {
  return useQuery({ queryKey: qk.files, queryFn: api.files.list })
}

export function useTools() {
  return useQuery({ queryKey: qk.tools, queryFn: api.tools.list, staleTime: 5 * 60_000 })
}

export function useUsage(days = 30) {
  return useQuery({ queryKey: qk.usage(days), queryFn: () => api.usage.get(days) })
}
