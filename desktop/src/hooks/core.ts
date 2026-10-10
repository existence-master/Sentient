/** React Query hooks for §2 core resources. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'
import { api, isNotImplemented } from '@/lib/api'
import { useConnection } from '@/stores/connection'
import type { OnboardingRequest, RuleProposal, Session, StopState } from '@/lib/types'
import { qk } from './queryKeys'

export function useBootstrap() {
  return useQuery({ queryKey: qk.bootstrap, queryFn: api.bootstrap, staleTime: 60_000 })
}

// ---------------------------------------------------------------------------- §17 stop everything
const NOT_STOPPED: StopState = { stopped: false, stopped_at: null, source: null }

/**
 * Live from `stop.updated` (lib/events.ts). Fetched only once the engine is ready, and again every time it
 * becomes ready (a restarted engine reads its saved state). Only an engine without §17 reads as not stopped;
 * any other failure stays an error and is retried, so a stopped Sentient is never shown as running.
 */
export function useStopState() {
  const ready = useConnection((s) => s.backend.state === 'ready')
  const qc = useQueryClient()
  useEffect(() => {
    if (ready) void qc.invalidateQueries({ queryKey: qk.stop })
  }, [ready, qc])
  return useQuery({
    queryKey: qk.stop,
    queryFn: () => api.stop.get().catch((err) => (isNotImplemented(err) ? NOT_STOPPED : Promise.reject(err))),
    enabled: ready,
    staleTime: Infinity,
    retry: true,
    retryDelay: (n) => Math.min(1000 * 2 ** n, 10_000)
  })
}

export function useStopAll() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.stop.all('desktop'),
    onSuccess: (state) => qc.setQueryData<StopState>(qk.stop, { stopped: state.stopped, stopped_at: state.stopped_at, source: state.source })
  })
}

export function useResume() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.stop.resume('desktop'),
    onSuccess: (state) => qc.setQueryData<StopState>(qk.stop, state)
  })
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

/** Undecided "Make this a rule?" cards for a chat. Kept fresh by `rule_proposal.updated` (lib/events.ts). */
export function useRuleProposals(sessionId: string | undefined) {
  return useQuery({
    queryKey: qk.ruleProposals(sessionId ?? ''),
    queryFn: () => api.ruleProposals.list(sessionId as string),
    enabled: !!sessionId,
    retry: false
  })
}

export function useDecideRuleProposal() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, decision }: { id: string; decision: 'accept' | 'decline'; sessionId: string }) =>
      api.ruleProposals.decide(id, decision),
    onSuccess: (p, { sessionId }) => {
      qc.setQueryData<RuleProposal[]>(qk.ruleProposals(sessionId), (old) => old?.filter((x) => x.id !== p.id))
      if (p.status === 'accepted') void qc.invalidateQueries({ queryKey: qk.config })
    }
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
