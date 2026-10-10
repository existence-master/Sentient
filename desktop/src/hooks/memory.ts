/** React Query hooks for §7 memory. `memory.updated` invalidates `qk.memories.all`. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, isNotImplemented, type UploadOptions } from '@/lib/api'
import type { MemoryQuery, Persona, WorkspaceFileId } from '@/lib/types'
import { qk } from './queryKeys'

export function useMemories(query: MemoryQuery = {}) {
  return useQuery({ queryKey: qk.memories.list(query), queryFn: () => api.memories.list(query) })
}

export function useMemoryTopics() {
  return useQuery({ queryKey: qk.memories.topics, queryFn: api.memories.topics })
}

export function useMemoryGraph() {
  return useQuery({ queryKey: qk.memories.graph, queryFn: api.memories.graph })
}

export function useMemorySummaries(limit = 50) {
  return useQuery({ queryKey: qk.memories.summaries, queryFn: () => api.memories.summaries(limit) })
}

/** Memories waiting for the user's review. Never fails loudly: an older engine without the route counts as empty. */
export function useMemoryReview() {
  return useQuery({
    queryKey: qk.memories.review,
    queryFn: async () => {
      try {
        return await api.memories.review.list()
      } catch (err) {
        if (isNotImplemented(err)) return { items: [], count: 0, expire_days: 30 }
        throw err
      }
    }
  })
}

export function useMemoryReviewActions() {
  const qc = useQueryClient()
  const invalidate = () => {
    void qc.invalidateQueries({ queryKey: qk.memories.all })
    void qc.invalidateQueries({ queryKey: qk.userModel })
  }
  return {
    approve: useMutation({
      mutationFn: ({ kind, id, content }: { kind: 'fact' | 'insight'; id: number | string; content?: string }) => api.memories.review.approve(kind, id, content),
      onSuccess: invalidate
    }),
    discard: useMutation({ mutationFn: ({ kind, id }: { kind: 'fact' | 'insight'; id: number | string }) => api.memories.review.discard(kind, id), onSuccess: invalidate }),
    approveAll: useMutation({ mutationFn: (from: string) => api.memories.review.approveAll(from), onSuccess: invalidate })
  }
}

export function useWorkspace() {
  return useQuery({ queryKey: qk.memories.workspace, queryFn: api.memories.workspace })
}

/** Offline fallback used when `GET /api/memories/personas` isn't available yet. */
export const FALLBACK_PERSONAS: Persona[] = [
  { id: 'friendly', name: 'Friendly companion', description: 'Warm, direct and brief. Talks like a capable friend.', soul_md: '' },
  { id: 'professional', name: 'Chief of staff', description: 'Polished, structured, anticipates next steps. Good for work-heavy days.', soul_md: '' },
  { id: 'concise', name: 'Minimalist', description: 'As few words as possible. Just the answer.', soul_md: '' },
  { id: 'coach', name: 'Encouraging coach', description: 'Supportive, keeps you accountable to your goals and habits.', soul_md: '' }
]

export function usePersonas() {
  return useQuery({
    queryKey: qk.memories.personas,
    queryFn: async () => {
      try {
        const list = await api.memories.personas()
        return Array.isArray(list) && list.length ? list : FALLBACK_PERSONAS
      } catch (err) {
        if (isNotImplemented(err)) return FALLBACK_PERSONAS
        throw err
      }
    },
    staleTime: 10 * 60_000
  })
}

export function useMemoryActions() {
  const qc = useQueryClient()
  const invalidate = () => void qc.invalidateQueries({ queryKey: qk.memories.all })
  return {
    create: useMutation({ mutationFn: ({ content, source }: { content: string; source?: string }) => api.memories.create(content, source), onSuccess: invalidate }),
    update: useMutation({ mutationFn: ({ id, content }: { id: number; content: string }) => api.memories.update(id, content), onSuccess: invalidate }),
    remove: useMutation({ mutationFn: (id: number) => api.memories.delete(id), onSuccess: invalidate }),
    removeBySource: useMutation({ mutationFn: (source: string) => api.memories.deleteBySource(source), onSuccess: invalidate }),
    importFile: useMutation({
      mutationFn: ({ file, opts }: { file: File; opts?: UploadOptions }) => api.memories.import(file, opts),
      onSuccess: invalidate
    }),
    writeWorkspace: useMutation({
      mutationFn: ({ which, content }: { which: WorkspaceFileId; content: string }) => api.memories.writeWorkspace(which, content),
      onSuccess: () => void qc.invalidateQueries({ queryKey: qk.memories.workspace })
    })
  }
}
