/** Memory page state (URL-backed so every view is linkable) and optimistic mutations. */
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router'
import { toast } from 'sonner'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import type { Memory, MemoryGraph } from '@/lib/types'

export type MemoryView = 'graph' | 'list' | 'timeline'
export type MemoryTab = 'memories' | 'review' | 'conversations' | 'profile'
export type MemoryTypeFilter = 'all' | 'long-term' | 'short-term'

export function useMemoryParams() {
  const [params, setParams] = useSearchParams()
  const get = (k: string) => params.get(k) ?? ''

  const set = useCallback(
    (patch: Record<string, string | number | null | undefined>, replace = true) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          for (const [k, v] of Object.entries(patch)) {
            if (v === null || v === undefined || v === '') next.delete(k)
            else next.set(k, String(v))
          }
          return next
        },
        { replace }
      ),
    [setParams]
  )

  const scope = get('scope') // a memory scope (#155), '' for all
  const view = (['graph', 'list', 'timeline'].includes(get('view')) ? get('view') : 'graph') as MemoryView
  const tab = (['memories', 'review', 'conversations', 'profile'].includes(get('tab')) ? get('tab') : 'memories') as MemoryTab
  const type = (['long-term', 'short-term'].includes(get('type')) ? get('type') : 'all') as MemoryTypeFilter
  const selected = Number(get('m')) || null

  return {
    view,
    tab,
    type,
    topic: get('topic'),
    scope,
    source: get('source'),
    q: get('q'),
    selected,
    addOpen: get('add') === '1',
    importOpen: get('import') === '1',
    /** ?import=assistant: bring memory from ChatGPT, Claude or OpenClaw (with ?app=&path= it opens on the preview). */
    assistantImportOpen: get('import') === 'assistant',
    importApp: get('app'),
    importPath: get('path'),
    forgetOpen: get('forget') === '1',
    set
  }
}

export function useDebounced<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(t)
  }, [value, ms])
  return v
}

/** Ticks every `ms` so countdowns stay live. */
export function useNow(ms = 30_000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), ms)
    return () => window.clearInterval(t)
  }, [ms])
  return now
}

type Snapshot = Array<[readonly unknown[], unknown]>

/** Update/delete with optimistic cache edits across every memory list + the graph. */
export function useOptimisticMemoryActions() {
  const qc = useQueryClient()

  const snapshot = async (): Promise<Snapshot> => {
    await qc.cancelQueries({ queryKey: qk.memories.all })
    return qc.getQueriesData({ queryKey: qk.memories.all })
  }
  const restore = (snap: Snapshot | undefined) => snap?.forEach(([key, data]) => qc.setQueryData(key, data))
  const settle = () => void qc.invalidateQueries({ queryKey: qk.memories.all })

  const patchLists = (fn: (list: Memory[]) => Memory[]) => {
    for (const [key, data] of qc.getQueriesData<unknown>({ queryKey: ['memories', 'list'] })) {
      if (Array.isArray(data)) qc.setQueryData(key, fn(data as Memory[]))
    }
  }

  const update = useMutation({
    mutationFn: ({ id, content }: { id: number; content: string }) => api.memories.update(id, content),
    onMutate: async ({ id, content }) => {
      const snap = await snapshot()
      patchLists((list) =>
        list.map((m) =>
          m.id === id ? ({ ...m, previous_content: m.content, content, updated_at: new Date().toISOString() } as Memory) : m
        )
      )
      qc.setQueryData<MemoryGraph>(qk.memories.graph, (g) =>
        g ? { ...g, nodes: g.nodes.map((n) => (n.id === id ? { ...n, content, title: content, label: content.slice(0, 25) } : n)) } : g
      )
      return snap
    },
    onError: (e, _v, snap) => {
      restore(snap)
      toast.error("Couldn't update memory", { description: errorMessage(e) })
    },
    onSuccess: () => toast.success('Memory updated', { description: 'Topics and expiry were re-analyzed.' }),
    onSettled: settle
  })

  const remove = useMutation({
    mutationFn: (id: number) => api.memories.delete(id),
    onMutate: async (id) => {
      const snap = await snapshot()
      patchLists((list) => list.filter((m) => m.id !== id))
      qc.setQueryData<MemoryGraph>(qk.memories.graph, (g) =>
        g
          ? {
              nodes: g.nodes.filter((n) => n.id !== id),
              links: g.links.filter((l) => linkEnd(l.source) !== id && linkEnd(l.target) !== id)
            }
          : g
      )
      return snap
    },
    onError: (e, _v, snap) => {
      restore(snap)
      toast.error("Couldn't forget that memory", { description: errorMessage(e) })
    },
    onSuccess: () => toast.success('Memory forgotten'),
    onSettled: settle
  })

  const move = useMutation({
    mutationFn: ({ id, scope }: { id: number; scope: string }) => api.memories.move(id, scope),
    onMutate: async ({ id, scope }) => {
      const snap = await snapshot()
      patchLists((list) => list.map((m) => (m.id === id ? ({ ...m, scope } as Memory) : m)))
      return snap
    },
    onError: (e, _v, snap) => {
      restore(snap)
      toast.error("Couldn't move that memory", { description: errorMessage(e) })
    },
    onSuccess: (m) => toast.success(m.scope === 'always' ? 'Every chat can use this memory now' : `Moved to ${scopeName(m.scope)}`),
    onSettled: settle
  })

  const setState = useMutation({
    mutationFn: ({ id, state }: { id: number; state: 'current' | 'disputed' }) => api.memories.setState(id, state),
    onMutate: async ({ id, state }) => {
      const snap = await snapshot()
      patchLists((list) => list.map((m) => (m.id === id ? ({ ...m, state, ...(state === 'current' ? { confidence: 1 } : {}) } as Memory) : m)))
      return snap
    },
    onError: (e, _v, snap) => {
      restore(snap)
      toast.error("Couldn't change that memory", { description: errorMessage(e) })
    },
    onSuccess: (m) =>
      toast.success(m.state === 'disputed' ? 'Sentient will leave this out of chats and tasks' : 'Sentient will use this memory again'),
    onSettled: settle
  })

  return useMemo(() => ({ update, remove, move, setState }), [update, remove, move, setState])
}

function scopeName(scope: string | undefined): string {
  return !scope || scope === 'personal' ? 'Personal' : scope === 'work' ? 'Work' : scope
}

/** force-graph mutates link ends into node objects. */
export function linkEnd(end: unknown): number {
  if (end && typeof end === 'object' && 'id' in end) return Number((end as { id: unknown }).id)
  return Number(end)
}
