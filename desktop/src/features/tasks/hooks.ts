import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router'
import { useBootstrap } from '@/hooks/core'
import { useIntegrations } from '@/hooks/integrations'
import { useTasks } from '@/hooks/tasks'
import type { Task } from '@/lib/types'
import { needsAttention, taskKind, taskSource, type TaskKind, type TaskSource } from './meta'
import { resolveZone } from './schedule'
import { pluginIdFor } from './tools'

/** The user's timezone (config `assistant.timezone`, `auto` -> system). */
export function useUserTimezone(): string {
  const { data } = useBootstrap()
  return resolveZone(data?.assistant.timezone)
}

/** Re-render every `ms` (relative times, live durations). */
export function useNow(ms = 30_000): Date {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const t = window.setInterval(() => setNow(new Date()), ms)
    return () => window.clearInterval(t)
  }, [ms])
  return now
}

export function useAttentionCount(): number {
  const { data } = useTasks()
  return useMemo(() => (data ?? []).filter(needsAttention).length, [data])
}

export interface MissingIntegration {
  id: string
  label: string
}

/** Integrations the plan or trigger needs that aren't connected (v2 "Connect tools to approve"). */
export function useMissingIntegrations(task: Pick<Task, 'plan' | 'schedule'> | undefined): MissingIntegration[] {
  const integrations = useIntegrations()
  return useMemo(() => {
    if (!task || !integrations.data) return []
    const needed = new Set<string>()
    for (const step of task.plan ?? []) if (step.tool) needed.add(pluginIdFor(step.tool))
    if (task.schedule?.type === 'triggered' && task.schedule.source) needed.add(pluginIdFor(task.schedule.source))
    const out: MissingIntegration[] = []
    for (const id of needed) {
      const integ = integrations.data.find((i) => i.id === id)
      if (integ && !integ.connected && integ.auth_type !== 'builtin') out.push({ id, label: integ.display_name })
    }
    return out
  }, [task, integrations.data])
}

// ---------------------------------------------------------------------------- URL state
export type TasksView = 'list' | 'calendar' | 'board'
export type StatusFilter = 'all' | 'attention' | 'planning' | 'running' | 'scheduled' | 'completed' | 'failed' | 'archived'

export const STATUS_FILTERS: Array<{ value: StatusFilter; label: string }> = [
  { value: 'all', label: 'Any status' },
  { value: 'attention', label: 'Needs attention' },
  { value: 'planning', label: 'Planning' },
  { value: 'running', label: 'Running' },
  { value: 'scheduled', label: 'Scheduled or active' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'archived', label: 'Archived' }
]

export interface TaskFilterState {
  q: string
  status: StatusFilter
  kind: 'all' | TaskKind
  source: 'all' | TaskSource
}

const statusPredicates: Record<StatusFilter, (t: Task) => boolean> = {
  all: () => true,
  attention: needsAttention,
  planning: (t) => t.status === 'planning',
  running: (t) => t.status === 'processing',
  scheduled: (t) => t.status === 'pending' || t.status === 'active',
  completed: (t) => t.status === 'completed' || t.status === 'completed_with_errors',
  failed: (t) => t.status === 'error' || t.runs.some((r) => r.status === 'error'),
  archived: (t) => t.status === 'archived'
}

export function applyFilters(tasks: Task[], f: TaskFilterState): Task[] {
  const q = f.q.trim().toLowerCase()
  return tasks.filter((t) => {
    if (!statusPredicates[f.status](t)) return false
    if (f.kind !== 'all' && taskKind(t) !== f.kind) return false
    if (f.source !== 'all' && taskSource(t) !== f.source) return false
    if (!q) return true
    const hay = [t.name, t.description, ...t.plan.map((p) => `${p.tool} ${p.description}`)].join(' ').toLowerCase()
    return hay.includes(q)
  })
}

export function useTaskUrlState() {
  const [params, setParams] = useSearchParams()
  const view = (['list', 'calendar', 'board'].includes(params.get('view') ?? '') ? params.get('view') : 'list') as TasksView
  const filters: TaskFilterState = {
    q: params.get('q') ?? '',
    status: (STATUS_FILTERS.some((s) => s.value === params.get('status')) ? params.get('status') : 'all') as StatusFilter,
    kind: (params.get('kind') as TaskFilterState['kind']) || 'all',
    source: (params.get('source') as TaskFilterState['source']) || 'all'
  }
  const set = (patch: Record<string, string | null | undefined>, replace = true) => {
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        for (const [k, v] of Object.entries(patch)) {
          if (v === null || v === undefined || v === '' || v === 'all') next.delete(k)
          else next.set(k, v)
        }
        return next
      },
      { replace }
    )
  }
  return {
    params,
    view,
    filters,
    selected: params.get('task'),
    day: params.get('day'),
    month: params.get('month'),
    set,
    filtersActive: !!(filters.q || filters.status !== 'all' || filters.kind !== 'all' || filters.source !== 'all')
  }
}
