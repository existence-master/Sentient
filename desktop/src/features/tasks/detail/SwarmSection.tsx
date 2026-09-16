/** Swarm details: goal, items, sub-agent progress, activity and aggregated results. */
import { IconBraces, IconCircleCheck, IconCircleX, IconLoader2, IconTable, IconTarget, IconUsersGroup } from '@tabler/icons-react'
import { useMemo, useState } from 'react'
import { Badge, JsonView, ProgressBar } from '@/components/ui'
import type { SwarmDetails } from '@/lib/types'
import { cn, humanize, parseDate } from '@/lib/utils'
import { SectionHeading } from '../parts'

export function SwarmSection({ swarm, tz, running }: { swarm: SwarmDetails; tz: string; running: boolean }) {
  const agents = useMemo(() => {
    const map = new Map<string, { id: string; status: string; message: string }>()
    for (const u of swarm.progress_updates ?? []) {
      if (u.worker_id === 'aggregator') continue
      map.set(u.worker_id, { id: u.worker_id, status: u.status, message: u.message })
    }
    return [...map.values()].sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }))
  }, [swarm.progress_updates])
  const total = swarm.total_agents || agents.length
  const done = swarm.completed_agents
  const failed = agents.filter((a) => a.status === 'error').length
  const aggregating = (swarm.progress_updates ?? []).some((u) => u.worker_id === 'aggregator')

  return (
    <section className="space-y-4">
      <SectionHeading icon={<IconUsersGroup />} title="Swarm" description={`${swarm.items.length} items · ${total} agents`} />

      <div className="rounded-xl border border-info/20 bg-info/5 px-4 py-3">
        <div className="flex items-center gap-1.5 text-2xs font-semibold uppercase tracking-wider text-info">
          <IconTarget size={12} /> Goal
        </div>
        <p className="mt-1 text-sm leading-relaxed text-fg">{swarm.goal}</p>
      </div>

      <div className="rounded-xl border border-border bg-surface p-4">
        <div className="flex items-end gap-3">
          <div>
            <div className="text-2xl font-semibold tabular-nums tracking-tight text-fg">
              {done}
              <span className="text-base text-fg-subtle"> / {total}</span>
            </div>
            <div className="text-xs text-fg-subtle">agents finished{failed ? ` · ${failed} failed` : ''}</div>
          </div>
          <span className="flex-1" />
          {running && <Badge tone="info" icon={<IconLoader2 className="animate-spin" />}>{aggregating ? 'Aggregating' : 'Working'}</Badge>}
        </div>
        <ProgressBar value={total ? done / total : running ? null : 0} tone={failed ? 'warning' : running ? 'info' : 'success'} size="md" className="mt-3" />
        {agents.length > 0 && (
          <div className="mt-3 grid grid-cols-2 gap-1.5 @xl:grid-cols-3">
            {agents.map((a, i) => (
              <div key={a.id} className="flex min-w-0 items-center gap-2 rounded-lg border border-border bg-elevated px-2 py-1.5">
                {a.status === 'completed' ? (
                  <IconCircleCheck size={14} className="shrink-0 text-success" />
                ) : a.status === 'error' ? (
                  <IconCircleX size={14} className="shrink-0 text-danger" />
                ) : (
                  <IconLoader2 size={14} className="shrink-0 animate-spin text-info" />
                )}
                <div className="min-w-0">
                  <div className="text-xs font-medium text-fg">{humanize(a.id)}</div>
                  <div className="truncate text-2xs text-fg-subtle">{typeof swarm.items[i] === 'string' ? (swarm.items[i] as string) : a.message}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {swarm.aggregated_results.length > 0 && <AggregatedResults results={swarm.aggregated_results} />}

      {(swarm.progress_updates?.length ?? 0) > 0 && (
        <details className="group rounded-xl border border-border bg-surface">
          <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-sm font-medium text-fg">
            Agent activity <span className="rounded-full bg-active px-1.5 text-2xs text-fg-muted">{swarm.progress_updates.length}</span>
            <span className="flex-1" />
            <span className="text-xs text-fg-subtle group-open:hidden">Show</span>
            <span className="hidden text-xs text-fg-subtle group-open:inline">Hide</span>
          </summary>
          <ol className="max-h-64 space-y-1 overflow-y-auto border-t border-border px-3 py-2 font-mono text-xs">
            {swarm.progress_updates.map((u, i) => (
              <li key={i} className="flex gap-2">
                <span className="shrink-0 tabular-nums text-fg-faint">{parseDate(u.timestamp)?.toLocaleTimeString(undefined, { timeZone: tz, hour: 'numeric', minute: '2-digit', second: '2-digit' })}</span>
                <span className={cn('w-20 shrink-0', u.status === 'error' ? 'text-danger' : u.status === 'completed' ? 'text-success' : 'text-info')}>{u.worker_id}</span>
                <span className="min-w-0 break-words text-fg-muted">{u.message}</span>
              </li>
            ))}
          </ol>
        </details>
      )}
    </section>
  )
}

function AggregatedResults({ results }: { results: unknown[] }) {
  const [raw, setRaw] = useState(false)
  const columns = useMemo(() => {
    if (!results.every((r) => r && typeof r === 'object' && !Array.isArray(r))) return null
    const keys = new Set<string>()
    for (const r of results) Object.keys(r as object).forEach((k) => keys.add(k))
    const flat = results.every((r) => Object.values(r as object).every((v) => v === null || typeof v !== 'object'))
    return flat && keys.size <= 8 ? [...keys] : null
  }, [results])

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-surface">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <span className="text-sm font-medium text-fg">Aggregated results</span>
        <span className="rounded-full bg-active px-1.5 text-2xs text-fg-muted">{results.length}</span>
        <span className="flex-1" />
        {columns && (
          <button type="button" onClick={() => setRaw((r) => !r)} className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs text-fg-subtle hover:bg-hover hover:text-fg">
            {raw ? <IconTable size={13} /> : <IconBraces size={13} />}
            {raw ? 'Table' : 'JSON'}
          </button>
        )}
      </div>
      {columns && !raw ? (
        <div className="max-h-96 overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-elevated">
              <tr>
                {columns.map((c) => (
                  <th key={c} className="whitespace-nowrap border-b border-border px-3 py-1.5 font-medium text-fg-subtle">
                    {humanize(c)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {results.map((r, i) => (
                <tr key={i} className="hover:bg-hover">
                  {columns.map((c, j) => (
                    <td key={c} className={cn('px-3 py-2 align-top text-fg-muted', j === 0 && 'whitespace-nowrap font-medium text-fg')}>
                      {String((r as Record<string, unknown>)[c] ?? '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="max-h-96 overflow-auto px-3 py-2.5">
          <JsonView value={results} collapsedDepth={2} />
        </div>
      )}
    </div>
  )
}
