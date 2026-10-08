/** Board: read-only kanban columns by status. */
import { useMemo } from 'react'
import { ProgressBar } from '@/components/ui'
import type { Task } from '@/lib/types'
import { cn } from '@/lib/utils'
import { taskHeadline, taskTiming } from './ListView'
import { BOARD_COLUMNS, KIND_META, displayName, isScriptJob, sortTasks, statusMeta, taskKind } from './meta'
import { KindTile, PriorityFlag, ScriptBadge } from './parts'
import { ToolStack } from './tools'

const dot: Record<string, string> = {
  neutral: 'bg-fg-faint',
  accent: 'bg-accent',
  success: 'bg-success',
  warning: 'bg-warning',
  danger: 'bg-danger',
  info: 'bg-info'
}

export function BoardView({ tasks, tz, now, selectedId, onSelect }: { tasks: Task[]; tz: string; now: Date; selectedId: string | null; onSelect: (t: Task) => void }) {
  const columns = useMemo(
    () => BOARD_COLUMNS.map((c) => ({ ...c, tasks: sortTasks(tasks.filter((t) => c.statuses.includes(t.status))) })),
    [tasks]
  )

  return (
    <div className="h-full overflow-x-auto pb-1">
      <div className="flex h-full min-h-[520px] gap-3">
        {columns.map((col) => (
          <section key={col.id} aria-label={col.label} className="flex min-w-[14rem] flex-1 basis-0 flex-col rounded-xl border border-border bg-sunken/40">
            <header className="flex items-center gap-2 px-3 pb-2 pt-2.5">
              <span className={cn('size-2 rounded-full', dot[col.tone])} />
              <h3 className="text-sm font-semibold text-fg">{col.label}</h3>
              <span className="rounded-full bg-active px-1.5 text-2xs font-medium tabular-nums text-fg-muted">{col.tasks.length}</span>
            </header>
            <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-2 pb-2">
              {col.tasks.length === 0 && <div className="rounded-lg border border-dashed border-border px-3 py-6 text-center text-xs text-fg-faint">Nothing here</div>}
              {col.tasks.map((t) => {
                const kind = taskKind(t)
                const K = KIND_META[kind].icon
                const swarm = t.swarm_details
                const meta = statusMeta(t.status, t)
                return (
                  <button
                    key={t.task_id}
                    type="button"
                    onClick={() => onSelect(t)}
                    className={cn(
                      'group flex flex-col gap-2 rounded-lg border bg-surface p-3 text-left shadow-soft transition-colors hover:border-border-strong',
                      t.task_id === selectedId ? 'border-accent/50' : 'border-border'
                    )}
                  >
                    <div className="flex items-start gap-2.5">
                      <KindTile task={t} size="sm" />
                      <div className="min-w-0 flex-1">
                        <div className="line-clamp-2 text-sm font-medium leading-snug text-fg">{displayName(t)}</div>
                        {isScriptJob(t) && <ScriptBadge className="mt-1" />}
                      </div>
                      <PriorityFlag priority={t.priority} className="mt-1" />
                    </div>
                    <p className={cn('line-clamp-2 text-xs leading-relaxed', t.status === 'error' ? 'text-danger/90' : 'text-fg-subtle')}>{taskHeadline(t)}</p>
                    {swarm && swarm.total_agents > 0 && t.status === 'processing' && <ProgressBar value={swarm.completed_agents / swarm.total_agents} tone="info" size="xs" />}
                    <div className="flex items-center gap-2 border-t border-border pt-2 text-2xs text-fg-subtle">
                      <K size={12} className="shrink-0" />
                      <span className="shrink-0">{!t.enabled && col.id === 'scheduled' ? meta.label : KIND_META[kind].label}</span>
                      <span className="min-w-0 flex-1 truncate text-right tabular-nums">{taskTiming(t, tz, now)}</span>
                      <ToolStack tools={t.plan.map((p) => p.tool)} max={3} />
                    </div>
                  </button>
                )
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}
