/** List view: tasks grouped by what they need (attention, running, scheduled…), collapsible. */
import { IconChevronRight, IconSearchOff } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useMemo, useState } from 'react'
import { Button, EmptyState, ProgressBar } from '@/components/ui'
import type { Task } from '@/lib/types'
import { cn, relativeTime, truncate } from '@/lib/utils'
import { GROUPS, displayName, isScriptJob, latestRun, processingRun, scriptOf, sortTasks, taskGroup, waitingRun, type GroupId } from './meta'
import { KindTile, PriorityFlag, ScriptBadge, TaskStatusBadge } from './parts'
import { describeSchedule, nextRunDate, upcomingPhrase } from './schedule'
import { ToolStack, toolIdentity } from './tools'

const STORAGE_KEY = 'sentient.tasks.collapsed'

function readCollapsed(): Partial<Record<GroupId, boolean>> {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') as Partial<Record<GroupId, boolean>>
  } catch {
    return {}
  }
}

export function ListView({
  tasks,
  selectedId,
  onSelect,
  tz,
  now,
  filtersActive,
  onClearFilters
}: {
  tasks: Task[]
  selectedId: string | null
  onSelect: (task: Task) => void
  tz: string
  now: Date
  filtersActive: boolean
  onClearFilters: () => void
}) {
  const [collapsed, setCollapsed] = useState(readCollapsed)
  const groups = useMemo(() => {
    const by = new Map<GroupId, Task[]>()
    for (const t of tasks) {
      const g = taskGroup(t)
      by.set(g, [...(by.get(g) ?? []), t])
    }
    return GROUPS.map((g) => ({ ...g, tasks: sortTasks(by.get(g.id) ?? []) })).filter((g) => g.tasks.length)
  }, [tasks])

  const toggle = (id: GroupId, open: boolean) => {
    const next = { ...collapsed, [id]: !open }
    setCollapsed(next)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    } catch {
      /* ignore */
    }
  }

  if (!groups.length) {
    return filtersActive ? (
      <EmptyState
        icon={<IconSearchOff />}
        title="No tasks match these filters"
        description="Try a different search or clear the filters."
        action={<Button size="sm" onClick={onClearFilters}>Clear filters</Button>}
      />
    ) : null
  }

  return (
    <div className="space-y-5">
      {groups.map((g) => {
        const open = filtersActive || (collapsed[g.id] === undefined ? g.defaultOpen : !collapsed[g.id])
        return (
          <section key={g.id} aria-label={g.label}>
            <button
              type="button"
              onClick={() => toggle(g.id, open)}
              aria-expanded={open}
              className="group mb-2 flex w-full items-center gap-2 rounded-md px-1 py-0.5 text-left"
            >
              <IconChevronRight size={14} className={cn('text-fg-subtle transition-transform duration-150', open && 'rotate-90')} />
              <span className={cn('text-sm font-semibold', g.id === 'attention' ? 'text-accent-text' : 'text-fg')}>{g.label}</span>
              <span
                className={cn(
                  'rounded-full px-1.5 text-2xs font-semibold tabular-nums',
                  g.id === 'attention' ? 'bg-accent text-accent-fg' : 'bg-active text-fg-muted'
                )}
              >
                {g.tasks.length}
              </span>
              <span className="hidden truncate text-xs text-fg-faint opacity-0 transition-opacity group-hover:opacity-100 @2xl:inline">{g.hint}</span>
            </button>
            <AnimatePresence initial={false}>
              {open && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.18 }}
                  className="overflow-hidden"
                >
                  <div
                    className={cn(
                      'divide-y divide-border overflow-hidden rounded-xl border bg-surface',
                      g.id === 'attention' ? 'border-accent/20' : 'border-border'
                    )}
                  >
                    {g.tasks.map((t) => (
                      <TaskRow key={t.task_id} task={t} selected={t.task_id === selectedId} onSelect={onSelect} tz={tz} now={now} />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        )
      })}
    </div>
  )
}

/** Second line of a task row: what's going on right now. */
export function taskHeadline(task: Task): string {
  const run = processingRun(task)
  switch (task.status) {
    case 'planning':
      return task.task_type === 'swarm' ? 'Splitting the work across agents…' : 'Sentient is drafting a plan…'
    case 'approval_pending':
      return isScriptJob(task) ? 'A small script is ready for your review' : `${task.plan.length}-step plan ready for your review`
    case 'clarification_pending': {
      const open = task.clarifying_questions.filter((q) => !q.answer).length
      return `${open} question${open === 1 ? '' : 's'} before Sentient can plan this`
    }
    case 'waiting_for_user': {
      const pending = waitingRun(task)?.pending_question
      if (pending?.kind === 'stuck' && pending.reason) return truncate(`Stuck: ${pending.reason}`, 140)
      const q = pending?.question
      return q ? truncate(`Asks: ${q}`, 140) : 'Waiting for your answer'
    }
    case 'error':
      return truncate(task.error || latestRun(task)?.error || 'The last run failed', 140)
    case 'processing': {
      if (task.task_type === 'swarm' && task.swarm_details) {
        return `${task.swarm_details.completed_agents} of ${task.swarm_details.total_agents} agents finished`
      }
      const boilerplate = /^(Resuming the run|Executor has picked up)/
      const last = [...(run?.progress_updates ?? [])].reverse().find((u) => !(u.message?.type === 'info' && boilerplate.test(u.message.content ?? '')))?.message
      if (last?.type === 'tool_call') return `Using ${toolIdentity(last.tool_name).label}…`
      if (last?.type === 'tool_result') return `Reading ${toolIdentity(last.tool_name).label} results…`
      if (last?.content) return truncate(last.content.replace(/[*_`#]/g, ''), 140)
      return 'Working…'
    }
  }
  const script = scriptOf(task)
  if (script) {
    if (script.last_error) return truncate(`Last check didn’t work: ${script.last_error}`, 140)
    const when = describeSchedule(task.schedule).text
    return script.last_run_at ? `${when} · checked ${relativeTime(script.last_run_at)}` : `${when} · not checked yet`
  }
  const s = describeSchedule(task.schedule, { swarm: task.task_type === 'swarm' })
  if (task.task_type === 'swarm' && task.swarm_details?.total_agents) {
    return `${task.swarm_details.total_agents} agents · ${task.swarm_details.items.length} items`
  }
  return s.detail ? `${s.text} · ${s.detail}` : s.text
}

export function taskTiming(task: Task, tz: string, now: Date): string {
  const next = task.status !== 'archived' ? nextRunDate(task, now) : null
  if (next && next.getTime() > now.getTime() - 60_000) return `Next ${upcomingPhrase(next, tz, now)}`
  if (task.status === 'processing') {
    const run = processingRun(task)
    return run ? `Started ${relativeTime(run.execution_start_time ?? run.created_at, now.getTime())}` : 'Running'
  }
  const asked = waitingRun(task)?.pending_question?.asked_at
  if (task.status === 'waiting_for_user' && asked) return `Asked ${relativeTime(asked, now.getTime())}`
  if (task.last_execution_at) return `Ran ${relativeTime(task.last_execution_at, now.getTime())}`
  return `Created ${relativeTime(task.created_at, now.getTime())}`
}

function TaskRow({ task, selected, onSelect, tz, now }: { task: Task; selected: boolean; onSelect: (t: Task) => void; tz: string; now: Date }) {
  const swarm = task.task_type === 'swarm' ? task.swarm_details : null
  const temp = task.task_id.startsWith('temp-')
  const tools = task.plan.map((p) => p.tool)
  if (task.schedule?.type === 'triggered' && task.schedule.source) tools.unshift(task.schedule.source)

  return (
    <button
      type="button"
      disabled={temp}
      onClick={() => onSelect(task)}
      aria-current={selected || undefined}
      className={cn(
        'group relative flex w-full items-center gap-3 px-3.5 py-2.5 text-left transition-colors',
        selected ? 'bg-active' : 'hover:bg-hover',
        temp && 'opacity-70'
      )}
    >
      {selected && <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-accent" />}
      <KindTile task={task} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-sm font-medium text-fg">{displayName(task)}</span>
          <PriorityFlag priority={task.priority} />
          {isScriptJob(task) && <ScriptBadge />}
        </div>
        <div className={cn('mt-0.5 truncate text-xs', task.status === 'error' ? 'text-danger/90' : 'text-fg-subtle')}>{taskHeadline(task)}</div>
        {swarm && swarm.total_agents > 0 && task.status === 'processing' && (
          <ProgressBar value={swarm.completed_agents / swarm.total_agents} tone="info" size="xs" className="mt-1.5 max-w-60" />
        )}
      </div>
      <ToolStack tools={tools} className="hidden @3xl:flex" />
      <span className="hidden w-44 shrink-0 truncate text-right text-xs tabular-nums text-fg-subtle @2xl:block">{temp ? 'Creating…' : taskTiming(task, tz, now)}</span>
      <span className="flex w-[9.5rem] shrink-0 justify-end">
        <TaskStatusBadge task={task} />
      </span>
      <IconChevronRight size={15} className="shrink-0 text-fg-faint transition-transform group-hover:translate-x-0.5 group-hover:text-fg-subtle" />
    </button>
  )
}
