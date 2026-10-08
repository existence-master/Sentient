/**
 * Tasks: long-running work Sentient plans, schedules and carries out.
 *
 *   /tasks                       list (default), ?view=calendar|board, filters ?q&status&kind&source
 *   /tasks?task=<id>             side panel with the task (notifications link here)
 *   /tasks?view=calendar&day=…   day timeline, &month=YYYY-MM for the month grid
 *   /tasks/:taskId               full-page task (?tab=overview|runs|chat, &run=<run_id>)
 *   ?prompt=…&compose=1          prefill and focus the new-task composer
 */
import {
  IconAlertTriangle,
  IconBolt,
  IconCalendarMonth,
  IconCalendarRepeat,
  IconLayoutKanban,
  IconListDetails,
  IconRefresh,
  IconRoute,
  IconUsersGroup
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useMemo, type ReactNode } from 'react'
import { useNavigate, useParams } from 'react-router'
import { Button, EmptyState, SegmentedControl, Skeleton } from '@/components/ui'
import { BoardView } from '@/features/tasks/BoardView'
import { CalendarView } from '@/features/tasks/CalendarView'
import { DayView } from '@/features/tasks/DayView'
import { TaskDetail } from '@/features/tasks/detail/TaskDetail'
import { applyFilters, useNow, useTaskUrlState, useUserTimezone, type TasksView } from '@/features/tasks/hooks'
import { ListView } from '@/features/tasks/ListView'
import { needsAttention } from '@/features/tasks/meta'
import { TaskComposer } from '@/features/tasks/TaskComposer'
import { TaskToolbar } from '@/features/tasks/TaskToolbar'
import { useTasks } from '@/hooks/tasks'
import { errorMessage, isNotImplemented } from '@/lib/api'
import type { Task } from '@/lib/types'
import { cn } from '@/lib/utils'

export function TasksPage() {
  const { taskId } = useParams()
  if (taskId) return <TaskDetail key={taskId} taskId={taskId} layout="page" />
  return <TasksHome />
}

function TasksHome() {
  const url = useTaskUrlState()
  const tasks = useTasks()
  const tz = useUserTimezone()
  const now = useNow(30_000)
  const navigate = useNavigate()
  const all = useMemo(() => tasks.data ?? [], [tasks.data])
  const { q, status, kind, source } = url.filters
  const filtered = useMemo(() => applyFilters(all, { q, status, kind, source }), [all, q, status, kind, source])
  const nonArchived = useMemo(() => (status === 'archived' ? filtered : filtered.filter((t) => t.status !== 'archived')), [filtered, status])
  const selected = url.selected && !url.selected.startsWith('temp-') ? url.selected : null

  const counts = useMemo(
    () => ({
      attention: all.filter(needsAttention).length,
      running: all.filter((t) => t.status === 'processing' || t.status === 'planning').length,
      scheduled: all.filter((t) => (t.status === 'active' || t.status === 'pending') && t.enabled).length
    }),
    [all]
  )

  const select = (t: Task) => url.set({ task: t.task_id }, false)
  const close = () => url.set({ task: null })

  useEffect(() => {
    if (!selected) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return
      if (document.querySelector('[role="dialog"], [role="menu"], [role="listbox"]')) return
      const target = e.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA'].includes(target.tagName)) return
      url.set({ task: null })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected])

  const clearFilters = () => url.set({ q: null, status: null, kind: null, source: null })
  const fullHeight = url.view !== 'list'

  let content: ReactNode
  if (tasks.isLoading) {
    content = <ListSkeleton />
  } else if (tasks.isError) {
    content = (
      <EmptyState
        icon={<IconAlertTriangle />}
        title={isNotImplemented(tasks.error) ? "The tasks engine isn't available yet" : "Couldn't load your tasks"}
        description={errorMessage(tasks.error)}
        action={
          <Button size="sm" leftIcon={<IconRefresh size={15} />} onClick={() => void tasks.refetch()}>
            Try again
          </Button>
        }
      />
    )
  } else if (!all.length) {
    content = <FirstTask />
  } else if (url.view === 'calendar') {
    content = url.day ? (
      <DayView tasks={nonArchived} tz={tz} now={now} day={url.day} onDayChange={(day) => url.set({ day })} onBack={() => url.set({ day: null })} onSelectTask={select} />
    ) : (
      <CalendarView tasks={nonArchived} tz={tz} now={now} month={url.month} onMonthChange={(month) => url.set({ month })} onOpenDay={(day) => url.set({ day }, false)} onSelectTask={select} />
    )
  } else if (url.view === 'board') {
    content = <BoardView tasks={nonArchived} tz={tz} now={now} selectedId={selected} onSelect={select} />
  } else {
    content = <ListView tasks={filtered} selectedId={selected} onSelect={select} tz={tz} now={now} filtersActive={url.filtersActive} onClearFilters={clearFilters} />
  }

  return (
    <div className="flex h-full min-h-0">
      <div className="@container flex min-w-0 flex-1 flex-col">
        <header className="shrink-0 px-8 pb-3 pt-6">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
            <div className="min-w-0 flex-1">
              <h1 className="text-xl font-semibold tracking-tight text-fg">Tasks</h1>
              <p className="mt-0.5 flex flex-wrap items-center gap-x-1.5 text-sm text-fg-muted">
                {all.length === 0 ? (
                  'Long-running work Sentient plans, schedules and carries out for you.'
                ) : (
                  <>
                    <SummaryLink active={status === 'attention'} tone={counts.attention ? 'accent' : 'muted'} onClick={() => url.set({ status: status === 'attention' ? null : 'attention', view: 'list' === url.view ? null : url.view })}>
                      {counts.attention} need{counts.attention === 1 ? 's' : ''} your attention
                    </SummaryLink>
                    <span className="text-fg-faint">·</span>
                    <SummaryLink active={status === 'running'} onClick={() => url.set({ status: status === 'running' ? null : 'running' })}>
                      {counts.running} in progress
                    </SummaryLink>
                    <span className="text-fg-faint">·</span>
                    <SummaryLink active={status === 'scheduled'} onClick={() => url.set({ status: status === 'scheduled' ? null : 'scheduled' })}>
                      {counts.scheduled} scheduled
                    </SummaryLink>
                  </>
                )}
              </p>
            </div>
            <SegmentedControl<TasksView>
              aria-label="View"
              value={url.view}
              onChange={(view) => url.set({ view: view === 'list' ? null : view, day: null }, false)}
              options={[
                { value: 'list', label: 'List', icon: <IconListDetails size={15} /> },
                { value: 'calendar', label: 'Calendar', icon: <IconCalendarMonth size={15} /> },
                { value: 'board', label: 'Board', icon: <IconLayoutKanban size={15} /> }
              ]}
            />
          </div>
          <TaskComposer className="mt-4" initialPrompt={url.params.get('prompt') ?? ''} autoFocus={url.params.get('compose') === '1'} />
          {all.length > 0 && (
            <div className="mt-4">
              <TaskToolbar filters={url.filters} onChange={(patch) => url.set(patch)} active={url.filtersActive} resultCount={filtered.length} />
            </div>
          )}
        </header>
        <div className={cn('min-h-0 flex-1 px-8 pb-8 pt-2', fullHeight ? 'flex flex-col overflow-hidden' : 'overflow-y-auto')}>
          <motion.div key={`${url.view}:${url.day ?? ''}`} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className={cn(fullHeight && 'min-h-0 flex-1 overflow-y-auto')}>
            {content}
          </motion.div>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {selected && (
          <motion.aside
            key="task-panel"
            initial={{ opacity: 0, x: 28 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 28 }}
            transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}
            className="w-[min(600px,52%)] shrink-0 border-l border-border bg-surface shadow-[-12px_0_32px_-24px_rgb(0_0_0/0.5)]"
            aria-label="Task details"
          >
            <TaskDetail key={selected} taskId={selected} layout="panel" onClose={close} />
          </motion.aside>
        )}
      </AnimatePresence>
      <NavigateGuard onMissing={() => navigate('/tasks', { replace: true })} />
    </div>
  )
}

/** Keeps `/tasks?task=temp-…` from sticking after an optimistic create. */
function NavigateGuard({ onMissing }: { onMissing: () => void }) {
  const url = useTaskUrlState()
  useEffect(() => {
    if (url.selected?.startsWith('temp-')) onMissing()
  }, [url.selected, onMissing])
  return null
}

function SummaryLink({ children, onClick, active, tone = 'muted' }: { children: ReactNode; onClick: () => void; active?: boolean; tone?: 'accent' | 'muted' }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'rounded px-0.5 underline-offset-4 transition-colors hover:underline',
        tone === 'accent' ? 'font-medium text-accent-text' : 'text-fg-muted hover:text-fg',
        active && 'underline'
      )}
    >
      {children}
    </button>
  )
}

function ListSkeleton() {
  return (
    <div className="space-y-5">
      {[3, 2, 4].map((n, g) => (
        <div key={g} className="space-y-2">
          <Skeleton className="h-4 w-44" />
          <div className="divide-y divide-border overflow-hidden rounded-xl border border-border">
            {Array.from({ length: n }, (_, i) => (
              <div key={i} className="flex items-center gap-3 px-3.5 py-3">
                <Skeleton className="size-9 rounded-xl" />
                <div className="flex-1 space-y-1.5">
                  <Skeleton className={cn('h-3.5', ['w-2/5', 'w-1/3', 'w-1/2', 'w-2/5'][i])} />
                  <Skeleton className="h-3 w-3/5" />
                </div>
                <Skeleton className="h-5.5 w-24 rounded-full" />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

const STARTERS = [
  { icon: IconCalendarRepeat, title: 'On a schedule', prompt: 'Every weekday at 9am, summarise my unread email and flag anything urgent' },
  { icon: IconBolt, title: 'When something happens', prompt: 'When an email with an invoice arrives, save the PDF to Google Drive' },
  { icon: IconRoute, title: 'A one-off job', prompt: 'Plan a 3-day trip to Goa in November with flights, stays and a day-by-day itinerary' },
  { icon: IconUsersGroup, title: 'With parallel agents', prompt: 'Research these 5 note-taking apps and compare pricing, sync and offline support' }
]

function FirstTask() {
  return (
    <div className="mx-auto max-w-3xl py-10 text-center">
      <h2 className="text-lg font-semibold tracking-tight text-fg">Hand off your first task</h2>
      <p className="mx-auto mt-1.5 max-w-lg text-sm text-fg-muted">
        Describe it in plain words. Sentient drafts a step-by-step plan, asks for your approval, then runs it once, on a schedule or whenever something happens.
      </p>
      <div className="mt-6 grid gap-3 text-left sm:grid-cols-2">
        {STARTERS.map((s) => (
          <button
            key={s.title}
            type="button"
            onClick={() => window.dispatchEvent(new CustomEvent('sentient:new-task', { detail: s.prompt }))}
            className="group flex gap-3 rounded-xl border border-border bg-surface p-3.5 transition-colors hover:border-border-strong hover:bg-elevated"
          >
            <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
              <s.icon size={17} />
            </span>
            <span className="min-w-0">
              <span className="block text-sm font-medium text-fg">{s.title}</span>
              <span className="mt-0.5 block text-xs leading-relaxed text-fg-subtle group-hover:text-fg-muted">“{s.prompt}”</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
