/** Reverse-chronological runs: trigger event, live log, result. */
import {
  IconAlertCircle,
  IconBolt,
  IconCalendarEvent,
  IconChevronRight,
  IconClock,
  IconExternalLink,
  IconFile,
  IconHandFinger,
  IconHistory,
  IconLink,
  IconMail,
  IconMapPin,
  IconPlayerStop,
  IconRepeat,
  IconUsers,
  IconWorldSearch
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, EmptyState, JsonView, Markdown, Tooltip } from '@/components/ui'
import { api, errorMessage } from '@/lib/api'
import { useRetryRun } from '@/lib/leap/hooks-b'
import { retryOf } from '@/lib/leap/types-b'
import { getBridge } from '@/lib/bridge'
import type { Run, Task, TaskRunResult } from '@/lib/types'
import { cn, formatDuration, parseDate, relativeTime, truncate } from '@/lib/utils'
import { useNow } from '../hooks'
import { runDurationMs } from '../meta'
import { LiveDot, RunStatusBadge, SectionHeading } from '../parts'
import { formatInZone, sourceLabel } from '../schedule'
import { ToolIcon, toolIdentity, useToolNames } from '../tools'
import { RunLog } from './RunLog'

export function RunHistory({
  task,
  tz,
  focusRunId,
  onCancelRun,
  cancelling
}: {
  task: Task
  tz: string
  focusRunId?: string | null
  onCancelRun: (runId: string) => void
  cancelling: boolean
}) {
  const runs = [...task.runs].reverse()
  if (!runs.length) {
    return (
      <EmptyState
        compact
        icon={<IconHistory />}
        title="No runs yet"
        description={task.status === 'approval_pending' ? 'Approve the plan and the first run will show up here.' : 'Runs and their live progress will show up here.'}
      />
    )
  }
  return (
    <section className="space-y-2.5">
      <SectionHeading icon={<IconHistory />} title="Runs" count={runs.length} description="Newest first" />
      <div className="space-y-2">
        {runs.map((run, i) => (
          <RunCard
            key={run.run_id}
            task={task}
            run={run}
            number={runs.length - i}
            tz={tz}
            defaultOpen={focusRunId ? focusRunId === run.run_id : i === 0}
            onCancel={() => onCancelRun(run.run_id)}
            cancelling={cancelling}
          />
        ))}
      </div>
    </section>
  )
}

function triggerSummary(task: Task, run: Run): { icon: typeof IconBolt; text: string } {
  const data = run.trigger_event_data
  if (data && Object.keys(data).length) {
    const subject = typeof data.subject === 'string' ? data.subject : typeof data.summary === 'string' ? data.summary : null
    const source = task.schedule?.type === 'triggered' ? task.schedule.source : undefined
    return { icon: IconBolt, text: subject ? `${sourceLabel(source)}: ${truncate(subject, 60)}` : `Triggered by ${sourceLabel(source)}` }
  }
  if (task.schedule?.type === 'recurring') return { icon: IconClock, text: 'Scheduled run' }
  if (task.task_type === 'swarm') return { icon: IconUsers, text: 'Swarm run' }
  return { icon: IconHandFinger, text: task.schedule?.type === 'once' && task.schedule.run_at ? 'Scheduled run' : 'Started on approval' }
}

function RunCard({ task, run, number, tz, defaultOpen, onCancel, cancelling }: { task: Task; run: Run; number: number; tz: string; defaultOpen: boolean; onCancel: () => void; cancelling: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  const live = run.status === 'processing'
  const now = useNow(live ? 1000 : 60_000)
  const started = parseDate(run.execution_start_time ?? run.created_at)
  const duration = runDurationMs(run, now.getTime())
  const trig = triggerSummary(task, run)
  const TrigIcon = trig.icon
  const retry = useRetryRun()
  const retryable = (run.status === 'error' || run.status === 'cancelled') && task.task_type !== 'swarm'
  const retriedFrom = retryOf(run)
  const retriedNumber = retriedFrom ? task.runs.findIndex((r) => r.run_id === retriedFrom) + 1 : 0
  const doRetry = () =>
    retry.mutate(
      { taskId: task.task_id, runId: run.run_id },
      {
        onSuccess: () => toast.success('Trying again', { description: 'Sentient picks up where this run stopped, so finished steps aren’t repeated.' }),
        onError: (e) => toast.error('Couldn’t retry this run', { description: errorMessage(e) })
      }
    )
  const retryButton = (
    <Button size="xs" variant="secondary" leftIcon={<IconRepeat size={12} />} loading={retry.isPending} onClick={doRetry}>
      Try again
    </Button>
  )

  return (
    <div className={cn('overflow-hidden rounded-xl border bg-surface transition-colors', live ? 'border-info/30' : 'border-border', open && !live && 'border-border-strong')}>
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="flex w-full items-center gap-3 px-3.5 py-2.5 text-left transition-colors hover:bg-hover">
        <IconChevronRight size={14} className={cn('shrink-0 text-fg-subtle transition-transform', open && 'rotate-90')} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-fg">Run #{number}</span>
            {retriedFrom && (
              <Tooltip content="Continued from a run that didn’t finish, without repeating the steps that worked">
                <span>
                  <Badge size="xs" tone="info" icon={<IconRepeat />}>
                    {retriedNumber > 0 ? `Retry of #${retriedNumber}` : 'Retry'}
                  </Badge>
                </span>
              </Tooltip>
            )}
            {live && <LiveDot />}
          </div>
          <div className="mt-0.5 flex min-w-0 items-center gap-1.5 text-xs text-fg-subtle">
            <TrigIcon size={12} className="shrink-0" />
            <span className="truncate">{trig.text}</span>
            <span className="text-fg-faint">·</span>
            <span className="shrink-0 tabular-nums" title={started?.toLocaleString()}>
              {started ? `${formatInZone(started, tz, { month: 'short', day: 'numeric' })}, ${formatInZone(started, tz, { hour: 'numeric', minute: '2-digit' })}` : 'Pending'}
            </span>
            {duration !== null && (
              <>
                <span className="text-fg-faint">·</span>
                <span className="shrink-0 tabular-nums">{live ? `running ${formatDuration(duration)}` : formatDuration(duration)}</span>
              </>
            )}
          </div>
        </div>
        {run.result?.tools_used?.length ? (
          <span className="hidden items-center gap-1 @xl:flex">
            {run.result.tools_used.slice(0, 4).map((t) => (
              <ToolIcon key={t} name={t} size={14} />
            ))}
          </span>
        ) : null}
        <RunStatusBadge status={run.status} />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.18 }} className="overflow-hidden">
            <div className="space-y-4 border-t border-border px-3.5 pb-4 pt-3.5">
              {live && (
                <div className="flex items-center gap-2 rounded-lg border border-info/20 bg-info/6 px-3 py-2 text-sm">
                  <span className="min-w-0 flex-1 text-fg-muted">
                    Started {relativeTime(run.execution_start_time ?? run.created_at, now.getTime())}. Updates stream in as Sentient works.
                  </span>
                  <Button size="xs" variant="danger" leftIcon={<IconPlayerStop size={12} />} loading={cancelling} onClick={onCancel}>
                    Cancel run
                  </Button>
                </div>
              )}
              {run.trigger_event_data && Object.keys(run.trigger_event_data).length > 0 && (
                <TriggerEventCard source={task.schedule?.type === 'triggered' ? task.schedule.source : undefined} data={run.trigger_event_data} tz={tz} />
              )}
              {run.error && (
                <Alert tone="danger" icon={<IconAlertCircle />} title={run.status === 'cancelled' ? 'This run was stopped' : 'This run failed'} action={retryable ? retryButton : undefined}>
                  {run.error}
                </Alert>
              )}
              {retryable && !run.error && (
                <div className="flex items-center gap-2 rounded-lg border border-border bg-sunken/40 px-3 py-2 text-sm">
                  <span className="min-w-0 flex-1 text-fg-muted">This run was stopped before it finished.</span>
                  {retryButton}
                </div>
              )}
              {run.result && <RunResult result={run.result} />}
              <div>
                <div className="mb-2 flex items-center gap-2 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">Execution log</div>
                <RunLog taskId={task.task_id} run={run} live={live} tz={tz} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ---------------------------------------------------------------------------- trigger event
const str = (v: unknown) => (typeof v === 'string' ? v : '')

export function TriggerEventCard({ source, data, tz }: { source?: string; data: Record<string, unknown>; tz: string }) {
  const open = (url: string) => /^https?:/.test(url) && void getBridge().openExternal(url)
  const isEmail = source === 'gmail' || ('subject' in data && ('from' in data || 'sender_email' in data))
  const isEvent = source === 'gcalendar' || ('summary' in data && 'start' in data)

  if (isEmail) {
    const from = str(data.from)
    const name = from.replace(/<.*>/, '').replace(/"/g, '').trim() || str(data.sender_email)
    const email = str(data.sender_email) || /<(.+?)>/.exec(from)?.[1] || ''
    const date = parseDate(str(data.date))
    const labels = Array.isArray(data.labels) ? (data.labels as unknown[]).map(String).filter((l) => !/^CATEGORY_/.test(l)) : []
    return (
      <div className="rounded-xl border border-border bg-elevated">
        <div className="flex items-center gap-2 border-b border-border px-3 py-1.5 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">
          <IconBolt size={12} className="text-accent-text" /> Triggered by this email
        </div>
        <div className="flex gap-3 px-3 py-3">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-full bg-[#ea4335]/12 text-[#ea4335]">
            <IconMail size={17} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline gap-2">
              <span className="truncate text-sm font-semibold text-fg">{name}</span>
              {email && email !== name && <span className="truncate text-xs text-fg-subtle">{email}</span>}
              <span className="flex-1" />
              {date && <span className="shrink-0 text-xs tabular-nums text-fg-subtle">{formatInZone(date, tz, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}</span>}
            </div>
            <div className="mt-0.5 truncate text-sm text-fg">{str(data.subject) || '(no subject)'}</div>
            {str(data.snippet) && <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-fg-muted">{str(data.snippet)}</p>}
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              {str(data.to) && <span className="text-2xs text-fg-subtle">to {str(data.to)}</span>}
              {labels.map((l) => (
                <Badge key={l} size="xs">
                  {l.toLowerCase()}
                </Badge>
              ))}
              <span className="flex-1" />
              {str(data.url) && (
                <Button size="xs" variant="ghost" rightIcon={<IconExternalLink size={12} />} onClick={() => open(str(data.url))}>
                  Open in Gmail
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (isEvent) {
    const start = parseDate(str(data.start))
    const end = parseDate(str(data.end))
    const attendees = Array.isArray(data.attendees) ? (data.attendees as unknown[]).map((a) => (typeof a === 'string' ? a : str((a as Record<string, unknown>)?.email))) : []
    return (
      <div className="rounded-xl border border-border bg-elevated">
        <div className="flex items-center gap-2 border-b border-border px-3 py-1.5 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">
          <IconBolt size={12} className="text-accent-text" /> Triggered by this event
        </div>
        <div className="flex gap-3 px-3 py-3">
          {start ? (
            <span className="flex w-11 shrink-0 flex-col items-center overflow-hidden rounded-lg border border-border bg-surface">
              <span className="w-full bg-[#4285f4] py-px text-center text-[9px] font-bold uppercase tracking-wider text-white">{formatInZone(start, tz, { month: 'short' })}</span>
              <span className="py-0.5 text-base font-semibold tabular-nums text-fg">{formatInZone(start, tz, { day: 'numeric' })}</span>
            </span>
          ) : (
            <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-[#4285f4]/12 text-[#4285f4]">
              <IconCalendarEvent size={17} />
            </span>
          )}
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-semibold text-fg">{str(data.summary) || 'Untitled event'}</div>
            {start && (
              <div className="mt-0.5 flex items-center gap-1.5 text-xs text-fg-muted">
                <IconClock size={12} />
                {formatInZone(start, tz, { weekday: 'short', hour: 'numeric', minute: '2-digit' })}
                {end && ` – ${formatInZone(end, tz, { hour: 'numeric', minute: '2-digit' })}`}
              </div>
            )}
            {str(data.location) && (
              <div className="mt-0.5 flex items-center gap-1.5 text-xs text-fg-muted">
                <IconMapPin size={12} />
                {str(data.location)}
              </div>
            )}
            {str(data.description) && <p className="mt-1 line-clamp-2 text-xs text-fg-subtle">{str(data.description)}</p>}
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              {attendees.length > 0 && (
                <span className="flex items-center gap-1 text-2xs text-fg-subtle">
                  <IconUsers size={12} /> {attendees.length} attendee{attendees.length === 1 ? '' : 's'}
                  {str(data.organizer_email) && ` · organised by ${str(data.organizer_email)}`}
                </span>
              )}
              <span className="flex-1" />
              {str(data.url) && (
                <Button size="xs" variant="ghost" rightIcon={<IconExternalLink size={12} />} onClick={() => open(str(data.url))}>
                  Open in Calendar
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-border bg-elevated">
      <div className="flex items-center gap-2 border-b border-border px-3 py-1.5 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">
        <IconBolt size={12} className="text-accent-text" /> Trigger event {source && `from ${sourceLabel(source)}`}
      </div>
      <div className="max-h-56 overflow-auto px-3 py-2.5">
        <JsonView value={data} collapsedDepth={1} />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------- result
export function RunResult({ result }: { result: TaskRunResult }) {
  const { names } = useToolNames()
  const open = (url: string) => /^https?:/.test(url) && void getBridge().openExternal(url)
  const links = [...(result.links_created ?? []).map((l) => ({ ...l, created: true })), ...(result.links_found ?? []).map((l) => ({ ...l, created: false }))]
  return (
    <div className="space-y-3">
      {result.summary && (
        <div className="rounded-xl border border-border bg-elevated px-4 py-3">
          <div className="mb-1.5 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">Result</div>
          <Markdown className="text-sm">{result.summary}</Markdown>
        </div>
      )}
      {(links.length > 0 || (result.files_created?.length ?? 0) > 0) && (
        <div className="grid gap-3 @5xl:grid-cols-2">
          {(result.files_created?.length ?? 0) > 0 && (
            <div className="rounded-xl border border-border bg-surface">
              <div className="border-b border-border px-3 py-1.5 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">Files created</div>
              <ul className="divide-y divide-border">
                {result.files_created.map((f) => (
                  <li key={f.filename} className="flex items-center gap-2.5 px-3 py-2">
                    <span className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
                      <IconFile size={14} />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-mono text-xs text-fg">{f.filename}</div>
                      {f.description && <div className="truncate text-2xs text-fg-subtle">{f.description}</div>}
                    </div>
                    <Button size="xs" variant="ghost" onClick={() => open(api.files.contentUrl(f.filename))} rightIcon={<IconExternalLink size={12} />}>
                      Open
                    </Button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {links.length > 0 && (
            <div className="rounded-xl border border-border bg-surface">
              <div className="border-b border-border px-3 py-1.5 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">Links</div>
              <ul className="divide-y divide-border">
                {links.map((l, i) => (
                  <li key={`${l.url}-${i}`}>
                    <button type="button" onClick={() => open(l.url)} className="flex w-full items-center gap-2.5 px-3 py-2 text-left hover:bg-hover">
                      {l.created ? <IconLink size={14} className="shrink-0 text-accent-text" /> : <IconWorldSearch size={14} className="shrink-0 text-info" />}
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-xs font-medium text-fg">{l.description || l.url}</div>
                        <div className="truncate text-2xs text-fg-subtle">
                          {l.created ? 'Created' : 'Found'} · {l.url.replace(/^https?:\/\//, '')}
                        </div>
                      </div>
                      <IconExternalLink size={12} className="shrink-0 text-fg-faint" />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      {(result.tools_used?.length ?? 0) > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-fg-subtle">Used</span>
          {result.tools_used.map((t) => (
            <span key={t} className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2 py-0.5 text-xs text-fg-muted">
              <ToolIcon name={t} size={12} />
              {toolIdentity(t, names).label}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
