import { IconChevronRight, IconClockHour4, IconUserBolt, IconUsersGroup } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Badge, Button, Markdown } from '@/components/ui'
import { errorMessage } from '@/lib/api'
import { wasDenied, type ToolCallView } from '@/lib/chatFold'
import type { DelegateResult, SubagentStatus } from '@/lib/types'
import { cn, safeJsonParse } from '@/lib/utils'
import { useCancelSubagent, useSubagent } from '../subagents'
import { asRecord, DeclinedText, Shimmer, StatusGlyph } from './bits'

export const isSubagentTool = (name: string) => name === 'delegate_task' || name === 'delegate_tasks'

function statusFromTool(tool: ToolCallView): SubagentStatus {
  if (tool.status === 'running' || tool.status === 'awaiting_approval') return 'running'
  if (tool.status === 'error') return 'error'
  if (tool.status === 'denied') return 'cancelled'
  return 'completed'
}

/** `delegate_task` / `delegate_tasks` (§10): helpers working on part of the request. */
export function SubagentCard({ tool }: { tool: ToolCallView }) {
  const args = tool.arguments as { goal?: string; background?: boolean }
  if (tool.status === 'denied' || wasDenied(tool.result)) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-border bg-surface px-3.5 py-2.5">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-warning/12 text-warning">
          {tool.name === 'delegate_tasks' ? <IconUsersGroup size={17} /> : <IconUserBolt size={17} />}
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium">
            <DeclinedText tool={tool} />
          </div>
          <div className="text-xs text-fg-subtle">No helper was started, so nothing changed</div>
        </div>
        <StatusGlyph status="denied" />
      </div>
    )
  }
  if (tool.name === 'delegate_tasks') return <SubagentGroup tool={tool} />
  const res = asRecord<DelegateResult>(tool.result)
  return (
    <HelperRow
      goal={args.goal ?? 'A helper task'}
      background={!!args.background}
      subagentId={res?.subagent_id ?? tool.progress?.subagent?.subagentId}
      status={res?.status ?? statusFromTool(tool)}
      summary={res?.summary ?? null}
      error={res?.error ?? (tool.status === 'error' && typeof tool.result === 'string' ? tool.result : null)}
      messages={tool.progress?.subagent?.messages ?? []}
    />
  )
}

function HelperRow({
  goal,
  background,
  subagentId,
  status: initialStatus,
  summary: initialSummary,
  error,
  messages,
  nested
}: {
  goal: string
  background: boolean
  subagentId?: string
  status: SubagentStatus
  summary: string | null
  error: string | null
  messages: string[]
  nested?: boolean
}) {
  // Background helpers keep going after the tool returns: follow them through the subagent API/events.
  const follow = !!subagentId && (background || initialStatus === 'running')
  const sub = useSubagent(follow ? subagentId : undefined)
  const cancel = useCancelSubagent()
  const [open, setOpen] = useState(false)
  const status = sub.data?.status ?? initialStatus
  const summary = sub.data?.summary ?? initialSummary
  const steps = sub.data?.tool_calls ?? (messages.length || undefined)
  const running = status === 'running'
  const latest = messages[messages.length - 1]

  const subline =
    status === 'running'
      ? null
      : status === 'completed'
        ? `Done${steps ? ` after ${steps} ${steps === 1 ? 'step' : 'steps'}` : ''}`
        : status === 'cancelled'
          ? 'Stopped'
          : (sub.data?.error ?? error ?? 'Ran into a problem')

  return (
    <div className={cn(!nested && 'overflow-hidden rounded-xl border border-border bg-surface')}>
      <div className={cn('flex items-center gap-3', nested ? 'px-3 py-2' : 'px-3.5 py-2.5')}>
        <div
          className={cn(
            'relative flex shrink-0 items-center justify-center rounded-lg',
            nested ? 'size-6.5' : 'size-8',
            status === 'error' ? 'bg-danger/10 text-danger' : 'bg-accent/10 text-accent-text'
          )}
        >
          <IconUserBolt size={nested ? 14 : 17} />
          {running && <span className="absolute -right-0.5 -top-0.5 size-2 animate-pulse rounded-full bg-accent ring-2 ring-surface" />}
        </div>
        <div className="min-w-0 flex-1">
          {!nested && <div className="text-2xs font-medium uppercase tracking-wide text-fg-subtle">{background ? 'Helper in the background' : 'Helper'}</div>}
          <div className={cn('text-fg', nested ? 'truncate text-xs' : 'line-clamp-2 text-sm font-medium')}>{goal}</div>
          <div className={cn('truncate text-xs', status === 'error' ? 'text-danger' : 'text-fg-subtle')}>
            {running ? (
              <Shimmer>{latest ?? (background ? 'Working in the background' : 'Working on it')}</Shimmer>
            ) : (
              subline
            )}
            {running && steps ? <span className="text-fg-faint"> · {steps} steps so far</span> : null}
          </div>
        </div>
        {background && running && !nested && (
          <Badge size="xs" icon={<IconClockHour4 />}>
            Keeps going if you leave
          </Badge>
        )}
        {running && subagentId && (
          <Button
            size="xs"
            variant="ghost"
            loading={cancel.isPending}
            onClick={() =>
              cancel.mutate(subagentId, { onError: (e) => toast.error("Couldn't stop the helper", { description: errorMessage(e) }) })
            }
          >
            Stop
          </Button>
        )}
        {!running && summary && (
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="flex h-7 items-center gap-1 rounded-md px-2 text-xs text-fg-subtle transition-colors hover:bg-hover hover:text-fg"
          >
            {open ? 'Hide' : 'What it found'}
            <IconChevronRight size={13} className={cn('transition-transform', open && 'rotate-90')} />
          </button>
        )}
      </div>
      <AnimatePresence initial={false}>
        {open && summary && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className={cn('border-t border-border bg-sunken/35 py-3 text-sm', nested ? 'px-3' : 'px-4')}>
              <Markdown>{summary}</Markdown>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function SubagentGroup({ tool }: { tool: ToolCallView }) {
  const tasks = ((tool.arguments as { tasks?: Array<{ goal?: string }> }).tasks ?? []).map((t) => t.goal ?? 'A helper task')
  const raw = typeof tool.result === 'string' ? safeJsonParse<unknown>(tool.result, null) : tool.result
  const list: DelegateResult[] = Array.isArray(raw)
    ? (raw as DelegateResult[])
    : Array.isArray((raw as { results?: unknown })?.results)
      ? ((raw as { results: DelegateResult[] }).results)
      : []
  const fallback = statusFromTool(tool)
  const done = list.filter((r) => r?.status === 'completed').length
  return (
    <div className="overflow-hidden rounded-xl border border-border bg-surface">
      <div className="flex items-center gap-3 px-3.5 py-2.5">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent-text">
          <IconUsersGroup size={17} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-fg">
            {fallback === 'running' ? <Shimmer>{`${tasks.length} helpers working in parallel`}</Shimmer> : `${tasks.length} helpers worked in parallel`}
          </div>
          {fallback !== 'running' && <div className="text-xs text-fg-subtle">{done || tasks.length} finished</div>}
        </div>
      </div>
      <div className="divide-y divide-border border-t border-border">
        {tasks.map((goal, i) => (
          <HelperRow
            key={i}
            nested
            goal={goal}
            background={false}
            subagentId={list[i]?.subagent_id}
            status={list[i]?.status ?? fallback}
            summary={list[i]?.summary ?? null}
            error={list[i]?.error ?? null}
            messages={[]}
          />
        ))}
      </div>
    </div>
  )
}
