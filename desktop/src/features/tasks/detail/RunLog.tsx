/** Execution log: ProgressUpdates as a readable timeline, streaming live with auto-scroll. */
import {
  IconAlertTriangle,
  IconArrowDown,
  IconBulb,
  IconCheck,
  IconChevronRight,
  IconCircleCheckFilled,
  IconInfoCircle,
  IconX
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Button, JsonView, Markdown, Spinner } from '@/components/ui'
import { argsPreview, toolMeta } from '@/features/chat/toolMeta'
import { useRunEvents } from '@/hooks/tasks'
import type { ProgressUpdate, Run } from '@/lib/types'
import { cn, parseDate } from '@/lib/utils'
import { ToolIcon } from '../tools'

const EMBEDDED_LIMIT = 200

type Entry =
  | { kind: 'info' | 'thought' | 'final_answer' | 'error'; key: string; at: string; content: string }
  | { kind: 'tool'; key: string; at: string; name: string; args?: Record<string, unknown>; result?: unknown; hasResult: boolean; isError: boolean; resultAt?: string }

function buildEntries(updates: ProgressUpdate[]): Entry[] {
  const out: Entry[] = []
  const open: Array<Extract<Entry, { kind: 'tool' }>> = []
  updates.forEach((u, i) => {
    const m = u.message ?? { type: 'info' }
    const key = `${i}`
    if (m.type === 'tool_call') {
      const e: Extract<Entry, { kind: 'tool' }> = { kind: 'tool', key, at: u.timestamp, name: m.tool_name ?? 'tool', args: m.parameters, hasResult: false, isError: false }
      out.push(e)
      open.push(e)
    } else if (m.type === 'tool_result') {
      const idx = open.findIndex((c) => c.name === m.tool_name)
      const call = idx >= 0 ? open.splice(idx, 1)[0] : open.shift()
      if (call) {
        call.result = m.result
        call.hasResult = true
        call.isError = !!m.is_error
        call.resultAt = u.timestamp
      } else {
        out.push({ kind: 'tool', key, at: u.timestamp, name: m.tool_name ?? 'tool', result: m.result, hasResult: true, isError: !!m.is_error })
      }
    } else {
      out.push({ kind: m.type, key, at: u.timestamp, content: m.content ?? (typeof m.result === 'string' ? m.result : '') })
    }
  })
  return out
}

export function RunLog({ taskId, run, live, tz, maxHeight = 460 }: { taskId: string; run: Run; live: boolean; tz: string; maxHeight?: number }) {
  const truncated = run.progress_updates.length >= EMBEDDED_LIMIT
  const [full, setFull] = useState(false)
  const events = useRunEvents(full ? taskId : undefined, full ? run.run_id : undefined)
  const updates = full && events.data ? events.data : run.progress_updates
  const entries = useMemo(() => buildEntries(updates), [updates])

  const scroller = useRef<HTMLDivElement>(null)
  const [atBottom, setAtBottom] = useState(true)
  const [unseen, setUnseen] = useState(0)
  const prevCount = useRef(entries.length)

  useLayoutEffect(() => {
    const el = scroller.current
    if (el && live) el.scrollTop = el.scrollHeight
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const added = entries.length - prevCount.current
    prevCount.current = entries.length
    if (added <= 0) return
    const el = scroller.current
    if (!el) return
    if (atBottom) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
    else setUnseen((n) => n + added)
  }, [entries.length, atBottom])

  const jump = () => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' })
    setUnseen(0)
  }

  if (!entries.length) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-dashed border-border px-4 py-5 text-sm text-fg-subtle">
        {live ? <Spinner size={14} className="text-info" /> : <IconInfoCircle size={15} />}
        {live ? 'Waiting for the first step…' : 'No execution log for this run.'}
      </div>
    )
  }

  const lastIndex = entries.length - 1

  return (
    <div className="relative">
      {(truncated || full) && (
        <div className="mb-2 flex items-center gap-2 text-xs text-fg-subtle">
          {full ? (events.isLoading ? <><Spinner size={12} /> Loading the full log…</> : `Full log · ${updates.length} updates`) : `Showing the latest ${EMBEDDED_LIMIT} updates`}
          {!full && (
            <button type="button" onClick={() => setFull(true)} className="font-medium text-accent-text hover:underline">
              Load full log
            </button>
          )}
        </div>
      )}
      <div
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget
          const bottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32
          setAtBottom(bottom)
          if (bottom) setUnseen(0)
        }}
        style={{ maxHeight }}
        className="overflow-y-auto rounded-xl border border-border bg-sunken/40 px-3 py-3"
      >
        <ol className="relative">
          {entries.map((e, i) => (
            <LogEntry key={e.key} entry={e} last={i === lastIndex} live={live && i === lastIndex} tz={tz} />
          ))}
        </ol>
      </div>
      <AnimatePresence>
        {!atBottom && (live || unseen > 0) && (
          <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }} className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center">
            <Button size="xs" variant="secondary" className="pointer-events-auto rounded-full shadow-pop" leftIcon={<IconArrowDown size={12} />} onClick={jump}>
              {unseen > 0 ? `${unseen} new · Jump to latest` : 'Jump to latest'}
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function stamp(iso: string, tz: string) {
  const d = parseDate(iso)
  return d ? d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit', second: '2-digit', timeZone: tz }) : ''
}

function Node({ children, tone = 'neutral' }: { children: React.ReactNode; tone?: 'neutral' | 'accent' | 'success' | 'danger' | 'info' }) {
  const cls = {
    neutral: 'border-border bg-surface text-fg-subtle',
    accent: 'border-accent/30 bg-accent/10 text-accent-text',
    success: 'border-success/30 bg-success/10 text-success',
    danger: 'border-danger/30 bg-danger/10 text-danger',
    info: 'border-info/30 bg-info/10 text-info'
  }[tone]
  return <span className={cn('relative z-10 flex size-6 shrink-0 items-center justify-center rounded-full border', cls)}>{children}</span>
}

function LogEntry({ entry, last, live, tz }: { entry: Entry; last: boolean; live: boolean; tz: string }) {
  const [open, setOpen] = useState(false)
  const time = <span className="shrink-0 pt-0.5 font-mono text-2xs tabular-nums text-fg-faint">{stamp(entry.at, tz)}</span>
  const rail = !last && <span aria-hidden className="absolute bottom-0 left-[11.5px] top-6 w-px bg-border" />

  if (entry.kind === 'tool') {
    const meta = toolMeta(entry.name)
    const running = !entry.hasResult && live
    const preview = argsPreview(entry.args)
    return (
      <li className="relative flex gap-2.5 pb-3">
        {rail}
        <Node tone={entry.isError ? 'danger' : 'neutral'}>
          <ToolIcon name={entry.name} size={13} />
        </Node>
        <div className="min-w-0 flex-1">
          <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="flex w-full min-w-0 items-center gap-2 text-left">
            <span className={cn('shrink-0 text-sm font-medium', running ? 'text-fg-muted' : 'text-fg')}>{running ? meta.running : meta.done}</span>
            {preview && <span className="min-w-0 flex-1 truncate font-mono text-xs text-fg-subtle">{preview}</span>}
            {!preview && <span className="flex-1" />}
            {running ? (
              <Spinner size={12} className="text-info" />
            ) : entry.isError ? (
              <IconX size={13} className="text-danger" />
            ) : entry.hasResult ? (
              <IconCheck size={13} className="text-success" />
            ) : null}
            <IconChevronRight size={13} className={cn('shrink-0 text-fg-faint transition-transform', open && 'rotate-90')} />
            {time}
          </button>
          <AnimatePresence initial={false}>
            {open && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.15 }} className="overflow-hidden">
                <div className="mt-2 space-y-2.5 rounded-lg border border-border bg-surface px-3 py-2.5">
                  <div className="font-mono text-2xs text-fg-subtle">{entry.name}</div>
                  {entry.args && Object.keys(entry.args).length > 0 && (
                    <div>
                      <div className="mb-1 text-2xs font-medium uppercase tracking-wide text-fg-subtle">Arguments</div>
                      <JsonView value={entry.args} collapsedDepth={3} />
                    </div>
                  )}
                  <div>
                    <div className={cn('mb-1 text-2xs font-medium uppercase tracking-wide', entry.isError ? 'text-danger' : 'text-fg-subtle')}>{entry.isError ? 'Error' : 'Result'}</div>
                    {!entry.hasResult ? (
                      <div className="text-xs text-fg-subtle">{live ? 'Waiting for the result…' : 'No result recorded'}</div>
                    ) : typeof entry.result === 'string' ? (
                      <pre className={cn('selectable max-h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-xs', entry.isError ? 'text-danger' : 'text-fg-muted')}>{entry.result}</pre>
                    ) : (
                      <div className="max-h-72 overflow-auto">
                        <JsonView value={entry.result} collapsedDepth={2} />
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </li>
    )
  }

  if (entry.kind === 'thought') {
    return (
      <li className="relative flex gap-2.5 pb-3">
        {rail}
        <Node tone="accent">
          <IconBulb size={13} />
        </Node>
        <div className="min-w-0 flex-1">
          <button type="button" onClick={() => setOpen((o) => !o)} className="flex w-full items-start gap-2 text-left">
            <span className={cn('min-w-0 flex-1 text-sm italic text-fg-muted', !open && 'line-clamp-2')}>{entry.content}</span>
            {time}
          </button>
        </div>
      </li>
    )
  }

  if (entry.kind === 'final_answer') {
    return (
      <li className="relative flex gap-2.5 pb-3">
        {rail}
        <Node tone="success">
          <IconCircleCheckFilled size={13} />
        </Node>
        <div className="min-w-0 flex-1 rounded-xl border border-success/20 bg-success/6 px-3 py-2.5">
          <div className="mb-1 flex items-center gap-2 text-2xs font-semibold uppercase tracking-wide text-success">
            Final answer <span className="flex-1" />
            {time}
          </div>
          <Markdown className="text-sm">{entry.content}</Markdown>
        </div>
      </li>
    )
  }

  if (entry.kind === 'error') {
    return (
      <li className="relative flex gap-2.5 pb-3">
        {rail}
        <Node tone="danger">
          <IconAlertTriangle size={13} />
        </Node>
        <div className="flex min-w-0 flex-1 items-start gap-2">
          <span className="min-w-0 flex-1 text-sm text-danger">{entry.content}</span>
          {time}
        </div>
      </li>
    )
  }

  return (
    <li className="relative flex gap-2.5 pb-3">
      {rail}
      <Node tone={live ? 'info' : 'neutral'}>{live ? <Spinner size={11} /> : <IconInfoCircle size={13} />}</Node>
      <div className="flex min-w-0 flex-1 items-start gap-2 pt-0.5">
        <span className={cn('min-w-0 flex-1 text-sm', live ? 'animate-pulse text-fg' : 'text-fg-muted')}>{entry.content}</span>
        {time}
      </div>
    </li>
  )
}
