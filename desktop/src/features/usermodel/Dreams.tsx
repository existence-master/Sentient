/** Dreams: the nightly memory consolidation, as a timeline of first-person journal entries. */
import { IconAlertTriangle, IconChevronDown, IconMoonStars, IconSparkles } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Alert, Button, EmptyState, Markdown, Skeleton, Spinner } from '@/components/ui'
import { useConfig } from '@/hooks/core'
import { errorMessage } from '@/lib/api'
import { notReady, useDreams, useRunDream } from '@/lib/leap/hooks-b'
import type { Dream } from '@/lib/leap/types-b'
import { cn, formatDuration, formatTime, parseDate } from '@/lib/utils'
import { dreamStatChips } from './meta'

function dreamDay(d: Dream, now = new Date()): string {
  const at = parseDate(d.started_at)
  if (!at) return 'Some time ago'
  const start = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime()
  const days = Math.round((start(now) - start(at)) / 86_400_000)
  const night = at.getHours() < 6
  if (days === 0) return night ? 'Last night' : 'Today'
  if (days === 1) return night ? 'The night before' : 'Yesterday'
  return at.toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' })
}

export function DreamsView() {
  const dreams = useDreams()
  const run = useRunDream()
  const config = useConfig()
  const dreaming = (config.data as unknown as { dreaming?: { enabled?: boolean; time?: string } } | undefined)?.dreaming
  const running = dreams.data?.some((d) => d.status === 'running') ?? false

  const consolidate = () =>
    run.mutate(undefined, {
      onSuccess: () => toast.success('Tidying up my memory now', { description: 'This usually takes a minute or two. I’ll write about it when I’m done.' }),
      onError: (e) => toast.error('Couldn’t start right now', { description: errorMessage(e) })
    })

  return (
    <div className="space-y-6">
      <div className="relative flex flex-wrap items-center gap-4 overflow-hidden rounded-2xl border border-border bg-surface px-5 py-4">
        <div aria-hidden className="pointer-events-none absolute -left-10 -top-16 size-48 rounded-full opacity-40 blur-3xl" style={{ background: 'radial-gradient(circle, #6d5ae6, transparent 70%)' }} />
        <span className="relative flex size-10 shrink-0 items-center justify-center rounded-xl border border-[#8b7bf0]/30 bg-[#8b7bf0]/12 text-[#a99cf5]">
          <IconMoonStars size={20} />
        </span>
        <div className="relative min-w-0 flex-1 basis-72">
          <div className="text-sm font-semibold text-fg">While you sleep, I tidy up</div>
          <p className="mt-0.5 text-sm text-fg-muted">
            Each night I look back over what I’ve learned: merge repeats, settle things that don’t agree, and let go of what is no longer true.
            {dreaming?.enabled === false ? ' Nightly tidying is turned off in Settings.' : dreaming?.time ? ` Next time: tonight at ${formatClock(dreaming.time)}.` : ''}
          </p>
        </div>
        <Button className="relative" variant="secondary" leftIcon={running ? <Spinner size={14} /> : <IconSparkles size={15} />} disabled={running || dreams.isError} loading={run.isPending} onClick={consolidate}>
          {running ? 'Tidying up…' : 'Consolidate now'}
        </Button>
      </div>

      {dreams.isLoading ? (
        <div className="space-y-4">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-40 rounded-2xl" />
          ))}
        </div>
      ) : dreams.isError ? (
        notReady(dreams.error) ? (
          <EmptyState icon={<IconMoonStars />} title="Dreams arrive with the next engine update" description="Once it’s here, you’ll find a short note from Sentient every morning about what it tidied up overnight." />
        ) : (
          <Alert tone="danger" title="Couldn’t load the dream journal" action={<Button size="sm" onClick={() => void dreams.refetch()}>Retry</Button>}>
            {errorMessage(dreams.error)}
          </Alert>
        )
      ) : !dreams.data?.length ? (
        <EmptyState icon={<IconMoonStars />} title="No dreams yet" description="After the first night, you’ll find a short note here about what Sentient tidied up. You can also start one now." />
      ) : (
        <ol className="relative space-y-5">
          <span aria-hidden className="absolute bottom-6 left-[19px] top-6 w-px bg-gradient-to-b from-border-strong via-border to-transparent" />
          <AnimatePresence initial={false}>
            {dreams.data.map((d, i) => (
              <DreamEntry key={d.id} dream={d} defaultOpen={i === 0} />
            ))}
          </AnimatePresence>
        </ol>
      )}
    </div>
  )
}

function formatClock(hhmm: string): string {
  const m = /^(\d{1,2}):(\d{2})/.exec(hhmm)
  if (!m) return hhmm
  return new Date(2000, 0, 1, Number(m[1]), Number(m[2])).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
}

function DreamEntry({ dream: d, defaultOpen }: { dream: Dream; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  const chips = dreamStatChips(d.stats)
  const duration = d.finished_at ? (parseDate(d.finished_at)?.getTime() ?? 0) - (parseDate(d.started_at)?.getTime() ?? 0) : null
  const long = d.journal_md.length > 280

  return (
    <motion.li layout="position" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="relative flex list-none gap-4">
      <span
        className={cn(
          'relative z-10 flex size-10 shrink-0 items-center justify-center rounded-full border bg-bg',
          d.status === 'error' ? 'border-danger/30 text-danger' : d.status === 'running' ? 'border-accent/40 text-accent-text' : 'border-[#8b7bf0]/30 text-[#a99cf5]'
        )}
      >
        {d.status === 'running' ? <Spinner size={16} /> : d.status === 'error' ? <IconAlertTriangle size={17} /> : d.trigger === 'manual' ? <IconSparkles size={17} /> : <IconMoonStars size={17} />}
      </span>
      <div className="min-w-0 flex-1 pb-1">
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 pt-1">
          <h3 className="text-sm font-semibold text-fg">{dreamDay(d)}</h3>
          <span className="text-xs text-fg-subtle">
            {formatTime(d.started_at)}
            {d.trigger === 'manual' ? ' · you asked' : ''}
            {duration && duration > 0 ? ` · took ${formatDuration(duration)}` : ''}
          </span>
        </div>

        {d.status === 'running' ? (
          <div className="mt-2 overflow-hidden rounded-2xl border border-accent/20 bg-accent/[0.05] px-4 py-3.5">
            <div className="text-sm text-fg">Tidying up right now…</div>
            <div className="mt-0.5 text-xs text-fg-subtle">Reading through recent memories. I’ll write a note when I’m done.</div>
            <div className="skeleton mt-3 h-1 rounded-full" />
          </div>
        ) : d.status === 'error' ? (
          <Alert className="mt-2" tone="danger" title="I couldn’t finish tidying up">
            {d.error || 'Something went wrong. I’ll try again tonight.'}
          </Alert>
        ) : (
          <div className="mt-2 rounded-2xl border border-border bg-surface">
            {chips.length > 0 && (
              <div className="flex flex-wrap gap-1.5 border-b border-border px-4 py-2.5">
                {chips.map((c) => (
                  <span key={c.key} className="inline-flex h-6 items-center rounded-full border border-border bg-elevated px-2.5 text-2xs font-medium text-fg-muted">
                    {c.text}
                  </span>
                ))}
              </div>
            )}
            {d.journal_md.trim() ? (
              <div className="relative px-5 py-4">
                <div className={cn('overflow-hidden', !open && long && 'max-h-[4.8rem]')}>
                  <Markdown className="!text-sm !leading-[1.75] text-fg-muted [&_strong]:text-fg">{d.journal_md}</Markdown>
                </div>
                {!open && long && <div aria-hidden className="pointer-events-none absolute inset-x-5 bottom-10 h-10 bg-gradient-to-t from-surface to-transparent" />}
                {long && (
                  <button type="button" onClick={() => setOpen((o) => !o)} className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-fg-subtle hover:text-fg">
                    {open ? 'Show less' : 'Read the whole note'}
                    <IconChevronDown size={13} className={cn('transition-transform', open && 'rotate-180')} />
                  </button>
                )}
              </div>
            ) : (
              <p className="px-5 py-4 text-sm text-fg-subtle">Nothing needed tidying.</p>
            )}
          </div>
        )}
      </div>
    </motion.li>
  )
}
