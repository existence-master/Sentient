/**
 * Activity (§21, #147): one timeline of what Sentient did and what it will do.
 *
 *   /activity?kinds=tasks,briefs&app=gmail&q=rent
 *
 * The main column is the past, newest first, a page at a time. The side column is what needs you now, what is
 * scheduled in the coming week and the routines that are paused. Every row says what happened and why, and opens
 * the chat, task, memory or setting it came from.
 */
import {
  IconActivity,
  IconAlertTriangle,
  IconArrowBackUp,
  IconBell,
  IconBrain,
  IconCalendarEvent,
  IconCheck,
  IconCoin,
  IconDatabase,
  IconListCheck,
  IconMessageCircle,
  IconPlayerPause,
  IconRadar,
  IconSearch,
  IconSend,
  IconShieldCheck,
  IconSparkles,
  IconSun,
  IconX,
  type Icon
} from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router'
import { toast } from 'sonner'
import { Badge, Button, EmptyState, Input, PageHeader, Select, Skeleton, type Tone } from '@/components/ui'
import { qk } from '@/hooks/queryKeys'
import { useTimeline, useTimelineApps, useUpcoming } from '@/hooks/timeline'
import { api, errorMessage } from '@/lib/api'
import type { TimelineFilters, TimelineItem, TimelineKind, TimelineRef, TimelineStatus } from '@/lib/types'
import { cn, formatTime, parseDate } from '@/lib/utils'

const KINDS: Array<{ id: TimelineKind; label: string; icon: Icon }> = [
  { id: 'chat', label: 'Chats', icon: IconMessageCircle },
  { id: 'tasks', label: 'Tasks', icon: IconListCheck },
  { id: 'briefs', label: 'Briefs', icon: IconSun },
  { id: 'monitors', label: 'Watches', icon: IconRadar },
  { id: 'approvals', label: 'Approvals', icon: IconShieldCheck },
  { id: 'channels', label: 'Messaging apps', icon: IconSend },
  { id: 'notifications', label: 'Notifications', icon: IconBell },
  { id: 'memory', label: 'Memory', icon: IconBrain },
  { id: 'skills', label: 'Skills', icon: IconSparkles },
  { id: 'backups', label: 'Backups', icon: IconDatabase },
  { id: 'cost', label: 'Spending', icon: IconCoin }
]
const KIND_IDS = new Set<string>(KINDS.map((k) => k.id))
const KIND_ICON = Object.fromEntries(KINDS.map((k) => [k.id, k.icon])) as Record<TimelineKind, Icon>

const STATUS: Partial<Record<TimelineStatus, { label: string; tone: Tone }>> = {
  failed: { label: "Didn't work", tone: 'danger' },
  waiting: { label: 'Waiting for you', tone: 'warning' },
  declined: { label: 'You said no', tone: 'neutral' },
  running: { label: 'Running', tone: 'info' },
  stopped: { label: 'Stopped', tone: 'neutral' },
  paused: { label: 'Paused', tone: 'neutral' }
}

const ALL_APPS = '__all'
const UPCOMING_SHOWN = 8

/** Where a row opens. */
export function refRoute(ref: TimelineRef | null): string | null {
  if (!ref) return null
  const id = ref.id ? encodeURIComponent(ref.id) : ''
  switch (ref.type) {
    case 'chat':
      return id ? `/chat/${id}` : '/chat'
    case 'task':
      return id ? `/tasks?task=${id}` : '/tasks'
    case 'memory':
      return ref.review ? '/memory?tab=review' : id ? `/memory?view=list&m=${id}` : '/memory'
    case 'dream':
      return '/about/dreams'
    case 'about':
      return '/about'
    case 'skill':
      return id ? `/skills?skill=${id}` : '/skills'
    case 'backups':
      return '/settings/backups'
    case 'usage':
      return '/settings/usage'
    case 'approvals':
      return '/settings/approvals'
    case 'notification':
      return '/notifications'
    default:
      return null
  }
}

function useFilters() {
  const [params, setParams] = useSearchParams()
  const kinds = (params.get('kinds') ?? '').split(',').filter((k): k is TimelineKind => KIND_IDS.has(k))
  const app = params.get('app') ?? ''
  const q = params.get('q') ?? ''
  const set = (patch: Record<string, string | null>) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        for (const [k, v] of Object.entries(patch)) {
          if (v) next.set(k, v)
          else next.delete(k)
        }
        return next
      },
      { replace: true }
    )
  return { kinds, app, q, set }
}

export function ActivityPage() {
  const f = useFilters()
  const [search, setSearch] = useState(f.q)
  useEffect(() => setSearch((s) => (s.trim() === f.q ? s : f.q)), [f.q])
  const { set } = f
  useEffect(() => {
    if (search.trim() === f.q) return
    const t = window.setTimeout(() => set({ q: search.trim() || null }), 300)
    return () => window.clearTimeout(t)
  }, [search, f.q, set])

  const kindKey = f.kinds.join(',')
  const filters: TimelineFilters = useMemo(
    () => ({ kinds: kindKey ? (kindKey.split(',') as TimelineKind[]) : [], app: f.app || undefined, q: f.q || undefined }),
    [kindKey, f.app, f.q]
  )
  const apps = useTimelineApps()
  const filtered = f.kinds.length > 0 || !!f.app || !!f.q

  const toggleKind = (id: TimelineKind) => {
    const next = f.kinds.includes(id) ? f.kinds.filter((k) => k !== id) : [...f.kinds, id]
    f.set({ kinds: next.length ? next.join(',') : null })
  }

  const appOptions = [
    { value: ALL_APPS, label: 'All apps' },
    ...(apps.data?.apps ?? []).map((a) => ({ value: a.id, label: a.label })),
    ...(f.app && !(apps.data?.apps ?? []).some((a) => a.id === f.app) ? [{ value: f.app, label: f.app }] : [])
  ]

  return (
    <div className="h-full overflow-y-auto">
      <PageHeader
        icon={<IconActivity />}
        title="Activity"
        description="What Sentient did, why, and what it will do next. Click anything to open it."
      >
        <div className="flex flex-wrap items-center gap-2">
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search activity"
            leftIcon={<IconSearch size={15} />}
            wrapperClassName="w-72"
            aria-label="Search activity"
          />
          <Select
            value={f.app || ALL_APPS}
            onValueChange={(v) => f.set({ app: v === ALL_APPS ? null : v })}
            options={appOptions}
            className="w-44"
            aria-label="Filter by app"
          />
          {filtered && (
            <Button variant="ghost" leftIcon={<IconX size={14} />} onClick={() => f.set({ kinds: null, app: null, q: null })}>
              Clear filters
            </Button>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by kind">
          {KINDS.map((k) => {
            const active = f.kinds.includes(k.id)
            return (
              <button
                key={k.id}
                type="button"
                aria-pressed={active}
                onClick={() => toggleKind(k.id)}
                className={cn(
                  'flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-xs transition-colors',
                  active ? 'border-accent/40 bg-accent/12 font-medium text-fg' : 'border-border text-fg-muted hover:border-border-strong hover:text-fg'
                )}
              >
                <k.icon size={13} className={active ? 'text-accent-text' : 'text-fg-subtle'} />
                {k.label}
              </button>
            )
          })}
        </div>
      </PageHeader>
      <div className="@container px-8 pb-12">
        <div className="grid items-start gap-6 @3xl:grid-cols-[minmax(0,1fr)_340px]">
          <aside className="space-y-4 @3xl:sticky @3xl:top-4 @3xl:order-2">
            <Upcoming filters={filters} filtered={filtered} />
          </aside>
          <section className="min-w-0 max-w-3xl @3xl:order-1">
            <Past filters={filters} filtered={filtered} />
          </section>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------- the past
function Past({ filters, filtered }: { filters: TimelineFilters; filtered: boolean }) {
  const { data, isLoading, isError, error, refetch, fetchNextPage, hasNextPage, isFetchingNextPage } = useTimeline(filters)
  const items = useMemo(() => data?.pages.flatMap((p) => p.items) ?? [], [data])
  const groups = useMemo(() => groupByDay(items), [items])
  // a page can come back empty while older items exist (a filter skipped many rows): keep reading
  const lastEmpty = !!data && data.pages[data.pages.length - 1].items.length === 0
  useEffect(() => {
    if (lastEmpty && hasNextPage && !isFetchingNextPage && (data?.pages.length ?? 0) < 20) void fetchNextPage()
  }, [lastEmpty, hasNextPage, isFetchingNextPage, fetchNextPage, data])

  if (isLoading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 6 }, (_, i) => (
          <Skeleton key={i} className="h-16 w-full rounded-xl" />
        ))}
      </div>
    )
  }
  if (isError) {
    return (
      <EmptyState
        icon={<IconAlertTriangle />}
        title="Couldn't load your activity"
        description={errorMessage(error)}
        action={<Button onClick={() => void refetch()}>Try again</Button>}
      />
    )
  }
  if (!items.length && !hasNextPage) {
    return (
      <EmptyState
        icon={<IconActivity />}
        title={filtered ? 'Nothing matches these filters' : 'Nothing here yet'}
        description={
          filtered
            ? 'Try another kind or app, or clear the filters.'
            : 'When Sentient uses an app in a chat, runs a task, saves a memory or backs up your data, it shows up here.'
        }
      />
    )
  }
  return (
    <div className="space-y-6">
      {groups.map((g) => (
        <div key={g.key}>
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-fg-subtle">{g.label}</h2>
          <div className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">
            {g.items.map((item) => (
              <Row key={item.id} item={item} />
            ))}
          </div>
        </div>
      ))}
      {hasNextPage && (
        <div className="flex justify-center">
          <Button variant="secondary" loading={isFetchingNextPage} onClick={() => void fetchNextPage()}>
            Show older activity
          </Button>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------- the future
function Upcoming({ filters, filtered }: { filters: TimelineFilters; filtered: boolean }) {
  const { data, isLoading, isError, error } = useUpcoming(filters)
  const [showAll, setShowAll] = useState(false)
  const [showPaused, setShowPaused] = useState(false)
  if (isLoading) return <Skeleton className="h-48 w-full rounded-xl" />
  if (isError) return <div className="rounded-xl border border-border bg-surface p-4 text-sm text-fg-muted">{errorMessage(error)}</div>
  const waiting = (data?.items ?? []).filter((i) => i.status === 'waiting')
  const later = (data?.items ?? []).filter((i) => i.status !== 'waiting')
  const shown = showAll ? later : later.slice(0, UPCOMING_SHOWN)
  const groups = groupByDay(shown)
  const paused = data?.paused ?? []
  return (
    <>
      {waiting.length > 0 && (
        <Panel title="Needs you now" count={waiting.length}>
          {waiting.map((item) => (
            <Row key={item.id} item={item} compact />
          ))}
        </Panel>
      )}
      <Panel title="Coming up" count={later.length} hint="In the next 7 days">
        {later.length === 0 ? (
          <div className="px-4 py-5 text-sm text-fg-subtle">{filtered ? 'Nothing coming up matches these filters.' : 'Nothing is scheduled for the coming week.'}</div>
        ) : (
          <>
            {groups.map((g) => (
              <div key={g.key}>
                <div className="bg-sunken/50 px-4 py-1.5 text-2xs font-semibold uppercase tracking-wide text-fg-subtle">{g.label}</div>
                {g.items.map((item) => (
                  <Row key={item.id} item={item} compact />
                ))}
              </div>
            ))}
            {later.length > UPCOMING_SHOWN && (
              <button type="button" onClick={() => setShowAll((v) => !v)} className="w-full px-4 py-2 text-left text-xs font-medium text-accent-text hover:bg-hover">
                {showAll ? 'Show less' : `Show all ${later.length}`}
              </button>
            )}
          </>
        )}
      </Panel>
      {paused.length > 0 && (
        <Panel title="Paused" count={paused.length} hint="These would run but are paused">
          {(showPaused ? paused : paused.slice(0, 4)).map((item) => (
            <Row key={item.id} item={item} compact />
          ))}
          {paused.length > 4 && (
            <button type="button" onClick={() => setShowPaused((v) => !v)} className="w-full px-4 py-2 text-left text-xs font-medium text-accent-text hover:bg-hover">
              {showPaused ? 'Show less' : `Show all ${paused.length}`}
            </button>
          )}
        </Panel>
      )}
    </>
  )
}

function Panel({ title, count, hint, children }: { title: string; count: number; hint?: string; children: React.ReactNode }) {
  return (
    <section className="overflow-hidden rounded-xl border border-border bg-surface">
      <header className="flex items-baseline gap-2 border-b border-border px-4 py-2.5">
        <h2 className="text-sm font-semibold text-fg">{title}</h2>
        <span className="text-xs tabular-nums text-fg-subtle">{count}</span>
        {hint && <span className="ml-auto text-2xs text-fg-faint">{hint}</span>}
      </header>
      <div className="divide-y divide-border">{children}</div>
    </section>
  )
}

// ---------------------------------------------------------------------------- one row
function Row({ item, compact }: { item: TimelineItem; compact?: boolean }) {
  const navigate = useNavigate()
  const to = refRoute(item.ref)
  const KindIcon = item.status === 'paused' ? IconPlayerPause : item.upcoming && item.status === 'scheduled' ? IconCalendarEvent : KIND_ICON[item.kind] ?? IconActivity
  const status = item.status ? STATUS[item.status] : undefined
  const tone = item.status === 'failed' ? 'text-danger bg-danger/10' : item.status === 'waiting' ? 'text-warning bg-warning/10' : 'text-fg-muted bg-active'
  const row = (
    <button
      type="button"
      disabled={!to}
      onClick={() => to && navigate(to)}
      className={cn('flex w-full items-start gap-3 text-left transition-colors enabled:hover:bg-hover', compact ? 'px-4 py-2.5' : 'px-4 py-3')}
    >
      <span className={cn('mt-0.5 flex shrink-0 items-center justify-center rounded-lg', compact ? 'size-6' : 'size-7', tone)}>
        <KindIcon size={compact ? 13 : 15} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-start gap-2">
          <span className={cn('min-w-0 flex-1 font-medium text-fg', compact ? 'text-xs leading-5' : 'text-sm')}>{item.title}</span>
          {item.at && <span className="shrink-0 pt-px text-2xs tabular-nums text-fg-subtle">{formatTime(item.at)}</span>}
        </span>
        {item.why && <span className={cn('mt-0.5 block text-fg-muted', compact ? 'text-2xs' : 'text-xs')}>{item.why}</span>}
        {item.detail && !compact && <span className="mt-1 line-clamp-2 block text-xs text-fg-subtle">{item.detail}</span>}
        {item.detail && compact && <span className="mt-0.5 line-clamp-1 block text-2xs text-fg-subtle">{item.detail}</span>}
        {(status || item.app_label) && (
          <span className="mt-1.5 flex flex-wrap items-center gap-1.5">
            {status && (
              <Badge size="xs" tone={status.tone}>
                {status.label}
              </Badge>
            )}
            {item.app_label && <Badge size="xs">{item.app_label}</Badge>}
          </span>
        )}
      </span>
    </button>
  )
  if (!item.undo || compact) return row
  return (
    <div className="flex items-start">
      <div className="min-w-0 flex-1">{row}</div>
      <RestoreButton item={item} />
    </div>
  )
}

/** Brings back what a chat moved to the trash, with the app's own restore (#142). */
function RestoreButton({ item }: { item: TimelineItem }) {
  const qc = useQueryClient()
  const [state, setState] = useState<'idle' | 'busy' | 'done'>('idle')
  return (
    <div className="shrink-0 py-3 pr-4">
      <Button
        size="sm"
        variant="secondary"
        leftIcon={state === 'done' ? <IconCheck size={14} /> : <IconArrowBackUp size={14} />}
        loading={state === 'busy'}
        disabled={state !== 'idle'}
        onClick={async () => {
          setState('busy')
          try {
            await api.timeline.restore(item.id)
            setState('done')
            toast.success('Restored', { description: 'It is back out of the trash.' })
            void qc.invalidateQueries({ queryKey: qk.timeline.all })
          } catch (err) {
            setState('idle')
            toast.error("Couldn't restore it", { description: errorMessage(err) })
          }
        }}
      >
        {state === 'done' ? 'Restored' : (item.undo?.label ?? 'Restore')}
      </Button>
    </div>
  )
}

// ---------------------------------------------------------------------------- days
function dayLabel(d: Date, now = new Date()): string {
  const start = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime()
  const diff = Math.round((start(d) - start(now)) / 86_400_000)
  if (diff === 0) return 'Today'
  if (diff === -1) return 'Yesterday'
  if (diff === 1) return 'Tomorrow'
  return d.toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric', year: d.getFullYear() !== now.getFullYear() ? 'numeric' : undefined })
}

function groupByDay(items: TimelineItem[]): Array<{ key: string; label: string; items: TimelineItem[] }> {
  const out: Array<{ key: string; label: string; items: TimelineItem[] }> = []
  for (const item of items) {
    const d = parseDate(item.at)
    const key = d ? `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}` : 'none'
    const last = out[out.length - 1]
    if (last && last.key === key) last.items.push(item)
    else out.push({ key, label: d ? dayLabel(d) : 'No time', items: [item] })
  }
  return out
}
