import {
  IconBrain,
  IconChartDots3,
  IconDots,
  IconFileImport,
  IconFilterOff,
  IconHourglassHigh,
  IconLayoutGrid,
  IconMessages,
  IconPlus,
  IconSearch,
  IconSparkles,
  IconTimeline,
  IconTrashX,
  IconUserCircle,
  IconUserHeart,
  IconX
} from '@tabler/icons-react'
import { useNavigate } from 'react-router'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'motion/react'
import { lazy, Suspense, useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  Alert,
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  EmptyState,
  IconButton,
  Input,
  PageHeader,
  SegmentedControl,
  Select,
  Skeleton,
  Spinner,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger
} from '@/components/ui'
import { ConversationsTab } from '@/features/memory/ConversationsTab'
import { AddMemoryDialog, ForgetSourceDialog, ImportDialog } from '@/features/memory/dialogs'
import { useDebounced, useMemoryParams, useNow } from '@/features/memory/hooks'
import { MemoryDetail } from '@/features/memory/MemoryDetail'
import { MemoryList, MemoryTimeline } from '@/features/memory/MemoryList'
import { EXPIRING_SOON_MS, countdown, expiresInMs, sourceMeta, TOPIC_META, TOPIC_ORDER, topicMeta, topicText, topicTint, type MemoryWithHistory } from '@/features/memory/meta'
import { ProfileTab } from '@/features/memory/ProfileTab'
import { useMemories, useMemoryGraph, useMemorySummaries, useMemoryTopics } from '@/hooks/memory'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import type { Memory } from '@/lib/types'
import { cn, relativeTime, truncate } from '@/lib/utils'
import { live } from '@/lib/ws'

const MemoryGraph = lazy(() => import('@/features/memory/MemoryGraph'))

export function MemoryPage() {
  const p = useMemoryParams()
  const navigate = useNavigate()
  const [search, setSearch] = useState(p.q)
  const q = useDebounced(search.trim(), 350)
  const [forgetInitial, setForgetInitial] = useState<string | undefined>()

  useEffect(() => {
    if (q !== p.q) p.set({ q })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q])

  const all = useMemories({ limit: 2000 })
  const searched = useQuery({
    queryKey: qk.memories.list({ q: p.q, limit: 200 }),
    queryFn: () => api.memories.list({ q: p.q, limit: 200 }),
    enabled: !!p.q,
    placeholderData: keepPreviousData
  })
  const searching = !!p.q
  const topics = useMemoryTopics()
  const summaries = useMemorySummaries(100)
  const graph = useMemoryGraph()
  const now = useNow()

  const memories = useMemo(() => all.data ?? [], [all.data])
  const base = searching ? (searched.data ?? []) : memories

  const filtered = useMemo(
    () =>
      base.filter(
        (m) =>
          (!p.topic || m.topics.includes(p.topic)) &&
          (!p.source || m.source === p.source) &&
          (p.type === 'all' || m.memory_type === p.type)
      ),
    [base, p.topic, p.source, p.type]
  )
  const filtering = !!(p.topic || p.source || p.type !== 'all' || searching)
  const matches = useMemo(() => (filtering ? new Set(filtered.map((m) => m.id)) : null), [filtering, filtered])

  const sources = useMemo(() => {
    const counts = new Map<string, number>()
    memories.forEach((m) => counts.set(m.source, (counts.get(m.source) ?? 0) + 1))
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([source, count]) => ({ source, count }))
  }, [memories])

  const topicCounts = useMemo(() => {
    const map = new Map<string, number>()
    if (topics.data) topics.data.forEach((t) => map.set(t.name, t.count))
    else memories.forEach((m) => m.topics.forEach((t) => map.set(t, (map.get(t) ?? 0) + 1)))
    return map
  }, [topics.data, memories])

  const selected = useMemo<MemoryWithHistory | null>(
    () => (p.selected ? ((memories.find((m) => m.id === p.selected) ?? searched.data?.find((m) => m.id === p.selected)) as MemoryWithHistory | undefined) ?? null : null),
    [p.selected, memories, searched.data]
  )

  // "Just learned" flash for live updates while the page is open.
  const [flash, setFlash] = useState<string | null>(null)
  useEffect(
    () =>
      live.onDomain('memory.updated', (e) => {
        const d = e.data as { action: string; content?: string | null; count?: number }
        const text =
          d.action === 'ADD' && d.content ? `Just learned: ${d.content}` : d.action === 'UPDATE' && d.content ? `Updated: ${d.content}` : d.count ? `${d.count} memories changed` : null
        if (text) setFlash(text)
      }),
    []
  )
  useEffect(() => {
    if (!flash) return
    const t = window.setTimeout(() => setFlash(null), 6000)
    return () => window.clearTimeout(t)
  }, [flash])

  const select = (id: number | null) => p.set({ m: id }, false)
  const clearFilters = () => {
    setSearch('')
    p.set({ topic: null, source: null, type: null, q: null })
  }

  return (
    <div className="h-full overflow-y-auto">
      <PageHeader
        icon={<IconBrain />}
        title="Memory"
        description="Everything Sentient knows about you. Correct it or forget it at any time."
        actions={
          <>
            <Button variant="ghost" leftIcon={<IconUserHeart size={15} />} onClick={() => navigate('/about')}>
              About you
            </Button>
            <Button variant="secondary" leftIcon={<IconFileImport size={15} />} onClick={() => p.set({ import: 1 }, false)}>
              Import
            </Button>
            <Button variant="primary" leftIcon={<IconPlus size={15} />} onClick={() => p.set({ add: 1 }, false)}>
              Add memory
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <IconButton label="More" variant="ghost" icon={<IconDots size={16} />} />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem icon={<IconTrashX />} danger disabled={!sources.length} onSelect={() => (setForgetInitial(undefined), p.set({ forget: 1 }, false))}>
                  Forget everything from a source…
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        }
      />

      <Tabs value={p.tab} onValueChange={(v) => p.set({ tab: v === 'memories' ? null : v })} className="px-8 pb-12">
        <TabsList className="mb-5">
          <TabsTrigger value="memories">
            <IconBrain size={15} /> Memories {all.data && <Count n={memories.length} />}
          </TabsTrigger>
          <TabsTrigger value="conversations">
            <IconMessages size={15} /> Conversations {summaries.data && <Count n={summaries.data.length} />}
          </TabsTrigger>
          <TabsTrigger value="profile">
            <IconUserCircle size={15} /> Profile
          </TabsTrigger>
        </TabsList>

        <TabsContent value="memories" className="space-y-4">
          {all.isError ? (
            <Alert tone="danger" title="Memory isn't available right now" action={<Button size="sm" onClick={() => void all.refetch()}>Retry</Button>}>
              {errorMessage(all.error)}
            </Alert>
          ) : all.isLoading ? (
            <LoadingState />
          ) : !memories.length ? (
            <EmptyState
              icon={<IconBrain />}
              title="Sentient doesn't remember anything yet"
              description="Tell it about yourself in chat, add a memory yourself, or import a document like your resume or notes."
              action={
                <>
                  <Button variant="secondary" leftIcon={<IconFileImport size={15} />} onClick={() => p.set({ import: 1 }, false)}>
                    Import a document
                  </Button>
                  <Button variant="primary" leftIcon={<IconPlus size={15} />} onClick={() => p.set({ add: 1 }, false)}>
                    Add memory
                  </Button>
                </>
              }
            />
          ) : (
            <>
              <Stats memories={memories} now={now} topicCounts={topicCounts} flash={flash} onTopic={(t) => p.set({ topic: t })} onExpiring={() => p.set({ type: 'short-term', view: 'list' })} onSelect={select} />

              {/* toolbar */}
              <div className="sticky top-0 z-20 -mx-8 space-y-3 border-b border-transparent bg-surface/90 px-8 pb-3 pt-2 backdrop-blur">
                <div className="flex flex-wrap items-center gap-2">
                  <Input
                    wrapperClassName="min-w-52 max-w-sm flex-1 basis-52"
                    leftIcon={searched.isFetching && searching ? <Spinner size={14} /> : <IconSearch />}
                    placeholder="Search by meaning…"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    onKeyDown={(e) => e.key === 'Escape' && setSearch('')}
                    rightSlot={
                      search && (
                        <button type="button" aria-label="Clear search" onClick={() => setSearch('')} className="flex size-6 items-center justify-center rounded text-fg-subtle hover:text-fg">
                          <IconX size={13} />
                        </button>
                      )
                    }
                  />
                  <Select
                    className="w-40"
                    aria-label="Source"
                    value={p.source || '__all'}
                    onValueChange={(v) => p.set({ source: v === '__all' ? null : v })}
                    options={[
                      { value: '__all', label: 'All sources' },
                      ...sources.map((s) => {
                        const meta = sourceMeta(s.source)
                        return { value: s.source, label: `${meta.label} (${s.count})`, icon: <meta.icon size={14} /> }
                      })
                    ]}
                  />
                  <Select
                    className="w-32"
                    aria-label="Memory type"
                    value={p.type}
                    onValueChange={(v) => p.set({ type: v === 'all' ? null : v })}
                    options={[
                      { value: 'all', label: 'Any type' },
                      { value: 'long-term', label: 'Long-term' },
                      { value: 'short-term', label: 'Short-term' }
                    ]}
                  />
                  <div className="flex-1" />
                  <SegmentedControl
                    aria-label="View"
                    value={p.view}
                    onChange={(v) => p.set({ view: v === 'graph' ? null : v })}
                    options={[
                      { value: 'graph', label: 'Graph', icon: <IconChartDots3 size={14} /> },
                      { value: 'list', label: 'List', icon: <IconLayoutGrid size={14} /> },
                      { value: 'timeline', label: 'Timeline', icon: <IconTimeline size={14} /> }
                    ]}
                  />
                </div>
                <div className="scrollbar-none flex items-center gap-1.5 overflow-x-auto pr-6 [mask-image:linear-gradient(to_right,black_calc(100%-40px),transparent)]">
                  <Chip active={!p.topic} onClick={() => p.set({ topic: null })}>
                    All <span className="text-fg-subtle">{memories.length}</span>
                  </Chip>
                  {TOPIC_ORDER.map((t) => {
                    const meta = TOPIC_META[t]
                    const n = topicCounts.get(t) ?? 0
                    const active = p.topic === t
                    return (
                      <Chip key={t} title={t} active={active} color={meta.color} disabled={!n} onClick={() => p.set({ topic: active ? null : t })}>
                        <span className="size-2 rounded-full" style={{ background: meta.color }} />
                        {meta.short}
                        <span className={active ? '' : 'text-fg-subtle'}>{n}</span>
                      </Chip>
                    )
                  })}
                </div>
              </div>

              {filtering && (
                <div className="flex items-center gap-2 text-xs text-fg-muted">
                  <span>
                    {searching && searched.isLoading ? 'Searching…' : `${filtered.length} of ${memories.length} memories`}
                    {searching && <> matching “{p.q}”</>}
                  </span>
                  <Button size="xs" variant="ghost" leftIcon={<IconFilterOff size={12} />} onClick={clearFilters}>
                    Clear filters
                  </Button>
                </div>
              )}

              {p.view === 'graph' ? (
                <div className="overflow-hidden rounded-2xl border border-border bg-sunken/40" style={{ height: 'max(460px, calc(100vh - 440px))' }}>
                  {graph.isError ? (
                    <div className="p-6">
                      <Alert tone="danger" title="Couldn't build the memory graph">
                        {errorMessage(graph.error)}
                      </Alert>
                    </div>
                  ) : !graph.data ? (
                    <GraphFallback />
                  ) : (
                    <Suspense fallback={<GraphFallback />}>
                      <MemoryGraph data={graph.data} matches={matches} selectedId={p.selected} onSelect={select} />
                    </Suspense>
                  )}
                </div>
              ) : searching && searched.isLoading ? (
                <LoadingState cardsOnly />
              ) : !filtered.length ? (
                <EmptyState
                  compact
                  icon={<IconSearch />}
                  title="No memories match"
                  description={searching ? 'Try different words. Search matches meaning, not just exact text.' : 'Try another topic, source or type.'}
                  action={
                    <Button size="sm" variant="secondary" onClick={clearFilters}>
                      Clear filters
                    </Button>
                  }
                />
              ) : p.view === 'list' ? (
                <MemoryList memories={filtered} selectedId={p.selected} onSelect={select} />
              ) : (
                <MemoryTimeline memories={filtered} selectedId={p.selected} onSelect={select} />
              )}
            </>
          )}
        </TabsContent>

        <TabsContent value="conversations">
          <ConversationsTab />
        </TabsContent>
        <TabsContent value="profile">
          <ProfileTab />
        </TabsContent>
      </Tabs>

      <MemoryDetail
        memory={selected}
        open={!!p.selected}
        onOpenChange={(o) => !o && select(null)}
        onForgetSource={(s) => {
          setForgetInitial(s)
          p.set({ m: null, forget: 1 }, false)
        }}
      />
      <AddMemoryDialog open={p.addOpen} onOpenChange={(o) => p.set({ add: o ? 1 : null })} onShow={(id) => p.set({ add: null, m: id })} />
      <ImportDialog open={p.importOpen} onOpenChange={(o) => p.set({ import: o ? 1 : null })} />
      <ForgetSourceDialog key={forgetInitial ?? 'none'} open={p.forgetOpen} onOpenChange={(o) => p.set({ forget: o ? 1 : null })} sources={sources} initial={forgetInitial} />
    </div>
  )
}

function Count({ n }: { n: number }) {
  return <span className="rounded-full bg-active px-1.5 text-2xs tabular-nums text-fg-muted">{n}</span>
}

function Chip({ active, color, disabled, onClick, children, title }: { active: boolean; color?: string; disabled?: boolean; onClick: () => void; children: ReactNode; title?: string }) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        'inline-flex h-7 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 text-xs font-medium transition-colors disabled:opacity-40',
        active ? 'text-fg' : 'border-border bg-surface text-fg-muted hover:border-border-strong hover:text-fg'
      )}
      style={active ? (color ? { background: topicTint(color, 16), borderColor: topicTint(color, 45), color: topicText(color) } : { background: 'var(--active)', borderColor: 'var(--border-strong)' }) : undefined}
    >
      {children}
    </button>
  )
}

function Stats({
  memories,
  now,
  topicCounts,
  flash,
  onTopic,
  onExpiring,
  onSelect
}: {
  memories: Memory[]
  now: number
  topicCounts: Map<string, number>
  flash: string | null
  onTopic: (t: string) => void
  onExpiring: () => void
  onSelect: (id: number) => void
}) {
  const shortTerm = memories.filter((m) => m.memory_type === 'short-term')
  const expiring = shortTerm
    .map((m) => ({ m, ms: expiresInMs(m, now) ?? Infinity }))
    .filter((x) => x.ms < EXPIRING_SOON_MS)
    .sort((a, b) => a.ms - b.ms)
  const latest = memories.reduce<Memory | null>((acc, m) => (!acc || m.created_at > acc.created_at ? m : acc), null)
  const totalTopics = [...topicCounts.values()].reduce((a, b) => a + b, 0) || 1

  return (
    <div className="grid grid-cols-2 gap-2.5 lg:grid-cols-4">
      <StatCard label="Memories" icon={<IconBrain size={14} />}>
        <div className="text-2xl font-semibold tabular-nums tracking-tight text-fg">{memories.length}</div>
        <div className="truncate text-xs text-fg-subtle" title={`${memories.length - shortTerm.length} long-term · ${shortTerm.length} short-term`}>
          {memories.length - shortTerm.length} long-term · {shortTerm.length} short
        </div>
      </StatCard>
      <StatCard label="By topic" icon={<IconChartDots3 size={14} />}>
        <div className="mt-1.5 flex h-2.5 overflow-hidden rounded-full bg-active">
          {TOPIC_ORDER.map((t) => {
            const n = topicCounts.get(t) ?? 0
            return n ? (
              <button key={t} type="button" title={`${t}: ${n}`} onClick={() => onTopic(t)} className="h-full transition-opacity hover:opacity-80" style={{ width: `${(n / totalTopics) * 100}%`, background: topicMeta(t).color }} />
            ) : null
          })}
        </div>
        <div className="mt-2 flex flex-wrap gap-x-2.5 gap-y-0.5 text-2xs text-fg-subtle">
          {TOPIC_ORDER.map((t) => [t, topicCounts.get(t) ?? 0] as const)
            .filter(([, n]) => n)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([t, n]) => (
              <span key={t}>
                {TOPIC_META[t].short} <span className="text-fg-muted">{n}</span>
              </span>
            ))}
        </div>
      </StatCard>
      <StatCard label="Expiring soon" icon={<IconHourglassHigh size={14} />} onClick={expiring.length ? onExpiring : undefined}>
        <div className={cn('text-2xl font-semibold tabular-nums tracking-tight', expiring.length ? 'text-warning' : 'text-fg')}>{expiring.length}</div>
        <div className="truncate text-xs text-fg-subtle">
          {expiring.length ? `next in ${countdown(expiring[0].ms)} · ${truncate(expiring[0].m.content.replace(/^Sarthak\s/, ''), 28)}` : 'Nothing expires in the next 2 days'}
        </div>
      </StatCard>
      <StatCard label="Last learned" icon={<IconSparkles size={14} />} onClick={latest ? () => onSelect(latest.id) : undefined}>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={flash ?? latest?.id} initial={{ opacity: 0, y: 3 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <div className={cn('text-2xl font-semibold tracking-tight', flash ? 'text-accent-text' : 'text-fg')}>{flash ? 'Just now' : latest ? relativeTime(latest.created_at) : '—'}</div>
            <div className="truncate text-xs text-fg-subtle">{flash ?? latest?.content}</div>
          </motion.div>
        </AnimatePresence>
      </StatCard>
    </div>
  )
}

function StatCard({ label, icon, children, onClick }: { label: string; icon: ReactNode; children: ReactNode; onClick?: () => void }) {
  const Comp = onClick ? 'button' : 'div'
  return (
    <Comp
      type={onClick ? 'button' : undefined}
      onClick={onClick}
      className={cn('min-w-0 rounded-xl border border-border bg-surface px-3.5 py-3 text-left', onClick && 'transition-colors hover:border-border-strong hover:bg-elevated')}
    >
      <div className="mb-1 flex items-center gap-1.5 text-xs text-fg-subtle">
        {icon}
        {label}
      </div>
      {children}
    </Comp>
  )
}

function GraphFallback() {
  return (
    <div className="flex size-full flex-col items-center justify-center gap-3 text-sm text-fg-subtle">
      <Spinner size={20} />
      Mapping how your memories connect…
    </div>
  )
}

function LoadingState({ cardsOnly }: { cardsOnly?: boolean }) {
  return (
    <div className="space-y-4">
      {!cardsOnly && (
        <div className="grid grid-cols-2 gap-2.5 lg:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-[84px] rounded-xl" />
          ))}
        </div>
      )}
      <div className="grid grid-cols-1 gap-2.5 md:grid-cols-2">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <Skeleton key={i} className="h-28 rounded-xl" />
        ))}
      </div>
    </div>
  )
}
