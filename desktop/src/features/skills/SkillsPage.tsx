import { IconArchive, IconArrowRight, IconEyeCheck, IconPlus, IconSearch, IconSparkles, IconTimeline } from '@tabler/icons-react'
import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router'
import { Alert, Button, EmptyState, Input, PageHeader, SegmentedControl, Skeleton, Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui'
import { CreateSkillDialog, ReviewNowButton } from '@/features/skills/actions'
import { EvolutionLog } from '@/features/skills/EvolutionLog'
import { PendingReview } from '@/features/skills/PendingReview'
import { SkillCard } from '@/features/skills/SkillCard'
import { SkillDetailSheet } from '@/features/skills/SkillDetail'
import { useSkills } from '@/hooks/skills'
import { errorMessage } from '@/lib/api'
import type { Skill } from '@/lib/types'
import { cn } from '@/lib/utils'

type Tab = 'active' | 'pending' | 'archived' | 'log'

export function SkillsPage() {
  const [params, setParams] = useSearchParams()
  const skills = useSkills()
  const [query, setQuery] = useState('')
  const [author, setAuthor] = useState<'all' | 'user' | 'assistant' | 'community'>('all')

  const tab = (['active', 'pending', 'archived', 'log'].includes(params.get('tab') ?? '') ? params.get('tab') : 'active') as Tab
  const open = params.get('skill')
  const editing = params.get('edit') === '1'
  const focus = params.get('focus')
  const set = (patch: Record<string, string | null>, replace = false) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        Object.entries(patch).forEach(([k, v]) => (v === null ? next.delete(k) : next.set(k, v)))
        return next
      },
      { replace }
    )

  const data = skills.data
  const pendingNames = useMemo(() => new Set(data?.pending.map((s) => s.name)), [data])
  const staleCount = data?.active.filter((s) => s.state === 'stale').length ?? 0

  const filter = (list: Skill[]) => {
    const q = query.trim().toLowerCase()
    return list
      .filter((s) => author === 'all' || s.author === author)
      .filter((s) => !q || s.name.includes(q) || s.description.toLowerCase().includes(q) || s.tags.some((t) => t.toLowerCase().includes(q)))
  }
  const active = useMemo(
    () =>
      filter(data?.active ?? []).sort((a, b) => {
        if ((a.state === 'stale') !== (b.state === 'stale')) return a.state === 'stale' ? 1 : -1
        return (b.last_used_at ?? '').localeCompare(a.last_used_at ?? '') || a.name.localeCompare(b.name)
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [data, query, author]
  )
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const archived = useMemo(() => filter(data?.archived ?? []), [data, query, author])

  const openSkill = (name: string, pending?: boolean) => (pending ? set({ tab: 'pending', focus: name, skill: null }) : set({ skill: name, edit: null }))

  return (
    <div className="h-full overflow-y-auto">
      <PageHeader
        icon={<IconSparkles />}
        title="Skills"
        description="Step-by-step procedures Sentient follows for your recurring work. It drafts new ones as it learns; you decide what it adopts."
        actions={
          <>
            <ReviewNowButton onProposed={(names) => set({ tab: 'pending', focus: names[0] ?? null })} />
            <Button variant="primary" leftIcon={<IconPlus size={15} />} onClick={() => set({ create: '1' })}>
              Create skill
            </Button>
          </>
        }
      />

      <Tabs value={tab} onValueChange={(v) => set({ tab: v === 'active' ? null : v, focus: null }, true)} className="px-8 pb-12">
        <TabsList className="mb-5">
          <TabsTrigger value="active">
            Active {data && <Count n={data.active.length} />}
          </TabsTrigger>
          <TabsTrigger value="pending">
            <IconEyeCheck size={15} /> Pending review {data && <Count n={data.pending.length} accent={data.pending.length > 0} />}
          </TabsTrigger>
          <TabsTrigger value="archived">
            <IconArchive size={15} /> Archived {data && <Count n={data.archived.length} />}
          </TabsTrigger>
          <TabsTrigger value="log">
            <IconTimeline size={15} /> Evolution log
          </TabsTrigger>
        </TabsList>

        {tab !== 'log' && skills.isError && (
          <Alert tone="danger" title="Skills aren't available right now" action={<Button size="sm" onClick={() => void skills.refetch()}>Retry</Button>}>
            {errorMessage(skills.error)}
          </Alert>
        )}

        <TabsContent value="active" className="space-y-4">
          {skills.isLoading ? (
            <CardSkeletons />
          ) : data ? (
            <>
              {data.pending.length > 0 && (
                <button
                  type="button"
                  onClick={() => set({ tab: 'pending' })}
                  className="group flex w-full items-center gap-3 rounded-xl border border-accent/30 bg-accent/8 px-4 py-3 text-left transition-colors hover:bg-accent/12"
                >
                  <span className="flex size-8 items-center justify-center rounded-lg bg-accent/15 text-accent-text">
                    <IconSparkles size={17} />
                  </span>
                  <span className="flex-1 text-sm text-fg">
                    <b className="font-semibold">
                      {data.pending.length} {data.pending.length === 1 ? 'skill is' : 'skills are'} waiting for your review.
                    </b>{' '}
                    <span className="text-fg-muted">Sentient won’t use them until you approve.</span>
                  </span>
                  <IconArrowRight size={16} className="text-accent-text transition-transform group-hover:translate-x-0.5" />
                </button>
              )}
              {data.active.length > 0 && (
                <Toolbar query={query} setQuery={setQuery} author={author} setAuthor={setAuthor} note={staleCount ? `${staleCount} stale` : undefined} />
              )}
              {!data.active.length ? (
                <EmptyState
                  icon={<IconSparkles />}
                  title="No skills yet"
                  description="Create one for a routine you repeat, or keep working: Sentient proposes skills after multi-step chats and tasks."
                  action={
                    <Button variant="primary" leftIcon={<IconPlus size={15} />} onClick={() => set({ create: '1' })}>
                      Create skill
                    </Button>
                  }
                />
              ) : !active.length ? (
                <EmptyState compact icon={<IconSearch />} title="No skills match" description="Try a different search or author." />
              ) : (
                <Grid>
                  {active.map((s) => (
                    <SkillCard key={s.name} skill={s} hasUpdate={pendingNames.has(s.name)} onOpen={() => openSkill(s.name)} />
                  ))}
                </Grid>
              )}
            </>
          ) : null}
        </TabsContent>

        <TabsContent value="pending">{skills.isLoading ? <Skeleton className="h-96 rounded-2xl" /> : data ? <PendingReview list={data} focus={focus} onOpenSkill={openSkill} /> : null}</TabsContent>

        <TabsContent value="archived" className="space-y-4">
          {skills.isLoading ? (
            <CardSkeletons />
          ) : data && !data.archived.length ? (
            <EmptyState icon={<IconArchive />} title="Nothing archived" description="Skills you archive, or that the curator retires after going unused, are kept here and can be restored." />
          ) : data ? (
            <>
              <p className="text-sm text-fg-muted">Archived skills are never used. Open one to restore it.</p>
              <Grid>
                {archived.map((s) => (
                  <SkillCard key={s.name} skill={s} onOpen={() => openSkill(s.name)} />
                ))}
              </Grid>
            </>
          ) : null}
        </TabsContent>

        <TabsContent value="log">
          <EvolutionLog onOpenSkill={openSkill} />
        </TabsContent>
      </Tabs>

      <SkillDetailSheet
        name={open}
        list={data}
        editing={editing}
        onEditingChange={(e) => set({ edit: e ? '1' : null }, true)}
        onOpenChange={(o) => !o && set({ skill: null, edit: null })}
        onReview={(name) => set({ skill: null, edit: null, tab: 'pending', focus: name })}
      />
      <CreateSkillDialog open={params.get('create') === '1'} onOpenChange={(o) => set({ create: o ? '1' : null }, true)} onCreated={(name) => set({ create: null, skill: name, tab: null })} />
    </div>
  )
}

function Count({ n, accent }: { n: number; accent?: boolean }) {
  return <span className={cn('rounded-full px-1.5 text-2xs tabular-nums', accent ? 'bg-accent text-accent-fg' : 'bg-active text-fg-muted')}>{n}</span>
}

function Grid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-1 gap-3 md:grid-cols-2 2xl:grid-cols-3">{children}</div>
}

function CardSkeletons() {
  return (
    <Grid>
      {[0, 1, 2, 3].map((i) => (
        <Skeleton key={i} className="h-44 rounded-xl" />
      ))}
    </Grid>
  )
}

function Toolbar({
  query,
  setQuery,
  author,
  setAuthor,
  note
}: {
  query: string
  setQuery: (q: string) => void
  author: 'all' | 'user' | 'assistant' | 'community'
  setAuthor: (a: 'all' | 'user' | 'assistant' | 'community') => void
  note?: string
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Input wrapperClassName="max-w-xs" leftIcon={<IconSearch />} placeholder="Search skills" value={query} onChange={(e) => setQuery(e.target.value)} />
      <SegmentedControl
        aria-label="Author"
        value={author}
        onChange={setAuthor}
        options={[
          { value: 'all', label: 'All' },
          { value: 'user', label: 'By you' },
          { value: 'assistant', label: 'By Sentient' },
          { value: 'community', label: 'Community' }
        ]}
      />
      <div className="flex-1" />
      {note && <span className="text-xs text-fg-subtle">{note}</span>}
    </div>
  )
}
