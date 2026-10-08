/**
 * "About you": how Sentient understands you (the user model), open questions and nightly dreams.
 *
 *   /about            your picture: summary, questions, insights by dimension
 *   /about/dreams     the dream journal
 *   ?dimension=<id>   filter to one part of your life
 */
import { IconBrain, IconMoonStars, IconUserHeart } from '@tabler/icons-react'
import { AnimatePresence } from 'motion/react'
import { useMemo, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router'
import { toast } from 'sonner'
import { Alert, Button, EmptyState, Skeleton, Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui'
import { DreamsView } from '@/features/usermodel/Dreams'
import { InsightCard, type InsightActions } from '@/features/usermodel/InsightCard'
import { DIMENSION_ORDER, dimensionMeta, tint } from '@/features/usermodel/meta'
import { AddInsightDialog, Portrait } from '@/features/usermodel/Portrait'
import { OpenQuestions } from '@/features/usermodel/Questions'
import { errorMessage, isNotImplemented } from '@/lib/api'
import { useUserModel, useUserModelActions } from '@/hooks/userModel'
import type { Insight, InsightDimension } from '@/lib/types'
import { cn } from '@/lib/utils'

export function AboutPage() {
  const { tab: rawTab } = useParams()
  const tab = rawTab === 'dreams' ? 'dreams' : 'picture'
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const model = useUserModel()
  const actions = useUserModelActions()
  const [adding, setAdding] = useState(false)
  const [showRetired, setShowRetired] = useState(false)
  const dimension = params.get('dimension')

  const insights = useMemo(() => model.data?.insights ?? [], [model.data])
  const live = insights.filter((i) => i.status !== 'retired')
  const retired = insights.filter((i) => i.status === 'retired')

  const groups = useMemo(() => {
    const by = new Map<string, Insight[]>()
    for (const i of live) by.set(i.dimension, [...(by.get(i.dimension) ?? []), i])
    const order = [...DIMENSION_ORDER, ...[...by.keys()].filter((k) => !DIMENSION_ORDER.includes(k as InsightDimension))]
    return order
      .filter((d) => by.has(d))
      .map((d) => ({
        dimension: d,
        items: (by.get(d) ?? []).sort((a, b) => Number(b.status === 'confirmed' || b.source === 'user') - Number(a.status === 'confirmed' || a.source === 'user') || b.confidence - a.confidence)
      }))
  }, [live])
  const shown = dimension ? groups.filter((g) => g.dimension === dimension) : groups

  const setStatus = (i: Insight, status: Insight['status'], success: string, undo?: Insight['status']) =>
    actions.patch.mutate(
      { id: i.id, patch: { status } },
      {
        onSuccess: () =>
          toast.success(success, undo ? { action: { label: 'Undo', onClick: () => actions.patch.mutate({ id: i.id, patch: { status: undo } }) } } : undefined),
        onError: (e) => toast.error('Couldn’t save that', { description: errorMessage(e) })
      }
    )

  const insightActions: InsightActions = {
    onConfirm: (i) => setStatus(i, 'confirmed', 'Thanks for confirming'),
    onRetire: (i) => setStatus(i, 'retired', 'Got it, I’ll stop assuming that', i.status),
    onRestore: (i) => setStatus(i, 'active', 'Restored'),
    onEdit: (i, statement) =>
      actions.patch.mutateAsync({ id: i.id, patch: { statement, status: 'confirmed' } }).then(
        () => toast.success('Updated in your words'),
        (e) => toast.error('Couldn’t save that', { description: errorMessage(e) })
      ),
    onDelete: (i) =>
      actions.remove.mutate(i.id, {
        onSuccess: () => toast.success('Forgotten'),
        onError: (e) => toast.error('Couldn’t forget that', { description: errorMessage(e) })
      })
  }

  const setDimension = (d: string | null) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (d) next.set('dimension', d)
        else next.delete('dimension')
        return next
      },
      { replace: true }
    )

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1080px] px-8 pb-16 pt-7">
        <Portrait model={model.data} loading={model.isLoading} onAdd={() => setAdding(true)} />

        <Tabs value={tab} onValueChange={(v) => navigate(v === 'dreams' ? '/about/dreams' : '/about', { replace: true })} className="mt-6">
          <TabsList className="mb-6">
            <TabsTrigger value="picture">
              <IconUserHeart size={15} /> Your picture
            </TabsTrigger>
            <TabsTrigger value="dreams">
              <IconMoonStars size={15} /> Dreams
            </TabsTrigger>
          </TabsList>

          <TabsContent value="picture" className="space-y-9">
            {model.isLoading ? (
              <div className="grid gap-3 md:grid-cols-2">
                {[0, 1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-24 rounded-xl" />
                ))}
              </div>
            ) : model.isError ? (
              isNotImplemented(model.error) ? (
                <EmptyState
                  icon={<IconUserHeart />}
                  title="This part of Sentient is on its way"
                  description="Soon you’ll see how Sentient understands you here, with the reasons behind each thing it notices. Your memories are safe in the meantime."
                  action={
                    <Button variant="secondary" leftIcon={<IconBrain size={15} />} onClick={() => navigate('/memory')}>
                      Open Memory
                    </Button>
                  }
                />
              ) : (
                <Alert tone="danger" title="Couldn’t load your picture" action={<Button size="sm" onClick={() => void model.refetch()}>Retry</Button>}>
                  {errorMessage(model.error)}
                </Alert>
              )
            ) : !live.length && !model.data?.questions.length ? (
              <EmptyState
                icon={<IconUserHeart />}
                title="Nothing here yet"
                description="Chat with Sentient for a while and it will start noticing what matters to you. Or tell it a few things yourself."
                action={
                  <Button variant="primary" onClick={() => setAdding(true)}>
                    Add something about you
                  </Button>
                }
              />
            ) : (
              <>
                <OpenQuestions questions={model.data?.questions ?? []} insights={insights} />

                <section className="space-y-4" aria-label="What Sentient has noticed">
                  <div className="flex flex-wrap items-end justify-between gap-3">
                    <div>
                      <h2 className="text-md font-semibold text-fg">What I’ve noticed</h2>
                      <p className="mt-0.5 text-sm text-fg-subtle">Hover over anything that isn’t you and choose “This is wrong”. I’ll stop assuming it.</p>
                    </div>
                  </div>
                  <nav
                    aria-label="Parts of your life"
                    className="scrollbar-none -mx-1 flex gap-1.5 overflow-x-auto px-1 pb-1 pr-8 [mask-image:linear-gradient(to_right,black_calc(100%-48px),transparent)]"
                  >
                    <Chip active={!dimension} onClick={() => setDimension(null)}>
                      Everything <span className="text-fg-subtle">{live.length}</span>
                    </Chip>
                    {groups.map((g) => {
                      const m = dimensionMeta(g.dimension)
                      const active = dimension === g.dimension
                      return (
                        <Chip key={g.dimension} active={active} color={m.color} onClick={() => setDimension(active ? null : g.dimension)}>
                          <m.icon size={13} style={{ color: m.color }} />
                          {m.label}
                          <span className={active ? '' : 'text-fg-subtle'}>{g.items.length}</span>
                        </Chip>
                      )
                    })}
                  </nav>

                  <div className="space-y-8 pt-2">
                    {shown.map((g) => {
                      const m = dimensionMeta(g.dimension)
                      return (
                        <section key={g.dimension} aria-label={m.label}>
                          <div className="mb-2.5 flex items-center gap-2.5">
                            <span className="flex size-7 items-center justify-center rounded-lg" style={{ background: tint(m.color, 15), color: m.color }}>
                              <m.icon size={15} />
                            </span>
                            <h3 className="text-sm font-semibold text-fg">{m.label}</h3>
                            <span className="text-xs text-fg-subtle">{m.blurb}</span>
                          </div>
                          <ul className="grid gap-2.5 md:grid-cols-2">
                            <AnimatePresence initial={false}>
                              {g.items.map((i) => (
                                <InsightCard key={i.id} insight={i} actions={insightActions} />
                              ))}
                            </AnimatePresence>
                          </ul>
                        </section>
                      )
                    })}
                  </div>
                </section>

                {retired.length > 0 && (
                  <section className="border-t border-border pt-6">
                    <button type="button" onClick={() => setShowRetired((s) => !s)} className="text-sm font-medium text-fg-muted hover:text-fg">
                      {showRetired ? 'Hide' : 'Show'} things you told me were wrong ({retired.length})
                    </button>
                    {showRetired && (
                      <ul className="mt-3 grid gap-2.5 md:grid-cols-2">
                        {retired.map((i) => (
                          <InsightCard key={i.id} insight={i} actions={insightActions} retired />
                        ))}
                      </ul>
                    )}
                  </section>
                )}
              </>
            )}
          </TabsContent>

          <TabsContent value="dreams">
            <DreamsView />
          </TabsContent>
        </Tabs>
      </div>
      <AddInsightDialog open={adding} onOpenChange={setAdding} initialDimension={(dimension as InsightDimension) || undefined} />
    </div>
  )
}

function Chip({ active, color, onClick, children }: { active: boolean; color?: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        'inline-flex h-7 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 text-xs font-medium transition-colors',
        active ? 'text-fg' : 'border-border bg-surface text-fg-muted hover:border-border-strong hover:text-fg'
      )}
      style={active ? { background: color ? tint(color, 14) : 'var(--active)', borderColor: color ? tint(color, 40) : 'var(--border-strong)' } : undefined}
    >
      {children}
    </button>
  )
}
