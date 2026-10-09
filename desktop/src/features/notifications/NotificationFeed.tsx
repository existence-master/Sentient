import { IconBell, IconBolt, IconListCheck, IconShieldQuestion, IconSparkles } from '@tabler/icons-react'
import { AnimatePresence } from 'motion/react'
import { useMemo, useState, type ReactNode } from 'react'
import { Button, EmptyState, Skeleton } from '@/components/ui'
import { useNotifications } from '@/hooks/notifications'
import { errorMessage } from '@/lib/api'
import { cn } from '@/lib/utils'
import { BriefCard } from './BriefCard'
import { NotificationItem } from './NotificationItem'
import { FILTERS, groupByDay, matchesFilter, needsAction, type FeedFilter } from './utils'

const EMPTY: Record<FeedFilter, { icon: ReactNode; title: string; description: string }> = {
  all: {
    icon: <IconBell />,
    title: "You're all caught up",
    description: 'Suggestions from your apps, task results, approvals and new skills will show up here.'
  },
  suggestions: {
    icon: <IconBolt />,
    title: 'No suggestions right now',
    description: 'When something in Gmail or Calendar needs a hand, Sentient suggests it here.'
  },
  tasks: { icon: <IconListCheck />, title: 'No task updates', description: 'Results and failures of your tasks appear here.' },
  approvals: { icon: <IconShieldQuestion />, title: 'Nothing waiting for you', description: 'Plans and actions that need your OK appear here.' },
  skills: { icon: <IconSparkles />, title: 'No new skills', description: 'Skills Sentient learns from your chats wait here for review.' }
}

export function NotificationFeed({
  variant,
  onNavigate,
  initialFilter = 'all'
}: {
  variant: 'panel' | 'page'
  onNavigate?: () => void
  initialFilter?: FeedFilter
}) {
  const { data, isLoading, isError, error, refetch } = useNotifications()
  const [filter, setFilter] = useState<FeedFilter>(initialFilter)
  // the Daily Brief has its own card at the top instead of a row in the feed
  const list = useMemo(() => (data?.notifications ?? []).filter((n) => n.kind !== 'brief'), [data])

  const counts = useMemo(() => {
    const c = {} as Record<FeedFilter, number>
    for (const f of FILTERS) c[f.id] = list.filter((n) => matchesFilter(n, f.id) && (f.id === 'all' ? !n.read : needsAction(n) || !n.read)).length
    return c
  }, [list])

  const groups = useMemo(() => groupByDay(list.filter((n) => matchesFilter(n, filter))), [list, filter])
  const panel = variant === 'panel'

  return (
    <div>
      <div className={cn('flex flex-wrap', panel ? 'sticky top-0 z-10 gap-1 border-b border-border bg-surface/95 px-3 py-2.5 backdrop-blur [&>button]:px-2' : 'mb-4 gap-1.5')}>
        {FILTERS.map((f) => {
          const active = filter === f.id
          return (
            <button
              key={f.id}
              type="button"
              onClick={() => setFilter(f.id)}
              className={cn(
                'flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-xs transition-colors',
                active ? 'border-accent/40 bg-accent/12 font-medium text-fg' : 'border-border text-fg-muted hover:border-border-strong hover:text-fg'
              )}
            >
              {f.label}
              {counts[f.id] > 0 && (
                <span className={cn('min-w-4 rounded-full px-1 text-center text-[10px] font-semibold leading-4 tabular-nums', active ? 'bg-accent text-accent-fg' : 'bg-active text-fg-muted')}>
                  {counts[f.id]}
                </span>
              )}
            </button>
          )
        })}
      </div>

      <div className={cn(panel && 'px-3 pb-6 pt-1')}>
        {filter === 'all' && (
          <div className={cn(panel && 'pt-2')}>
            <BriefCard compact={panel} onNavigate={onNavigate} />
          </div>
        )}
        {isLoading ? (
          <div className="space-y-2 pt-3">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-20 rounded-xl" />
            ))}
          </div>
        ) : isError ? (
          <EmptyState
            compact
            icon={<IconBell />}
            title="Couldn't load notifications"
            description={errorMessage(error)}
            action={
              <Button size="sm" onClick={() => void refetch()}>
                Try again
              </Button>
            }
          />
        ) : !groups.length ? (
          <EmptyState compact={panel} {...EMPTY[filter]} />
        ) : (
          groups.map((g) => (
            <section key={g.label} className="pt-3">
              <h3 className={cn('pb-2 text-2xs font-semibold uppercase tracking-wider text-fg-subtle', panel ? 'px-1' : 'px-0.5')}>{g.label}</h3>
              <ul className="space-y-2">
                <AnimatePresence initial={false}>
                  {g.items.map((n) => (
                    <NotificationItem key={n.id} n={n} dense={panel} onNavigate={onNavigate} />
                  ))}
                </AnimatePresence>
              </ul>
            </section>
          ))
        )}
      </div>
    </div>
  )
}
