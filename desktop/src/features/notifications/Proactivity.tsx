import {
  IconAlertTriangle,
  IconBolt,
  IconBrain,
  IconMoon,
  IconRefresh,
  IconRestore,
  IconSettings,
  IconThumbDown,
  IconThumbUp
} from '@tabler/icons-react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Badge, Button, Card, EmptyState, IconButton, Skeleton, StatusDot, Tooltip } from '@/components/ui'
import { BrandIcon } from '@/features/integrations/BrandIcon'
import { useProactivityActions, useProactivityPreferences, useProactivityStatus } from '@/hooks/notifications'
import { errorMessage, isNotImplemented } from '@/lib/api'
import type { ProactivitySource } from '@/lib/types'
import { cn, humanize, relativeTime } from '@/lib/utils'
import { SOURCE_LABEL } from './utils'

function usePollNow() {
  const { pollNow } = useProactivityActions()
  const run = () =>
    pollNow.mutate(undefined, {
      onSuccess: (r) =>
        toast.success('Checked your apps', { description: r.events ? `${r.events} new item${r.events === 1 ? '' : 's'} to look at.` : 'Nothing new since the last check.' }),
      onError: (e) => toast.error("Couldn't check your apps", { description: errorMessage(e) })
    })
  return { run, pending: pollNow.isPending }
}

function sourceLine(s: ProactivitySource): string {
  if (!s.connected) return 'Not connected'
  return s.last_poll_at ? `Checked ${relativeTime(s.last_poll_at)}` : 'Not checked yet'
}

/** One-line status for the notifications panel. */
export function ProactivityStrip({ onNavigate }: { onNavigate?: () => void }) {
  const navigate = useNavigate()
  const { data, isError } = useProactivityStatus()
  const poll = usePollNow()
  if (isError || !data) return null
  const errors = data.sources.filter((s) => s.last_error)

  return (
    <div className="flex items-center gap-2.5 border-b border-border bg-sunken/40 px-4 py-2">
      <StatusDot tone={!data.enabled ? 'neutral' : errors.length ? 'warning' : 'success'} />
      <span className="text-xs font-medium text-fg">{data.enabled ? (data.quiet_now ? 'Quiet hours' : 'Proactivity on') : 'Proactivity off'}</span>
      <span className="flex items-center gap-1">
        {data.sources.map((s) => (
          <Tooltip key={s.source} content={`${SOURCE_LABEL[s.source] ?? humanize(s.source)}: ${s.last_error ?? sourceLine(s)}`}>
            <span className={cn('relative flex size-6 items-center justify-center rounded-md border border-border bg-elevated', !s.connected && 'opacity-45')}>
              <BrandIcon id={s.source} size={13} bare />
              {s.last_error && <span className="absolute -right-0.5 -top-0.5 size-2 rounded-full border border-surface bg-danger" />}
            </span>
          </Tooltip>
        ))}
      </span>
      <span className="min-w-0 flex-1 truncate text-2xs text-fg-subtle">{data.suggestions_today} today</span>
      <IconButton size="xs" label="Check now" icon={<IconRefresh size={13} />} loading={poll.pending} onClick={poll.run} />
      <IconButton
        size="xs"
        label="Proactivity settings"
        icon={<IconSettings size={13} />}
        onClick={() => {
          onNavigate?.()
          navigate('/settings/proactivity')
        }}
      />
    </div>
  )
}

export function ProactivityStatusCard() {
  const navigate = useNavigate()
  const { data, isLoading, isError, error } = useProactivityStatus()
  const poll = usePollNow()

  return (
    <Card>
      <div className="flex items-start gap-3 px-4 pt-4">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/12 text-accent-text">
          <IconBolt size={17} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-sm font-semibold text-fg">Proactivity</span>
            {data && <Badge tone={data.enabled ? 'success' : 'neutral'}>{data.enabled ? 'On' : 'Off'}</Badge>}
            {data?.quiet_now && (
              <Badge tone="info" icon={<IconMoon />}>
                Quiet hours
              </Badge>
            )}
          </div>
          <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">Sentient watches your connected apps and suggests what to do.</p>
        </div>
        <IconButton size="sm" label="Proactivity settings" icon={<IconSettings size={15} />} onClick={() => navigate('/settings/proactivity')} />
      </div>

      <div className="px-4 pb-4 pt-3">
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-10 rounded-lg" />
            <Skeleton className="h-10 rounded-lg" />
          </div>
        ) : isError || !data ? (
          <p className="text-xs text-fg-subtle">{isNotImplemented(error) ? 'Proactivity is not available in this engine.' : errorMessage(error)}</p>
        ) : (
          <>
            <ul className="space-y-1.5">
              {data.sources.map((s) => (
                <li key={s.source} className="rounded-lg border border-border bg-sunken/40 px-3 py-2">
                  <div className="flex items-center gap-2.5">
                    <BrandIcon id={s.source} size={26} />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium text-fg">{SOURCE_LABEL[s.source] ?? humanize(s.source)}</div>
                      <div className="text-2xs text-fg-subtle">{sourceLine(s)}</div>
                    </div>
                    {!s.connected ? (
                      <Button size="xs" variant="secondary" onClick={() => navigate(`/integrations?connect=${s.source}`)}>
                        Connect
                      </Button>
                    ) : (
                      <StatusDot tone={s.last_error ? 'danger' : 'success'} />
                    )}
                  </div>
                  {s.last_error && (
                    <div className="mt-1.5 flex items-start gap-1.5 text-2xs leading-relaxed text-danger">
                      <IconAlertTriangle size={12} className="mt-0.5 shrink-0" />
                      <span className="min-w-0">{s.last_error}</span>
                    </div>
                  )}
                </li>
              ))}
            </ul>
            <div className="mt-3 flex items-center gap-2">
              <span className="min-w-0 flex-1 text-xs text-fg-subtle">
                <span className="font-semibold text-fg">{data.suggestions_today}</span> suggestion{data.suggestions_today === 1 ? '' : 's'} today
                {data.heartbeat_minutes ? ` · check-in every ${data.heartbeat_minutes} min` : ''}
              </span>
              <Button size="sm" variant="secondary" leftIcon={<IconRefresh size={14} />} loading={poll.pending} disabled={!data.enabled} onClick={poll.run}>
                Check now
              </Button>
            </div>
          </>
        )}
      </div>
    </Card>
  )
}

export function LearnedPreferences({ id }: { id?: string }) {
  const { data, isLoading, isError, error } = useProactivityPreferences()
  const { resetPreference } = useProactivityActions()

  return (
    <Card id={id} className="scroll-mt-6">
      <div className="flex items-start gap-3 px-4 pt-4">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-fg-muted">
          <IconBrain size={17} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-fg">What Sentient has learned</div>
          <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">
            Each approve or dismiss tunes how sure Sentient must be before suggesting that kind of thing again.
          </p>
        </div>
      </div>
      <div className="px-4 pb-4 pt-3">
        {isLoading ? (
          <div className="space-y-2">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-12 rounded-lg" />
            ))}
          </div>
        ) : isError ? (
          <p className="text-xs text-fg-subtle">{errorMessage(error)}</p>
        ) : !data?.length ? (
          <EmptyState compact icon={<IconBrain />} title="Nothing learned yet" description="Approve or dismiss a few suggestions and Sentient will tune itself." />
        ) : (
          <ul className="divide-y divide-border">
            {data.map((p) => {
              const pos = Math.max(0, Math.min(1, (p.threshold - 0.4) / 0.55))
              return (
                <li key={p.suggestion_type} className="group flex items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium text-fg">{humanize(p.suggestion_type)}</span>
                      <span
                        className={cn(
                          'rounded-full px-1.5 text-[10px] font-semibold leading-4 tabular-nums',
                          p.score > 0 ? 'bg-success/12 text-success' : p.score < 0 ? 'bg-danger/12 text-danger' : 'bg-active text-fg-muted'
                        )}
                      >
                        {p.score > 0 ? `+${p.score}` : p.score}
                      </span>
                    </div>
                    <div className="mt-1 flex items-center gap-3 text-2xs text-fg-subtle">
                      <span className="flex items-center gap-1">
                        <IconThumbUp size={11} /> {p.approvals}
                      </span>
                      <span className="flex items-center gap-1">
                        <IconThumbDown size={11} /> {p.dismissals}
                      </span>
                      <Tooltip content={`Suggests this when at least ${Math.round(p.threshold * 100)}% sure (range 40–95%)`}>
                        <span className="flex items-center gap-1.5">
                          <span className="relative h-1 w-14 rounded-full bg-active">
                            <span className="absolute -top-[3px] size-2.5 -translate-x-1/2 rounded-full border-2 border-surface bg-accent" style={{ left: `${pos * 100}%` }} />
                          </span>
                          {Math.round(p.threshold * 100)}%+
                        </span>
                      </Tooltip>
                    </div>
                  </div>
                  <IconButton
                    size="xs"
                    label="Forget what was learned"
                    icon={<IconRestore size={13} />}
                    className="opacity-60 group-hover:opacity-100"
                    loading={resetPreference.isPending && resetPreference.variables === p.suggestion_type}
                    onClick={() =>
                      resetPreference.mutate(p.suggestion_type, {
                        onSuccess: () => toast(`Reset “${humanize(p.suggestion_type)}”`, { description: 'Sentient starts fresh for this kind of suggestion.' }),
                        onError: (e) => toast.error("Couldn't reset", { description: errorMessage(e) })
                      })
                    }
                  />
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </Card>
  )
}
