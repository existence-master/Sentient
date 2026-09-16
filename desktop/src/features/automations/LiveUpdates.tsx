/**
 * "Live updates" (docs/API.md §5, §16): whether Gmail and Google Calendar change feeds and the IMAP push watcher are
 * keeping Sentient current. Hidden when nothing feed-capable is connected or the engine has no feeds endpoint.
 */
import { IconBroadcast, IconRefresh } from '@tabler/icons-react'
import { toast } from 'sonner'
import { Button, StatusDot, Tooltip } from '@/components/ui'
import { BrandIcon } from '@/features/integrations/BrandIcon'
import { errorMessage } from '@/lib/api'
import { useFeedSync, useFeeds } from '@/lib/leap/hooks-b'
import type { FeedStatus } from '@/lib/leap/types-b'
import { cn, relativeTime } from '@/lib/utils'

function describe(f: FeedStatus): { tone: 'success' | 'danger' | 'warning' | 'neutral'; text: string; hint: string } {
  const push = f.kind === 'imap_idle'
  const when = f.last_success_at ? `Last update ${relativeTime(f.last_success_at)}.` : ''
  switch (f.status) {
    case 'ok':
      return { tone: 'success', text: push ? 'Instant' : 'Live', hint: `${push ? 'New mail arrives the moment it lands.' : 'Checked every minute, without using AI.'} ${when}`.trim() }
    case 'starting':
      return { tone: 'warning', text: 'Starting', hint: 'Getting ready. The first check only remembers where things stand.' }
    case 'error':
      return { tone: 'danger', text: 'Having trouble', hint: f.last_error ?? 'The last check didn’t work. Trying again soon.' }
    case 'off':
      return { tone: 'neutral', text: 'Off', hint: 'Live updates are turned off, so Sentient checks now and then instead.' }
    default:
      return { tone: 'neutral', text: 'Not connected', hint: 'Connect it to get live updates.' }
  }
}

export function LiveUpdatesRow() {
  const feeds = useFeeds()
  const sync = useFeedSync()
  const rows = (feeds.data ?? []).filter((f) => f.connected)
  if (!rows.length) return null
  const failing = rows.filter((f) => f.status === 'error')

  return (
    <section aria-label="Live updates" className="flex flex-wrap items-center gap-x-5 gap-y-3 rounded-xl border border-border bg-surface px-4 py-3">
      <div className="flex min-w-0 flex-1 basis-64 items-center gap-3">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-success/10 text-success">
          <IconBroadcast size={17} />
        </span>
        <div className="min-w-0">
          <div className="text-sm font-medium text-fg">Live updates</div>
          <div className="text-xs text-fg-subtle">New emails and events reach Sentient within moments, without using AI.</div>
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-end gap-2">
        {rows.map((f) => {
          const d = describe(f)
          return (
            <Tooltip key={f.source} content={f.note ? `${d.hint} ${f.note}` : d.hint}>
              <span className={cn('inline-flex h-8 items-center gap-2 rounded-full border bg-elevated/60 pl-1.5 pr-3 text-xs', f.status === 'error' ? 'border-danger/30' : 'border-border')}>
                <BrandIcon id={f.source} size={22} />
                <span className="font-medium text-fg">{f.display_name}</span>
                <StatusDot tone={d.tone} pulse={d.tone === 'success'} />
                <span className={f.status === 'error' ? 'text-danger' : 'text-fg-muted'}>{d.text}</span>
              </span>
            </Tooltip>
          )
        })}
        {failing.length > 0 && (
          <Button
            size="xs"
            variant="ghost"
            leftIcon={<IconRefresh size={13} />}
            loading={sync.isPending}
            onClick={() =>
              failing.forEach((f) =>
                sync.mutate(f.source, {
                  onSuccess: (r) => (r.ok ? toast.success(`${f.display_name} is back`) : toast.error(`${f.display_name} still can’t connect`, { description: r.error })),
                  onError: (e) => toast.error('Couldn’t check now', { description: errorMessage(e) })
                })
              )
            }
          >
            Try now
          </Button>
        )}
      </div>
    </section>
  )
}
