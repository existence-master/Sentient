import { IconUserBolt } from '@tabler/icons-react'
import { toast } from 'sonner'
import { Button, Popover, PopoverContent, PopoverTrigger } from '@/components/ui'
import { errorMessage } from '@/lib/api'
import type { Subagent } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'
import { Shimmer } from './cards/bits'
import { useCancelSubagent, useSessionSubagents } from './subagents'

const STATUS_TEXT: Record<Subagent['status'], string> = {
  running: 'Working',
  completed: 'Finished',
  error: 'Ran into a problem',
  cancelled: 'Stopped'
}

/** Small header pill listing helpers that work in the background for this chat. */
export function HelpersTray({ sessionId }: { sessionId: string | undefined }) {
  const { data } = useSessionSubagents(sessionId)
  const cancel = useCancelSubagent()
  const background = (data ?? []).filter((s) => s.background)
  const running = background.filter((s) => s.status === 'running')
  if (!background.length) return null
  const recent = [...running, ...background.filter((s) => s.status !== 'running')].slice(0, 6)

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className={cn(
            'no-drag flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-xs font-medium transition-colors',
            running.length ? 'border-accent/30 bg-accent/10 text-accent-text hover:bg-accent/15' : 'border-border text-fg-subtle hover:bg-hover hover:text-fg'
          )}
        >
          <span className="relative flex">
            <IconUserBolt size={14} />
            {running.length > 0 && <span className="absolute -right-0.5 -top-0.5 size-1.5 animate-pulse rounded-full bg-accent" />}
          </span>
          {running.length ? `${running.length} ${running.length === 1 ? 'helper' : 'helpers'} working` : 'Helpers'}
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 p-1.5">
        <div className="px-2 pb-1.5 pt-1">
          <div className="text-sm font-medium text-fg">Helpers in the background</div>
          <div className="text-xs text-fg-subtle">They keep working while you do other things. Results show up in this chat.</div>
        </div>
        <div className="space-y-0.5">
          {recent.map((s) => (
            <div key={s.subagent_id} className="flex items-start gap-2.5 rounded-lg px-2 py-2 hover:bg-hover">
              <span className={cn('mt-1.5 size-1.5 shrink-0 rounded-full', s.status === 'running' ? 'animate-pulse bg-accent' : s.status === 'completed' ? 'bg-success' : s.status === 'error' ? 'bg-danger' : 'bg-fg-faint')} />
              <div className="min-w-0 flex-1">
                <div className="line-clamp-2 text-sm text-fg">{s.goal}</div>
                <div className="text-xs text-fg-subtle">
                  {s.status === 'running' ? <Shimmer>{STATUS_TEXT.running}</Shimmer> : STATUS_TEXT[s.status]}
                  {s.tool_calls ? ` · ${s.tool_calls} steps` : ''} · {relativeTime(s.status === 'running' ? s.started_at : (s.finished_at ?? s.started_at))}
                </div>
              </div>
              {s.status === 'running' && (
                <Button
                  size="xs"
                  variant="ghost"
                  onClick={() => cancel.mutate(s.subagent_id, { onError: (e) => toast.error("Couldn't stop the helper", { description: errorMessage(e) }) })}
                >
                  Stop
                </Button>
              )}
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  )
}
