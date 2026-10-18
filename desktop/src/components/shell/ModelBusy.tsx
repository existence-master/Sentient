/**
 * #149 One local model job at a time: a small title bar indicator, shown while the local model works on something
 * other than your chat, while something waits for it, or while Sentient's own upkeep waits on battery (#296).
 */
import { Tooltip, StatusDot } from '@/components/ui'
import { useModelBusy } from '@/hooks/models'
import type { ModelBusy, ModelJobKind } from '@/lib/types'
import { modelShortName } from '@/lib/models'
import { useConnection } from '@/stores/connection'

const JOB: Record<ModelJobKind, { short: string; doing: string }> = {
  chat: { short: 'your chat', doing: 'answering you' },
  interactive: { short: 'a request', doing: 'working on something you asked for' },
  task: { short: 'a task', doing: 'working on a task' },
  suggestions: { short: 'suggestions', doing: 'looking for suggestions' },
  memory: { short: 'memory', doing: 'tidying your memory' },
  skills: { short: 'skills', doing: 'reviewing skills' },
  titles: { short: 'a chat title', doing: 'naming a chat' },
  background: { short: 'background work', doing: 'doing background work' }
}

const BATTERY_TIP: Record<'battery' | 'battery_low', string> = {
  battery:
    "Sentient's own upkeep (memory, suggestions, chat titles) is waiting because this computer is on battery. " +
    'It runs now and then once the battery is above your limit and you have stopped chatting for a few minutes. ' +
    'It never waits longer than the limit in Settings > Models. Your chats and the tasks you scheduled run on time.',
  battery_low:
    "Sentient's own upkeep (memory, suggestions, chat titles) is waiting because the battery is low. " +
    'It never waits longer than the limit in Settings > Models. Your chats and the tasks you scheduled still run on time.'
}

export function modelBusyText(s: ModelBusy): { label: string; detail: string | null; tip: string } | null {
  // #296: on battery only Sentient's own upkeep waits, never your chats or tasks; say so honestly
  const battery = s.deferred_reason === 'battery' || s.deferred_reason === 'battery_low' ? s.deferred_reason : null
  if (!(s.busy && (s.job !== 'chat' || s.waiting > 0)) && !battery) return null
  const lines: string[] = []
  if (s.busy && s.job) {
    const name = modelShortName(s.model) || 'Your local model'
    lines.push(`${name} is ${JOB[s.job]?.doing ?? 'busy'}. It does one job at a time so it stays fast.`)
  }
  if (s.waiting > 0) lines.push(`${s.waiting} more waiting. Your chats always go first.`)
  if (battery) lines.push(BATTERY_TIP[battery])
  else if (s.deferred > 0) lines.push('Background work waits until you have stopped chatting for a moment.')
  if (s.busy) return { label: 'Model busy', detail: s.job ? JOB[s.job]?.short ?? null : null, tip: lines.join(' ') }
  return { label: 'Waiting', detail: battery === 'battery_low' ? 'battery low' : 'on battery', tip: lines.join(' ') }
}

export function ModelBusyIndicator() {
  const ready = useConnection((s) => s.backend.state === 'ready')
  const { data } = useModelBusy()
  const text = ready && data ? modelBusyText(data) : null
  if (!text || !data) return null
  return (
    <Tooltip content={text.tip} side="bottom">
      <span
        role="status"
        aria-label={`${text.label}${text.detail ? `: ${text.detail}` : ''}`}
        className="no-drag flex h-6.5 max-w-60 items-center gap-1.5 rounded-full border border-border bg-surface/60 px-2.5 text-xs text-fg-muted"
      >
        <StatusDot tone={data.busy ? 'info' : 'warning'} pulse={data.busy} />
        <span className="shrink-0 font-medium text-fg">
          {text.label}
          {!data.busy && text.detail ? ':' : ''}
        </span>
        {text.detail && <span className="truncate text-fg-subtle">{text.detail}</span>}
        {data.waiting > 0 && (
          <span className="shrink-0 rounded-full bg-active px-1.5 text-[10px] font-medium text-fg-muted">+{data.waiting}</span>
        )}
      </span>
    </Tooltip>
  )
}
