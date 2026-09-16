import { IconHourglassHigh } from '@tabler/icons-react'
import { cn } from '@/lib/utils'
import type { Memory } from '@/lib/types'
import { countdown, EXPIRING_SOON_MS, expiresInMs, sourceMeta, topicMeta, topicText, topicTint } from './meta'

export function TopicPill({ topic, size = 'sm', className }: { topic: string; size?: 'xs' | 'sm'; className?: string }) {
  const meta = topicMeta(topic)
  return (
    <span
      className={cn(
        'inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full border font-medium',
        size === 'xs' ? 'h-4.5 px-1.5 text-2xs' : 'h-5.5 px-2 text-xs',
        className
      )}
      style={{ background: topicTint(meta.color, 12), borderColor: topicTint(meta.color, 28), color: topicText(meta.color) }}
    >
      <meta.icon size={size === 'xs' ? 10 : 12} stroke={2} />
      {topic}
    </span>
  )
}

export function SourceChip({ source, className }: { source: string; className?: string }) {
  const meta = sourceMeta(source)
  return (
    <span title={meta.description} className={cn('inline-flex min-w-0 items-center gap-1 text-xs text-fg-subtle', className)}>
      <meta.icon size={12} className="shrink-0" />
      <span className="truncate">{meta.label}</span>
    </span>
  )
}

export function ExpiryBadge({ memory, now, className }: { memory: Memory; now: number; className?: string }) {
  const ms = expiresInMs(memory, now)
  if (memory.memory_type !== 'short-term' || ms === null) return null
  const soon = ms < EXPIRING_SOON_MS
  return (
    <span
      title={`Short-term memory: forgotten automatically ${ms > 0 ? `in ${countdown(ms)}` : 'soon'}`}
      className={cn(
        'inline-flex h-5.5 shrink-0 items-center gap-1 rounded-full border px-2 font-mono text-2xs tabular-nums',
        soon ? 'border-warning/30 bg-warning/10 text-warning' : 'border-border bg-active text-fg-muted',
        className
      )}
    >
      <IconHourglassHigh size={11} />
      {ms > 0 ? countdown(ms) : 'expiring'}
    </span>
  )
}
