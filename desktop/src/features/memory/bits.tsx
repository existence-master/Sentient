import { IconBriefcase, IconFolder, IconHome, IconHourglassHigh, IconInfinity } from '@tabler/icons-react'
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

/** The plain name of a memory scope (#155): Personal, Work, Every chat, or a project's own name. */
export function scopeLabel(scope: string | null | undefined): string {
  if (!scope || scope === 'personal') return 'Personal'
  if (scope === 'work') return 'Work'
  if (scope === 'always') return 'Every chat'
  return scope
}

export function ScopeIcon({ scope, size = 15, className }: { scope: string | null | undefined; size?: number; className?: string }) {
  if (!scope || scope === 'personal') return <IconHome size={size} className={className} />
  if (scope === 'work') return <IconBriefcase size={size} className={className} />
  if (scope === 'always') return <IconInfinity size={size} className={className} />
  return <IconFolder size={size} className={className} />
}

/**
 * Where a memory is kept, shown unless it is the default (Personal). It reads "Used in Work" in a dashed, square
 * outline next to the source, so it never looks like a topic pill such as "Work & Learning" (#392).
 */
export function ScopeBadge({ scope, className }: { scope: string | null | undefined; className?: string }) {
  if (!scope || scope === 'personal') return null
  const where = scope === 'always' ? 'every chat' : scopeLabel(scope)
  return (
    <span
      title={scope === 'always' ? 'Every chat uses this memory' : `Only ${scopeLabel(scope)} chats use this memory`}
      data-scope-badge={scope}
      className={cn(
        'inline-flex h-5 min-w-0 max-w-44 shrink items-center gap-1 rounded-md border border-dashed border-border-strong px-1.5 text-2xs text-fg-subtle',
        className
      )}
    >
      <ScopeIcon scope={scope} size={11} className="shrink-0" />
      <span className="truncate">
        Used in <span className="font-medium text-fg-muted">{where}</span>
      </span>
    </span>
  )
}
