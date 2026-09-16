import { IconPointFilled } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useMemo } from 'react'
import type { Memory } from '@/lib/types'
import { cn, formatTime, relativeTime } from '@/lib/utils'
import { ExpiryBadge, SourceChip, TopicPill } from './bits'
import { useNow } from './hooks'
import { dayKey, dayLabel, primaryTopic, topicMeta } from './meta'

function MemoryCard({ memory, selected, onSelect, now, compact }: { memory: Memory; selected: boolean; onSelect: (id: number) => void; now: number; compact?: boolean }) {
  const color = topicMeta(primaryTopic(memory.topics)).color
  return (
    <motion.button
      layout="position"
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98 }}
      type="button"
      onClick={() => onSelect(memory.id)}
      className={cn(
        'group relative flex w-full flex-col gap-2.5 overflow-hidden rounded-xl border bg-surface p-3.5 pl-4 text-left transition-colors hover:border-border-strong hover:bg-elevated',
        selected ? 'border-accent/50 bg-elevated ring-3 ring-accent/10' : 'border-border'
      )}
    >
      <span aria-hidden className="absolute inset-y-0 left-0 w-[3px]" style={{ background: color }} />
      <p className={cn('text-sm leading-relaxed text-fg', compact ? 'line-clamp-2' : 'line-clamp-3')}>{memory.content}</p>
      <div className="flex flex-wrap items-center gap-1.5">
        {memory.topics.map((t) => (
          <TopicPill key={t} topic={t} size="xs" />
        ))}
        <ExpiryBadge memory={memory} now={now} className="h-4.5" />
        {typeof memory.similarity === 'number' && (
          <span className="rounded-full bg-accent/12 px-1.5 text-2xs font-medium text-accent-text">{Math.round(memory.similarity * 100)}% match</span>
        )}
      </div>
      <div className="flex items-center gap-2 text-xs text-fg-subtle">
        <SourceChip source={memory.source} className="min-w-0" />
        <IconPointFilled size={6} className="shrink-0 text-fg-faint" />
        <span className="shrink-0" title={memory.created_at}>
          {compact ? formatTime(memory.created_at) : relativeTime(memory.created_at)}
        </span>
        {memory.updated_at && memory.created_at && memory.updated_at > memory.created_at && (
          <span className="shrink-0 text-fg-faint">· edited {relativeTime(memory.updated_at)}</span>
        )}
      </div>
    </motion.button>
  )
}

export function MemoryList({ memories, selectedId, onSelect }: { memories: Memory[]; selectedId: number | null; onSelect: (id: number) => void }) {
  const now = useNow()
  return (
    <div className="grid grid-cols-1 gap-2.5 md:grid-cols-2 2xl:grid-cols-3">
      <AnimatePresence initial={false}>
        {memories.map((m) => (
          <MemoryCard key={m.id} memory={m} selected={m.id === selectedId} onSelect={onSelect} now={now} />
        ))}
      </AnimatePresence>
    </div>
  )
}

export function MemoryTimeline({ memories, selectedId, onSelect }: { memories: Memory[]; selectedId: number | null; onSelect: (id: number) => void }) {
  const now = useNow()
  const days = useMemo(() => {
    const groups = new Map<string, Memory[]>()
    const sorted = [...memories].sort((a, b) => (a.created_at < b.created_at ? 1 : -1))
    for (const m of sorted) {
      const k = dayKey(m.created_at)
      if (!groups.has(k)) groups.set(k, [])
      groups.get(k)!.push(m)
    }
    return [...groups.entries()]
  }, [memories])

  return (
    <ol className="relative space-y-6">
      <span aria-hidden className="absolute bottom-2 left-[7px] top-2 w-px bg-border" />
      {days.map(([key, items]) => {
        const topics = new Map<string, number>()
        items.forEach((m) => topics.set(primaryTopic(m.topics), (topics.get(primaryTopic(m.topics)) ?? 0) + 1))
        return (
          <li key={key} className="relative pl-7">
            <span aria-hidden className="absolute left-0 top-1 flex size-[15px] items-center justify-center rounded-full border border-border-strong bg-surface">
              <span className="size-1.5 rounded-full bg-accent" />
            </span>
            <div className="mb-2.5 flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
              <h3 className="text-sm font-semibold text-fg">{dayLabel(key)}</h3>
              <span className="text-xs text-fg-subtle">
                learned {items.length} {items.length === 1 ? 'memory' : 'memories'}
              </span>
              <span className="flex items-center gap-1">
                {[...topics.entries()].map(([t, n]) => (
                  <span key={t} title={`${t}: ${n}`} className="h-1.5 rounded-full" style={{ width: 6 + n * 6, background: topicMeta(t).color }} />
                ))}
              </span>
            </div>
            <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
              {items.map((m) => (
                <MemoryCard key={m.id} memory={m} selected={m.id === selectedId} onSelect={onSelect} now={now} compact />
              ))}
            </div>
          </li>
        )
      })}
    </ol>
  )
}
