import { Tooltip } from '@/components/ui'
import { useMemoryReview } from '@/hooks/memory'
import { cn } from '@/lib/utils'

/** Count of memories waiting for review, shown on the sidebar Memory item. */
export function MemoryNavBadge({ collapsed }: { collapsed: boolean }) {
  const { data } = useMemoryReview()
  const count = data?.count ?? 0
  if (!count) return null
  const label = `${count} ${count === 1 ? 'memory' : 'memories'} waiting for your review`
  return (
    <Tooltip content={label} side="right">
      <span
        aria-label={label}
        className={cn(
          'pointer-events-auto flex h-4.5 min-w-4.5 items-center justify-center rounded-full bg-accent px-1 text-[10px] font-semibold leading-none text-accent-fg tabular-nums',
          collapsed ? 'absolute right-0.5 top-0.5 h-4 min-w-4' : 'ml-auto'
        )}
      >
        {count > 99 ? '99+' : count}
      </span>
    </Tooltip>
  )
}
