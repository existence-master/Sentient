import { Tooltip } from '@/components/ui'
import { cn } from '@/lib/utils'
import { useAttentionCount } from './hooks'

/** Count of tasks needing attention, shown on the sidebar Tasks item. */
export function TasksNavBadge({ collapsed }: { collapsed: boolean }) {
  const count = useAttentionCount()
  if (!count) return null
  const label = `${count} task${count === 1 ? '' : 's'} need${count === 1 ? 's' : ''} your attention`
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
