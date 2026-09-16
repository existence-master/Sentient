import { Tooltip } from '@/components/ui'
import { useSkills } from '@/hooks/skills'
import { cn } from '@/lib/utils'

/** Count of skills waiting for review, shown on the sidebar Skills item. */
export function SkillsNavBadge({ collapsed }: { collapsed: boolean }) {
  const { data } = useSkills()
  const count = data?.pending.length ?? 0
  if (!count) return null
  const label = `${count} skill${count === 1 ? '' : 's'} waiting for your review`
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
