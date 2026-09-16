import { IconChartBar, IconClock, IconEye, IconFilePencil, IconTool, IconZzz } from '@tabler/icons-react'
import { motion } from 'motion/react'
import { Badge, Tooltip } from '@/components/ui'
import type { Skill } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'
import { authorMeta } from './meta'

export function AuthorBadge({ author, className }: { author: string; className?: string }) {
  const meta = authorMeta(author)
  return (
    <Tooltip content={meta.description}>
      <span
        className={cn(
          'inline-flex h-5.5 shrink-0 items-center gap-1 rounded-full border px-2 text-xs font-medium',
          author === 'assistant' ? 'border-accent/25 bg-accent/10 text-accent-text' : author === 'community' ? 'border-info/25 bg-info/10 text-info' : 'border-border bg-active text-fg-muted',
          className
        )}
      >
        <meta.icon size={12} />
        {meta.label}
      </span>
    </Tooltip>
  )
}

export function SkillStat({ icon, value, label }: { icon: React.ReactNode; value: React.ReactNode; label: string }) {
  return (
    <Tooltip content={label}>
      <span className="inline-flex items-center gap-1 tabular-nums">
        {icon}
        {value}
      </span>
    </Tooltip>
  )
}

export function SkillCard({ skill, onOpen, hasUpdate }: { skill: Skill; onOpen: () => void; hasUpdate?: boolean }) {
  const stale = skill.state === 'stale'
  return (
    <motion.button
      layout="position"
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      type="button"
      onClick={onOpen}
      className={cn(
        'group flex h-full flex-col gap-3 rounded-xl border border-border bg-surface p-4 text-left transition-colors hover:border-border-strong hover:bg-elevated',
        stale && 'bg-surface/60'
      )}
    >
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className={cn('truncate font-mono text-sm font-semibold', stale ? 'text-fg-muted' : 'text-fg')}>{skill.name}</div>
          <div className="mt-0.5 text-2xs text-fg-subtle">v{skill.version}</div>
        </div>
        {hasUpdate && (
          <Badge tone="info" size="xs" icon={<IconFilePencil />}>
            Update
          </Badge>
        )}
        {stale && (
          <Tooltip content="Not used recently. The curator archives stale skills that stay unused.">
            <span>
              <Badge tone="neutral" size="xs" icon={<IconZzz />}>
                Stale
              </Badge>
            </span>
          </Tooltip>
        )}
        <AuthorBadge author={skill.author} />
      </div>
      <p className="line-clamp-2 flex-1 text-sm leading-relaxed text-fg-muted">{skill.description || 'No description.'}</p>
      {(skill.tags.length > 0 || skill.requires_tools.length > 0) && (
        <div className="flex flex-wrap gap-1">
          {skill.tags.map((t) => (
            <span key={t} className="rounded-md bg-active px-1.5 py-0.5 text-2xs text-fg-muted">
              #{t}
            </span>
          ))}
          {skill.requires_tools.map((t) => (
            <span key={t} className="inline-flex items-center gap-1 rounded-md border border-border px-1.5 py-0.5 text-2xs text-fg-subtle">
              <IconTool size={10} />
              {t}
            </span>
          ))}
        </div>
      )}
      <div className="flex items-center gap-3 border-t border-border pt-2.5 text-xs text-fg-subtle">
        <SkillStat icon={<IconChartBar size={13} />} value={skill.use_count} label="Times used" />
        <SkillStat icon={<IconEye size={13} />} value={skill.view_count} label="Times read by Sentient" />
        <SkillStat icon={<IconFilePencil size={13} />} value={skill.patch_count} label="Times improved" />
        <div className="flex-1" />
        <span className="inline-flex items-center gap-1">
          <IconClock size={12} />
          {skill.last_used_at ? relativeTime(skill.last_used_at) : 'never used'}
        </span>
      </div>
    </motion.button>
  )
}
