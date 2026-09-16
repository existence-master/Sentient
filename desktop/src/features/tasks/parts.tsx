/** Small building blocks shared by the task views. */
import { IconFlag3Filled, IconPlayerPause, IconRadar } from '@tabler/icons-react'
import type { ReactNode } from 'react'
import { Badge, Tooltip } from '@/components/ui'
import type { Task } from '@/lib/types'
import { cn } from '@/lib/utils'
import { KIND_META, priorityMeta, runStatusMeta, statusMeta, taskKind, type StatusMeta } from './meta'

function MetaBadge({ meta, size = 'sm', className }: { meta: StatusMeta; size?: 'xs' | 'sm'; className?: string }) {
  const I = meta.icon
  return (
    <Badge tone={meta.tone} size={size} className={className} icon={<I className={cn(meta.live && meta.label === 'Running' && 'animate-spin [animation-duration:2.4s]')} />}>
      {meta.label}
    </Badge>
  )
}

export function TaskStatusBadge({ task, size, className }: { task: Pick<Task, 'status' | 'enabled'>; size?: 'xs' | 'sm'; className?: string }) {
  return <MetaBadge meta={statusMeta(task.status, task)} size={size} className={className} />
}

export function RunStatusBadge({ status, size, className }: { status: string; size?: 'xs' | 'sm'; className?: string }) {
  return <MetaBadge meta={runStatusMeta(status)} size={size} className={className} />
}

const toneBg: Record<string, string> = {
  neutral: 'bg-elevated text-fg-muted border-border',
  accent: 'bg-accent/10 text-accent-text border-accent/25',
  success: 'bg-success/10 text-success border-success/20',
  warning: 'bg-warning/10 text-warning border-warning/25',
  danger: 'bg-danger/10 text-danger border-danger/25',
  info: 'bg-info/10 text-info border-info/25'
}

/** Rounded tile with the task-kind icon, tinted by status. */
export function KindTile({ task, size = 'md', className }: { task: Task; size?: 'sm' | 'md' | 'lg'; className?: string }) {
  const kind = taskKind(task)
  const I = KIND_META[kind].icon
  const meta = statusMeta(task.status, task)
  const dims = { sm: 'size-7 rounded-lg', md: 'size-9 rounded-xl', lg: 'size-11 rounded-2xl' }[size]
  const icon = { sm: 14, md: 17, lg: 21 }[size]
  return (
    <span className={cn('relative inline-flex shrink-0 items-center justify-center border', dims, toneBg[meta.tone], className)}>
      <I size={icon} stroke={1.75} />
      {meta.live && (
        <span className="absolute -right-0.5 -top-0.5 flex size-2.5">
          <span className="absolute inset-0 animate-ping rounded-full bg-info opacity-60" />
          <span className="relative inline-flex size-2.5 rounded-full border-2 border-surface bg-info" />
        </span>
      )}
      {!task.enabled && (
        <span className="absolute -bottom-1 -right-1 flex size-4 items-center justify-center rounded-full border border-border bg-surface text-fg-subtle">
          <IconPlayerPause size={9} />
        </span>
      )}
    </span>
  )
}

/** Marks a script job ("Watcher") in the list, board and calendar views. */
export function ScriptBadge({ className }: { className?: string }) {
  return (
    <Tooltip content="A small script checks this on a schedule, without using AI each time">
      <span className={cn('inline-flex', className)}>
        <Badge tone="info" size="xs" icon={<IconRadar />}>
          Watcher
        </Badge>
      </span>
    </Tooltip>
  )
}

export function PriorityFlag({ priority, className, showLow }: { priority: number; className?: string; showLow?: boolean }) {
  const meta = priorityMeta(priority)
  if (priority === 1 || (priority === 2 && !showLow)) return null
  return (
    <Tooltip content={`${meta.label} priority`}>
      <span className={cn('inline-flex', priority === 0 ? 'text-danger' : 'text-fg-faint', className)}>
        <IconFlag3Filled size={12} />
      </span>
    </Tooltip>
  )
}

export function SectionHeading({ icon, title, count, actions, className, description }: { icon?: ReactNode; title: ReactNode; count?: number; actions?: ReactNode; className?: string; description?: ReactNode }) {
  return (
    <div className={cn('flex items-center gap-2', className)}>
      {icon && <span className="flex text-fg-subtle [&_svg]:size-4">{icon}</span>}
      <h3 className="text-sm font-semibold text-fg">{title}</h3>
      {count !== undefined && <span className="rounded-full bg-active px-1.5 text-2xs font-medium tabular-nums text-fg-muted">{count}</span>}
      {description && <span className="min-w-0 truncate text-xs text-fg-subtle">{description}</span>}
      <span className="flex-1" />
      {actions}
    </div>
  )
}

export function LiveDot({ className, label = 'Live' }: { className?: string; label?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-2xs font-semibold uppercase tracking-wider text-info', className)}>
      <span className="relative flex size-2">
        <span className="absolute inset-0 animate-ping rounded-full bg-info opacity-60" />
        <span className="relative inline-flex size-2 rounded-full bg-info" />
      </span>
      {label}
    </span>
  )
}
