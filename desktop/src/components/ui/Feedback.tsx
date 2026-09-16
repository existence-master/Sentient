import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------- Spinner
export function Spinner({ size = 16, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      role="status"
      aria-label="Loading"
      className={cn('animate-spin text-current', className)}
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  )
}

// ---------------------------------------------------------------------------- Skeleton
export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn('skeleton rounded-md', className)} />
}

// ---------------------------------------------------------------------------- Badge
export type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'info'

const toneClass: Record<Tone, string> = {
  neutral: 'bg-active text-fg-muted border-border',
  accent: 'bg-accent/12 text-accent-text border-accent/25',
  success: 'bg-success/12 text-success border-success/25',
  warning: 'bg-warning/12 text-warning border-warning/25',
  danger: 'bg-danger/12 text-danger border-danger/25',
  info: 'bg-info/12 text-info border-info/25'
}

export function Badge({
  tone = 'neutral',
  size = 'sm',
  icon,
  className,
  children
}: {
  tone?: Tone
  size?: 'xs' | 'sm'
  icon?: ReactNode
  className?: string
  children: ReactNode
}) {
  return (
    <span
      className={cn(
        'inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full border font-medium [&_svg]:size-3',
        size === 'xs' ? 'h-4.5 px-1.5 text-2xs' : 'h-5.5 px-2 text-xs',
        toneClass[tone],
        className
      )}
    >
      {icon}
      {children}
    </span>
  )
}

// ---------------------------------------------------------------------------- StatusDot
const dotTone: Record<Tone, string> = {
  neutral: 'bg-fg-faint',
  accent: 'bg-accent',
  success: 'bg-success',
  warning: 'bg-warning',
  danger: 'bg-danger',
  info: 'bg-info'
}

export function StatusDot({ tone = 'neutral', pulse, className }: { tone?: Tone; pulse?: boolean; className?: string }) {
  return (
    <span className={cn('relative inline-flex size-2 shrink-0', className)}>
      {pulse && <span className={cn('absolute inset-0 animate-ping rounded-full opacity-50', dotTone[tone])} />}
      <span className={cn('relative inline-flex size-2 rounded-full', dotTone[tone])} />
    </span>
  )
}

// ---------------------------------------------------------------------------- Kbd
export function Kbd({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <kbd
      className={cn(
        'inline-flex h-5 min-w-5 items-center justify-center rounded-[5px] border border-border-strong bg-elevated px-1 font-sans text-2xs font-medium text-fg-muted shadow-[0_1px_0_var(--border-strong)]',
        className
      )}
    >
      {children}
    </kbd>
  )
}

/** Renders "Ctrl+Shift+K" as separate keys. */
export function Shortcut({ keys, className }: { keys: string; className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-0.5', className)}>
      {keys.split('+').map((k) => (
        <Kbd key={k}>{k}</Kbd>
      ))}
    </span>
  )
}

// ---------------------------------------------------------------------------- ProgressBar
export function ProgressBar({
  value,
  tone = 'accent',
  className,
  size = 'sm'
}: {
  /** 0..1, or null/undefined for indeterminate */
  value?: number | null
  tone?: Tone
  className?: string
  size?: 'xs' | 'sm' | 'md'
}) {
  const indeterminate = value === null || value === undefined || Number.isNaN(value)
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.round((value as number) * 100)}
      className={cn('relative w-full overflow-hidden rounded-full bg-active', { xs: 'h-0.5', sm: 'h-1', md: 'h-1.5' }[size], className)}
    >
      {indeterminate ? (
        <div className={cn('absolute inset-y-0 w-2/5 animate-indeterminate rounded-full', dotTone[tone])} />
      ) : (
        <div
          className={cn('h-full rounded-full transition-[width] duration-300', dotTone[tone])}
          style={{ width: `${Math.max(0, Math.min(1, value as number)) * 100}%` }}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------- EmptyState
export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
  compact
}: {
  icon?: ReactNode
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  className?: string
  compact?: boolean
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center text-center', compact ? 'gap-2 py-8' : 'gap-3 py-16', className)}>
      {icon && (
        <div
          className={cn(
            'flex items-center justify-center rounded-2xl border border-border bg-elevated text-fg-muted shadow-soft',
            compact ? 'size-10 [&_svg]:size-5' : 'size-14 [&_svg]:size-6.5'
          )}
        >
          {icon}
        </div>
      )}
      <div className="space-y-1">
        <div className={cn('font-medium text-fg', compact ? 'text-sm' : 'text-md')}>{title}</div>
        {description && <div className="mx-auto max-w-sm text-sm text-fg-subtle">{description}</div>}
      </div>
      {action && <div className="mt-1 flex items-center gap-2">{action}</div>}
    </div>
  )
}

// ---------------------------------------------------------------------------- InlineAlert
export function Alert({
  tone = 'info',
  icon,
  title,
  children,
  action,
  className
}: {
  tone?: Tone
  icon?: ReactNode
  title?: ReactNode
  children?: ReactNode
  action?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex items-start gap-3 rounded-xl border px-3.5 py-3 text-sm', toneClass[tone], className)}>
      {icon && <span className="mt-0.5 flex shrink-0 [&_svg]:size-4">{icon}</span>}
      <div className="min-w-0 flex-1 space-y-0.5">
        {title && <div className="font-medium">{title}</div>}
        {children && <div className="text-fg-muted">{children}</div>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  )
}
