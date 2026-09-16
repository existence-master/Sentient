import type { ComponentProps, ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Spinner } from './Feedback'
import { Tooltip } from './Tooltip'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'subtle' | 'danger' | 'outline'
export type ButtonSize = 'xs' | 'sm' | 'md' | 'lg'

const variantClass: Record<ButtonVariant, string> = {
  primary: 'bg-accent text-accent-fg hover:bg-accent-hover shadow-soft font-medium',
  secondary: 'bg-elevated text-fg border border-border-strong hover:bg-overlay shadow-soft',
  ghost: 'text-fg-muted hover:text-fg hover:bg-hover',
  subtle: 'bg-hover text-fg hover:bg-active',
  danger: 'bg-danger/10 text-danger border border-danger/25 hover:bg-danger/18',
  outline: 'border border-border-strong text-fg hover:bg-hover'
}

const sizeClass: Record<ButtonSize, string> = {
  xs: 'h-6 px-2 text-xs gap-1 rounded-md',
  sm: 'h-7 px-2.5 text-sm gap-1.5 rounded-md',
  md: 'h-8.5 px-3.5 text-sm gap-2 rounded-lg',
  lg: 'h-10 px-5 text-base gap-2 rounded-lg'
}

export interface ButtonProps extends ComponentProps<'button'> {
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
  leftIcon?: ReactNode
  rightIcon?: ReactNode
}

export function Button({
  variant = 'secondary',
  size = 'md',
  loading = false,
  leftIcon,
  rightIcon,
  className,
  children,
  disabled,
  type = 'button',
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(
        'no-drag inline-flex shrink-0 select-none items-center justify-center whitespace-nowrap transition-[background-color,border-color,color,box-shadow,opacity] duration-150 disabled:pointer-events-none disabled:opacity-45 [&_svg]:shrink-0',
        variantClass[variant],
        sizeClass[size],
        className
      )}
      {...props}
    >
      {loading ? <Spinner size={size === 'lg' ? 16 : 14} /> : leftIcon}
      {children}
      {!loading && rightIcon}
    </button>
  )
}

export interface IconButtonProps extends Omit<ComponentProps<'button'>, 'children'> {
  icon: ReactNode
  /** Accessible label; also shown as a tooltip unless `tooltip={false}`. */
  label: string
  tooltip?: boolean | ReactNode
  shortcut?: string
  variant?: 'ghost' | 'subtle' | 'secondary' | 'primary' | 'danger'
  size?: 'xs' | 'sm' | 'md' | 'lg'
  active?: boolean
  loading?: boolean
  side?: 'top' | 'bottom' | 'left' | 'right'
}

const iconSize = { xs: 'size-6 rounded-md', sm: 'size-7 rounded-md', md: 'size-8 rounded-lg', lg: 'size-10 rounded-lg' }

export function IconButton({
  icon,
  label,
  tooltip = true,
  shortcut,
  variant = 'ghost',
  size = 'md',
  active,
  loading,
  className,
  disabled,
  side,
  type = 'button',
  ...props
}: IconButtonProps) {
  const button = (
    <button
      type={type}
      aria-label={label}
      aria-pressed={active}
      disabled={disabled || loading}
      className={cn(
        'no-drag inline-flex shrink-0 items-center justify-center transition-colors duration-150 disabled:pointer-events-none disabled:opacity-40',
        variantClass[variant],
        variant === 'primary' && 'shadow-none',
        iconSize[size],
        active && 'bg-active text-fg',
        className
      )}
      {...props}
    >
      {loading ? <Spinner size={14} /> : icon}
    </button>
  )
  if (!tooltip) return button
  return (
    <Tooltip content={tooltip === true ? label : tooltip} shortcut={shortcut} side={side}>
      {button}
    </Tooltip>
  )
}
