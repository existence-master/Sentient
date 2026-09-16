import * as SelectPrimitive from '@radix-ui/react-select'
import { IconCheck, IconChevronDown } from '@tabler/icons-react'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { fieldBase } from './Input'

export interface SelectOption<T extends string = string> {
  value: T
  label: ReactNode
  description?: ReactNode
  icon?: ReactNode
  disabled?: boolean
}

export interface SelectProps<T extends string = string> {
  value: T | undefined
  onValueChange: (value: T) => void
  options: SelectOption<T>[]
  placeholder?: string
  size?: 'sm' | 'md'
  disabled?: boolean
  className?: string
  contentClassName?: string
  id?: string
  'aria-label'?: string
}

/** Radix Select styled for Sentient. Option values must be non-empty strings. */
export function Select<T extends string = string>({
  value,
  onValueChange,
  options,
  placeholder = 'Select…',
  size = 'md',
  disabled,
  className,
  contentClassName,
  id,
  'aria-label': ariaLabel
}: SelectProps<T>) {
  const current = options.find((o) => o.value === value)
  return (
    <SelectPrimitive.Root value={value} onValueChange={(v) => onValueChange(v as T)} disabled={disabled}>
      <SelectPrimitive.Trigger
        id={id}
        aria-label={ariaLabel}
        className={cn(
          fieldBase,
          'inline-flex items-center justify-between gap-2 px-3 text-left data-[placeholder]:text-fg-subtle',
          size === 'sm' ? 'h-7 text-xs' : 'h-8.5',
          className
        )}
      >
        <span className="flex min-w-0 items-center gap-2 truncate">
          {current?.icon}
          <SelectPrimitive.Value placeholder={placeholder}>{current?.label}</SelectPrimitive.Value>
        </span>
        <SelectPrimitive.Icon className="text-fg-subtle">
          <IconChevronDown size={14} />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content
          position="popper"
          sideOffset={6}
          className={cn(
            'z-50 max-h-[min(360px,var(--radix-select-content-available-height))] min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-xl border border-border-strong bg-overlay p-1 text-sm shadow-pop data-[state=closed]:animate-pop-out data-[state=open]:animate-pop-in',
            contentClassName
          )}
        >
          <SelectPrimitive.Viewport>
            {options.map((o) => (
              <SelectPrimitive.Item
                key={o.value}
                value={o.value}
                disabled={o.disabled}
                className="relative flex cursor-pointer select-none items-center gap-2 rounded-lg py-1.5 pl-2.5 pr-8 text-fg outline-none data-[disabled]:pointer-events-none data-[highlighted]:bg-active data-[disabled]:opacity-45"
              >
                {o.icon && <span className="flex text-fg-muted">{o.icon}</span>}
                <div className="min-w-0">
                  <SelectPrimitive.ItemText>{o.label}</SelectPrimitive.ItemText>
                  {o.description && <div className="text-xs text-fg-subtle">{o.description}</div>}
                </div>
                <SelectPrimitive.ItemIndicator className="absolute right-2.5 text-accent-text">
                  <IconCheck size={14} />
                </SelectPrimitive.ItemIndicator>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  )
}
