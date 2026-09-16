import * as SliderPrimitive from '@radix-ui/react-slider'
import * as SwitchPrimitive from '@radix-ui/react-switch'
import * as TabsPrimitive from '@radix-ui/react-tabs'
import { motion } from 'motion/react'
import { useId, type ComponentProps, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------- Switch
export interface SwitchProps extends Omit<ComponentProps<typeof SwitchPrimitive.Root>, 'onChange'> {
  size?: 'sm' | 'md'
}

export function Switch({ size = 'md', className, ...props }: SwitchProps) {
  return (
    <SwitchPrimitive.Root
      className={cn(
        'no-drag relative inline-flex shrink-0 items-center rounded-full border border-transparent bg-active transition-colors duration-200 data-[state=checked]:bg-accent disabled:opacity-45',
        size === 'sm' ? 'h-4.5 w-8' : 'h-5.5 w-9.5',
        className
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        className={cn(
          'block rounded-full bg-white shadow-[0_1px_3px_rgb(0_0_0/0.35)] transition-transform duration-200 ease-[var(--ease-snappy)] data-[state=unchecked]:translate-x-0.5',
          size === 'sm' ? 'size-3.5 data-[state=checked]:translate-x-[15px]' : 'size-4.5 data-[state=checked]:translate-x-[17px]'
        )}
      />
    </SwitchPrimitive.Root>
  )
}

// ---------------------------------------------------------------------------- Slider
export interface SliderProps {
  value: number
  onValueChange: (value: number) => void
  onValueCommit?: (value: number) => void
  min?: number
  max?: number
  step?: number
  disabled?: boolean
  className?: string
  'aria-label'?: string
}

export function Slider({ value, onValueChange, onValueCommit, min = 0, max = 1, step = 0.01, disabled, className, ...rest }: SliderProps) {
  return (
    <SliderPrimitive.Root
      value={[value]}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onValueChange={([v]) => onValueChange(v)}
      onValueCommit={([v]) => onValueCommit?.(v)}
      className={cn('relative flex h-5 w-full touch-none select-none items-center data-[disabled]:opacity-45', className)}
    >
      <SliderPrimitive.Track className="relative h-1 grow overflow-hidden rounded-full bg-active">
        <SliderPrimitive.Range className="absolute h-full rounded-full bg-accent" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        aria-label={rest['aria-label']}
        className="block size-3.5 rounded-full border-2 border-accent bg-white shadow-soft transition-transform hover:scale-110 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-accent/25"
      />
    </SliderPrimitive.Root>
  )
}

// ---------------------------------------------------------------------------- Tabs
export const Tabs = TabsPrimitive.Root

export function TabsList({ className, ...props }: ComponentProps<typeof TabsPrimitive.List>) {
  return <TabsPrimitive.List className={cn('flex items-center gap-1 border-b border-border', className)} {...props} />
}

export function TabsTrigger({ className, ...props }: ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      className={cn(
        'relative -mb-px inline-flex h-9 items-center gap-1.5 border-b-2 border-transparent px-3 text-sm text-fg-muted transition-colors hover:text-fg data-[state=active]:border-accent data-[state=active]:text-fg',
        className
      )}
      {...props}
    />
  )
}

export function TabsContent({ className, ...props }: ComponentProps<typeof TabsPrimitive.Content>) {
  return <TabsPrimitive.Content className={cn('outline-none', className)} {...props} />
}

// ---------------------------------------------------------------------------- SegmentedControl
export interface SegmentOption<T extends string> {
  value: T
  label: ReactNode
  icon?: ReactNode
  disabled?: boolean
}

export interface SegmentedControlProps<T extends string> {
  value: T
  onChange: (value: T) => void
  options: SegmentOption<T>[]
  size?: 'sm' | 'md'
  className?: string
  fullWidth?: boolean
  'aria-label'?: string
}

export function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
  size = 'md',
  className,
  fullWidth,
  ...rest
}: SegmentedControlProps<T>) {
  const id = useId()
  return (
    <div
      role="radiogroup"
      aria-label={rest['aria-label']}
      className={cn(
        'no-drag inline-flex items-center gap-0.5 rounded-lg border border-border bg-sunken/70 p-0.5',
        fullWidth && 'flex w-full',
        className
      )}
    >
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={o.disabled}
            onClick={() => onChange(o.value)}
            className={cn(
              'relative inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors disabled:opacity-40',
              size === 'sm' ? 'h-6 px-2 text-xs' : 'h-7 px-3 text-sm',
              fullWidth && 'flex-1',
              active ? 'text-fg' : 'text-fg-subtle hover:text-fg-muted'
            )}
          >
            {active && (
              <motion.span
                layoutId={`seg-${id}`}
                transition={{ type: 'spring', stiffness: 520, damping: 40 }}
                className="absolute inset-0 rounded-md border border-border-strong bg-elevated shadow-soft"
              />
            )}
            <span className="relative flex items-center gap-1.5">
              {o.icon}
              {o.label}
            </span>
          </button>
        )
      })}
    </div>
  )
}
