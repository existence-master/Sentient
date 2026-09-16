import * as PopoverPrimitive from '@radix-ui/react-popover'
import { IconCheck, IconPencilPlus, IconSelector } from '@tabler/icons-react'
import { Command } from 'cmdk'
import { useMemo, useState, type ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { fieldBase } from './Input'

export interface ComboboxOption {
  value: string
  label: string
  hint?: string
  badge?: string
  disabled?: boolean
}

export interface ComboboxGroup {
  id: string
  label: string
  options: ComboboxOption[]
}

export interface ComboboxProps {
  value: string
  onChange: (value: string) => void
  groups: ComboboxGroup[]
  placeholder?: string
  searchPlaceholder?: string
  /** Allow typing any value ("Use …"). */
  allowCustom?: boolean
  emptyText?: string
  disabled?: boolean
  className?: string
  size?: 'sm' | 'md'
  /** Custom rendering of the selected value inside the trigger. */
  renderValue?: (value: string, option?: ComboboxOption) => ReactNode
  /** Extra content under the list (e.g. "Pull a model…"). */
  footer?: ReactNode
  /** Optional "none" choice (e.g. use primary). */
  noneOption?: { label: string; value?: string }
  id?: string
  'aria-label'?: string
}

/** Searchable picker that also accepts free text. Used for model selection. */
export function Combobox({
  value,
  onChange,
  groups,
  placeholder = 'Choose…',
  searchPlaceholder = 'Search or type…',
  allowCustom = true,
  emptyText = 'No matches',
  disabled,
  className,
  size = 'md',
  renderValue,
  footer,
  noneOption,
  id,
  'aria-label': ariaLabel
}: ComboboxProps) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const all = useMemo(() => groups.flatMap((g) => g.options), [groups])
  const selected = all.find((o) => o.value === value)
  const trimmed = search.trim()
  const exact = all.some((o) => o.value === trimmed || o.label === trimmed)

  const choose = (v: string) => {
    onChange(v)
    setOpen(false)
    setSearch('')
  }

  return (
    <PopoverPrimitive.Root open={open} onOpenChange={(o) => (setOpen(o), o || setSearch(''))}>
      <PopoverPrimitive.Trigger
        id={id}
        aria-label={ariaLabel}
        disabled={disabled}
        className={cn(
          fieldBase,
          'inline-flex items-center justify-between gap-2 px-3 text-left',
          size === 'sm' ? 'h-7 text-xs' : 'h-8.5',
          className
        )}
      >
        <span className={cn('min-w-0 flex-1 truncate', !value && 'text-fg-subtle')}>
          {value ? (renderValue ? renderValue(value, selected) : (selected?.label ?? value)) : (noneOption?.label ?? placeholder)}
        </span>
        <IconSelector size={14} className="shrink-0 text-fg-subtle" />
      </PopoverPrimitive.Trigger>
      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Content
          align="start"
          sideOffset={6}
          className="z-50 w-[max(var(--radix-popover-trigger-width),300px)] overflow-hidden rounded-xl border border-border-strong bg-overlay shadow-pop data-[state=closed]:animate-pop-out data-[state=open]:animate-pop-in"
        >
          <Command loop className="flex max-h-[min(420px,var(--radix-popover-content-available-height))] flex-col">
            <div className="border-b border-border px-3">
              <Command.Input
                value={search}
                onValueChange={setSearch}
                placeholder={searchPlaceholder}
                className="h-10 w-full bg-transparent text-sm text-fg outline-none placeholder:text-fg-subtle"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && allowCustom && trimmed && !exact && !all.some((o) => o.label.toLowerCase().includes(trimmed.toLowerCase()))) {
                    e.preventDefault()
                    choose(trimmed)
                  }
                }}
              />
            </div>
            <Command.List className="min-h-0 flex-1 overflow-y-auto p-1">
              <Command.Empty className="px-3 py-6 text-center text-sm text-fg-subtle">
                {allowCustom && trimmed ? 'Press Enter to use this value' : emptyText}
              </Command.Empty>
              {allowCustom && trimmed && !exact && (
                <Command.Group heading="Custom">
                  <Command.Item
                    value={`__custom__${trimmed}`}
                    onSelect={() => choose(trimmed)}
                    className="flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm text-fg"
                  >
                    <IconPencilPlus size={14} className="text-fg-muted" />
                    Use <span className="font-mono text-xs text-accent-text">{trimmed}</span>
                  </Command.Item>
                </Command.Group>
              )}
              {noneOption && (
                <Command.Group>
                  <Command.Item
                    value={`__none__ ${noneOption.label}`}
                    onSelect={() => choose(noneOption.value ?? '')}
                    className="flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm text-fg-muted"
                  >
                    <span className="flex-1">{noneOption.label}</span>
                    {!value && <IconCheck size={14} className="text-accent-text" />}
                  </Command.Item>
                </Command.Group>
              )}
              {groups.map((g) => (
                <Command.Group key={g.id} heading={g.label}>
                  {g.options.map((o) => (
                    <Command.Item
                      key={`${g.id}:${o.value}`}
                      value={`${o.label} ${o.value} ${g.label}`}
                      disabled={o.disabled}
                      onSelect={() => choose(o.value)}
                      className="flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm text-fg"
                    >
                      <span className="min-w-0 flex-1 truncate">{o.label}</span>
                      {o.hint && <span className="shrink-0 text-xs text-fg-subtle">{o.hint}</span>}
                      {o.badge && (
                        <span className="shrink-0 rounded-full bg-warning/12 px-1.5 py-px text-2xs text-warning">{o.badge}</span>
                      )}
                      <IconCheck size={14} className={cn('shrink-0 text-accent-text', o.value !== value && 'invisible')} />
                    </Command.Item>
                  ))}
                </Command.Group>
              ))}
            </Command.List>
            {footer && <div className="border-t border-border p-1">{footer}</div>}
          </Command>
        </PopoverPrimitive.Content>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  )
}
