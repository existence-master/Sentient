import { useCallback, useLayoutEffect, useRef, type ComponentProps, type ReactNode, type Ref } from 'react'
import { cn } from '@/lib/utils'

export const fieldBase =
  'w-full rounded-lg border border-border bg-field text-sm text-fg shadow-[inset_0_1px_0_rgb(0_0_0/0.04)] transition-[border-color,box-shadow,background-color] duration-150 placeholder:text-fg-subtle hover:border-border-strong focus:border-accent/60 focus:outline-none focus:ring-3 focus:ring-accent/15 disabled:cursor-not-allowed disabled:opacity-50'

export const fieldInvalid = 'border-danger/60 hover:border-danger/70 focus:border-danger/70 focus:ring-danger/15'

export interface InputProps extends Omit<ComponentProps<'input'>, 'size'> {
  leftIcon?: ReactNode
  rightSlot?: ReactNode
  invalid?: boolean
  size?: 'sm' | 'md' | 'lg'
  wrapperClassName?: string
}

const heights = { sm: 'h-7 text-xs', md: 'h-8.5', lg: 'h-10 text-base' }

export function Input({ leftIcon, rightSlot, invalid, size = 'md', className, wrapperClassName, ...props }: InputProps) {
  return (
    <div className={cn('relative flex w-full items-center', wrapperClassName)}>
      {leftIcon && (
        <span className="pointer-events-none absolute left-2.5 flex text-fg-subtle [&_svg]:size-4">{leftIcon}</span>
      )}
      <input
        spellCheck={false}
        className={cn(fieldBase, heights[size], 'px-3', leftIcon && 'pl-8.5', rightSlot && 'pr-9', invalid && fieldInvalid, className)}
        aria-invalid={invalid || undefined}
        {...props}
      />
      {rightSlot && <span className="absolute right-1.5 flex items-center">{rightSlot}</span>}
    </div>
  )
}

export interface TextareaProps extends ComponentProps<'textarea'> {
  autoGrow?: boolean
  maxHeight?: number
  minHeight?: number
  invalid?: boolean
  ref?: Ref<HTMLTextAreaElement>
}

export function Textarea({
  autoGrow = false,
  maxHeight = 320,
  minHeight,
  invalid,
  className,
  value,
  ref,
  ...props
}: TextareaProps) {
  const inner = useRef<HTMLTextAreaElement | null>(null)
  const setRef = useCallback(
    (el: HTMLTextAreaElement | null) => {
      inner.current = el
      if (typeof ref === 'function') ref(el)
      else if (ref) (ref as { current: HTMLTextAreaElement | null }).current = el
    },
    [ref]
  )

  const measure = useCallback(() => {
    const el = inner.current
    if (!autoGrow || !el || el.clientWidth === 0) return
    el.style.height = 'auto'
    const next = Math.min(Math.max(el.scrollHeight, minHeight ?? 0), maxHeight)
    el.style.height = `${next}px`
    el.style.overflowY = el.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [autoGrow, maxHeight, minHeight])

  useLayoutEffect(() => {
    measure()
  }, [value, props.placeholder, measure])

  // Re-measure when the width changes (window resize, sidebar toggle, first layout while hidden).
  useLayoutEffect(() => {
    const el = inner.current
    if (!autoGrow || !el || typeof ResizeObserver === 'undefined') return
    let lastWidth = el.clientWidth
    const ro = new ResizeObserver(() => {
      if (el.clientWidth !== lastWidth) {
        lastWidth = el.clientWidth
        measure()
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [autoGrow, measure])

  return (
    <textarea
      ref={setRef}
      value={value}
      className={cn(fieldBase, 'block resize-none px-3 py-2 leading-relaxed', invalid && fieldInvalid, className)}
      aria-invalid={invalid || undefined}
      {...props}
    />
  )
}
