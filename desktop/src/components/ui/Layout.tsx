import * as ScrollAreaPrimitive from '@radix-ui/react-scroll-area'
import { useCallback, useEffect, useRef, useState, type ReactNode, type Ref, type UIEvent } from 'react'
import { cn, initials } from '@/lib/utils'

// ---------------------------------------------------------------------------- Card
export function Card({ className, children, interactive, ...props }: React.ComponentProps<'div'> & { interactive?: boolean }) {
  return (
    <div
      className={cn(
        'rounded-xl border border-border bg-surface',
        interactive && 'transition-colors hover:border-border-strong hover:bg-elevated',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({
  icon,
  title,
  description,
  actions,
  className
}: {
  icon?: ReactNode
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex items-start gap-3 px-4 pt-4', className)}>
      {icon && (
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-fg-muted [&_svg]:size-4.5">
          {icon}
        </div>
      )}
      <div className="min-w-0 flex-1">
        <div className="text-sm font-semibold text-fg">{title}</div>
        {description && <div className="mt-0.5 text-sm text-fg-subtle">{description}</div>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
    </div>
  )
}

export function CardBody({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn('p-4', className)}>{children}</div>
}

// ---------------------------------------------------------------------------- Field / FormRow
export function Field({
  label,
  description,
  error,
  htmlFor,
  children,
  className,
  optional
}: {
  label?: ReactNode
  description?: ReactNode
  error?: ReactNode
  htmlFor?: string
  children: ReactNode
  className?: string
  optional?: boolean
}) {
  return (
    <div className={cn('space-y-1.5', className)}>
      {label && (
        <label htmlFor={htmlFor} className="flex items-baseline gap-1.5 text-sm font-medium text-fg">
          {label}
          {optional && <span className="text-xs font-normal text-fg-subtle">optional</span>}
        </label>
      )}
      {children}
      {error ? (
        <p className="text-xs text-danger">{error}</p>
      ) : (
        description && <p className="text-xs leading-relaxed text-fg-subtle">{description}</p>
      )}
    </div>
  )
}

/** Settings row: label and description on the left, control on the right. */
export function FormRow({
  label,
  description,
  error,
  htmlFor,
  children,
  className,
  stack,
  id
}: {
  label: ReactNode
  description?: ReactNode
  error?: ReactNode
  htmlFor?: string
  children: ReactNode
  className?: string
  /** Put the control under the label (wide controls). */
  stack?: boolean
  id?: string
}) {
  return (
    <div
      id={id}
      className={cn(
        'flex gap-x-8 gap-y-2.5 px-4 py-3.5',
        stack ? 'flex-col' : 'flex-col items-start sm:flex-row sm:items-center',
        className
      )}
    >
      <div className={cn('min-w-0', !stack && 'sm:flex-1')}>
        <label htmlFor={htmlFor} className="block text-sm font-medium text-fg">
          {label}
        </label>
        {description && <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">{description}</p>}
        {error && <p className="mt-1 text-xs text-danger">{error}</p>}
      </div>
      <div className={cn('flex min-w-0 items-center', stack ? 'w-full' : 'w-full shrink-0 sm:w-auto sm:max-w-[55%] sm:justify-end')}>
        {children}
      </div>
    </div>
  )
}

/** Group of FormRows inside a card with dividers. */
export function FormSection({
  title,
  description,
  actions,
  children,
  className,
  id
}: {
  title?: ReactNode
  description?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  id?: string
}) {
  return (
    <section id={id} className={cn('space-y-2.5', className)}>
      {(title || actions) && (
        <div className="flex items-end justify-between gap-4 px-1">
          <div>
            {title && <h3 className="text-sm font-semibold text-fg">{title}</h3>}
            {description && <p className="mt-0.5 text-xs text-fg-subtle">{description}</p>}
          </div>
          {actions}
        </div>
      )}
      <div className="divide-y divide-border rounded-xl border border-border bg-surface">{children}</div>
    </section>
  )
}

// ---------------------------------------------------------------------------- PageHeader
export function PageHeader({
  title,
  description,
  icon,
  actions,
  className,
  children
}: {
  title: ReactNode
  description?: ReactNode
  icon?: ReactNode
  actions?: ReactNode
  className?: string
  children?: ReactNode
}) {
  return (
    <header className={cn('flex flex-col gap-4 px-8 pb-5 pt-7', className)}>
      <div className="flex items-start gap-3.5">
        {icon && (
          <div className="flex size-10 shrink-0 items-center justify-center rounded-xl border border-border bg-elevated text-accent-text shadow-soft [&_svg]:size-5">
            {icon}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <h1 className="text-xl font-semibold tracking-tight text-fg">{title}</h1>
          {description && <p className="mt-1 max-w-2xl text-sm text-fg-muted">{description}</p>}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      </div>
      {children}
    </header>
  )
}

// ---------------------------------------------------------------------------- ScrollArea
export function ScrollArea({
  children,
  className,
  viewportClassName,
  viewportRef,
  onScroll,
  orientation = 'vertical'
}: {
  children: ReactNode
  className?: string
  viewportClassName?: string
  viewportRef?: Ref<HTMLDivElement>
  onScroll?: (e: UIEvent<HTMLDivElement>) => void
  orientation?: 'vertical' | 'horizontal' | 'both'
}) {
  return (
    <ScrollAreaPrimitive.Root type="scroll" scrollHideDelay={700} className={cn('relative overflow-hidden', className)}>
      <ScrollAreaPrimitive.Viewport ref={viewportRef} onScroll={onScroll} className={cn('size-full [&>div]:!block', viewportClassName)}>
        {children}
      </ScrollAreaPrimitive.Viewport>
      {orientation !== 'horizontal' && <ScrollBar orientation="vertical" />}
      {orientation !== 'vertical' && <ScrollBar orientation="horizontal" />}
      <ScrollAreaPrimitive.Corner />
    </ScrollAreaPrimitive.Root>
  )
}

function ScrollBar({ orientation }: { orientation: 'vertical' | 'horizontal' }) {
  return (
    <ScrollAreaPrimitive.Scrollbar
      orientation={orientation}
      className={cn(
        'z-10 flex touch-none select-none p-0.5 transition-opacity',
        orientation === 'vertical' ? 'w-2.5' : 'h-2.5 flex-col'
      )}
    >
      <ScrollAreaPrimitive.Thumb className="relative flex-1 rounded-full bg-border-strong hover:bg-fg-faint" />
    </ScrollAreaPrimitive.Scrollbar>
  )
}

// ---------------------------------------------------------------------------- SplitPane
export function SplitPane({
  left,
  right,
  defaultSize = 320,
  min = 220,
  max = 560,
  storageKey,
  className
}: {
  left: ReactNode
  right: ReactNode
  defaultSize?: number
  min?: number
  max?: number
  storageKey?: string
  className?: string
}) {
  const [size, setSize] = useState(() => {
    if (!storageKey) return defaultSize
    const saved = Number(localStorage.getItem(`split:${storageKey}`))
    return saved >= min && saved <= max ? saved : defaultSize
  })
  const dragging = useRef(false)
  const container = useRef<HTMLDivElement>(null)

  const onMove = useCallback(
    (e: PointerEvent) => {
      if (!dragging.current || !container.current) return
      const rect = container.current.getBoundingClientRect()
      setSize(Math.max(min, Math.min(max, e.clientX - rect.left)))
    },
    [min, max]
  )
  const stop = useCallback(() => {
    dragging.current = false
    document.body.style.cursor = ''
  }, [])

  useEffect(() => {
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', stop)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', stop)
    }
  }, [onMove, stop])

  useEffect(() => {
    if (storageKey) localStorage.setItem(`split:${storageKey}`, String(Math.round(size)))
  }, [size, storageKey])

  return (
    <div ref={container} className={cn('flex h-full min-h-0 w-full', className)}>
      <div style={{ width: size }} className="h-full min-h-0 shrink-0">
        {left}
      </div>
      <div
        role="separator"
        aria-orientation="vertical"
        onPointerDown={() => {
          dragging.current = true
          document.body.style.cursor = 'col-resize'
        }}
        onDoubleClick={() => setSize(defaultSize)}
        className="group relative w-px shrink-0 cursor-col-resize bg-border"
      >
        <div className="absolute inset-y-0 -left-1.5 -right-1.5 group-hover:bg-accent/20" />
      </div>
      <div className="h-full min-h-0 min-w-0 flex-1">{right}</div>
    </div>
  )
}

// ---------------------------------------------------------------------------- Avatar
export function Avatar({
  name,
  src,
  size = 28,
  className
}: {
  name?: string | null
  src?: string
  size?: number
  className?: string
}) {
  return (
    <span
      style={{ width: size, height: size, fontSize: Math.max(10, size * 0.38) }}
      className={cn(
        'inline-flex shrink-0 items-center justify-center overflow-hidden rounded-full border border-border bg-gradient-to-br from-active to-hover font-semibold text-fg-muted',
        className
      )}
    >
      {src ? <img src={src} alt={name ?? ''} className="size-full object-cover" /> : initials(name)}
    </span>
  )
}

export function Divider({ className, label }: { className?: string; label?: ReactNode }) {
  if (!label) return <div className={cn('h-px w-full bg-border', className)} />
  return (
    <div className={cn('flex items-center gap-3 text-xs text-fg-subtle', className)}>
      <div className="h-px flex-1 bg-border" />
      {label}
      <div className="h-px flex-1 bg-border" />
    </div>
  )
}
