import * as DialogPrimitive from '@radix-ui/react-dialog'
import * as DropdownPrimitive from '@radix-ui/react-dropdown-menu'
import * as PopoverPrimitive from '@radix-ui/react-popover'
import * as TooltipPrimitive from '@radix-ui/react-tooltip'
import { IconX } from '@tabler/icons-react'
import { useState, type ComponentProps, type ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Button } from './Button'

// ---------------------------------------------------------------------------- Dialog
export interface DialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title?: ReactNode
  description?: ReactNode
  children?: ReactNode
  footer?: ReactNode
  size?: 'sm' | 'md' | 'lg' | 'xl'
  hideClose?: boolean
  className?: string
  /** Prevent closing on outside click (forms with unsaved input). */
  modalLock?: boolean
}

const dialogWidth = { sm: 'max-w-sm', md: 'max-w-lg', lg: 'max-w-2xl', xl: 'max-w-4xl' }

export function Dialog({ open, onOpenChange, title, description, children, footer, size = 'md', hideClose, className, modalLock }: DialogProps) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 grid place-items-center overflow-y-auto bg-black/55 p-6 backdrop-blur-[2px] data-[state=closed]:animate-overlay-out data-[state=open]:animate-overlay-in">
          <DialogPrimitive.Content
            onPointerDownOutside={modalLock ? (e) => e.preventDefault() : undefined}
            aria-describedby={description ? undefined : undefined}
            className={cn(
              'relative w-full rounded-2xl border border-border-strong bg-elevated shadow-pop outline-none data-[state=closed]:animate-pop-out data-[state=open]:animate-pop-in',
              dialogWidth[size],
              className
            )}
          >
            {(title || !hideClose) && (
              <div className="flex items-start gap-3 px-5 pb-1 pt-4.5">
                <div className="min-w-0 flex-1">
                  {title ? (
                    <DialogPrimitive.Title className="text-md font-semibold text-fg">{title}</DialogPrimitive.Title>
                  ) : (
                    <DialogPrimitive.Title className="sr-only">Dialog</DialogPrimitive.Title>
                  )}
                  {description ? (
                    <DialogPrimitive.Description className="mt-1 text-sm text-fg-muted">{description}</DialogPrimitive.Description>
                  ) : (
                    <DialogPrimitive.Description className="sr-only">{typeof title === 'string' ? title : 'Dialog'}</DialogPrimitive.Description>
                  )}
                </div>
                {!hideClose && (
                  <DialogPrimitive.Close
                    aria-label="Close"
                    className="-mr-1.5 -mt-0.5 flex size-7 items-center justify-center rounded-md text-fg-subtle hover:bg-hover hover:text-fg"
                  >
                    <IconX size={16} />
                  </DialogPrimitive.Close>
                )}
              </div>
            )}
            {children && <div className="px-5 py-3">{children}</div>}
            {footer && <div className="flex items-center justify-end gap-2 border-t border-border px-5 py-3">{footer}</div>}
          </DialogPrimitive.Content>
        </DialogPrimitive.Overlay>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}

// ---------------------------------------------------------------------------- ConfirmDialog
export interface ConfirmDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: ReactNode
  description?: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  tone?: 'danger' | 'primary'
  onConfirm: () => void | Promise<void>
}

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  tone = 'danger',
  onConfirm
}: ConfirmDialogProps) {
  const [busy, setBusy] = useState(false)
  return (
    <Dialog
      open={open}
      onOpenChange={(o) => !busy && onOpenChange(o)}
      title={title}
      description={description}
      size="sm"
      hideClose
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={busy}>
            {cancelLabel}
          </Button>
          <Button
            autoFocus
            variant={tone === 'danger' ? 'danger' : 'primary'}
            loading={busy}
            onClick={async () => {
              setBusy(true)
              try {
                await onConfirm()
                onOpenChange(false)
              } finally {
                setBusy(false)
              }
            }}
          >
            {confirmLabel}
          </Button>
        </>
      }
    />
  )
}

// ---------------------------------------------------------------------------- Sheet / Drawer
export interface SheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  side?: 'right' | 'left'
  title?: ReactNode
  description?: ReactNode
  actions?: ReactNode
  children?: ReactNode
  width?: number
  className?: string
}

export function Sheet({ open, onOpenChange, side = 'right', title, description, actions, children, width = 400, className }: SheetProps) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/35 data-[state=closed]:animate-overlay-out data-[state=open]:animate-overlay-in" />
        <DialogPrimitive.Content
          style={{ width }}
          className={cn(
            'fixed bottom-0 top-0 z-50 flex max-w-[calc(100vw-48px)] flex-col border-border-strong bg-surface shadow-pop outline-none',
            side === 'right'
              ? 'right-0 border-l data-[state=closed]:animate-sheet-out data-[state=open]:animate-sheet-in'
              : 'left-0 border-r data-[state=closed]:animate-sheet-left-out data-[state=open]:animate-sheet-left-in',
            className
          )}
        >
          <div className="flex h-14 shrink-0 items-center gap-2 border-b border-border px-4">
            <div className="min-w-0 flex-1">
              <DialogPrimitive.Title className="truncate text-md font-semibold">{title ?? 'Panel'}</DialogPrimitive.Title>
              <DialogPrimitive.Description className={description ? 'truncate text-xs text-fg-subtle' : 'sr-only'}>
                {description ?? 'Side panel'}
              </DialogPrimitive.Description>
            </div>
            {actions}
            <DialogPrimitive.Close aria-label="Close" className="flex size-7 items-center justify-center rounded-md text-fg-subtle hover:bg-hover hover:text-fg">
              <IconX size={16} />
            </DialogPrimitive.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}

// ---------------------------------------------------------------------------- Tooltip
export const TooltipProvider = TooltipPrimitive.Provider

export function Tooltip({
  content,
  children,
  side = 'top',
  shortcut,
  delayDuration,
  className
}: {
  content: ReactNode
  children: ReactNode
  side?: 'top' | 'bottom' | 'left' | 'right'
  shortcut?: string
  delayDuration?: number
  className?: string
}) {
  if (!content) return <>{children}</>
  return (
    <TooltipPrimitive.Root delayDuration={delayDuration}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          side={side}
          sideOffset={6}
          className={cn(
            'z-[60] flex max-w-xs items-center gap-2 rounded-md border border-border-strong bg-overlay px-2 py-1 text-xs text-fg shadow-pop data-[state=closed]:animate-overlay-out data-[state=delayed-open]:animate-pop-in',
            className
          )}
        >
          {content}
          {shortcut && <span className="text-fg-subtle">{shortcut}</span>}
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  )
}

// ---------------------------------------------------------------------------- Popover
export const Popover = PopoverPrimitive.Root
export const PopoverTrigger = PopoverPrimitive.Trigger
export const PopoverAnchor = PopoverPrimitive.Anchor
export const PopoverClose = PopoverPrimitive.Close

export function PopoverContent({ className, sideOffset = 8, ...props }: ComponentProps<typeof PopoverPrimitive.Content>) {
  return (
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Content
        sideOffset={sideOffset}
        className={cn(
          'z-50 rounded-xl border border-border-strong bg-overlay p-3 shadow-pop outline-none data-[state=closed]:animate-pop-out data-[state=open]:animate-pop-in',
          className
        )}
        {...props}
      />
    </PopoverPrimitive.Portal>
  )
}

// ---------------------------------------------------------------------------- DropdownMenu
export const DropdownMenu = DropdownPrimitive.Root
export const DropdownMenuTrigger = DropdownPrimitive.Trigger
export const DropdownMenuGroup = DropdownPrimitive.Group

export function DropdownMenuContent({ className, sideOffset = 6, ...props }: ComponentProps<typeof DropdownPrimitive.Content>) {
  return (
    <DropdownPrimitive.Portal>
      <DropdownPrimitive.Content
        sideOffset={sideOffset}
        className={cn(
          'z-50 min-w-44 rounded-xl border border-border-strong bg-overlay p-1 text-sm shadow-pop data-[state=closed]:animate-pop-out data-[state=open]:animate-pop-in',
          className
        )}
        {...props}
      />
    </DropdownPrimitive.Portal>
  )
}

export function DropdownMenuItem({
  className,
  icon,
  shortcut,
  danger,
  children,
  ...props
}: ComponentProps<typeof DropdownPrimitive.Item> & { icon?: ReactNode; shortcut?: string; danger?: boolean }) {
  return (
    <DropdownPrimitive.Item
      className={cn(
        'flex cursor-pointer select-none items-center gap-2 rounded-lg px-2.5 py-1.5 outline-none data-[disabled]:pointer-events-none data-[highlighted]:bg-active data-[disabled]:opacity-45 [&_svg]:size-4',
        danger ? 'text-danger' : 'text-fg',
        className
      )}
      {...props}
    >
      {icon && <span className={cn('flex', danger ? 'text-danger' : 'text-fg-muted')}>{icon}</span>}
      <span className="flex-1">{children}</span>
      {shortcut && <span className="text-xs text-fg-subtle">{shortcut}</span>}
    </DropdownPrimitive.Item>
  )
}

export function DropdownMenuSeparator({ className }: { className?: string }) {
  return <DropdownPrimitive.Separator className={cn('mx-1 my-1 h-px bg-border', className)} />
}

export function DropdownMenuLabel({ className, ...props }: ComponentProps<typeof DropdownPrimitive.Label>) {
  return <DropdownPrimitive.Label className={cn('px-2.5 pb-1 pt-1.5 text-2xs font-medium uppercase tracking-wide text-fg-subtle', className)} {...props} />
}
