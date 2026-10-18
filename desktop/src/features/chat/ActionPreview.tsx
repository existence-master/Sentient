/**
 * What an action will send or change, the way a person reads it (#142): an email's recipients, subject and full
 * text; a message's chat; what a delete removes. The engine builds it from the call's own arguments, so it shows
 * exactly what goes out. Plain text only, never Markdown.
 */
import { IconCreditCard, IconMail, IconMessage, IconSend, IconTrash, IconTrashX, type Icon } from '@tabler/icons-react'
import type { ActionPreview as Preview } from '@/lib/types'
import { cn } from '@/lib/utils'

const ICONS: Record<string, Icon> = {
  email: IconMail,
  message: IconMessage,
  post: IconSend,
  delete: IconTrashX,
  trash: IconTrash,
  purchase: IconCreditCard
}

export function ActionPreview({ preview, className, bodyLines = 12 }: { preview: Preview; className?: string; bodyLines?: number }) {
  const Glyph = ICONS[preview.kind] ?? IconSend
  return (
    <div className={cn('overflow-hidden rounded-lg border border-border bg-surface', className)}>
      <div className="flex items-center gap-1.5 border-b border-border bg-sunken/40 px-3 py-1.5 text-2xs font-medium uppercase tracking-wide text-fg-subtle">
        <Glyph size={12} />
        {preview.title}
      </div>
      {preview.fields.length > 0 && (
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 px-3 py-2 text-sm">
          {preview.fields.map((f) => (
            <div key={f.label} className="contents">
              <dt className="text-fg-subtle">{f.label}</dt>
              <dd className="selectable min-w-0 break-words font-medium text-fg">{f.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {preview.body && (
        <div
          className={cn('selectable overflow-auto whitespace-pre-wrap break-words px-3 py-2 text-sm leading-relaxed text-fg', preview.fields.length > 0 && 'border-t border-border')}
          style={{ maxHeight: `${bodyLines * 1.6}em` }}
        >
          {preview.body}
        </div>
      )}
    </div>
  )
}
