import { IconCheck, IconCopy, IconFile, IconPhoto } from '@tabler/icons-react'
import { memo, useState } from 'react'
import { IconButton } from '@/components/ui'
import { api } from '@/lib/api'
import type { AttachmentView, UserMessageView } from '@/lib/chatFold'
import { cn, copyText, formatBytes } from '@/lib/utils'

const IMAGE_RE = /\.(png|jpe?g|gif|webp|bmp|svg)$/i

export function isImageAttachment(a: AttachmentView): boolean {
  return (a.mime ?? '').startsWith('image/') || IMAGE_RE.test(a.name)
}

export function AttachmentChip({ a, className }: { a: AttachmentView; className?: string }) {
  const image = isImageAttachment(a)
  const src = a.previewUrl ?? (image ? api.files.contentUrl(a.name) : undefined)
  const base = a.name.split('/').pop() ?? a.name
  if (image && src) {
    return (
      <div className={cn('overflow-hidden rounded-xl border border-border bg-sunken', className)} title={base}>
        <img src={src} alt={base} className="h-28 max-w-56 object-cover" />
      </div>
    )
  }
  return (
    <div className={cn('flex h-11 max-w-60 items-center gap-2.5 rounded-xl border border-border bg-surface px-3', className)} title={base}>
      <div className="flex size-7 shrink-0 items-center justify-center rounded-md bg-accent/10 text-accent-text">
        {image ? <IconPhoto size={15} /> : <IconFile size={15} />}
      </div>
      <div className="min-w-0">
        <div className="truncate text-xs font-medium text-fg">{base}</div>
        {a.size !== undefined && <div className="text-2xs text-fg-subtle">{formatBytes(a.size)}</div>}
      </div>
    </div>
  )
}

export const UserMessage = memo(function UserMessage({ message }: { message: UserMessageView }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className={cn('group/user flex flex-col items-end gap-1.5', message.pending && 'opacity-80')}>
      {message.attachments.length > 0 && (
        <div className="flex max-w-[85%] flex-wrap justify-end gap-1.5">
          {message.attachments.map((a) => (
            <AttachmentChip key={a.name} a={a} />
          ))}
        </div>
      )}
      {message.text.trim() && (
        <div className="selectable max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md border border-border bg-elevated px-4 py-2.5 text-md leading-relaxed text-fg shadow-soft">
          {message.text}
        </div>
      )}
      <div className="flex h-6 items-center opacity-0 transition-opacity group-hover/user:opacity-100">
        <IconButton
          size="xs"
          label={copied ? 'Copied' : 'Copy'}
          icon={copied ? <IconCheck size={13} /> : <IconCopy size={13} />}
          onClick={async () => {
            if (await copyText(message.text)) {
              setCopied(true)
              setTimeout(() => setCopied(false), 1400)
            }
          }}
        />
      </div>
    </div>
  )
})
