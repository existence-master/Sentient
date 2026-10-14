/**
 * "Ready for you": a call a "Hand it to me" rule handed over instead of running (#140). Sentient prepared it; the
 * user takes the last step. Shows what was prepared, with Copy and, for an email, Open in your email app.
 */
import { IconCopy, IconHandFinger, IconMail } from '@tabler/icons-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui'
import { getBridge } from '@/lib/bridge'
import type { HandOff } from '@/lib/types'
import { ActionPreview } from './ActionPreview'

/** The result of a handed-off call, or null for any other result (also when it comes back as stored JSON text). */
export function asHandOff(result: unknown): HandOff | null {
  let value = result
  if (typeof value === 'string' && value.trim().startsWith('{')) {
    try {
      value = JSON.parse(value)
    } catch {
      return null
    }
  }
  if (value && typeof value === 'object' && !Array.isArray(value) && (value as { handed_off?: unknown }).handed_off === true) {
    return value as HandOff
  }
  return null
}

function plainText(h: HandOff): string {
  const fields = h.preview?.fields ?? h.details ?? []
  const lines = fields.map((f) => `${f.label}: ${f.value}`)
  const body = h.preview?.body?.trim()
  return [lines.join('\n'), body].filter(Boolean).join('\n\n')
}

/** A mailto: link for an email that fits in one (the email app opens with it filled in), else null. */
function mailto(h: HandOff): string | null {
  if (h.preview?.kind !== 'email') return null
  const field = (label: string) => h.preview?.fields.find((f) => f.label === label)?.value ?? ''
  const to = field('To')
  if (!to) return null
  const params = new URLSearchParams()
  if (field('Cc')) params.set('cc', field('Cc'))
  if (field('Subject')) params.set('subject', field('Subject'))
  if (h.preview.body) params.set('body', h.preview.body)
  const url = `mailto:${encodeURIComponent(to).replace(/%40/g, '@').replace(/%2C/g, ',')}?${params.toString().replace(/\+/g, '%20')}`
  return url.length <= 1800 ? url : null
}

export function HandOffCard({ handOff }: { handOff: HandOff }) {
  const what = (handOff.preview?.title ?? handOff.title ?? 'action').toLowerCase()
  const link = mailto(handOff)
  const text = plainText(handOff)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      toast.success('Copied')
    } catch {
      toast.error("Couldn't copy it")
    }
  }

  const open = () => {
    if (!link) return
    void getBridge().openExternal(link)
  }

  return (
    <div className="overflow-hidden rounded-xl border border-accent/30 bg-accent/[0.05] shadow-soft">
      <div className="flex items-start gap-3 px-4 pt-3.5">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/12 text-accent-text">
          <IconHandFinger size={17} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-fg">Ready for you: {what}</div>
          <div className="text-xs text-fg-muted">
            You set {handOff.label} to Hand it to me, so Sentient prepared it and didn't do it. The last step is yours.
          </div>
        </div>
      </div>
      {handOff.preview ? (
        <ActionPreview preview={handOff.preview} className="mx-4 mt-3" bodyLines={10} />
      ) : (
        (handOff.details?.length ?? 0) > 0 && (
          <dl className="mx-4 mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 rounded-lg border border-border bg-surface px-3 py-2 text-sm">
            {handOff.details?.map((d) => (
              <div key={d.label} className="contents">
                <dt className="text-fg-subtle">{d.label}</dt>
                <dd className="selectable min-w-0 whitespace-pre-wrap break-words font-medium text-fg">{d.value}</dd>
              </div>
            ))}
          </dl>
        )
      )}
      <div className="flex flex-wrap items-center gap-2 px-4 py-3">
        {link && (
          <Button size="sm" variant="primary" leftIcon={<IconMail size={14} />} onClick={open}>
            Open in your email app
          </Button>
        )}
        <Button size="sm" variant={link ? 'ghost' : 'primary'} leftIcon={<IconCopy size={14} />} disabled={!text} onClick={() => void copy()}>
          Copy
        </Button>
      </div>
    </div>
  )
}
