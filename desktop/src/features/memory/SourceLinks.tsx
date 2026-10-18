/**
 * Where a memory came from, exactly (#152): every source with the words it said, each opening what it points at
 * (the chat message, the email, the note). And how sure Sentient is of it.
 */
import { IconArrowUpRight, IconQuote } from '@tabler/icons-react'
import { useNavigate } from 'react-router'
import { getBridge } from '@/lib/bridge'
import type { Memory, MemorySourceLink, MemorySourceRef } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'
import { SourceChip } from './bits'

/** What opening a source does, or null when there's nothing to open. */
function opener(ref: MemorySourceRef | null, navigate: (to: string) => void): { label: string; open: () => void } | null {
  if (!ref) return null
  if (ref.kind === 'message' && ref.session_id) {
    const to = `/chat/${ref.session_id}${ref.message_id ? `?message=${encodeURIComponent(ref.message_id)}` : ''}`
    return { label: 'Open the chat', open: () => navigate(to) }
  }
  if (ref.url && /^(https?|obsidian):\/\//.test(ref.url)) {
    return { label: ref.kind === 'email' ? 'Open the email' : ref.kind === 'note' ? 'Open the note' : 'Open', open: () => void getBridge().openExternal(ref.url!) }
  }
  if (ref.session_id) return { label: 'Open the chat that read it', open: () => navigate(`/chat/${ref.session_id}`) }
  return null
}

function where(link: MemorySourceLink): string | null {
  const ref = link.ref
  if (!ref) return null
  if (ref.kind === 'email' && ref.title) return `Email: ${ref.title}`
  if (ref.kind === 'note') return ref.title ? `Note: ${ref.title}` : ref.path ?? null
  if (ref.kind === 'file' && ref.file) return ref.file
  if (ref.kind === 'app' && ref.title) return ref.title
  return null
}

export function SourceLinks({ memory, onForgetSource }: { memory: Memory; onForgetSource: (source: string) => void }) {
  const navigate = useNavigate()
  const links: MemorySourceLink[] =
    memory.sources && memory.sources.length > 0 ? memory.sources : [{ source: memory.source, ref: memory.ref ?? null, at: memory.created_at, own: true }]
  return (
    <ul className="space-y-2.5">
      {links.map((link) => {
        const go = opener(link.ref, navigate)
        const place = where(link)
        return (
          <li key={link.source} className="rounded-lg border border-border bg-sunken/40 px-3 py-2.5">
            <div className="flex items-center gap-2">
              <SourceChip source={link.source} className="text-sm text-fg" />
              <span className="shrink-0 text-xs text-fg-subtle">· {relativeTime(link.at)}</span>
              {go && (
                <button
                  type="button"
                  onClick={go.open}
                  className="ml-auto inline-flex shrink-0 items-center gap-0.5 text-xs text-accent-text underline-offset-2 hover:underline"
                >
                  {go.label} <IconArrowUpRight size={12} />
                </button>
              )}
            </div>
            {place && <div className="mt-1 truncate text-xs text-fg-muted">{place}</div>}
            {link.ref?.quote && (
              <p className="mt-1.5 flex gap-1.5 text-sm italic text-fg-muted">
                <IconQuote size={13} className="mt-0.5 shrink-0 text-fg-faint" />
                <span className="selectable">{link.ref.quote}</span>
              </p>
            )}
            {link.own && (
              <button
                type="button"
                onClick={() => onForgetSource(link.source)}
                className="mt-1.5 text-xs text-fg-subtle underline-offset-2 hover:text-danger hover:underline"
              >
                Forget all from this source
              </button>
            )}
          </li>
        )
      })}
    </ul>
  )
}

export function confidenceWords(c: number): string {
  if (c >= 0.9) return 'Very sure'
  if (c >= 0.7) return 'Fairly sure'
  return 'Not sure yet'
}

export function Confidence({ memory, className }: { memory: Memory; className?: string }) {
  const c = memory.confidence ?? 0.8
  const n = memory.sources?.length ?? 1
  return (
    <div className={cn('space-y-1.5', className)}>
      <div className="flex items-center justify-between text-sm">
        <span className="text-fg">{confidenceWords(c)}</span>
        <span className="tabular-nums text-fg-subtle">{Math.round(c * 100)}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-sunken" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(c * 100)} aria-label="How sure Sentient is">
        <div className={cn('h-full rounded-full', c >= 0.9 ? 'bg-success' : c >= 0.7 ? 'bg-accent' : 'bg-warning')} style={{ width: `${Math.round(c * 100)}%` }} />
      </div>
      <div className="text-xs text-fg-subtle">
        {n > 1 ? `${n} places told Sentient this.` : 'One place told Sentient this.'} It gets surer each time you or another source say it again.
      </div>
    </div>
  )
}
