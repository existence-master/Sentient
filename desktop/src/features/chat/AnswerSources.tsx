/**
 * Sources under a reply (#174): the pages, emails, files and apps it used, as small chips, plus the memories it had
 * in mind. Recorded by the engine from tool results (docs/API.md §2 "Answer sources"), never the model's own claim.
 * A chip shows the excerpt Sentient read; "Open" goes to the page, email or file.
 */
import { IconArrowUpRight, IconBrain, IconFileText, IconMail, IconNotebook, IconPlugConnected, IconWorld, type Icon } from '@tabler/icons-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Button, Popover, PopoverContent, PopoverTrigger } from '@/components/ui'
import { getBridge } from '@/lib/bridge'
import type { AnswerSource, MemorySource } from '@/lib/types'
import { cn } from '@/lib/utils'
import { openOutputFile } from './files'
import { MemorySources } from './MemorySources'
import { toolMeta } from './toolMeta'

const KIND: Record<AnswerSource['kind'] | 'note', { icon: Icon; label: string }> = {
  web: { icon: IconWorld, label: 'Web page' },
  email: { icon: IconMail, label: 'Email' },
  file: { icon: IconFileText, label: 'File' },
  note: { icon: IconNotebook, label: 'Your notes' },
  app: { icon: IconPlugConnected, label: 'App' }
}

function host(url?: string): string {
  if (!url) return ''
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return ''
  }
}

/** A note's path inside its folder ("Projects/Kestrel.md"), from its full path and the folder's name. */
function notePath(path?: string, folder?: string): string {
  const parts = (path ?? '').split(/[\\/]/)
  const at = folder ? parts.lastIndexOf(folder) : -1
  return (at >= 0 ? parts.slice(at + 1) : parts.slice(-1)).join('/')
}

function chipText(s: AnswerSource): string {
  if (s.kind === 'web') return s.title && s.title !== host(s.url) ? s.title : host(s.url) || s.title
  return s.title
}

export function AnswerSources({
  sources,
  memorySources,
  assistantName
}: {
  sources?: AnswerSource[]
  memorySources?: MemorySource[]
  assistantName: string
}) {
  const [memoryOpen, setMemoryOpen] = useState(false)
  const items = sources ?? []
  const memories = memorySources ?? []
  if (!items.length && !memories.length) return null
  const n = memories.length

  return (
    <div className="text-sm">
      <div className="flex flex-wrap items-center gap-1.5" aria-label="Sources">
        {items.map((s, i) => (
          <SourceChip key={`${s.kind}:${s.url ?? s.file ?? s.title}:${i}`} source={s} />
        ))}
        {n > 0 && (
          <button
            type="button"
            onClick={() => setMemoryOpen((o) => !o)}
            aria-expanded={memoryOpen}
            className={cn(chipClass, memoryOpen && 'border-accent/40 text-fg')}
          >
            <IconBrain size={13} className="shrink-0 text-fg-subtle" />
            <span className="truncate">
              {n} {n === 1 ? 'memory' : 'memories'}
            </span>
          </button>
        )}
      </div>
      {n > 0 && <MemorySources sources={memories} assistantName={assistantName} open={memoryOpen} onOpenChange={setMemoryOpen} />}
    </div>
  )
}

const chipClass =
  'inline-flex h-7 max-w-[240px] items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 text-xs text-fg-muted transition-colors hover:border-border-strong hover:bg-elevated hover:text-fg'

function SourceChip({ source: s }: { source: AnswerSource }) {
  // a note from a folder the user added (#176): opens in Obsidian (obsidian:// link) or its own app
  const note = s.kind === 'file' && !!s.path
  const obsidian = note && !!s.url?.startsWith('obsidian://')
  const meta = (note ? KIND.note : KIND[s.kind]) ?? KIND.app
  const Icon = meta.icon
  // a note opens through the desktop app only (the browser can't reach files on this computer)
  const canOpen = note
    ? getBridge().isDesktop
    : (s.kind === 'web' || s.kind === 'email') ? !!s.url : s.kind === 'file' ? !!s.file : false
  const open = () => {
    if (note && s.path) {
      void getBridge()
        .openFolderFile({ path: s.path, url: s.url })
        .then((ok) => {
          if (!ok) toast.error("That note can't be opened", { description: 'It may have been moved, or its folder removed.' })
        })
    } else if (s.kind === 'file' && s.file) void openOutputFile(s.file)
    else if (s.url) void getBridge().openExternal(s.url)
  }
  const where = note
    ? [s.folder, notePath(s.path, s.folder)].filter(Boolean).join(' · ')
    : s.kind === 'web' ? host(s.url) : s.kind === 'email' ? s.from : s.kind === 'file' ? s.file : undefined
  const tools = s.kind === 'app' ? (s.tools?.length ? s.tools : [s.tool]) : []

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button type="button" className={chipClass} title={s.title}>
          <Icon size={13} className="shrink-0 text-fg-subtle" />
          <span className="truncate">{chipText(s)}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[340px] max-w-[90vw] space-y-2">
        <div className="flex items-center gap-1.5 text-2xs font-medium uppercase tracking-wider text-fg-subtle">
          <Icon size={12} />
          {meta.label}
        </div>
        <div className="selectable break-words text-sm font-medium text-fg">{s.title}</div>
        {where && <div className="selectable truncate text-xs text-fg-subtle">{where}</div>}
        {s.snippet && (
          <blockquote className="selectable max-h-40 overflow-y-auto border-l-2 border-border-strong pl-2.5 text-xs leading-relaxed text-fg-muted">
            {s.snippet}
          </blockquote>
        )}
        {tools.length > 0 && (
          <ul className="space-y-0.5 text-xs text-fg-muted">
            {tools.map((t) => (
              <li key={t}>{toolMeta(t).done}</li>
            ))}
          </ul>
        )}
        {canOpen && (
          <Button size="sm" variant="secondary" rightIcon={<IconArrowUpRight size={14} />} onClick={open}>
            {obsidian ? 'Open in Obsidian' : note ? 'Open note' : s.kind === 'email' ? 'Open email' : s.kind === 'file' ? 'Open file' : 'Open page'}
          </Button>
        )}
      </PopoverContent>
    </Popover>
  )
}
