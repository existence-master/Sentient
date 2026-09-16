import { IconCheck, IconCopy, IconExternalLink } from '@tabler/icons-react'
import { Fragment, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { getBridge } from '@/lib/bridge'
import { cn, copyText } from '@/lib/utils'

export interface ParsedInstructions {
  intro: string[]
  steps: string[]
  outro: string[]
}

/** Splits an integration's `instructions_md` into intro paragraphs, numbered steps and trailing notes. */
export function parseInstructions(md: string): ParsedInstructions {
  const out: ParsedInstructions = { intro: [], steps: [], outro: [] }
  let lastWasStep = false
  for (const raw of (md || '').split(/\r?\n/)) {
    const line = raw.trimEnd()
    if (!line.trim()) {
      lastWasStep = false
      continue
    }
    const m = /^\s*(\d+)[.)]\s+(.*)$/.exec(line)
    if (m) {
      out.steps.push(m[2])
      lastWasStep = true
    } else if (/^\s{2,}\S/.test(raw) && lastWasStep && out.steps.length) {
      out.steps[out.steps.length - 1] += ` ${line.trim()}`
    } else if (!out.steps.length) {
      out.intro.push(line.trim())
    } else {
      out.outro.push(line.trim())
      lastWasStep = false
    }
  }
  return out
}

export function openExternal(url: string) {
  void getBridge().openExternal(url)
}

function prettyUrl(url: string): string {
  try {
    const u = new URL(url)
    const path = u.pathname === '/' ? '' : u.pathname
    const s = `${u.hostname.replace(/^www\./, '')}${path}`
    return s.length > 42 ? `${s.slice(0, 41)}…` : s
  } catch {
    return url
  }
}

export function ExternalLinkChip({ url, label }: { url: string; label?: string }) {
  return (
    <button
      type="button"
      onClick={() => openExternal(url)}
      title={url}
      className="mx-0.5 inline-flex max-w-full items-center gap-1 rounded-md border border-accent/25 bg-accent/8 px-1.5 align-baseline text-xs font-medium text-accent-text transition-colors hover:border-accent/45 hover:bg-accent/14"
    >
      <span className="truncate">{label ?? prettyUrl(url)}</span>
      <IconExternalLink size={12} className="shrink-0" />
    </button>
  )
}

export function CopyChip({ value, className }: { value: string; className?: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      type="button"
      title="Copy"
      onClick={async () => {
        if (await copyText(value)) {
          setCopied(true)
          window.setTimeout(() => setCopied(false), 1400)
        } else toast.error("Couldn't copy to the clipboard")
      }}
      className={cn(
        'mx-0.5 inline-flex max-w-full items-center gap-1 rounded-md border border-border-strong bg-sunken px-1.5 align-baseline font-mono text-xs text-fg transition-colors hover:bg-active',
        className
      )}
    >
      <span className="break-all text-left">{value}</span>
      {copied ? <IconCheck size={12} className="shrink-0 text-success" /> : <IconCopy size={12} className="shrink-0 text-fg-subtle" />}
    </button>
  )
}

const TOKEN = /(\*\*[^*]+\*\*|`[^`]+`|https?:\/\/[^\s)<>]+)/g

/** Renders bold, `copyable code` and clickable external links inside one line of instructions. */
export function RichLine({ text }: { text: string }) {
  const parts: ReactNode[] = []
  let last = 0
  let key = 0
  for (const m of text.matchAll(TOKEN)) {
    const idx = m.index ?? 0
    if (idx > last) parts.push(<Fragment key={key++}>{text.slice(last, idx)}</Fragment>)
    const tok = m[0]
    if (tok.startsWith('**')) {
      parts.push(
        <strong key={key++} className="font-semibold text-fg">
          {tok.slice(2, -2)}
        </strong>
      )
    } else if (tok.startsWith('`')) {
      parts.push(<CopyChip key={key++} value={tok.slice(1, -1)} />)
    } else {
      let url = tok
      let trail = ''
      while (/[.,;:!?]$/.test(url)) {
        trail = url.slice(-1) + trail
        url = url.slice(0, -1)
      }
      parts.push(<ExternalLinkChip key={key++} url={url} />)
      if (trail) parts.push(<Fragment key={key++}>{trail}</Fragment>)
    }
    last = idx + tok.length
  }
  if (last < text.length) parts.push(<Fragment key={key++}>{text.slice(last)}</Fragment>)
  return <>{parts}</>
}

export function InstructionsGuide({ markdown, className }: { markdown: string; className?: string }) {
  const parsed = parseInstructions(markdown)
  if (!parsed.steps.length && !parsed.intro.length) return null
  return (
    <div className={cn('space-y-3', className)}>
      {parsed.intro.map((p, i) => (
        <p key={i} className="text-sm leading-relaxed text-fg-muted">
          <RichLine text={p} />
        </p>
      ))}
      {parsed.steps.length > 0 && (
        <ol className="relative space-y-0">
          {parsed.steps.map((s, i) => (
            <li key={i} className="relative flex gap-3 pb-3.5 last:pb-0">
              {i < parsed.steps.length - 1 && <span aria-hidden className="absolute bottom-0 left-[11px] top-7 w-px bg-border-strong" />}
              <span className="relative z-[1] mt-px flex size-6 shrink-0 items-center justify-center rounded-full border border-border-strong bg-elevated text-2xs font-semibold tabular-nums text-fg-muted">
                {i + 1}
              </span>
              <p className="min-w-0 flex-1 pt-0.5 text-sm leading-relaxed text-fg-muted">
                <RichLine text={s} />
              </p>
            </li>
          ))}
        </ol>
      )}
      {parsed.outro.map((p, i) => (
        <p key={i} className="rounded-lg border border-border bg-surface px-3 py-2 text-xs leading-relaxed text-fg-muted">
          <RichLine text={p} />
        </p>
      ))}
    </div>
  )
}
