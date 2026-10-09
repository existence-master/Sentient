import { IconCheck, IconHandStop, IconShieldQuestion, IconX } from '@tabler/icons-react'
import type { ReactNode } from 'react'
import { Spinner } from '@/components/ui'
import type { ToolCallView, ToolStatus } from '@/lib/chatFold'
import { cn, humanize, safeJsonParse, truncate } from '@/lib/utils'
import { toolMeta } from '../toolMeta'

/**
 * A plain label from a browser element ref. Models sometimes pass the whole snapshot line:
 * `[e4] button "Place order"` -> `Place order`; a bare `e4` has no label.
 */
export function refLabel(ref: unknown): string | null {
  if (typeof ref !== 'string') return null
  const s = ref.trim()
  const quoted = /["“]([^"”]{1,80})["”]/.exec(s)
  if (quoted) return quoted[1].trim()
  if (!s || /^\[?e\d+\]?$/i.test(s)) return null
  const rest = s
    .replace(/^\[?e\d+\]\s*/i, '')
    .replace(/^(button|link|textbox|searchbox|checkbox|combobox|menuitem|tab|option|radio|img|heading)\b\s*/i, '')
    .trim()
  return rest || null
}

/** What a tool call acts on, for "You declined: ..." (the approval's engine `target` wins). */
export function actionTarget(tool: ToolCallView, approval?: { target?: string | null } | null): string | null {
  const a = (tool.arguments ?? {}) as Record<string, unknown>
  const str = (v: unknown) => (typeof v === 'string' && v.trim() ? v.trim() : null)
  // a command's approval target is its folder; the command itself says more
  if (tool.name === 'terminal_run' && str(a.command)) return truncate(String(a.command), 80)
  if (approval?.target) return approval.target
  switch (tool.name) {
    case 'browser_click':
      return refLabel(a.ref) ?? str(a.text)
    case 'browser_type':
      return str(a.text) ? `Type “${truncate(String(a.text), 40)}”` : refLabel(a.ref)
    case 'browser_select':
      return str(a.option) ?? refLabel(a.ref)
    case 'browser_open':
      return str(a.url) ? `Open ${hostOf(String(a.url))}` : null
    case 'browser_press':
      return str(a.key) ? `Press ${a.key}` : null
    case 'execute_code':
      return str(a.purpose) ?? 'Running code'
    case 'delegate_task':
      return str(a.goal) ?? 'Asking a helper'
    case 'delegate_tasks':
      return 'Asking helpers'
    case 'device_take_photo':
      return 'Taking a photo'
    case 'device_capture_screen':
      return 'Looking at your screen'
    default:
      return null
  }
}

/** Neutral/amber "You declined: Place order". The action did not happen; this is not a failure. */
export function DeclinedText({ tool, approval, className }: { tool: ToolCallView; approval?: { target?: string | null } | null; className?: string }) {
  const target = actionTarget(tool, approval) ?? toolMeta(tool.name).running
  return (
    <span className={cn('truncate', className)}>
      <span className="text-warning">You declined:</span> <span className="text-fg-muted">{target}</span>
    </span>
  )
}

export function Shimmer({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={cn('animate-shimmer bg-clip-text text-transparent', className)}
      style={{
        backgroundImage: 'linear-gradient(90deg, var(--fg-subtle) 30%, var(--fg) 50%, var(--fg-subtle) 70%)',
        backgroundSize: '200% 100%'
      }}
    >
      {children}
    </span>
  )
}

export function StatusGlyph({ status, size = 14 }: { status: ToolStatus | 'cancelled'; size?: number }) {
  if (status === 'running') return <Spinner size={size - 1} className="text-fg-subtle" />
  if (status === 'awaiting_approval') return <IconShieldQuestion size={size} className="text-warning" />
  if (status === 'error') return <IconX size={size} className="text-danger" />
  if (status === 'denied') return <IconHandStop size={size} className="text-warning" />
  if (status === 'cancelled') return <IconX size={size} className="text-fg-subtle" />
  return <IconCheck size={size} className="text-success" />
}

/** Tool results arrive as objects (live) or JSON strings (history). */
export function asRecord<T extends object>(value: unknown): T | null {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value as T
  if (typeof value === 'string') {
    const v = value.trim()
    if (v.startsWith('{')) {
      const parsed = safeJsonParse<unknown>(v, null)
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed as T
    }
  }
  return null
}

export function hostOf(url: string | undefined | null): string {
  if (!url) return ''
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

export function prettyUrl(url: string | undefined | null, max = 64): string {
  if (!url) return ''
  try {
    const u = new URL(url)
    const s = `${u.hostname.replace(/^www\./, '')}${u.pathname === '/' ? '' : u.pathname}`
    return s.length > max ? `${s.slice(0, max - 1)}…` : s
  } catch {
    return url
  }
}

/** A flat object of short primitive values renders as small stat tiles; anything else returns null. */
export function StatTiles({ value }: { value: unknown }) {
  const rec = asRecord<Record<string, unknown>>(value)
  if (!rec) return null
  const entries = Object.entries(rec)
  if (!entries.length || entries.length > 6) return null
  if (!entries.every(([, v]) => ['string', 'number', 'boolean'].includes(typeof v) && String(v).length <= 40)) return null
  return (
    <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(entries.length, 3)}, minmax(0, 1fr))` }}>
      {entries.map(([k, v]) => (
        <div key={k} className="min-w-0 rounded-lg border border-border bg-elevated/60 px-3 py-2">
          <div className="truncate text-2xs text-fg-subtle">{humanize(k)}</div>
          <div className="mt-0.5 truncate text-sm font-semibold tabular-nums text-fg">{String(v)}</div>
        </div>
      ))}
    </div>
  )
}
