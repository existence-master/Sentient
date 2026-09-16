import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}

export const uid = (): string =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`

export const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

export function formatBytes(bytes: number | undefined | null, digits = 1): string {
  if (bytes === undefined || bytes === null || Number.isNaN(bytes)) return '-'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let v = bytes / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v >= 100 ? 0 : digits)} ${units[i]}`
}

export function formatNumber(n: number, compact = true): string {
  return new Intl.NumberFormat(undefined, compact ? { notation: 'compact', maximumFractionDigits: 1 } : {}).format(n)
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return ''
  if (ms < 1000) return `${Math.round(ms)} ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`
  const m = Math.floor(ms / 60_000)
  const s = Math.round((ms % 60_000) / 1000)
  return `${m}m ${s}s`
}

/** Backend timestamps are ISO-8601 UTC; tolerate missing zone designators. */
export function parseDate(iso: string | null | undefined): Date | null {
  if (!iso) return null
  const hasZone = /[zZ]|[+-]\d\d:?\d\d$/.test(iso)
  const d = new Date(hasZone ? iso : `${iso}Z`)
  return Number.isNaN(d.getTime()) ? null : d
}

const rtf = typeof Intl !== 'undefined' ? new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' }) : null

export function relativeTime(iso: string | null | undefined, now = Date.now()): string {
  const d = parseDate(iso)
  if (!d || !rtf) return ''
  const diff = (d.getTime() - now) / 1000
  const abs = Math.abs(diff)
  if (abs < 45) return 'just now'
  if (abs < 3600) return rtf.format(Math.round(diff / 60), 'minute')
  if (abs < 86_400) return rtf.format(Math.round(diff / 3600), 'hour')
  if (abs < 7 * 86_400) return rtf.format(Math.round(diff / 86_400), 'day')
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: abs > 300 * 86_400 ? 'numeric' : undefined })
}

export function formatTime(iso: string | null | undefined): string {
  const d = parseDate(iso)
  return d ? d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' }) : ''
}

export function formatDateTime(iso: string | null | undefined): string {
  const d = parseDate(iso)
  return d ? d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : ''
}

/** "Today" / "Yesterday" / "Previous 7 days" / "Older" buckets for chat lists. */
export function dateBucket(iso: string | null | undefined, now = new Date()): string {
  const d = parseDate(iso)
  if (!d) return 'Older'
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const t = d.getTime()
  if (t >= startOfToday) return 'Today'
  if (t >= startOfToday - 86_400_000) return 'Yesterday'
  if (t >= startOfToday - 7 * 86_400_000) return 'Previous 7 days'
  if (t >= startOfToday - 30 * 86_400_000) return 'Previous 30 days'
  return 'Older'
}

export function greeting(date = new Date()): string {
  const h = date.getHours()
  if (h < 5) return 'Good evening'
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}

export function truncate(s: string, max: number): string {
  return s.length > max ? `${s.slice(0, max - 1).trimEnd()}…` : s
}

export function safeJsonParse<T = unknown>(text: string | null | undefined, fallback: T): T {
  if (!text) return fallback
  try {
    return JSON.parse(text) as T
  } catch {
    return fallback
  }
}

/** snake_case / kebab-case / camelCase -> "Sentence case". */
export function humanize(key: string): string {
  const words = key
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .trim()
    .toLowerCase()
  return words.charAt(0).toUpperCase() + words.slice(1)
}

export function initials(name: string | null | undefined): string {
  const parts = (name ?? '').trim().split(/\s+/).filter(Boolean)
  if (!parts.length) return '?'
  return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : '')).toUpperCase()
}

export function isEditableTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  if (!el) return false
  return el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)
}

export const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)
export const modKey = isMac ? '⌘' : 'Ctrl'

/** Legacy IANA names some ICU builds still report. */
const TZ_ALIASES: Record<string, string> = {
  'Asia/Calcutta': 'Asia/Kolkata',
  'Asia/Saigon': 'Asia/Ho_Chi_Minh',
  'Asia/Katmandu': 'Asia/Kathmandu',
  'Asia/Rangoon': 'Asia/Yangon',
  'Europe/Kiev': 'Europe/Kyiv',
  'America/Buenos_Aires': 'America/Argentina/Buenos_Aires'
}

export function detectTimezone(): string {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
    return TZ_ALIASES[tz] ?? tz
  } catch {
    return 'UTC'
  }
}

export function listTimezones(): string[] {
  try {
    const fn = (Intl as unknown as { supportedValuesOf?: (k: string) => string[] }).supportedValuesOf
    if (fn) {
      const zones = fn('timeZone').map((z) => TZ_ALIASES[z] ?? z)
      return Array.from(new Set(zones)).sort()
    }
  } catch {
    /* older engines */
  }
  return ['UTC', 'Europe/London', 'Europe/Berlin', 'Asia/Kolkata', 'Asia/Tokyo', 'America/New_York', 'America/Los_Angeles']
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}
