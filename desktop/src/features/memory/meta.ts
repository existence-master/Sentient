/** Presentation metadata for memory: topic colors/icons, sources, expiry. */
import {
  IconBriefcase,
  IconCoin,
  IconFileText,
  IconHeartbeat,
  IconListCheck,
  IconMessageCircle,
  IconPalette,
  IconPencil,
  IconSparkles,
  IconTarget,
  IconUserCircle,
  IconUsers,
  IconBox,
  type Icon
} from '@tabler/icons-react'
import { MEMORY_TOPICS, type Memory } from '@/lib/types'
import { parseDate, relativeTime } from '@/lib/utils'

export interface TopicMeta {
  color: string
  icon: Icon
  short: string
}

export const TOPIC_META: Record<string, TopicMeta> = {
  'Personal Identity': { color: '#a78bfa', icon: IconUserCircle, short: 'Identity' },
  'Interests & Lifestyle': { color: '#f472b6', icon: IconPalette, short: 'Lifestyle' },
  'Work & Learning': { color: '#60a5fa', icon: IconBriefcase, short: 'Work' },
  'Health & Wellbeing': { color: '#34d399', icon: IconHeartbeat, short: 'Health' },
  'Relationships & Social Life': { color: '#fb923c', icon: IconUsers, short: 'Relationships' },
  Financial: { color: '#facc15', icon: IconCoin, short: 'Financial' },
  'Goals & Challenges': { color: '#22d3ee', icon: IconTarget, short: 'Goals' },
  Miscellaneous: { color: '#94a3b8', icon: IconBox, short: 'Misc' }
}

export const TOPIC_ORDER: string[] = [...MEMORY_TOPICS]

export function topicMeta(name: string | undefined): TopicMeta {
  return (name && TOPIC_META[name]) || TOPIC_META.Miscellaneous
}

export function primaryTopic(topics: string[] | undefined): string {
  return topics?.find((t) => TOPIC_META[t]) ?? 'Miscellaneous'
}

/** Text color for a topic pill that stays readable in both themes. */
export function topicText(color: string): string {
  return `color-mix(in oklab, ${color} 72%, var(--fg))`
}

export function topicTint(color: string, pct = 14): string {
  return `color-mix(in oklab, ${color} ${pct}%, transparent)`
}

// ---------------------------------------------------------------------------- sources
export interface SourceMeta {
  label: string
  icon: Icon
  description: string
}

export function sourceMeta(source: string): SourceMeta {
  if (source.startsWith('file:')) return { label: source.slice(5), icon: IconFileText, description: 'Imported document' }
  if (source.startsWith('task:')) return { label: 'Task', icon: IconListCheck, description: 'Learned while running a task' }
  switch (source) {
    case 'conversation':
      return { label: 'Conversation', icon: IconMessageCircle, description: 'Learned in chat' }
    case 'manual':
      return { label: 'Added by you', icon: IconPencil, description: 'You added this' }
    case 'onboarding':
      return { label: 'Onboarding', icon: IconSparkles, description: 'From your first-run setup' }
    default:
      return { label: source, icon: IconBox, description: source }
  }
}

// ---------------------------------------------------------------------------- expiry
export const EXPIRING_SOON_MS = 48 * 3600_000

export function expiresInMs(m: Pick<Memory, 'expires_at'>, now = Date.now()): number | null {
  const d = parseDate(m.expires_at)
  return d ? d.getTime() - now : null
}

export function countdown(ms: number): string {
  if (ms <= 0) return 'expired'
  const mins = Math.floor(ms / 60_000)
  if (mins < 60) return `${Math.max(1, mins)}m`
  const hours = Math.floor(mins / 60)
  if (hours < 48) return `${hours}h ${mins % 60}m`
  const days = Math.floor(hours / 24)
  return `${days}d ${hours % 24}h`
}

/** The engine keeps the previous text on UPDATE; not in the documented shape yet. */
export type MemoryWithHistory = Memory & { previous_content?: string | null }

/** Relative time only when it adds something ("3 days ago"); '' once it would just repeat the date. */
export function recentRelative(iso: string | null | undefined, now = Date.now()): string {
  const d = parseDate(iso)
  if (!d || Math.abs(now - d.getTime()) >= 7 * 86_400_000) return ''
  return relativeTime(iso, now)
}

export function dayKey(iso: string): string {
  const d = parseDate(iso)
  if (!d) return 'unknown'
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

export function dayLabel(key: string, now = new Date()): string {
  const [y, m, d] = key.split('-').map(Number)
  const date = new Date(y, m - 1, d)
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const diff = Math.round((today.getTime() - date.getTime()) / 86_400_000)
  if (diff === 0) return 'Today'
  if (diff === 1) return 'Yesterday'
  return date.toLocaleDateString(undefined, {
    weekday: diff < 7 ? 'long' : undefined,
    month: 'long',
    day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
  })
}
