/** Plain-language presentation of the user model: dimensions, confidence words, evidence kinds, dream stats. */
import {
  IconBan,
  IconBrain,
  IconBriefcase,
  IconCalendarEvent,
  IconCompass,
  IconHeartHandshake,
  IconMessage,
  IconMessageChatbot,
  IconMessages,
  IconSparkles,
  IconTarget,
  IconThumbUp,
  IconUsers,
  IconWorld,
  type Icon
} from '@tabler/icons-react'
import type { DreamStats, Insight, InsightDimension } from '@/lib/leap/types-b'
import { humanize } from '@/lib/utils'

export interface DimensionMeta {
  label: string
  /** One short line under the heading. */
  blurb: string
  icon: Icon
  color: string
  /** Placeholder when adding your own. */
  example: string
}

export const DIMENSION_META: Record<InsightDimension, DimensionMeta> = {
  preferences: { label: 'What you like', blurb: 'Tastes, tools and the way you like things done', icon: IconHeartHandshake, color: '#f59e0b', example: 'I like my morning briefing as a short bullet list' },
  communication: { label: 'How you communicate', blurb: 'Tone, length and style with different people', icon: IconMessageChatbot, color: '#60a5fa', example: 'Keep messages to my team short and casual' },
  goals: { label: 'What you are working toward', blurb: 'The things that matter to you right now', icon: IconTarget, color: '#34d399', example: 'Launch the beta before the end of the month' },
  routines: { label: 'Your rhythms', blurb: 'When you focus, rest and do the usual things', icon: IconCalendarEvent, color: '#a78bfa', example: 'I go for a run most mornings around 6' },
  relationships: { label: 'People in your life', blurb: 'Who matters and how you stay in touch', icon: IconUsers, color: '#f472b6', example: 'My sister lives in Bengaluru and we talk on weekends' },
  values: { label: 'What you care about', blurb: 'Principles that guide your choices', icon: IconCompass, color: '#2dd4bf', example: 'I would rather pay more for things that respect my privacy' },
  work_style: { label: 'How you work', blurb: 'Planning, deciding and getting things done', icon: IconBriefcase, color: '#fb923c', example: 'Show me a plan before you do anything big' },
  dislikes: { label: 'What to avoid', blurb: 'Things that bother you or waste your time', icon: IconBan, color: '#f87171', example: 'Please don’t schedule anything before 10 AM' },
  context: { label: 'Your situation', blurb: 'Where you are, what you do and what is going on', icon: IconWorld, color: '#94a3b8', example: 'I am moving to a new apartment in October' }
}

export const DIMENSION_ORDER = Object.keys(DIMENSION_META) as InsightDimension[]

export function dimensionMeta(d: string): DimensionMeta {
  return (DIMENSION_META as Record<string, DimensionMeta>)[d] ?? { label: humanize(d), blurb: '', icon: IconSparkles, color: '#94a3b8', example: '' }
}

export const tint = (color: string, pct: number) => `color-mix(in oklab, ${color} ${pct}%, transparent)`

/** How sure Sentient is, in words and as 1-4 filled steps. Never shows the raw number. */
export function certainty(i: Pick<Insight, 'confidence' | 'status' | 'source'>): { word: string; level: number; hint: string } {
  if (i.source === 'user') return { word: 'You told me', level: 4, hint: 'You added this yourself.' }
  if (i.status === 'confirmed') return { word: 'You confirmed this', level: 4, hint: 'You said this is right.' }
  if (i.status === 'disputed') return { word: 'Not sure anymore', level: 1, hint: 'Something you said or did doesn’t fit, so I am holding this loosely.' }
  const c = Number.isFinite(i.confidence) ? i.confidence : 0.5
  if (c >= 0.85) return { word: 'Confident', level: 4, hint: 'Many things you said point the same way.' }
  if (c >= 0.65) return { word: 'Fairly sure', level: 3, hint: 'A few clear signs point this way.' }
  if (c >= 0.45) return { word: 'Getting a sense', level: 2, hint: 'I have noticed this a couple of times.' }
  return { word: 'Just a hunch', level: 1, hint: 'I have only seen a hint of this so far.' }
}

export const EVIDENCE_META: Record<string, { label: string; icon: Icon }> = {
  message: { label: 'Something you said', icon: IconMessage },
  fact: { label: 'Something I remember', icon: IconBrain },
  summary: { label: 'From a past conversation', icon: IconMessages },
  feedback: { label: 'Your feedback', icon: IconThumbUp }
}

export const evidenceMeta = (kind: string) => EVIDENCE_META[kind] ?? { label: humanize(kind), icon: IconSparkles }

export const DREAM_STAT_LABELS: Array<{ key: keyof DreamStats; one: string; many: string }> = [
  { key: 'facts_reviewed', one: 'Looked over 1 memory', many: 'Looked over {n} memories' },
  { key: 'merged', one: 'Merged 1 duplicate', many: 'Merged {n} duplicates' },
  { key: 'contradictions_resolved', one: 'Settled 1 contradiction', many: 'Settled {n} contradictions' },
  { key: 'promoted', one: 'Kept 1 for the long term', many: 'Kept {n} for the long term' },
  { key: 'expired', one: 'Let 1 old note fade', many: 'Let {n} old notes fade' },
  { key: 'insights_updated', one: 'Updated 1 thing about you', many: 'Updated {n} things about you' }
]

export function dreamStatChips(stats: Partial<DreamStats> | undefined): Array<{ key: string; text: string }> {
  if (!stats) return []
  return DREAM_STAT_LABELS.flatMap(({ key, one, many }) => {
    const n = Number(stats[key] ?? 0)
    return n > 0 ? [{ key, text: n === 1 ? one : many.replace('{n}', n.toLocaleString()) }] : []
  })
}
