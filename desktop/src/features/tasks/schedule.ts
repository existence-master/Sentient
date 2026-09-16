/**
 * Schedule presentation and math, done in the schedule's own timezone.
 *
 * Mirrors sentient/tasks/schedule.py: naive `run_at` values are wall-clock times in the
 * schedule's timezone, recurring times are HH:MM in that zone, weekly uses day names.
 */
import { intervalMinutes, isIntervalSchedule } from '@/lib/leap/types-b'
import type { RecurringSchedule, Task, TaskSchedule, TriggeredSchedule, Weekday } from '@/lib/types'
import { detectTimezone, humanize, parseDate } from '@/lib/utils'

export const WEEKDAYS: Weekday[] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

// ---------------------------------------------------------------------------- timezones
const validZones = new Map<string, string>()

/** A zone Intl accepts. 'auto' -> system zone, 'UTC+05:30' -> '+05:30', unknown -> 'UTC'. */
export function resolveZone(tz: string | null | undefined): string {
  const raw = (tz ?? '').trim()
  const key = raw || 'auto'
  const cached = validZones.get(key)
  if (cached) return cached
  let candidate = key.toLowerCase() === 'auto' ? detectTimezone() : raw
  const offset = /^(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?$/i.exec(candidate)
  if (offset) candidate = `${offset[1]}${offset[2].padStart(2, '0')}:${offset[3] ?? '00'}`
  if (/^(utc|gmt|z)$/i.test(candidate)) candidate = 'UTC'
  let zone = 'UTC'
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: candidate })
    zone = candidate
  } catch {
    zone = 'UTC'
  }
  validZones.set(key, zone)
  return zone
}

export interface ZonedParts {
  year: number
  month: number // 1-12
  day: number
  hour: number
  minute: number
  weekday: number // 0 = Monday
}

const partsFormatters = new Map<string, Intl.DateTimeFormat>()

export function zonedParts(date: Date, tz: string): ZonedParts {
  const zone = resolveZone(tz)
  let f = partsFormatters.get(zone)
  if (!f) {
    f = new Intl.DateTimeFormat('en-US', {
      timeZone: zone,
      hourCycle: 'h23',
      year: 'numeric',
      month: 'numeric',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric',
      weekday: 'short'
    })
    partsFormatters.set(zone, f)
  }
  const out: Record<string, string> = {}
  for (const p of f.formatToParts(date)) out[p.type] = p.value
  const wd = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].indexOf(out.weekday)
  return {
    year: Number(out.year),
    month: Number(out.month),
    day: Number(out.day),
    hour: Number(out.hour) % 24,
    minute: Number(out.minute),
    weekday: wd < 0 ? 0 : wd
  }
}

/** UTC instant of a wall-clock time in `tz`. */
export function zonedTime(year: number, month: number, day: number, hour: number, minute: number, tz: string): Date {
  const wall = Date.UTC(year, month - 1, day, hour, minute)
  let guess = wall
  for (let i = 0; i < 3; i++) {
    const p = zonedParts(new Date(guess), tz)
    const seen = Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute)
    const diff = wall - seen
    if (diff === 0) break
    guess += diff
  }
  return new Date(guess)
}

/** "YYYY-MM-DD" of an instant in `tz`. */
export function dayKey(date: Date, tz: string): string {
  const p = zonedParts(date, tz)
  return `${p.year}-${String(p.month).padStart(2, '0')}-${String(p.day).padStart(2, '0')}`
}

export function parseDayKey(key: string): { year: number; month: number; day: number } | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(key)
  return m ? { year: Number(m[1]), month: Number(m[2]), day: Number(m[3]) } : null
}

/** Pure calendar arithmetic on day keys (no timezone involved). */
export function addDays(key: string, days: number): string {
  const p = parseDayKey(key)
  if (!p) return key
  const d = new Date(Date.UTC(p.year, p.month - 1, p.day + days))
  return d.toISOString().slice(0, 10)
}

export function weekdayOfKey(key: string): number {
  const p = parseDayKey(key)
  if (!p) return 0
  return (new Date(Date.UTC(p.year, p.month - 1, p.day)).getUTCDay() + 6) % 7
}

export function formatDayKey(key: string, opts: Intl.DateTimeFormatOptions): string {
  const p = parseDayKey(key)
  if (!p) return key
  return new Date(Date.UTC(p.year, p.month - 1, p.day, 12)).toLocaleDateString(undefined, { ...opts, timeZone: 'UTC' })
}

export function formatInZone(date: Date, tz: string, opts: Intl.DateTimeFormatOptions): string {
  return date.toLocaleString(undefined, { ...opts, timeZone: resolveZone(tz) })
}

/** "09:00" -> "9:00 AM" (locale aware). */
export function formatClock(time: string | undefined): string {
  const m = /^(\d{1,2})(?::(\d{2}))?/.exec(time ?? '')
  const h = m ? Math.min(23, Number(m[1])) : 9
  const min = m?.[2] ? Math.min(59, Number(m[2])) : 0
  return new Date(Date.UTC(2000, 0, 1, h, min)).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit', timeZone: 'UTC' })
}

export function parseClock(time: string | undefined): { hour: number; minute: number } {
  const m = /^(\d{1,2})(?::(\d{2}))?/.exec(time ?? '')
  return { hour: m ? Math.min(23, Number(m[1])) : 9, minute: m?.[2] ? Math.min(59, Number(m[2])) : 0 }
}

/** A once-schedule's run time as a Date (naive values are in the schedule's zone). */
export function runAtDate(schedule: TaskSchedule | null | undefined): Date | null {
  if (!schedule || schedule.type !== 'once' || !schedule.run_at) return null
  const text = schedule.run_at.trim()
  if (/[zZ]|[+-]\d\d:?\d\d$/.test(text)) return parseDate(text)
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{1,2}):(\d{2}))?/.exec(text)
  if (!m) return null
  return zonedTime(Number(m[1]), Number(m[2]), Number(m[3]), Number(m[4] ?? 9), Number(m[5] ?? 0), schedule.timezone ?? 'auto')
}

/** Date -> naive "YYYY-MM-DDTHH:MM" in `tz` (for <input type=datetime-local> and run_at). */
export function toNaiveLocal(date: Date, tz: string): string {
  const p = zonedParts(date, tz)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${p.year}-${pad(p.month)}-${pad(p.day)}T${pad(p.hour)}:${pad(p.minute)}`
}

// ---------------------------------------------------------------------------- plain language
function joinList(items: string[]): string {
  if (items.length <= 1) return items[0] ?? ''
  try {
    return new Intl.ListFormat(undefined, { style: 'long', type: 'conjunction' }).format(items)
  } catch {
    return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
  }
}

export function normalizeDays(days: RecurringSchedule['days']): Weekday[] {
  const out = new Set<Weekday>()
  for (const d of days ?? []) {
    const key = String(d).trim().toLowerCase()
    if (key === 'weekday' || key === 'weekdays') WEEKDAYS.slice(0, 5).forEach((w) => out.add(w))
    else if (key === 'weekend' || key === 'weekends') WEEKDAYS.slice(5).forEach((w) => out.add(w))
    else {
      const hit = WEEKDAYS.find((w) => key.length >= 2 && w.toLowerCase().startsWith(key.slice(0, 3)))
      if (hit) out.add(hit)
    }
  }
  return WEEKDAYS.filter((w) => out.has(w))
}

export function describeDays(days: Weekday[]): string {
  if (days.length === 7) return 'Every day'
  if (days.length === 5 && WEEKDAYS.slice(0, 5).every((d) => days.includes(d))) return 'Every weekday'
  if (days.length === 2 && days.includes('Saturday') && days.includes('Sunday')) return 'Every weekend'
  if (!days.length) return 'Every Monday'
  return `Every ${joinList(days)}`
}

export function zoneLabel(tz: string | undefined): string {
  const z = !tz || tz === 'auto' ? detectTimezone() : tz
  return z
}

export const SOURCE_LABELS: Record<string, string> = {
  gmail: 'Gmail',
  gcalendar: 'Google Calendar',
  slack: 'Slack',
  github: 'GitHub',
  notion: 'Notion',
  discord: 'Discord',
  whatsapp: 'WhatsApp',
  trello: 'Trello',
  webhook: 'Webhook'
}

export const EVENT_LABELS: Record<string, string> = {
  new_email: 'a new email arrives',
  new_event: 'a new calendar event is added',
  updated_event: 'a calendar event changes',
  new_message: 'a new message arrives',
  new_issue: 'a new issue is opened',
  new_pull_request: 'a new pull request is opened'
}

export const sourceLabel = (s: string | undefined) => (s ? (SOURCE_LABELS[s] ?? humanize(s)) : 'an app')

export function eventLabel(source: string | undefined, event: string | undefined): string {
  if (event && EVENT_LABELS[event]) return EVENT_LABELS[event]
  if (event) return `${humanize(event).toLowerCase()} happens`
  return `something happens in ${sourceLabel(source)}`
}

export interface ScheduleText {
  /** Main sentence, e.g. "Every weekday at 9:00 AM". */
  text: string
  /** Timezone shown alongside, when relevant. */
  zone?: string
  /** Extra line (filter rules for triggered tasks). */
  detail?: string
}

export function describeSchedule(schedule: TaskSchedule | null | undefined, opts: { swarm?: boolean } = {}): ScheduleText {
  if (opts.swarm && !schedule) return { text: 'Starts right away with parallel agents' }
  if (!schedule) return { text: 'Runs once, right after you approve it' }
  if (schedule.type === 'once') {
    const at = runAtDate(schedule)
    if (!at) return { text: 'Runs once, right after you approve it' }
    const tz = schedule.timezone ?? 'auto'
    const date = formatInZone(at, tz, { weekday: 'short', month: 'short', day: 'numeric' })
    const time = formatInZone(at, tz, { hour: 'numeric', minute: '2-digit' })
    return { text: `Once on ${date} at ${time}`, zone: zoneLabel(tz) }
  }
  if (isIntervalSchedule(schedule)) return { text: everyMinutesText(intervalMinutes(schedule)) }
  if (schedule.type === 'recurring') {
    const lead = schedule.frequency === 'daily' ? 'Every day' : describeDays(normalizeDays(schedule.days))
    return { text: `${lead} at ${formatClock(schedule.time)}`, zone: zoneLabel(schedule.timezone) }
  }
  const t = schedule as TriggeredSchedule
  const rules = filterRules(t.filter)
  const detail = rules.rules.length ? rules.rules.map((r) => r.text).join(rules.mode === 'any' ? ' or ' : ', ') : undefined
  if (t.source === 'webhook') return { text: 'When your webhook is called', detail }
  return { text: `When ${eventLabel(t.source, t.event)} in ${sourceLabel(t.source)}`, detail }
}

/** "Every 15 minutes", "Every hour", "Every 6 hours". */
export function everyMinutesText(minutes: number): string {
  if (minutes % 1440 === 0) return minutes === 1440 ? 'Once a day' : `Every ${minutes / 1440} days`
  if (minutes % 60 === 0) return minutes === 60 ? 'Every hour' : `Every ${minutes / 60} hours`
  return `Every ${minutes} minutes`
}

export function scheduleSentence(schedule: TaskSchedule | null | undefined, opts: { swarm?: boolean } = {}): string {
  const d = describeSchedule(schedule, opts)
  let s = d.text
  if (d.detail) s += ` matching ${d.detail}`
  if (d.zone) s += ` (${d.zone})`
  return s
}

// ---------------------------------------------------------------------------- trigger filter rules
export type RuleOp = 'is' | 'is_not' | 'contains' | 'matches' | 'one_of' | 'none_of'

export const RULE_OPS: Array<{ value: RuleOp; label: string; dsl: string }> = [
  { value: 'is', label: 'is', dsl: '$eq' },
  { value: 'is_not', label: 'is not', dsl: '$ne' },
  { value: 'contains', label: 'contains', dsl: '$contains' },
  { value: 'one_of', label: 'is one of', dsl: '$in' },
  { value: 'none_of', label: 'is none of', dsl: '$nin' },
  { value: 'matches', label: 'matches pattern', dsl: '$regex' }
]

export interface FilterRule {
  field: string
  op: RuleOp
  value: string
  text: string
}

export const FIELD_LABELS: Record<string, string> = {
  from: 'sender',
  sender_email: 'sender email',
  to: 'recipient',
  subject: 'subject',
  snippet: 'preview',
  body: 'body',
  labels: 'labels',
  summary: 'title',
  description: 'description',
  location: 'location',
  attendees: 'attendees',
  organizer_email: 'organizer'
}

export const SOURCE_FIELDS: Record<string, string[]> = {
  gmail: ['from', 'to', 'subject', 'body', 'labels'],
  gcalendar: ['summary', 'description', 'location', 'attendees', 'organizer_email'],
  webhook: ['name', 'body']
}

const fieldLabel = (f: string) => FIELD_LABELS[f] ?? humanize(f).toLowerCase()

function valueText(v: unknown): string {
  if (Array.isArray(v)) return joinList(v.map((x) => String(x))).replace(/ and /, ' or ')
  if (typeof v === 'string') return v
  return JSON.stringify(v)
}

function ruleText(field: string, op: RuleOp, value: unknown): string {
  const label = fieldLabel(field)
  const v = valueText(value)
  switch (op) {
    case 'is':
      return `${label} is ${v}`
    case 'is_not':
      return `${label} is not ${v}`
    case 'contains':
      return `${label} contains “${v}”`
    case 'matches':
      return `${label} matches /${v}/`
    case 'one_of':
      return `${label} is ${v}`
    case 'none_of':
      return `${label} is not ${v}`
  }
}

const DSL_TO_OP: Record<string, RuleOp> = { $eq: 'is', $ne: 'is_not', $contains: 'contains', $regex: 'matches', $in: 'one_of', $nin: 'none_of' }

export interface FilterRules {
  mode: 'all' | 'any'
  rules: FilterRule[]
  /** The filter uses constructs the simple editor can't represent ($not, nested groups…). */
  complex: boolean
}

/** Flatten a v2 filter object into readable rules. */
export function filterRules(filter: Record<string, unknown> | undefined | null): FilterRules {
  const out: FilterRules = { mode: 'all', rules: [], complex: false }
  if (!filter || typeof filter !== 'object') return out
  const visit = (cond: Record<string, unknown>, depth: number) => {
    for (const [key, sub] of Object.entries(cond)) {
      if (key === '$or' || key === '$and') {
        if (!Array.isArray(sub)) continue
        if (depth > 0 || Object.keys(cond).length > 1) out.complex = true
        if (key === '$or') out.mode = 'any'
        sub.forEach((c) => c && typeof c === 'object' && visit(c as Record<string, unknown>, depth + 1))
      } else if (key === '$not') {
        out.complex = true
        out.rules.push({ field: '', op: 'is_not', value: JSON.stringify(sub), text: `not (${filterRules(sub as Record<string, unknown>).rules.map((r) => r.text).join(', ')})` })
      } else if (sub && typeof sub === 'object' && !Array.isArray(sub)) {
        for (const [op, v] of Object.entries(sub as Record<string, unknown>)) {
          const mapped = DSL_TO_OP[op]
          if (!mapped) {
            out.complex = true
            continue
          }
          out.rules.push({ field: key, op: mapped, value: Array.isArray(v) ? v.join(', ') : String(v), text: ruleText(key, mapped, v) })
        }
      } else {
        out.rules.push({ field: key, op: 'is', value: String(sub), text: ruleText(key, 'is', sub) })
      }
    }
  }
  visit(filter, 0)
  return out
}

/** Build a filter object from simple rules. */
export function rulesToFilter(rules: Array<Pick<FilterRule, 'field' | 'op' | 'value'>>, mode: 'all' | 'any'): Record<string, unknown> {
  const conds = rules
    .filter((r) => r.field.trim() && r.value.trim())
    .map((r) => {
      const dsl = RULE_OPS.find((o) => o.value === r.op)?.dsl ?? '$eq'
      const value = dsl === '$in' || dsl === '$nin' ? r.value.split(',').map((s) => s.trim()).filter(Boolean) : r.value.trim()
      return { [r.field.trim()]: dsl === '$eq' ? value : { [dsl]: value } } as Record<string, unknown>
    })
  if (!conds.length) return {}
  if (mode === 'any' && conds.length > 1) return { $or: conds }
  const merged: Record<string, unknown> = {}
  for (const c of conds) {
    const [k, v] = Object.entries(c)[0]
    if (k in merged) {
      const prev = merged[k]
      const a = prev && typeof prev === 'object' ? (prev as Record<string, unknown>) : { $eq: prev }
      const b = v && typeof v === 'object' ? (v as Record<string, unknown>) : { $eq: v }
      merged[k] = { ...a, ...b }
    } else merged[k] = v
  }
  return merged
}

// ---------------------------------------------------------------------------- next runs and occurrences
/** Recurring occurrences in [from, to), computed in the schedule's timezone. */
export function recurringOccurrences(schedule: RecurringSchedule, from: Date, to: Date, limit = 400): Date[] {
  if (isIntervalSchedule(schedule)) {
    // Every N minutes. Short intervals would flood the calendar, so below 6 hours only the first check of each day is listed.
    const step = intervalMinutes(schedule) * 60_000
    const out: Date[] = []
    let t = Math.ceil(from.getTime() / step) * step
    let lastDay = ''
    while (t < to.getTime() && out.length < limit) {
      const d = new Date(t)
      const day = dayKey(d, (schedule as { timezone?: string }).timezone ?? 'auto')
      if (step >= 6 * 3_600_000 || day !== lastDay) out.push(d)
      lastDay = day
      t += step
    }
    return out
  }
  const tz = schedule.timezone ?? 'auto'
  const { hour, minute } = parseClock(schedule.time)
  const days = schedule.frequency === 'daily' ? WEEKDAYS : normalizeDays(schedule.days).length ? normalizeDays(schedule.days) : (['Monday'] as Weekday[])
  const wanted = new Set(days.map((d) => WEEKDAYS.indexOf(d)))
  const out: Date[] = []
  let key = addDays(dayKey(from, tz), -1)
  const endKey = addDays(dayKey(to, tz), 1)
  while (key <= endKey && out.length < limit) {
    if (wanted.has(weekdayOfKey(key))) {
      const p = parseDayKey(key)
      if (p) {
        const at = zonedTime(p.year, p.month, p.day, hour, minute, tz)
        if (at >= from && at < to) out.push(at)
      }
    }
    key = addDays(key, 1)
  }
  return out
}

export function nextRunDate(task: Pick<Task, 'next_execution_at' | 'schedule' | 'status' | 'enabled'>, now = new Date()): Date | null {
  if (!task.enabled) return null
  const stored = parseDate(task.next_execution_at)
  if (stored) return stored
  const s = task.schedule
  if (s?.type === 'recurring' && ['active', 'approval_pending', 'processing'].includes(task.status)) {
    if (isIntervalSchedule(s)) return new Date(now.getTime() + intervalMinutes(s) * 60_000)
    return recurringOccurrences(s, now, new Date(now.getTime() + 8 * 86_400_000), 1)[0] ?? null
  }
  if (s?.type === 'once' && task.status === 'pending') return runAtDate(s)
  return null
}

/** Relative "in 3 hours" / "tomorrow at 8:00 AM" phrasing for upcoming runs. */
export function upcomingPhrase(date: Date, tz: string, now = new Date()): string {
  const diff = date.getTime() - now.getTime()
  if (diff < 0) return 'due now'
  if (diff < 60 * 60_000) return `in ${Math.max(1, Math.round(diff / 60_000))} min`
  const time = formatInZone(date, tz, { hour: 'numeric', minute: '2-digit' })
  const today = dayKey(now, tz)
  const key = dayKey(date, tz)
  if (key === today) return `today at ${time}`
  if (key === addDays(today, 1)) return `tomorrow at ${time}`
  if (diff < 6 * 86_400_000) return `${formatInZone(date, tz, { weekday: 'long' })} at ${time}`
  return `${formatInZone(date, tz, { month: 'short', day: 'numeric' })} at ${time}`
}
