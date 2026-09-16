/** Calendar entries: past runs, upcoming recurring occurrences and scheduled one-off runs. */
import type { Task } from '@/lib/types'
import { parseDate } from '@/lib/utils'
import { addDays, dayKey, parseDayKey, recurringOccurrences, runAtDate, zonedParts, zonedTime } from './schedule'

export type CalendarItemKind = 'run' | 'upcoming' | 'scheduled' | 'created'

export interface CalendarItem {
  key: string
  task: Task
  at: Date
  kind: CalendarItemKind
  /** Run status for runs, `pending` for upcoming, task status otherwise. */
  status: string
  runId?: string
}

export function calendarItems(tasks: Task[], from: Date, to: Date, now: Date): CalendarItem[] {
  const out: CalendarItem[] = []
  const inRange = (d: Date | null): d is Date => !!d && d >= from && d < to
  for (const task of tasks) {
    const s = task.schedule
    for (const run of task.runs) {
      const at = parseDate(run.execution_start_time ?? run.created_at)
      if (inRange(at)) out.push({ key: `${task.task_id}:${run.run_id}`, task, at, kind: 'run', status: run.status, runId: run.run_id })
    }
    if (s?.type === 'recurring' && task.enabled && ['active', 'processing'].includes(task.status)) {
      const created = parseDate(task.created_at) ?? from
      const start = new Date(Math.max(from.getTime(), now.getTime(), created.getTime()))
      for (const at of recurringOccurrences(s, start, to)) {
        out.push({ key: `${task.task_id}:next:${at.toISOString()}`, task, at, kind: 'upcoming', status: 'pending' })
      }
      continue
    }
    if (s?.type === 'triggered') continue
    if (!task.runs.length) {
      const scheduled = task.status === 'pending' || task.status === 'approval_pending' ? (parseDate(task.next_execution_at) ?? runAtDate(s)) : null
      if (scheduled) {
        if (inRange(scheduled)) out.push({ key: `${task.task_id}:scheduled`, task, at: scheduled, kind: 'scheduled', status: task.status })
      } else if (task.status !== 'archived') {
        const created = parseDate(task.created_at)
        if (inRange(created)) out.push({ key: `${task.task_id}:created`, task, at: created, kind: 'created', status: task.status })
      }
    }
  }
  return out.sort((a, b) => a.at.getTime() - b.at.getTime())
}

export function groupByDay(items: CalendarItem[], tz: string): Map<string, CalendarItem[]> {
  const map = new Map<string, CalendarItem[]>()
  for (const item of items) {
    const k = dayKey(item.at, tz)
    map.set(k, [...(map.get(k) ?? []), item])
  }
  return map
}

/** Start (inclusive) and end (exclusive) instants of a local day. */
export function dayRange(key: string, tz: string): { from: Date; to: Date } {
  const p = parseDayKey(key) ?? { year: 1970, month: 1, day: 1 }
  const n = parseDayKey(addDays(key, 1)) ?? p
  return { from: zonedTime(p.year, p.month, p.day, 0, 0, tz), to: zonedTime(n.year, n.month, n.day, 0, 0, tz) }
}

export function monthKeyOf(date: Date, tz: string): string {
  const p = zonedParts(date, tz)
  return `${p.year}-${String(p.month).padStart(2, '0')}`
}

export function shiftMonth(month: string, delta: number): string {
  const [y, m] = month.split('-').map(Number)
  const d = new Date(Date.UTC(y, m - 1 + delta, 1))
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}`
}

export function monthLabel(month: string): string {
  const [y, m] = month.split('-').map(Number)
  return new Date(Date.UTC(y, m - 1, 15)).toLocaleDateString(undefined, { month: 'long', year: 'numeric', timeZone: 'UTC' })
}
