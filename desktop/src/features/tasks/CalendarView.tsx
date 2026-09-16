/** Month grid: past runs, upcoming recurring occurrences and scheduled one-off runs. Click a day for its timeline. */
import { IconChevronLeft, IconChevronRight, IconRadar } from '@tabler/icons-react'
import { isScriptJob } from '@/lib/leap/types-b'
import { useMemo } from 'react'
import { Button, IconButton, Tooltip } from '@/components/ui'
import type { Task } from '@/lib/types'
import { cn } from '@/lib/utils'
import { calendarItems, groupByDay, monthKeyOf, monthLabel, shiftMonth, type CalendarItem } from './calendar'
import { displayName, runStatusMeta, statusMeta } from './meta'
import { addDays, dayKey, formatInZone, parseDayKey, weekdayOfKey, zonedTime } from './schedule'

const WEEKDAY_SHORT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

const chipTone: Record<string, string> = {
  success: 'bg-success/10 text-fg border-success/25 [--dot:var(--success)]',
  danger: 'bg-danger/10 text-fg border-danger/25 [--dot:var(--danger)]',
  warning: 'bg-warning/10 text-fg border-warning/25 [--dot:var(--warning)]',
  info: 'bg-info/10 text-fg border-info/25 [--dot:var(--info)]',
  accent: 'bg-accent/10 text-fg border-accent/25 [--dot:var(--accent)]',
  neutral: 'bg-active text-fg-muted border-border [--dot:var(--fg-faint)]'
}

export function itemTone(item: CalendarItem): string {
  if (item.kind === 'upcoming') return 'neutral'
  if (item.kind === 'run') return runStatusMeta(item.status).tone
  return statusMeta(item.status).tone
}

export function itemLabel(item: CalendarItem): string {
  if (item.kind === 'upcoming') return 'Upcoming run'
  if (item.kind === 'scheduled') return 'Scheduled'
  if (item.kind === 'created') return statusMeta(item.status).label
  return `Run · ${runStatusMeta(item.status).label}`
}

export function CalendarView({
  tasks,
  tz,
  now,
  month,
  onMonthChange,
  onOpenDay,
  onSelectTask
}: {
  tasks: Task[]
  tz: string
  now: Date
  month: string | null
  onMonthChange: (month: string) => void
  onOpenDay: (day: string) => void
  onSelectTask: (task: Task) => void
}) {
  const current = month && /^\d{4}-\d{2}$/.test(month) ? month : monthKeyOf(now, tz)
  const today = dayKey(now, tz)

  const { days, byDay } = useMemo(() => {
    const first = `${current}-01`
    const start = addDays(first, -weekdayOfKey(first))
    const [y, m] = current.split('-').map(Number)
    const daysInMonth = new Date(Date.UTC(y, m, 0)).getUTCDate()
    const weeks = Math.ceil((weekdayOfKey(first) + daysInMonth) / 7)
    const keys = Array.from({ length: weeks * 7 }, (_, i) => addDays(start, i))
    const s = parseDayKey(keys[0])!
    const e = parseDayKey(addDays(keys[keys.length - 1], 1))!
    const items = calendarItems(tasks, zonedTime(s.year, s.month, s.day, 0, 0, tz), zonedTime(e.year, e.month, e.day, 0, 0, tz), now)
    return { days: keys, byDay: groupByDay(items, tz) }
    // `now` changes every 30s; occurrences only need to move when the day changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tasks, current, tz, today])

  const weeks = days.length / 7
  const monthItems = days.filter((d) => d.startsWith(current)).reduce((n, d) => n + (byDay.get(d)?.length ?? 0), 0)

  return (
    <div className="flex h-full min-h-[560px] flex-col">
      <div className="mb-3 flex items-center gap-2">
        <h2 className="text-lg font-semibold tracking-tight text-fg">{monthLabel(current)}</h2>
        <span className="text-xs text-fg-subtle">{monthItems ? `${monthItems} item${monthItems === 1 ? '' : 's'}` : 'Nothing scheduled'}</span>
        <span className="flex-1" />
        <Legend />
        <div className="ml-3 flex items-center gap-1">
          <Button size="sm" variant="outline" onClick={() => onMonthChange(monthKeyOf(now, tz))} disabled={current === monthKeyOf(now, tz)}>
            Today
          </Button>
          <IconButton size="sm" label="Previous month" icon={<IconChevronLeft size={16} />} onClick={() => onMonthChange(shiftMonth(current, -1))} />
          <IconButton size="sm" label="Next month" icon={<IconChevronRight size={16} />} onClick={() => onMonthChange(shiftMonth(current, 1))} />
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-border bg-surface">
        <div className="grid grid-cols-7 border-b border-border bg-sunken/40">
          {WEEKDAY_SHORT.map((d) => (
            <div key={d} className="px-2.5 py-1.5 text-2xs font-medium uppercase tracking-wider text-fg-subtle">
              {d}
            </div>
          ))}
        </div>
        <div className="grid min-h-0 flex-1 grid-cols-7" style={{ gridTemplateRows: `repeat(${weeks}, minmax(104px, 1fr))` }}>
          {days.map((key, i) => {
            const items = byDay.get(key) ?? []
            const inMonth = key.startsWith(current)
            const isToday = key === today
            const past = key < today
            const shown = items.slice(0, items.length > 3 ? 2 : 3)
            return (
              <div
                key={key}
                role="button"
                tabIndex={0}
                aria-label={`${key}: ${items.length} items`}
                onClick={() => onOpenDay(key)}
                onKeyDown={(e) => e.key === 'Enter' && onOpenDay(key)}
                className={cn(
                  'group relative flex min-w-0 flex-col gap-1 overflow-hidden border-border p-1.5 text-left transition-colors hover:bg-hover',
                  i % 7 !== 6 && 'border-r',
                  i < days.length - 7 && 'border-b',
                  !inMonth && 'bg-sunken/35'
                )}
              >
                <div className="flex items-center justify-between px-0.5">
                  <span
                    className={cn(
                      'flex size-6 items-center justify-center rounded-full text-xs font-medium tabular-nums',
                      isToday ? 'bg-accent font-semibold text-accent-fg' : inMonth ? (past ? 'text-fg-subtle' : 'text-fg') : 'text-fg-faint'
                    )}
                  >
                    {parseDayKey(key)?.day}
                  </span>
                  {items.length > 0 && <span className="text-2xs text-fg-faint opacity-0 transition-opacity group-hover:opacity-100">Open day</span>}
                </div>
                {shown.map((item) => (
                  <Tooltip key={item.key} content={`${displayName(item.task)} · ${itemLabel(item)} · ${formatInZone(item.at, tz, { hour: 'numeric', minute: '2-digit' })}`}>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelectTask(item.task)
                      }}
                      className={cn(
                        'flex h-5.5 w-full min-w-0 items-center gap-1.5 rounded-md border px-1.5 text-left text-2xs transition-[filter] hover:brightness-110',
                        chipTone[itemTone(item)],
                        item.kind === 'upcoming' && 'border-dashed bg-transparent',
                        !inMonth && 'opacity-60'
                      )}
                    >
                      <span className="size-1.5 shrink-0 rounded-full bg-[var(--dot)]" />
                      <span className="shrink-0 tabular-nums text-fg-subtle">{formatInZone(item.at, tz, { hour: 'numeric', minute: '2-digit' }).replace(/\s?(AM|PM)$/i, (x) => x.trim().toLowerCase()[0])}</span>
                      {isScriptJob(item.task) && <IconRadar size={11} aria-label="Watcher" className="shrink-0 text-info" />}
                      <span className="min-w-0 truncate font-medium">{displayName(item.task)}</span>
                    </button>
                  </Tooltip>
                ))}
                {items.length > shown.length && (
                  <span className="px-1 text-2xs font-medium text-fg-subtle group-hover:text-fg-muted">+{items.length - shown.length} more</span>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function Legend() {
  return (
    <div className="hidden items-center gap-3 text-2xs text-fg-subtle @2xl:flex">
      <span className="flex items-center gap-1.5">
        <span className="size-2 rounded-full bg-success" /> Completed
      </span>
      <span className="flex items-center gap-1.5">
        <span className="size-2 rounded-full bg-danger" /> Failed
      </span>
      <span className="flex items-center gap-1.5">
        <span className="size-2 rounded-full border border-dashed border-fg-subtle" /> Upcoming
      </span>
    </div>
  )
}
