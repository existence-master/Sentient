/** Day timeline: every run and upcoming run of one day, placed by time in the user's timezone. */
import { IconArrowLeft, IconCalendarOff, IconChevronLeft, IconChevronRight } from '@tabler/icons-react'
import { useEffect, useMemo, useRef } from 'react'
import { Button, EmptyState, IconButton } from '@/components/ui'
import type { Task } from '@/lib/types'
import { cn } from '@/lib/utils'
import { calendarItems, dayRange, type CalendarItem } from './calendar'
import { itemLabel, itemTone } from './CalendarView'
import { displayName } from './meta'
import { KindTile } from './parts'
import { addDays, dayKey, formatDayKey, formatInZone, zonedParts } from './schedule'

const HOUR = 56
const BLOCK_MIN = 50

const blockTone: Record<string, string> = {
  success: 'border-l-success bg-success/8',
  danger: 'border-l-danger bg-danger/8',
  warning: 'border-l-warning bg-warning/8',
  info: 'border-l-info bg-info/8',
  accent: 'border-l-accent bg-accent/8',
  neutral: 'border-l-fg-faint bg-elevated'
}

export function DayView({
  tasks,
  tz,
  now,
  day,
  onDayChange,
  onBack,
  onSelectTask
}: {
  tasks: Task[]
  tz: string
  now: Date
  day: string
  onDayChange: (day: string) => void
  onBack: () => void
  onSelectTask: (task: Task) => void
}) {
  const scroller = useRef<HTMLDivElement>(null)
  const today = dayKey(now, tz)
  const items = useMemo(() => {
    const { from, to } = dayRange(day, tz)
    return calendarItems(tasks, from, to, now)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tasks, day, tz, today])

  const placed = useMemo(() => layout(items, tz), [items, tz])
  const runs = items.filter((i) => i.kind === 'run').length
  const upcoming = items.filter((i) => i.kind === 'upcoming' || i.kind === 'scheduled').length
  const created = items.length - runs - upcoming

  useEffect(() => {
    const first = placed.length ? Math.min(...placed.map((p) => p.minutes)) : null
    const nowParts = zonedParts(now, tz)
    const hour = first !== null ? Math.floor(first / 60) - 1 : day === today ? nowParts.hour - 2 : 7
    scroller.current?.scrollTo({ top: Math.max(0, hour * HOUR - 12) })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [day])

  const nowParts = zonedParts(now, tz)
  const nowTop = ((nowParts.hour * 60 + nowParts.minute) / 60) * HOUR

  return (
    <div className="flex h-full min-h-[560px] flex-col">
      <div className="mb-3 flex items-center gap-2">
        <IconButton size="sm" label="Back to month" icon={<IconArrowLeft size={16} />} onClick={onBack} />
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold tracking-tight text-fg">{formatDayKey(day, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}</h2>
        </div>
        <span className="text-xs text-fg-subtle">
          {items.length
            ? [runs && `${runs} run${runs === 1 ? '' : 's'}`, upcoming && `${upcoming} upcoming`, created && `${created} created`].filter(Boolean).join(' · ')
            : 'Nothing on this day'}
        </span>
        <span className="flex-1" />
        <div className="flex items-center gap-1">
          <Button size="sm" variant="outline" onClick={() => onDayChange(today)} disabled={day === today}>
            Today
          </Button>
          <IconButton size="sm" label="Previous day" icon={<IconChevronLeft size={16} />} onClick={() => onDayChange(addDays(day, -1))} />
          <IconButton size="sm" label="Next day" icon={<IconChevronRight size={16} />} onClick={() => onDayChange(addDays(day, 1))} />
        </div>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden rounded-xl border border-border bg-surface">
        {!items.length && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
            <EmptyState compact icon={<IconCalendarOff />} title="A quiet day" description="No runs happened or are scheduled on this day." className="pointer-events-auto rounded-2xl bg-surface/90 px-8" />
          </div>
        )}
        <div ref={scroller} className="h-full overflow-y-auto">
          <div className="relative" style={{ height: 24 * HOUR }}>
            {Array.from({ length: 24 }, (_, h) => (
              <div key={h} className="absolute inset-x-0 flex border-t border-border first:border-t-0" style={{ top: h * HOUR, height: HOUR }}>
                <div className="w-16 shrink-0 -translate-y-2 pr-3 text-right text-2xs tabular-nums text-fg-faint">
                  {h === 0 ? '' : new Date(Date.UTC(2000, 0, 1, h)).toLocaleTimeString(undefined, { hour: 'numeric', timeZone: 'UTC' })}
                </div>
              </div>
            ))}
            {day === today && (
              <div className="pointer-events-none absolute left-14 right-0 z-20 flex items-center" style={{ top: nowTop }}>
                <span className="size-2 -translate-x-1 rounded-full bg-accent" />
                <span className="h-px flex-1 bg-accent/70" />
              </div>
            )}
            <div className="absolute inset-y-0 left-16 right-3">
              {placed.map((p) => {
                const tone = itemTone(p.item)
                return (
                  <button
                    key={p.item.key}
                    type="button"
                    onClick={() => onSelectTask(p.item.task)}
                    className={cn(
                      'absolute flex items-center gap-2.5 overflow-hidden rounded-lg border border-l-[3px] border-border px-2.5 text-left shadow-soft transition-[filter,transform] hover:brightness-110',
                      blockTone[tone],
                      p.item.kind === 'upcoming' && 'border-dashed'
                    )}
                    style={{
                      top: (p.minutes / 60) * HOUR + 1,
                      height: (BLOCK_MIN / 60) * HOUR - 3,
                      left: `calc(${(p.col / p.cols) * 100}% + 2px)`,
                      width: `calc(${100 / p.cols}% - 4px)`
                    }}
                  >
                    <KindTile task={p.item.task} size="sm" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-fg">{displayName(p.item.task)}</div>
                      <div className="truncate text-2xs text-fg-subtle">
                        <span className="tabular-nums">{formatInZone(p.item.at, tz, { hour: 'numeric', minute: '2-digit' })}</span> · {itemLabel(p.item)}
                      </div>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

interface Placed {
  item: CalendarItem
  minutes: number
  col: number
  cols: number
}

/** Side-by-side columns for items whose blocks overlap. */
function layout(items: CalendarItem[], tz: string): Placed[] {
  const rows = items.map((item) => {
    const p = zonedParts(item.at, tz)
    return { item, minutes: p.hour * 60 + p.minute, col: 0, cols: 1 }
  })
  let cluster: Placed[] = []
  let clusterEnd = -1
  const flush = () => {
    const ends: number[] = []
    for (const r of cluster) {
      let c = ends.findIndex((end) => end <= r.minutes)
      if (c === -1) {
        c = ends.length
        ends.push(0)
      }
      ends[c] = r.minutes + BLOCK_MIN
      r.col = c
    }
    cluster.forEach((r) => (r.cols = Math.max(1, ends.length)))
    cluster = []
  }
  for (const r of rows) {
    if (cluster.length && r.minutes >= clusterEnd) flush()
    cluster.push(r)
    clusterEnd = Math.max(clusterEnd, r.minutes + BLOCK_MIN)
  }
  flush()
  return rows
}
