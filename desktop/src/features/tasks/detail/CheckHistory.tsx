/** Script jobs (docs/API.md §16): the quiet history of recent checks, newest first. */
import { IconAlertCircle, IconArrowsDiff, IconBellRinging, IconCircleCheck, IconFlag, IconMinus, IconPlayerSkipForward, IconRadar, type Icon } from '@tabler/icons-react'
import type { ScriptCheck, ScriptCheckOutcome, TaskScript } from '@/lib/types'
import { cn, parseDate } from '@/lib/utils'
import { SectionHeading } from '../parts'
import { formatInZone } from '../schedule'

type Tone = 'quiet' | 'news' | 'danger'

function describe(outcome: ScriptCheckOutcome, then: TaskScript['then']): { icon: Icon; text: string; tone: Tone } {
  const told = then === 'run' ? 'Sentient ran the task' : 'You were told'
  switch (outcome) {
    case 'baseline':
      return { icon: IconFlag, text: 'First check, saved as the starting point', tone: 'quiet' }
    case 'nothing_new':
      return { icon: IconCircleCheck, text: 'Checked, nothing new', tone: 'quiet' }
    case 'quiet':
      return { icon: IconCircleCheck, text: 'Checked, all quiet', tone: 'quiet' }
    case 'minor':
      return { icon: IconMinus, text: 'Something changed, but not enough to tell you', tone: 'quiet' }
    case 'changed':
      return { icon: IconArrowsDiff, text: `Something changed. ${told}`, tone: 'news' }
    case 'alert':
      return { icon: IconBellRinging, text: `Raised an alert. ${told}`, tone: 'news' }
    case 'reported':
      return { icon: IconBellRinging, text: `Reported. ${told}`, tone: 'news' }
    case 'failed':
      return { icon: IconAlertCircle, text: 'The check didn’t work', tone: 'danger' }
    case 'skipped': // what it reads wasn't there, so the check didn't run (#161); the note says why
      return { icon: IconPlayerSkipForward, text: 'Skipped: what it reads wasn’t there', tone: 'danger' }
    default:
      return { icon: IconCircleCheck, text: 'Checked', tone: 'quiet' }
  }
}

function when(iso: string | undefined, tz: string): string {
  const d = parseDate(iso)
  return d ? `${formatInZone(d, tz, { month: 'short', day: 'numeric' })}, ${formatInZone(d, tz, { hour: 'numeric', minute: '2-digit' })}` : ''
}

export function CheckHistory({ script, tz }: { script: TaskScript; tz: string }) {
  const checks: ScriptCheck[] = [...(script.checks ?? [])].reverse()
  if (!checks.length) return null
  const total = checks.reduce((n, c) => n + (c.count ?? 1), 0)
  return (
    <section className="space-y-2.5">
      <SectionHeading icon={<IconRadar />} title="Checks" count={total} description="Newest first. Quiet checks don’t send you anything." />
      <ol className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">
        {checks.map((c, i) => {
          const d = describe(c.outcome, script.then)
          const count = c.count ?? 1
          return (
            <li key={`${c.at}-${i}`} className={cn('flex items-start gap-2.5 px-3.5 py-2', d.tone === 'quiet' && 'bg-sunken/30')}>
              <d.icon size={15} className={cn('mt-0.5 shrink-0', d.tone === 'news' ? 'text-accent-text' : d.tone === 'danger' ? 'text-danger' : 'text-fg-faint')} />
              <div className="min-w-0 flex-1">
                <div className={cn('text-sm', d.tone === 'quiet' ? 'text-fg-subtle' : d.tone === 'danger' ? 'text-danger' : 'font-medium text-fg')}>
                  {d.text}
                  {count > 1 && <span className="text-fg-faint"> · {count} times</span>}
                </div>
                {c.note && <div className={cn('mt-0.5 text-xs', d.tone === 'quiet' ? 'text-fg-faint' : 'text-fg-muted')}>{c.note}</div>}
              </div>
              <span className="shrink-0 text-right text-2xs tabular-nums text-fg-subtle">
                {count > 1 && c.last_at ? (
                  <>
                    {when(c.at, tz)}
                    <span className="block">to {when(c.last_at, tz)}</span>
                  </>
                ) : (
                  when(c.at, tz)
                )}
              </span>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
