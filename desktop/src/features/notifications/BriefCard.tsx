/**
 * The Daily Brief card at the top of Notifications (docs/API.md section 6): today's few lines with thumbs up and down
 * per line and per section, a set-up card before the first opt-in, and a dialog to change time, days and sections.
 * The brief itself is an ordinary recurring task, so pausing and deleting happen in Tasks.
 */
import {
  IconCalendarEvent,
  IconCloud,
  IconDots,
  IconEyeOff,
  IconListCheck,
  IconMail,
  IconNews,
  IconPencil,
  IconRefresh,
  IconSunrise,
  IconThumbDown,
  IconThumbUp,
  type Icon
} from '@tabler/icons-react'
import { useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import {
  Button,
  Card,
  Dialog,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  Field,
  IconButton,
  Input,
  SegmentedControl,
  Skeleton,
  Switch,
  Tooltip
} from '@/components/ui'
import { openExternal } from '@/features/integrations/InstructionsGuide'
import { useBrief, useBriefActions } from '@/hooks/notifications'
import { errorMessage } from '@/lib/api'
import type { Brief, BriefFeedback, BriefItem, BriefSectionId, BriefState } from '@/lib/types'
import { cn, parseDate } from '@/lib/utils'
import { taskRoute } from './utils'

const SECTIONS: Array<{ id: BriefSectionId; label: string; icon: Icon; hint: string; missing: string }> = [
  { id: 'calendar', label: 'Calendar', icon: IconCalendarEvent, hint: "Today's meetings and events", missing: 'Connect Google Calendar first' },
  { id: 'email', label: 'Email', icon: IconMail, hint: 'Emails that need you', missing: 'Connect Gmail or email first' },
  { id: 'tasks', label: 'Tasks', icon: IconListCheck, hint: 'Tasks due today or waiting for you', missing: '' },
  { id: 'weather', label: 'Weather', icon: IconCloud, hint: 'Weather where you live', missing: 'Add your city in Settings first' },
  { id: 'news', label: 'News', icon: IconNews, hint: 'A headline for each topic you pick', missing: 'Add a topic below' }
]
const SECTION_BY_ID = Object.fromEntries(SECTIONS.map((s) => [s.id, s])) as Record<BriefSectionId, (typeof SECTIONS)[number]>
const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

function scheduleLine(s: BriefState): string {
  const time = s.time ?? '07:30'
  const days = s.days ?? []
  const when = days.length === 7 ? 'every day' : days.length === 5 && WEEKDAYS.every((d) => days.includes(d)) ? 'on weekdays' : `on ${days.map((d) => d.slice(0, 3)).join(', ')}`
  return `${when} at ${time}`
}

function dayTitle(day: string): string {
  const d = parseDate(`${day}T12:00:00`)
  return d ? d.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' }) : day
}

function Thumbs({ value, onRate, label, disabled }: { value: BriefFeedback | null; onRate: (v: BriefFeedback) => void; label: string; disabled?: boolean }) {
  return (
    <span className={cn('flex shrink-0 items-center', !value && 'opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100')}>
      <IconButton
        size="xs"
        label={value === 'up' ? 'You liked this' : `More like ${label}`}
        icon={<IconThumbUp size={13} />}
        active={value === 'up'}
        disabled={disabled || !!value}
        onClick={() => onRate('up')}
        className={cn(value === 'up' && 'text-success')}
      />
      <IconButton
        size="xs"
        label={value === 'down' ? 'You disliked this' : `Less like ${label}`}
        icon={<IconThumbDown size={13} />}
        active={value === 'down'}
        disabled={disabled || !!value}
        onClick={() => onRate('down')}
        className={cn(value === 'down' && 'text-danger')}
      />
    </span>
  )
}

function BriefLine({ item, onRate, onOpen, busy }: { item: BriefItem; onRate: (v: BriefFeedback) => void; onOpen: () => void; busy: boolean }) {
  return (
    <li className="group flex items-start gap-2 rounded-lg px-2 py-1.5 hover:bg-hover">
      <div className="min-w-0 flex-1">
        {item.link ? (
          <button type="button" onClick={onOpen} className="block max-w-full truncate text-left text-sm text-fg hover:text-accent-text hover:underline">
            {item.text}
          </button>
        ) : (
          <p className="truncate text-sm text-fg">{item.text}</p>
        )}
        <p className="truncate text-2xs text-fg-subtle">{item.why}</p>
      </div>
      <Thumbs value={item.feedback} onRate={onRate} label="this" disabled={busy} />
    </li>
  )
}

function BriefBody({ brief, state, compact, onNavigate }: { brief: Brief; state: BriefState; compact?: boolean; onNavigate?: () => void }) {
  const navigate = useNavigate()
  const { feedback, setup } = useBriefActions()
  const go = (route: string) => {
    onNavigate?.()
    navigate(route)
  }
  const rate = (value: BriefFeedback, target: { item_id?: string; section?: BriefSectionId }) =>
    feedback.mutate(
      { brief_id: brief.id, value, ...target },
      {
        onSuccess: () => toast(value === 'up' ? 'Thanks, noted' : 'Got it', { description: value === 'up' ? 'Future briefs will show more like this.' : 'Future briefs will show less like this.' }),
        onError: (e) => toast.error("Couldn't save that", { description: errorMessage(e) })
      }
    )
  const turnOff = (section: BriefSectionId) =>
    setup.mutate(
      { sections: state.sections.filter((s) => s !== section) },
      {
        onSuccess: () => toast(`${SECTION_BY_ID[section].label} is off`, { description: 'It will be left out of your next briefs. Turn it back on with Edit.' }),
        onError: (e) => toast.error("Couldn't change that", { description: errorMessage(e) })
      }
    )

  if (!brief.items.length) {
    return <p className="px-4 pb-4 text-sm text-fg-muted">Nothing needs you this morning. Enjoy your day.</p>
  }
  return (
    <div className={cn('space-y-3 px-2 pb-3', compact && 'space-y-2')}>
      {brief.sections.map((s) => {
        const meta = SECTION_BY_ID[s.id]
        const SectionIcon = meta?.icon ?? IconSunrise
        return (
          <section key={s.id} className="group/section">
            <div className="group flex items-center gap-1.5 px-2 pb-0.5">
              <SectionIcon size={13} className="text-fg-subtle" />
              <h4 className="flex-1 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">{s.label}</h4>
              <Thumbs value={s.feedback} onRate={(v) => rate(v, { section: s.id })} label={s.label.toLowerCase()} disabled={feedback.isPending} />
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <IconButton size="xs" label={`${s.label} options`} icon={<IconDots size={13} />} className="opacity-0 group-hover:opacity-100 data-[state=open]:opacity-100" />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem icon={<IconEyeOff />} onSelect={() => turnOff(s.id)}>
                    Turn off {s.label}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <ul>
              {brief.items
                .filter((i) => i.section === s.id)
                .map((i) => (
                  <BriefLine
                    key={i.id}
                    item={i}
                    busy={feedback.isPending}
                    onRate={(v) => rate(v, { item_id: i.id })}
                    onOpen={() => (i.link?.startsWith('/') ? go(i.link) : i.link && openExternal(i.link))}
                  />
                ))}
            </ul>
          </section>
        )
      })}
      {!compact && brief.skipped.length > 0 && (
        <p className="px-2 text-2xs text-fg-subtle">
          Not included: {brief.skipped.map((s) => `${s.label} (${s.reason.replace(/\.$/, '')})`).join(', ')}.
        </p>
      )}
    </div>
  )
}

/** Set up the brief, or change its time, days, sections and topics. */
export function BriefSetupDialog({ open, onOpenChange, state }: { open: boolean; onOpenChange: (o: boolean) => void; state?: BriefState }) {
  const { setup } = useBriefActions()
  const [time, setTime] = useState(state?.time ?? '07:30')
  const [days, setDays] = useState<'weekdays' | 'daily'>(state?.days?.length === 7 ? 'daily' : 'weekdays')
  const [sections, setSections] = useState<BriefSectionId[]>(state?.set_up ? state.sections : ['calendar', 'email', 'tasks', 'weather'])
  const [topics, setTopics] = useState((state?.news_topics ?? []).join(', '))
  const toggle = (id: BriefSectionId, on: boolean) => setSections((s) => (on ? [...s, id] : s.filter((x) => x !== id)))
  const save = () =>
    setup.mutate(
      {
        time,
        days: days === 'daily' ? 'daily' : WEEKDAYS,
        sections,
        news_topics: topics.split(',').map((t) => t.trim()).filter(Boolean)
      },
      {
        onSuccess: (s) => {
          onOpenChange(false)
          toast.success(state?.set_up ? 'Daily Brief updated' : 'Daily Brief is set up', { description: `It arrives ${scheduleLine(s)}. You can pause or delete it in Tasks.` })
        },
        onError: (e) => toast.error("Couldn't save your Daily Brief", { description: errorMessage(e) })
      }
    )

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={state?.set_up ? 'Edit your Daily Brief' : 'Set up your Daily Brief'}
      description="A few lines each morning about your day. Sentient only reads; it never sends or changes anything."
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" loading={setup.isPending} disabled={!sections.length} onClick={save}>
            {state?.set_up ? 'Save' : 'Set up'}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Time" htmlFor="brief-time">
            <Input id="brief-time" type="time" value={time} onChange={(e) => setTime(e.target.value)} className="w-32" />
          </Field>
          <SegmentedControl
            aria-label="Days"
            value={days}
            onChange={setDays}
            options={[
              { value: 'weekdays', label: 'Weekdays' },
              { value: 'daily', label: 'Every day' }
            ]}
          />
        </div>
        <Field label="What to include">
          <ul className="divide-y divide-border rounded-lg border border-border">
            {SECTIONS.map((s) => {
              const on = sections.includes(s.id)
              const ready = state?.available?.[s.id] ?? s.id === 'tasks'
              const needsTopic = s.id === 'news' && !topics.trim()
              return (
                <li key={s.id} className="flex items-center gap-3 px-3 py-2">
                  <s.icon size={16} className="text-fg-muted" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-fg">{s.label}</p>
                    <p className="text-2xs text-fg-subtle">{on && (needsTopic || (!ready && s.id !== 'news')) ? s.missing : s.hint}</p>
                  </div>
                  <Switch size="sm" checked={on} onCheckedChange={(v) => toggle(s.id, v)} aria-label={`Include ${s.label}`} />
                </li>
              )
            })}
          </ul>
        </Field>
        {sections.includes('news') && (
          <Field label="News topics" htmlFor="brief-topics" description="Separate topics with commas, for example: climate, cricket.">
            <Input id="brief-topics" value={topics} onChange={(e) => setTopics(e.target.value)} placeholder="climate, cricket" />
          </Field>
        )}
      </div>
    </Dialog>
  )
}

/** The Daily Brief at the top of the notification feed. Renders nothing when the engine has no brief API. */
export function BriefCard({ compact, onNavigate }: { compact?: boolean; onNavigate?: () => void }) {
  const navigate = useNavigate()
  const { data, isLoading, isError } = useBrief()
  const { runNow } = useBriefActions()
  const [editing, setEditing] = useState(false)
  if (isLoading) return <Skeleton className="mb-3 h-24 rounded-xl" />
  if (isError || !data) return null

  const makeOne = () =>
    runNow.mutate(undefined, {
      onSuccess: () => toast('Making your brief', { description: 'It shows up here in a moment.' }),
      onError: (e) => toast.error("Couldn't make your brief", { description: errorMessage(e) })
    })
  const dialog = editing && <BriefSetupDialog open={editing} onOpenChange={setEditing} state={data} />

  if (!data.set_up) {
    if (compact) return null
    return (
      <Card className="mb-4 flex items-center gap-3 px-4 py-3.5">
        <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-accent/12 text-accent-text">
          <IconSunrise size={18} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-fg">Start your day with a Daily Brief</p>
          <p className="text-xs text-fg-muted">Today's meetings, emails that need you, tasks and the weather in a few lines, every weekday morning.</p>
        </div>
        <Button variant="primary" size="sm" onClick={() => setEditing(true)}>
          Set up my Daily Brief
        </Button>
        {dialog}
      </Card>
    )
  }

  const brief = data.today
  const paused = !data.enabled
  const subtitle = paused ? 'Paused. Turn it back on in Tasks.' : brief ? `${dayTitle(brief.day)} · next ${scheduleLine(data)}` : `Arrives ${scheduleLine(data)}`
  return (
    <Card className={cn('mb-4 overflow-hidden', brief && 'border-accent/30')}>
      <div className="flex items-center gap-3 px-4 pb-2 pt-3.5">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/12 text-accent-text">
          <IconSunrise size={17} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-fg">Daily Brief</p>
          <p className="truncate text-xs text-fg-subtle">{subtitle}</p>
        </div>
        <Tooltip content="Make a fresh brief now">
          <span>
            <IconButton size="sm" label="Make one now" tooltip={false} icon={<IconRefresh size={15} />} loading={runNow.isPending} onClick={makeOne} />
          </span>
        </Tooltip>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <IconButton size="sm" label="Daily Brief options" icon={<IconDots size={15} />} />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem icon={<IconPencil />} onSelect={() => setEditing(true)}>
              Edit time and sections
            </DropdownMenuItem>
            {data.task_id && (
              <DropdownMenuItem
                icon={<IconListCheck />}
                onSelect={() => {
                  onNavigate?.()
                  navigate(taskRoute(data.task_id as string))
                }}
              >
                Pause or delete in Tasks
              </DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      {brief ? (
        <BriefBody brief={brief} state={data} compact={compact} onNavigate={onNavigate} />
      ) : (
        <p className="px-4 pb-3.5 text-sm text-fg-muted">No brief yet today. Press refresh to make one now.</p>
      )}
      {dialog}
    </Card>
  )
}
