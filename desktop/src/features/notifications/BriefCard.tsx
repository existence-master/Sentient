/**
 * The brief card at the top of Notifications (docs/API.md section 6): the brief showing now (the morning Daily Brief or
 * the Evening Brief), with thumbs up and down per line and per section, a set-up card before the first opt-in, and a
 * dialog to change time, days and sections. Each brief is an ordinary recurring task, so pausing and deleting happen
 * in Tasks.
 */
import {
  IconCalendarEvent,
  IconCalendarDue,
  IconChecks,
  IconCloud,
  IconDots,
  IconEyeOff,
  IconFile,
  IconHourglass,
  IconListCheck,
  IconMail,
  IconMoon,
  IconNews,
  IconPencil,
  IconRefresh,
  IconSend,
  IconSunrise,
  IconThumbDown,
  IconThumbUp,
  type Icon
} from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import {
  Button,
  Card,
  Dialog,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
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
import { useBrief, useBriefActions, useNotificationActions, useNotifications } from '@/hooks/notifications'
import { errorMessage } from '@/lib/api'
import type { Brief, BriefFeedback, BriefItem, BriefKind, BriefSectionId, BriefState, BriefTaskState } from '@/lib/types'
import { cn, parseDate } from '@/lib/utils'
import { taskRoute } from './utils'

interface SectionMeta {
  id: BriefSectionId
  label: string
  icon: Icon
  hint: string
  missing: string
}

const SECTIONS: Record<BriefKind, SectionMeta[]> = {
  morning: [
    { id: 'calendar', label: 'Calendar', icon: IconCalendarEvent, hint: "Today's meetings and events", missing: 'Connect Google Calendar first' },
    { id: 'email', label: 'Email', icon: IconMail, hint: 'Emails that need you', missing: 'Connect Gmail or email first' },
    { id: 'tasks', label: 'Tasks', icon: IconListCheck, hint: 'Tasks due today or waiting for you', missing: '' },
    { id: 'weather', label: 'Weather', icon: IconCloud, hint: 'Weather where you live', missing: 'Add your city in Settings first' },
    { id: 'news', label: 'News', icon: IconNews, hint: 'A headline for each topic you pick', missing: 'Add a topic below' }
  ],
  evening: [
    { id: 'done', label: 'Done today', icon: IconChecks, hint: 'Tasks that finished or failed today', missing: '' },
    { id: 'sent', label: 'Sent today', icon: IconSend, hint: 'Replies and emails sent today', missing: '' },
    { id: 'files', label: 'New files', icon: IconFile, hint: 'Files Sentient made today', missing: '' },
    { id: 'waiting', label: 'Still waiting for you', icon: IconHourglass, hint: 'Questions, plans and suggestions to answer', missing: '' },
    { id: 'tomorrow', label: 'Tomorrow', icon: IconCalendarDue, hint: "Tomorrow's first events", missing: 'Connect Google Calendar first' }
  ]
}
const SECTION_BY_ID = Object.fromEntries([...SECTIONS.morning, ...SECTIONS.evening].map((s) => [s.id, s])) as Record<BriefSectionId, SectionMeta>
const DEFAULTS: Record<BriefKind, { time: string; days: 'weekdays' | 'daily'; sections: BriefSectionId[] }> = {
  morning: { time: '07:30', days: 'weekdays', sections: ['calendar', 'email', 'tasks', 'weather'] },
  evening: { time: '21:00', days: 'daily', sections: ['done', 'sent', 'files', 'waiting', 'tomorrow'] }
}
const NAME: Record<BriefKind, string> = { morning: 'Daily Brief', evening: 'Evening Brief' }
const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
type DaysChoice = 'weekdays' | 'daily' | 'custom'

function daysChoice(days: string[] | null | undefined): DaysChoice | null {
  if (!days?.length) return null
  if (days.length === 7) return 'daily'
  if (days.length === 5 && WEEKDAYS.every((d) => days.includes(d))) return 'weekdays'
  return 'custom'
}

function scheduleLine(s: BriefTaskState): string {
  const choice = daysChoice(s.days)
  const when = choice === 'daily' ? 'every day' : choice === 'weekdays' ? 'on weekdays' : `on ${(s.days ?? []).map((d) => d.slice(0, 3)).join(', ')}`
  return `${when} at ${s.time ?? '07:30'}`
}

function taskState(state: BriefState, kind: BriefKind): BriefTaskState {
  return kind === 'evening' ? state.evening : state
}

function dayTitle(day: string): string {
  const d = parseDate(`${day}T12:00:00`)
  return d ? d.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' }) : day
}

function Thumbs({ value, onRate, label, disabled }: { value: BriefFeedback | null; onRate: (v: BriefFeedback) => void; label: string; disabled?: boolean }) {
  // a rating can be changed: the latest one counts
  return (
    <span className={cn('flex shrink-0 items-center', !value && 'opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100')}>
      <IconButton
        size="xs"
        label={value === 'up' ? 'You liked this' : `More like ${label}`}
        icon={<IconThumbUp size={13} />}
        active={value === 'up'}
        disabled={disabled}
        onClick={() => value !== 'up' && onRate('up')}
        className={cn(value === 'up' && 'text-success')}
      />
      <IconButton
        size="xs"
        label={value === 'down' ? 'You disliked this' : `Less like ${label}`}
        icon={<IconThumbDown size={13} />}
        active={value === 'down'}
        disabled={disabled}
        onClick={() => value !== 'down' && onRate('down')}
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
  const kind: BriefKind = brief.kind ?? 'morning'
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
      { kind, sections: taskState(state, kind).sections.filter((s) => s !== section) },
      {
        onSuccess: () => toast(`${SECTION_BY_ID[section]?.label ?? 'That section'} is off`, { description: 'It will be left out of your next briefs. Turn it back on with Edit.' }),
        onError: (e) => toast.error("Couldn't change that", { description: errorMessage(e) })
      }
    )

  if (!brief.items.length) {
    return (
      <p className="px-4 pb-4 text-sm text-fg-muted">
        {kind === 'evening' ? 'A quiet day. Nothing is waiting for you tonight.' : 'Nothing needs you this morning. Enjoy your day.'}
      </p>
    )
  }
  return (
    <div className={cn('space-y-3 px-2 pb-3', compact && 'space-y-2')}>
      {brief.sections.map((s) => {
        const SectionIcon = SECTION_BY_ID[s.id]?.icon ?? IconSunrise
        return (
          <section key={s.id}>
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

/** Set up a brief, or change its time, days, sections and topics. */
export function BriefSetupDialog({ open, onOpenChange, state, kind = 'morning' }: { open: boolean; onOpenChange: (o: boolean) => void; state?: BriefState; kind?: BriefKind }) {
  const { setup } = useBriefActions()
  const current = state ? taskState(state, kind) : undefined
  const setUp = !!current?.set_up
  const initialDays = (setUp && daysChoice(current?.days)) || DEFAULTS[kind].days
  const [time, setTime] = useState(current?.time ?? DEFAULTS[kind].time)
  const [days, setDays] = useState<DaysChoice>(initialDays)
  const [daysTouched, setDaysTouched] = useState(false)
  const [sections, setSections] = useState<BriefSectionId[]>(setUp && current ? current.sections : DEFAULTS[kind].sections)
  const [topics, setTopics] = useState((state?.news_topics ?? []).join(', '))
  const toggle = (id: BriefSectionId, on: boolean) => setSections((s) => (on ? [...s, id] : s.filter((x) => x !== id)))
  const save = () =>
    setup.mutate(
      {
        kind,
        time,
        // days are sent only when chosen here, so custom days set in Tasks are kept
        ...((daysTouched || !setUp) && days !== 'custom' ? { days: days === 'daily' ? 'daily' : WEEKDAYS } : {}),
        sections,
        ...(kind === 'morning' ? { news_topics: topics.split(',').map((t) => t.trim()).filter(Boolean) } : {})
      },
      {
        onSuccess: (s) => {
          onOpenChange(false)
          toast.success(setUp ? `${NAME[kind]} updated` : `${NAME[kind]} is set up`, { description: `It arrives ${scheduleLine(taskState(s, kind))}. You can pause or delete it in Tasks.` })
        },
        onError: (e) => toast.error(`Couldn't save your ${NAME[kind]}`, { description: errorMessage(e) })
      }
    )

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={`${setUp ? 'Edit' : 'Set up'} your ${NAME[kind]}`}
      description={
        kind === 'evening'
          ? 'A few lines each evening wrapping up your day. Sentient only reads; it never sends or changes anything.'
          : 'A few lines each morning about your day. Sentient only reads; it never sends or changes anything.'
      }
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" loading={setup.isPending} disabled={!sections.length} onClick={save}>
            {setUp ? 'Save' : 'Set up'}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Time" htmlFor={`brief-time-${kind}`}>
            <Input id={`brief-time-${kind}`} type="time" value={time} onChange={(e) => setTime(e.target.value)} className="w-32" />
          </Field>
          <SegmentedControl
            aria-label="Days"
            value={days}
            onChange={(v) => {
              setDays(v)
              setDaysTouched(true)
            }}
            options={[
              { value: 'weekdays', label: 'Weekdays' },
              { value: 'daily', label: 'Every day' },
              ...(initialDays === 'custom' ? [{ value: 'custom' as const, label: 'Custom days' }] : [])
            ]}
          />
        </div>
        <Field label="What to include">
          <ul className="divide-y divide-border rounded-lg border border-border">
            {SECTIONS[kind].map((s) => {
              const on = sections.includes(s.id)
              const ready = state?.available?.[s.id] ?? !s.missing
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
        {kind === 'morning' && sections.includes('news') && (
          <Field label="News topics" htmlFor="brief-topics" description="Separate topics with commas, for example: climate, cricket.">
            <Input id="brief-topics" value={topics} onChange={(e) => setTopics(e.target.value)} placeholder="climate, cricket" />
          </Field>
        )}
      </div>
    </Dialog>
  )
}

/** Briefs are left out of the feed, so the card marks the one it shows as read. */
function useMarkBriefRead(briefId: string | undefined) {
  const { data } = useNotifications()
  const { markRead } = useNotificationActions()
  const unread = !!briefId && !!data?.notifications.some((n) => n.id === briefId && !n.read)
  useEffect(() => {
    if (unread && briefId) markRead.mutate(briefId)
  }, [unread, briefId]) // markRead is a fresh object each render; unread guards repeats
}

/** The brief card at the top of the notification feed. Renders nothing when the engine has no brief API. */
export function BriefCard({ compact, onNavigate }: { compact?: boolean; onNavigate?: () => void }) {
  const navigate = useNavigate()
  const { data, isLoading, isError } = useBrief()
  const { runNow } = useBriefActions()
  const [editing, setEditing] = useState<BriefKind | null>(null)
  useMarkBriefRead(data?.today?.id)
  if (isLoading) return <Skeleton className="mb-3 h-24 rounded-xl" />
  if (isError || !data) return null

  const makeOne = (kind: BriefKind) =>
    runNow.mutate(kind, {
      onSuccess: () => toast(`Making your ${NAME[kind]}`, { description: 'It shows up here in a moment.' }),
      onError: (e) => toast.error(`Couldn't make your ${NAME[kind]}`, { description: errorMessage(e) })
    })
  const dialog = editing && <BriefSetupDialog key={editing} open onOpenChange={(o) => !o && setEditing(null)} state={data} kind={editing} />

  if (!data.set_up && !data.evening.set_up) {
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
        <Button variant="primary" size="sm" onClick={() => setEditing('morning')}>
          Set up my Daily Brief
        </Button>
        {dialog}
      </Card>
    )
  }

  const brief = data.today
  const kind: BriefKind = brief?.kind ?? (data.set_up ? 'morning' : 'evening')
  const own = taskState(data, kind)
  const next = [data.set_up && `morning ${scheduleLine(data)}`, data.evening.set_up && `evening ${scheduleLine(data.evening)}`].filter(Boolean).join(', ')
  const subtitle = !own.enabled ? 'Paused. Turn it back on in Tasks.' : brief ? `${dayTitle(brief.day)} · next ${next}` : `Arrives ${next}`
  const KindIcon = kind === 'evening' ? IconMoon : IconSunrise
  return (
    <Card className={cn('mb-4 overflow-hidden', brief && 'border-accent/30')}>
      <div className="flex items-center gap-3 px-4 pb-2 pt-3.5">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/12 text-accent-text">
          <KindIcon size={17} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-fg">{NAME[kind]}</p>
          <p className="truncate text-xs text-fg-subtle">{subtitle}</p>
        </div>
        <Tooltip content={`Make a fresh ${NAME[kind]} now`}>
          <span>
            <IconButton size="sm" label="Make one now" tooltip={false} icon={<IconRefresh size={15} />} loading={runNow.isPending} onClick={() => makeOne(kind)} />
          </span>
        </Tooltip>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <IconButton size="sm" label="Brief options" icon={<IconDots size={15} />} />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem icon={<IconPencil />} onSelect={() => setEditing('morning')}>
              {data.set_up ? 'Edit Daily Brief' : 'Set up Daily Brief'}
            </DropdownMenuItem>
            <DropdownMenuItem icon={<IconMoon />} onSelect={() => setEditing('evening')}>
              {data.evening.set_up ? 'Edit Evening Brief' : 'Set up Evening Brief'}
            </DropdownMenuItem>
            {data.evening.set_up && kind === 'morning' && (
              <DropdownMenuItem icon={<IconRefresh />} onSelect={() => makeOne('evening')}>
                Make an Evening Brief now
              </DropdownMenuItem>
            )}
            {data.set_up && kind === 'evening' && (
              <DropdownMenuItem icon={<IconRefresh />} onSelect={() => makeOne('morning')}>
                Make a Daily Brief now
              </DropdownMenuItem>
            )}
            {own.task_id && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  icon={<IconListCheck />}
                  onSelect={() => {
                    onNavigate?.()
                    navigate(taskRoute(own.task_id as string))
                  }}
                >
                  Pause or delete in Tasks
                </DropdownMenuItem>
              </>
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
