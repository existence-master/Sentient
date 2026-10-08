/** Schedule editor (v2 ScheduleEditor, friendlier) and the read-only schedule summary. */
import { IconBolt, IconBraces, IconCalendarRepeat, IconCalendarTime, IconListDetails, IconPlus, IconTrash, IconWebhook } from '@tabler/icons-react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router'
import { Alert, Button, Combobox, IconButton, Input, SegmentedControl, Select, Textarea } from '@/components/ui'
import { useIntegrations } from '@/hooks/integrations'
import { useHooks } from '@/hooks/automations'
import type { RecurringSchedule, TaskSchedule, TriggeredSchedule, Weekday } from '@/lib/types'
import { cn, listTimezones } from '@/lib/utils'
import {
  FIELD_LABELS,
  RULE_OPS,
  SOURCE_FIELDS,
  SOURCE_LABELS,
  WEEKDAYS,
  filterRules,
  intervalMinutes,
  isIntervalSchedule,
  normalizeDays,
  rulesToFilter,
  scheduleSentence,
  sourceLabel,
  type FilterRules,
  type RuleOp
} from '../schedule'

const TRIGGER_EVENTS: Record<string, Array<{ value: string; label: string }>> = {
  gmail: [{ value: 'new_email', label: 'A new email arrives' }],
  gcalendar: [
    { value: 'new_event', label: 'A new event is added' },
    { value: 'updated_event', label: 'An event changes' }
  ],
  slack: [{ value: 'new_message', label: 'A new message arrives' }],
  github: [
    { value: 'new_issue', label: 'A new issue is opened' },
    { value: 'new_pull_request', label: 'A new pull request is opened' }
  ]
}

type Kind = 'once' | 'recurring' | 'triggered'

export function ScheduleEditor({ value, onChange, defaultTimezone }: { value: TaskSchedule | null; onChange: (s: TaskSchedule) => void; defaultTimezone: string }) {
  const schedule: TaskSchedule = value ?? { type: 'once', run_at: null, timezone: defaultTimezone }
  const tz = ('timezone' in schedule && schedule.timezone) || defaultTimezone

  const switchKind = (kind: Kind) => {
    if (kind === schedule.type) return
    if (kind === 'once') onChange({ type: 'once', run_at: null, timezone: tz })
    else if (kind === 'recurring') onChange({ type: 'recurring', frequency: 'daily', time: '09:00', timezone: tz })
    else onChange({ type: 'triggered', source: 'gmail', event: 'new_email', filter: {} })
  }

  return (
    <div className="space-y-4">
      <SegmentedControl<Kind>
        fullWidth
        aria-label="Schedule type"
        value={schedule.type}
        onChange={switchKind}
        options={[
          { value: 'once', label: 'Once', icon: <IconCalendarTime size={14} /> },
          { value: 'recurring', label: 'Repeats', icon: <IconCalendarRepeat size={14} /> },
          { value: 'triggered', label: 'When something happens', icon: <IconBolt size={14} /> }
        ]}
      />

      {schedule.type === 'once' && (
        <div className="space-y-3">
          <SegmentedControl<'asap' | 'at'>
            size="sm"
            value={schedule.run_at ? 'at' : 'asap'}
            onChange={(v) => onChange({ ...schedule, run_at: v === 'asap' ? null : defaultRunAt() })}
            options={[
              { value: 'asap', label: 'Right after approval' },
              { value: 'at', label: 'At a specific time' }
            ]}
          />
          {schedule.run_at && (
            <div className="flex flex-wrap items-center gap-2">
              <Input
                type="datetime-local"
                size="sm"
                value={schedule.run_at.slice(0, 16)}
                onChange={(e) => onChange({ ...schedule, run_at: e.target.value || null })}
                wrapperClassName="w-56"
                aria-label="Run at"
              />
              <ZonePicker value={tz} onChange={(z) => onChange({ ...schedule, timezone: z })} />
            </div>
          )}
        </div>
      )}

      {schedule.type === 'recurring' && <RecurringFields schedule={schedule} tz={tz} onChange={onChange} />}
      {schedule.type === 'triggered' && <TriggeredFields schedule={schedule} onChange={onChange} />}

      <div className="rounded-lg border border-border bg-sunken/50 px-3 py-2 text-sm text-fg-muted">
        <span className="text-fg-subtle">Summary: </span>
        {scheduleSentence(schedule)}
      </div>
    </div>
  )
}

function defaultRunAt(): string {
  const d = new Date(Date.now() + 24 * 3600_000)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T09:00`
}

function ZonePicker({ value, onChange }: { value: string; onChange: (z: string) => void }) {
  const zones = useMemo(() => listTimezones(), [])
  return (
    <Combobox
      size="sm"
      className="w-56"
      aria-label="Timezone"
      value={value}
      onChange={(z) => z && onChange(z)}
      searchPlaceholder="Search timezones"
      groups={[{ id: 'tz', label: 'Timezones', options: zones.map((z) => ({ value: z, label: z })) }]}
    />
  )
}

const INTERVAL_PICKS = [5, 15, 30, 60, 180, 360, 720, 1440]

function RecurringFields({ schedule, tz, onChange }: { schedule: RecurringSchedule; tz: string; onChange: (s: TaskSchedule) => void }) {
  const days = normalizeDays(schedule.days)
  const interval = isIntervalSchedule(schedule)
  const preset = interval ? 'interval' : schedule.frequency === 'daily' ? 'daily' : days.length === 5 && WEEKDAYS.slice(0, 5).every((d) => days.includes(d)) ? 'weekdays' : 'weekly'
  const toRecurring = (patch: Partial<RecurringSchedule>): TaskSchedule => {
    const { interval_minutes: _drop, ...rest } = schedule as RecurringSchedule & { interval_minutes?: number }
    void _drop
    return { ...rest, time: rest.time || '09:00', ...patch } as TaskSchedule
  }
  const setPreset = (p: string) => {
    if (p === 'interval') onChange({ type: 'recurring', frequency: 'interval', interval_minutes: 60, timezone: tz } as unknown as TaskSchedule)
    else if (p === 'daily') onChange(toRecurring({ frequency: 'daily', days: undefined }))
    else if (p === 'weekdays') onChange(toRecurring({ frequency: 'weekly', days: WEEKDAYS.slice(0, 5) }))
    else onChange(toRecurring({ frequency: 'weekly', days: days.length ? days : ['Monday'] }))
  }
  if (interval) {
    const minutes = intervalMinutes(schedule)
    const setMinutes = (m: number) => onChange({ ...(schedule as object), frequency: 'interval', interval_minutes: Math.max(5, Math.round(m) || 5) } as unknown as TaskSchedule)
    return (
      <div className="space-y-3">
        <RecurringPresets preset={preset} onChange={setPreset} />
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-fg-subtle">Check every</span>
          <Input
            type="number"
            size="sm"
            min={5}
            step={5}
            value={String(minutes)}
            onChange={(e) => setMinutes(Number(e.target.value))}
            wrapperClassName="w-24"
            className="text-right tabular-nums"
            aria-label="Minutes between checks"
          />
          <span className="text-sm text-fg-subtle">minutes</span>
          <div className="flex flex-wrap gap-1">
            {INTERVAL_PICKS.map((m) => (
              <button
                key={m}
                type="button"
                aria-pressed={m === minutes}
                onClick={() => setMinutes(m)}
                className={cn(
                  'h-7 rounded-md border px-2 text-xs transition-colors',
                  m === minutes ? 'border-accent/40 bg-accent/15 text-accent-text' : 'border-border text-fg-muted hover:border-border-strong hover:text-fg'
                )}
              >
                {m < 60 ? `${m} min` : m % 1440 === 0 ? `${m / 1440} day` : `${m / 60} h`}
              </button>
            ))}
          </div>
        </div>
        <p className="text-xs text-fg-subtle">Good for watching something, like a price or a status page. The shortest gap is 5 minutes.</p>
      </div>
    )
  }
  return <RecurringDayFields schedule={schedule} tz={tz} days={days} preset={preset} setPreset={setPreset} onChange={onChange} />
}

function RecurringPresets({ preset, onChange }: { preset: string; onChange: (p: string) => void }) {
  return (
    <SegmentedControl
      size="sm"
      value={preset}
      onChange={onChange}
      options={[
        { value: 'daily', label: 'Every day' },
        { value: 'weekdays', label: 'Weekdays' },
        { value: 'weekly', label: 'Choose days' },
        { value: 'interval', label: 'Every few minutes' }
      ]}
    />
  )
}

function RecurringDayFields({
  schedule,
  tz,
  days,
  preset,
  setPreset,
  onChange
}: {
  schedule: RecurringSchedule
  tz: string
  days: Weekday[]
  preset: string
  setPreset: (p: string) => void
  onChange: (s: TaskSchedule) => void
}) {
  const toggleDay = (d: Weekday) => {
    const next = days.includes(d) ? days.filter((x) => x !== d) : [...days, d]
    onChange({ ...schedule, frequency: 'weekly', days: WEEKDAYS.filter((w) => next.includes(w)) })
  }
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <RecurringPresets preset={preset} onChange={setPreset} />
        <span className="text-sm text-fg-subtle">at</span>
        <Input type="time" size="sm" value={schedule.time} onChange={(e) => onChange({ ...schedule, time: e.target.value || '09:00' })} wrapperClassName="w-32" aria-label="Time" />
        <ZonePicker value={tz} onChange={(z) => onChange({ ...schedule, timezone: z })} />
      </div>
      {schedule.frequency === 'weekly' && (
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Days">
          {WEEKDAYS.map((d) => (
            <button
              key={d}
              type="button"
              aria-pressed={days.includes(d)}
              onClick={() => toggleDay(d)}
              className={cn(
                'h-8 w-11 rounded-lg border text-xs font-medium transition-colors',
                days.includes(d) ? 'border-accent/40 bg-accent/15 text-accent-text' : 'border-border text-fg-muted hover:border-border-strong hover:text-fg'
              )}
            >
              {d.slice(0, 3)}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

interface DraftRule {
  field: string
  op: RuleOp
  value: string
}

function TriggeredFields({ schedule, onChange }: { schedule: TriggeredSchedule; onChange: (s: TaskSchedule) => void }) {
  const parsed = useMemo(() => filterRules(schedule.filter), [schedule.filter])
  const [advanced, setAdvanced] = useState(parsed.complex)
  const [json, setJson] = useState(() => JSON.stringify(schedule.filter ?? {}, null, 2))
  const [jsonError, setJsonError] = useState<string | null>(null)
  const rules: DraftRule[] = parsed.rules.map((r) => ({ field: r.field, op: r.op, value: r.value }))
  const source = schedule.source || 'gmail'
  const integrations = useIntegrations()
  const hooks = useHooks()
  const navigate = useNavigate()
  // Trigger sources/events: each integration's `triggers` (§5), with a built-in fallback.
  const triggerSources = useMemo(() => {
    const map: Record<string, { label: string; events: Array<{ value: string; label: string }> }> = {}
    for (const [id, evs] of Object.entries(TRIGGER_EVENTS)) map[id] = { label: SOURCE_LABELS[id] ?? id, events: evs }
    for (const i of integrations.data ?? []) {
      if (i.triggers?.length) map[i.id] = { label: i.display_name, events: i.triggers.map((t) => ({ value: t.event, label: t.label || t.event })) }
    }
    // the builtin `webhook` integration already lists one trigger per hook; the hooks list keeps names fresh
    if (hooks.data) map.webhook = { label: 'Webhook', events: hooks.data.map((h) => ({ value: h.id, label: `When “${h.name}” is called` })) }
    else if (!map.webhook) map.webhook = { label: 'Webhook', events: [] }
    if (!map[source]) map[source] = { label: sourceLabel(source), events: [] }
    if (schedule.event && !map[source].events.some((e) => e.value === schedule.event)) {
      map[source] = { ...map[source], events: [...map[source].events, { value: schedule.event, label: schedule.event }] }
    }
    return map
  }, [integrations.data, hooks.data, source, schedule.event])
  const events = triggerSources[source].events.length ? triggerSources[source].events : [{ value: 'new_item', label: 'new_item' }]
  const fields = SOURCE_FIELDS[source] ?? ['from', 'subject']
  const isWebhook = source === 'webhook'
  const hookOptions = triggerSources.webhook?.events ?? []

  const setRules = (next: DraftRule[], mode = parsed.mode) => {
    const filter = rulesToFilter(next, mode)
    setJson(JSON.stringify(filter, null, 2))
    onChange({ ...schedule, filter })
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-fg-subtle">When</span>
        <Select
          size="sm"
          className="w-44"
          aria-label="App"
          value={source}
          onValueChange={(v) =>
            onChange({ ...schedule, source: v, event: v === 'webhook' ? (hookOptions[0]?.value ?? '') : (triggerSources[v]?.events[0]?.value ?? schedule.event), filter: {} })
          }
          options={Object.entries(triggerSources).map(([value, s]) => ({ value, label: s.label }))}
        />
        {isWebhook ? (
          hookOptions.length > 0 && (
            <Select size="sm" className="w-56" aria-label="Webhook" placeholder="Choose a webhook" value={schedule.event || undefined} onValueChange={(v) => onChange({ ...schedule, event: v })} options={hookOptions} />
          )
        ) : (
          <Select size="sm" className="w-56" aria-label="Event" value={schedule.event || events[0].value} onValueChange={(v) => onChange({ ...schedule, event: v })} options={events} />
        )}
      </div>
      {isWebhook && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-sunken/40 px-3 py-2 text-xs text-fg-subtle">
          <IconWebhook size={14} className="text-accent-text" />
          <span className="min-w-0 flex-1">
            {hooks.isError
              ? 'Webhooks aren’t available in this version of the engine yet.'
              : hookOptions.length
                ? 'Runs each time another app calls this webhook’s private link.'
                : 'You don’t have a webhook yet. Create one, then pick it here.'}
          </span>
          <Button size="xs" variant="ghost" onClick={() => navigate('/integrations?section=webhooks')}>
            {hookOptions.length ? 'Manage webhooks' : 'Create a webhook'}
          </Button>
        </div>
      )}

      <div className="rounded-xl border border-border bg-surface">
        <div className="flex items-center gap-2 border-b border-border px-3 py-2">
          <span className="text-sm font-medium text-fg">Only when</span>
          {!advanced && rules.length > 1 && (
            <SegmentedControl
              size="sm"
              value={parsed.mode}
              onChange={(m) => setRules(rules, m)}
              options={[
                { value: 'all', label: 'all match' },
                { value: 'any', label: 'any match' }
              ]}
            />
          )}
          <span className="flex-1" />
          <button
            type="button"
            onClick={() => {
              if (advanced && jsonError) return
              setAdvanced((a) => !a)
            }}
            disabled={parsed.complex && advanced}
            className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs text-fg-subtle hover:bg-hover hover:text-fg disabled:opacity-50"
          >
            {advanced ? <IconListDetails size={13} /> : <IconBraces size={13} />}
            {advanced ? 'Rules' : 'Advanced JSON'}
          </button>
        </div>
        {advanced ? (
          <div className="space-y-2 p-3">
            {parsed.complex && <p className="text-xs text-fg-subtle">This filter uses nested conditions, so it can only be edited as JSON.</p>}
            <Textarea
              value={json}
              rows={6}
              invalid={!!jsonError}
              spellCheck={false}
              className="font-mono text-xs"
              onChange={(e) => {
                setJson(e.target.value)
                try {
                  const v = JSON.parse(e.target.value || '{}') as unknown
                  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('The filter must be a JSON object')
                  setJsonError(null)
                  onChange({ ...schedule, filter: v as Record<string, unknown> })
                } catch (err) {
                  setJsonError((err as Error).message)
                }
              }}
            />
            {jsonError ? <p className="text-xs text-danger">{jsonError}</p> : <FilterRulesView rules={filterRules(schedule.filter)} compact />}
          </div>
        ) : (
          <div className="space-y-2 p-3">
            {!rules.length && <p className="text-xs text-fg-subtle">No conditions: every {source === 'gcalendar' ? 'new event' : 'new item'} starts a run.</p>}
            {rules.map((r, i) => (
              <div key={i} className="flex items-center gap-2">
                <Combobox
                  size="sm"
                  className="w-36"
                  aria-label="Field"
                  value={r.field}
                  onChange={(v) => setRules(rules.map((x, j) => (j === i ? { ...x, field: v } : x)))}
                  groups={[{ id: 'f', label: 'Fields', options: fields.map((f) => ({ value: f, label: FIELD_LABELS[f] ?? f })) }]}
                />
                <Select size="sm" className="w-40" aria-label="Condition" value={r.op} onValueChange={(v) => setRules(rules.map((x, j) => (j === i ? { ...x, op: v } : x)))} options={RULE_OPS.map((o) => ({ value: o.value, label: o.label }))} />
                <Input
                  size="sm"
                  value={r.value}
                  placeholder={r.op === 'one_of' || r.op === 'none_of' ? 'a@x.com, b@y.com' : 'value'}
                  onChange={(e) => setRules(rules.map((x, j) => (j === i ? { ...x, value: e.target.value } : x)))}
                  aria-label="Value"
                />
                <IconButton size="sm" label="Remove condition" icon={<IconTrash size={14} />} onClick={() => setRules(rules.filter((_, j) => j !== i))} />
              </div>
            ))}
            <button
              type="button"
              onClick={() => setRules([...rules, { field: fields[0], op: 'contains', value: '' }])}
              className="flex items-center gap-1.5 rounded-md px-1.5 py-1 text-xs font-medium text-accent-text hover:bg-accent/10"
            >
              <IconPlus size={13} /> Add condition
            </button>
          </div>
        )}
      </div>
      {parsed.complex && !advanced && <Alert tone="warning">This filter has nested conditions; switch to JSON to edit it.</Alert>}
    </div>
  )
}

/** Readable filter rules as chips. */
export function FilterRulesView({ rules, compact }: { rules: FilterRules; compact?: boolean }) {
  if (!rules.rules.length) return <span className="text-xs text-fg-subtle">Every event matches</span>
  return (
    <div className={cn('flex flex-wrap items-center gap-1.5', compact && 'text-xs')}>
      <span className="text-xs text-fg-subtle">{rules.mode === 'any' ? 'Any of' : rules.rules.length > 1 ? 'All of' : 'Only if'}</span>
      {rules.rules.map((r, i) => (
        <span key={i} className="inline-flex max-w-full items-center rounded-md border border-border bg-elevated px-2 py-0.5 text-xs text-fg-muted">
          <span className="truncate">{r.text}</span>
        </span>
      ))}
    </div>
  )
}
