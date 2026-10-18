/** A routine's setup checklist (docs/API.md §4 "Setup checklist", #161): when it runs, what it reads, what you get,
 * what it may do without asking and what happens when what it reads isn't there. The engine fills it in from the plan;
 * you confirm it as it is, or change any answer. Try it first runs it once without sending or changing anything. */
import {
  IconAlertTriangle,
  IconApps,
  IconCheck,
  IconChecklist,
  IconClock,
  IconFlask,
  IconFolder,
  IconInbox,
  IconPencil,
  IconShieldCheck,
  IconWorld,
  type Icon
} from '@tabler/icons-react'
import { useState, type ReactNode } from 'react'
import { Badge, Button, Input, SegmentedControl, Select, Tooltip } from '@/components/ui'
import type { MissingSourcePolicy, Task, TaskSchedule, TaskSetup, TaskSetupAnswers, TaskSource } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'
import { describeSchedule } from '../schedule'
import type { useTaskOps } from '../useTaskOps'
import { useToolNames } from '../tools'
import { ScheduleEditor } from './ScheduleEditor'

type Ops = ReturnType<typeof useTaskOps>

const MISSING: Record<MissingSourcePolicy, string> = {
  skip: 'Skip that run quietly',
  notify: 'Skip that run and tell me',
  retry: 'Try again a little later, then tell me'
}

const SOURCE_KINDS: Array<{ value: TaskSource['kind']; label: string }> = [
  { value: 'page', label: 'A web page' },
  { value: 'app', label: 'An app' },
  { value: 'folder', label: 'A folder' },
  { value: 'none', label: 'Nothing' }
]

/** A routine (recurring or triggered, run by its plan or a check script) has a checklist; others don't. */
export function hasChecklist(task: Task): boolean {
  const kind = task.schedule?.type
  return (kind === 'recurring' || kind === 'triggered') && task.task_type !== 'swarm' && !task.original_context?.fixed_call && !!task.setup
}

function sourceIcon(kind: TaskSource['kind']): Icon {
  return kind === 'page' ? IconWorld : kind === 'app' ? IconApps : kind === 'folder' ? IconFolder : IconInbox
}

function sourceText(source: TaskSource): string {
  if (source.kind === 'none') return 'Nothing to check before it starts'
  return source.label || source.target
}

/** "the page", "Gmail", "the folder": what "isn't there" is about. */
function missingWhat(source: TaskSource): string {
  if (source.kind === 'page') return 'the page'
  if (source.kind === 'folder') return 'the folder'
  if (source.kind === 'app') return source.label || 'the app'
  return 'what it reads'
}

function approvalText(setup: TaskSetup): string {
  if (setup.approval === 'ask') return 'Asks you first before it sends, posts, buys, deletes or runs anything'
  if (!setup.acts.length) return 'Goes ahead with the steps in the plan'
  const apps = setup.acts.length === 1 ? setup.acts[0] : `${setup.acts.slice(0, -1).join(', ')} and ${setup.acts[setup.acts.length - 1]}`
  return `Goes ahead and uses ${apps} as the plan says`
}

export function SetupChecklist({ task, tz, ops, canTry, onTry }: { task: Task; tz: string; ops: Ops; canTry: boolean; onTry: () => void }) {
  const setup = task.setup as TaskSetup
  const [draft, setDraft] = useState<Draft | null>(null)
  const confirmed = !!setup.confirmed_at
  const id = task.task_id
  const saving = ops.isBusy('update', id)
  const d = describeSchedule(task.schedule)
  const SourceIcon = sourceIcon(setup.source.kind)

  const confirm = () => void ops.update(id, { setup: { confirm: true } }, 'Checklist confirmed')

  return (
    <section className={cn('overflow-hidden rounded-xl border bg-surface', confirmed ? 'border-border' : 'border-accent/35 shadow-[0_0_0_3px] shadow-accent/8')}>
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2.5">
        <IconChecklist size={16} className={confirmed ? 'text-fg-subtle' : 'text-accent-text'} />
        <h3 className="text-sm font-semibold text-fg">Setup checklist</h3>
        {confirmed ? (
          <Badge size="xs" tone="success" icon={<IconCheck />}>
            Confirmed {relativeTime(setup.confirmed_at)}
          </Badge>
        ) : (
          <span className="text-xs text-fg-subtle">Filled in from the plan. Check it, then confirm.</span>
        )}
        <span className="flex-1" />
        {!draft && (
          <Button size="xs" variant="ghost" leftIcon={<IconPencil size={13} />} onClick={() => setDraft(draftOf(task, tz))}>
            Change
          </Button>
        )}
      </div>

      {draft ? (
        <ChecklistEditor task={task} tz={tz} draft={draft} onChange={setDraft} onCancel={() => setDraft(null)} saving={saving} onSave={async (patch) => (await ops.update(id, patch, 'Checklist saved')) && setDraft(null)} />
      ) : (
        <>
          <dl className="divide-y divide-border text-sm">
            <Row icon={IconClock} label="When it runs">
              {d.text}
              {d.zone && <span className="ml-1.5 text-xs text-fg-subtle">{d.zone}</span>}
            </Row>
            <Row icon={SourceIcon} label="What it reads">
              <span className={cn(setup.source.kind === 'none' && 'text-fg-subtle', 'break-all')}>{sourceText(setup.source)}</span>
            </Row>
            <Row icon={IconInbox} label="What you get">
              {setup.output || <span className="text-fg-subtle">Not set</span>}
            </Row>
            <Row icon={IconShieldCheck} label="Without asking">
              {approvalText(setup)}
            </Row>
            {setup.source.kind !== 'none' && (
              <Row icon={IconAlertTriangle} label={`If ${missingWhat(setup.source)} isn’t there`}>
                {MISSING[setup.if_missing]}
              </Row>
            )}
          </dl>
          <div className="flex flex-wrap items-center gap-2 border-t border-border bg-sunken/40 px-4 py-2.5">
            <Tooltip content="Runs it once now without sending, changing or running anything, and shows what it would do">
              <span>
                <Button size="sm" variant="secondary" leftIcon={<IconFlask size={15} />} disabled={!canTry} loading={ops.isBusy('dryRun', id)} onClick={onTry}>
                  Try it first
                </Button>
              </span>
            </Tooltip>
            <span className="min-w-0 flex-1 text-xs text-fg-subtle">Nothing is sent or changed. You see what it would do.</span>
            {!confirmed && (
              <Button size="sm" variant="primary" leftIcon={<IconCheck size={15} />} loading={saving} onClick={confirm}>
                Looks right
              </Button>
            )}
          </div>
        </>
      )}
    </section>
  )
}

function Row({ icon: I, label, children }: { icon: Icon; label: string; children: ReactNode }) {
  return (
    <div className="flex gap-3 px-4 py-2.5">
      <dt className="flex w-44 shrink-0 items-center gap-2 text-fg-subtle">
        <I size={15} className="shrink-0" />
        {label}
      </dt>
      <dd className="min-w-0 flex-1 text-fg">{children}</dd>
    </div>
  )
}

// ---------------------------------------------------------------------------- editing
interface Draft {
  schedule: TaskSchedule
  kind: TaskSource['kind']
  target: string
  output: string
  approval: TaskSetup['approval']
  if_missing: MissingSourcePolicy
}

function draftOf(task: Task, tz: string): Draft {
  const setup = task.setup as TaskSetup
  return {
    schedule: task.schedule ?? { type: 'recurring', frequency: 'daily', time: '09:00', timezone: tz },
    kind: setup.source.kind,
    target: setup.source.target,
    output: setup.output,
    approval: setup.approval,
    if_missing: setup.if_missing
  }
}

function ChecklistEditor({
  task,
  tz,
  draft,
  onChange,
  onCancel,
  onSave,
  saving
}: {
  task: Task
  tz: string
  draft: Draft
  onChange: (d: Draft) => void
  onCancel: () => void
  onSave: (patch: { schedule?: TaskSchedule; setup?: TaskSetupAnswers }) => Promise<unknown>
  saving: boolean
}) {
  const { options } = useToolNames()
  const set = (patch: Partial<Draft>) => onChange({ ...draft, ...patch })
  const routine = draft.schedule.type === 'recurring' || draft.schedule.type === 'triggered'
  const needsTarget = draft.kind !== 'none'
  const valid = !needsTarget || draft.target.trim().length > 0
  const save = () => {
    const changedSchedule = JSON.stringify(draft.schedule) !== JSON.stringify(task.schedule)
    const setup: TaskSetupAnswers = {
      source: { kind: draft.kind, target: draft.target.trim() },
      output: draft.output,
      approval: draft.approval,
      if_missing: draft.if_missing,
      confirm: true
    }
    // a task that stops repeating has no checklist: only its schedule is saved
    return onSave({ ...(changedSchedule ? { schedule: draft.schedule } : {}), ...(routine ? { setup } : {}) })
  }

  return (
    <div className="space-y-5 px-4 py-4">
      <Field label="When it runs">
        <div className="rounded-xl border border-border-strong bg-elevated p-4">
          <ScheduleEditor value={draft.schedule} onChange={(schedule) => set({ schedule })} defaultTimezone={tz} />
        </div>
      </Field>
      <Field label="What it reads" hint="Sentient looks that it is there before each run.">
        <div className="space-y-2">
          <SegmentedControl aria-label="What it reads" value={draft.kind} onChange={(kind) => set({ kind, target: kind === task.setup?.source.kind ? (task.setup?.source.target ?? '') : '' })} options={SOURCE_KINDS} />
          {draft.kind === 'page' && <Input value={draft.target} onChange={(e) => set({ target: e.target.value })} placeholder="https://example.com/events" aria-label="Web page address" />}
          {draft.kind === 'folder' && <Input value={draft.target} onChange={(e) => set({ target: e.target.value })} placeholder="C:\Users\you\Documents\Reports" aria-label="Folder" />}
          {draft.kind === 'app' && <Select aria-label="App" value={draft.target} onValueChange={(target) => set({ target })} options={options} placeholder="Choose an app" className="w-64" />}
        </div>
      </Field>
      <Field label="What you get">
        <Input value={draft.output} onChange={(e) => set({ output: e.target.value })} placeholder="An email with a short summary" maxLength={300} aria-label="What you get" />
      </Field>
      <Field label="Without asking">
        <SegmentedControl
          aria-label="Without asking"
          value={draft.approval}
          onChange={(approval) => set({ approval })}
          options={[
            { value: 'plan', label: 'Go ahead with the plan' },
            { value: 'ask', label: 'Ask me first' }
          ]}
        />
        <p className="mt-1.5 text-xs text-fg-subtle">
          {draft.approval === 'ask' ? 'It asks you before it sends, posts, buys, deletes or runs anything.' : 'It does what the plan says. Your rules in Settings still apply.'}
        </p>
      </Field>
      {draft.kind !== 'none' && (
        <Field label="If it isn’t there">
          <SegmentedControl
            aria-label="If it isn't there"
            value={draft.if_missing}
            onChange={(if_missing) => set({ if_missing })}
            options={[
              { value: 'skip', label: 'Skip quietly' },
              { value: 'notify', label: 'Skip and tell me' },
              { value: 'retry', label: 'Try again later' }
            ]}
          />
        </Field>
      )}
      <div className="flex justify-end gap-1.5">
        <Button size="sm" variant="ghost" onClick={onCancel} disabled={saving}>
          Cancel
        </Button>
        <Button size="sm" variant="primary" leftIcon={<IconCheck size={15} />} loading={saving} disabled={!valid} onClick={() => void save()}>
          Save and confirm
        </Button>
      </div>
    </div>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-fg-subtle">{label}</span>
        {hint && <span className="text-xs text-fg-faint">{hint}</span>}
      </div>
      {children}
    </div>
  )
}
