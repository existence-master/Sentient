/** Script jobs (docs/API.md §16): what the script watches, when it acts, the code, the last check and "Test it". */
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import {
  IconAlertCircle,
  IconArrowsDiff,
  IconBell,
  IconBellRinging,
  IconCheck,
  IconChevronDown,
  IconCircleCheck,
  IconCode,
  IconCopy,
  IconPlayerPlay,
  IconRadar,
  IconRocket,
  IconShieldCheck,
  IconX,
  type Icon
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useMemo, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { IconPencil } from '@tabler/icons-react'
import { Alert, Button, IconButton, JsonView, Textarea } from '@/components/ui'
import { errorMessage } from '@/lib/api'
import { useScriptTest, useScriptUpdate } from '@/hooks/tasks'
import { scriptOf } from '../meta'
import type { SandboxResult, ScriptCondition, ScriptThen, Task, TaskScript } from '@/lib/types'
import { cn, copyText, formatDuration, relativeTime, truncate } from '@/lib/utils'
import { useNow, useUserTimezone } from '../hooks'
import { SectionHeading } from '../parts'
import { nextRunDate, upcomingPhrase } from '../schedule'

hljs.registerLanguage('python', python)

interface Choice<T extends string> {
  value: T
  title: string
  body: string
  icon: Icon
}

const CONDITIONS: Array<Choice<ScriptCondition>> = [
  { value: 'changed', title: 'Tell me when something changes', body: 'Each check is compared with the one before.', icon: IconArrowsDiff },
  { value: 'alert', title: 'Tell me when the script raises an alert', body: 'The script decides, for example when a price drops below your target.', icon: IconBellRinging }
]

const THENS: Array<Choice<ScriptThen>> = [
  { value: 'notify', title: 'Just notify me', body: 'You get a notification with what it found.', icon: IconBell },
  { value: 'run', title: 'Run the task', body: 'Sentient picks it up from there and follows the plan.', icon: IconRocket }
]

export function ScriptJobSection({ task }: { task: Task }) {
  const script = scriptOf(task)
  const update = useScriptUpdate()
  const test = useScriptTest()
  const now = useNow(60_000)
  const tz = useUserTimezone()
  const next = nextRunDate(task, now)
  const editable = !['processing', 'archived', 'planning', 'declined'].includes(task.status)
  const [draft, setDraft] = useState<string | null>(null)

  if (!script) {
    return (
      <section className="space-y-2.5">
        <SectionHeading icon={<IconRadar />} title="What Sentient watches" />
        <div className="rounded-xl border border-dashed border-border px-4 py-6 text-center text-sm text-fg-subtle">
          {task.status === 'planning' ? 'Sentient is writing the script. It will show up here in a moment.' : 'No script is attached to this task yet.'}
        </div>
      </section>
    )
  }

  const change = (patch: Partial<Pick<TaskScript, 'condition' | 'then'>>) =>
    update.mutate(
      { taskId: task.task_id, script: patch },
      {
        onSuccess: () => toast.success('Saved'),
        onError: (e) => toast.error('Couldn’t change that yet', { description: errorMessage(e) })
      }
    )

  const dirty = draft !== null && draft !== script.code
  const runTest = () =>
    test.mutate(
      { taskId: task.task_id, code: dirty ? (draft as string) : undefined },
      { onError: (e) => toast.error('Couldn’t run the test', { description: errorMessage(e) }) }
    )
  const saveCode = () =>
    update.mutate(
      { taskId: task.task_id, script: { code: draft as string } },
      {
        onSuccess: () => {
          setDraft(null)
          toast.success('Script saved', { description: 'The next check uses your changes and starts fresh.' })
        },
        onError: (e) => toast.error('That code has a problem', { description: errorMessage(e) })
      }
    )

  return (
    <section className="space-y-4">
      <SectionHeading icon={<IconRadar />} title="What Sentient watches" description="A small script checks on schedule. No AI is used unless it finds something." />

      {task.status === 'approval_pending' && (
        <Alert tone="accent" icon={<IconShieldCheck />} title="Have a look at the script before you approve">
          Once approved it runs on its schedule by itself. Scripts can look things up, but they can’t send, buy or delete anything. If the code has a mistake, approving tells you what to fix.
        </Alert>
      )}

      <div className="grid gap-3 @2xl:grid-cols-2">
        <ChoiceGroup label="Let me know" options={CONDITIONS} value={script.condition} disabled={!editable || update.isPending} onChange={(condition) => change({ condition })} />
        <ChoiceGroup label="Then" options={THENS} value={script.then} disabled={!editable || update.isPending} onChange={(then) => change({ then })} />
      </div>

      <LastCheck script={script} next={next ? upcomingPhrase(next, tz, now) : null} now={now} />

      <CodeViewer
        code={script.code}
        editing={draft !== null}
        draft={draft ?? ''}
        onDraft={setDraft}
        note={
          dirty
            ? 'You have unsaved changes. Test it tries them without saving.'
            : 'Test it runs the script once now. It won’t notify you or change what it remembers.'
        }
        actions={
          <>
            {draft === null ? (
              editable && (
                <Button size="xs" variant="ghost" leftIcon={<IconPencil size={13} />} onClick={() => setDraft(script.code)}>
                  Edit
                </Button>
              )
            ) : (
              <>
                <Button size="xs" variant="ghost" onClick={() => setDraft(null)} disabled={update.isPending}>
                  Cancel
                </Button>
                <Button size="xs" variant="secondary" loading={update.isPending} disabled={!dirty || !draft?.trim()} onClick={saveCode}>
                  Save
                </Button>
              </>
            )}
            <Button size="xs" variant="primary" leftIcon={<IconPlayerPlay size={13} />} loading={test.isPending} onClick={runTest}>
              Test it
            </Button>
          </>
        }
      />

      <AnimatePresence initial={false}>{test.data && <TestResult key="result" result={test.data} script={script} onClose={() => test.reset()} />}</AnimatePresence>
    </section>
  )
}

function ChoiceGroup<T extends string>({ label, options, value, disabled, onChange }: { label: string; options: Array<Choice<T>>; value: T; disabled: boolean; onChange: (v: T) => void }) {
  return (
    <div role="radiogroup" aria-label={label} className="rounded-xl border border-border bg-surface p-1.5">
      <div className="px-2 pb-1.5 pt-1 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className="space-y-1">
        {options.map((o) => {
          const active = o.value === value
          return (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={disabled && !active}
              onClick={() => !active && onChange(o.value)}
              className={cn(
                'flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors disabled:opacity-50',
                active ? 'bg-accent/10 ring-1 ring-accent/30' : 'hover:bg-hover'
              )}
            >
              <span className={cn('mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full border', active ? 'border-accent bg-accent text-accent-fg' : 'border-border-strong')}>
                {active && <IconCheck size={10} stroke={3} />}
              </span>
              <span className="min-w-0 flex-1">
                <span className={cn('block text-sm font-medium', active ? 'text-fg' : 'text-fg-muted')}>{o.title}</span>
                <span className="block text-xs text-fg-subtle">{o.body}</span>
              </span>
              <o.icon size={16} className={cn('mt-0.5 shrink-0', active ? 'text-accent-text' : 'text-fg-faint')} />
            </button>
          )
        })}
      </div>
    </div>
  )
}

function resultSummary(v: unknown): string {
  if (v === null || v === undefined) return 'Nothing yet'
  if (typeof v === 'object' && !Array.isArray(v)) {
    const o = v as Record<string, unknown>
    if ('alert' in o) return o.alert ? 'Raised an alert' : 'All quiet, no alert'
    return `${Object.keys(o).length} value${Object.keys(o).length === 1 ? '' : 's'} saved`
  }
  if (Array.isArray(v)) return `${v.length} item${v.length === 1 ? '' : 's'}`
  return truncate(String(v), 60)
}

function LastCheck({ script, next, now }: { script: TaskScript; next: string | null; now: Date }) {
  const hasObject = script.last_result !== null && script.last_result !== undefined && typeof script.last_result === 'object'
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-1 gap-2 @xl:grid-cols-3">
        <Tile label="Last checked" value={script.last_run_at ? relativeTime(script.last_run_at, now.getTime()) : 'Not yet'} />
        <Tile label="Next check" value={next ?? 'Not scheduled'} />
        <Tile label="Last result" value={script.last_error ? 'Didn’t work' : resultSummary(script.last_result)} danger={!!script.last_error} />
      </div>
      {script.last_error && (
        <Alert tone="danger" icon={<IconAlertCircle />} title="The last check didn’t work">
          {script.last_error}
        </Alert>
      )}
      {hasObject && (
        <details className="group rounded-xl border border-border bg-sunken/40">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 px-4 py-2 text-xs font-medium text-fg-muted hover:text-fg">
            <IconChevronDown size={13} className="-rotate-90 transition-transform group-open:rotate-0" />
            What it found last time
          </summary>
          <div className="border-t border-border px-4 py-3">
            <JsonView value={script.last_result} collapsedDepth={2} />
          </div>
        </details>
      )}
    </div>
  )
}

function Tile({ label, value, danger }: { label: string; value: string; danger?: boolean }) {
  return (
    <div className="rounded-xl border border-border bg-sunken/40 px-3.5 py-2.5">
      <div className="text-2xs font-medium uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className={cn('mt-0.5 truncate text-sm font-medium', danger ? 'text-danger' : 'text-fg')}>{value}</div>
    </div>
  )
}

const escapeHtml = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/** Read-only, highlighted, collapsible code with line numbers. */
export function CodeViewer({
  code,
  note,
  actions,
  collapsedLines = 16,
  editing = false,
  draft = '',
  onDraft
}: {
  code: string
  note?: string
  actions?: ReactNode
  collapsedLines?: number
  editing?: boolean
  draft?: string
  onDraft?: (v: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const source = code.replace(/\s+$/, '')
  const lines = source.split('\n')
  const html = useMemo(() => {
    try {
      return hljs.highlight(source, { language: 'python' }).value
    } catch {
      return escapeHtml(source)
    }
  }, [source])
  const long = lines.length > collapsedLines

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-sunken">
      <div className="flex min-h-10 flex-wrap items-center gap-x-2 gap-y-1 border-b border-border px-3 py-1.5">
        <IconCode size={15} className="text-fg-subtle" />
        <span className="text-sm font-medium text-fg">The script</span>
        <span className="text-2xs text-fg-subtle">Python · {lines.length} lines{editing ? ' · editing' : ''}</span>
        <span className="flex-1" />
        <IconButton
          size="xs"
          label={copied ? 'Copied' : 'Copy code'}
          icon={copied ? <IconCheck size={13} /> : <IconCopy size={13} />}
          onClick={() =>
            void copyText(source).then((ok) => {
              if (!ok) return
              setCopied(true)
              window.setTimeout(() => setCopied(false), 1400)
            })
          }
        />
        {actions}
      </div>
      {note && <div className="border-b border-border px-3.5 py-1.5 text-2xs text-fg-subtle">{note}</div>}
      {editing ? (
        <Textarea
          autoFocus
          autoGrow
          minHeight={320}
          maxHeight={640}
          spellCheck={false}
          value={draft}
          onChange={(e) => onDraft?.(e.target.value)}
          aria-label="Script code"
          className="rounded-none border-0 bg-transparent px-4 py-3 font-mono text-[12.5px] leading-[1.7] shadow-none hover:border-0 focus:border-0 focus:ring-0"
        />
      ) : (
      <div className="relative">
        <div className={cn('selectable flex overflow-x-auto font-mono text-[12.5px] leading-[1.7]', !open && long && 'max-h-[364px] overflow-y-hidden')}>
          <div aria-hidden className="select-none border-r border-border py-3 pl-3 pr-2.5 text-right text-fg-faint">
            {lines.map((_, i) => (
              <div key={i}>{i + 1}</div>
            ))}
          </div>
          <pre className="hljs m-0 min-w-0 flex-1 py-3 pl-3.5 pr-4 font-mono">
            <code dangerouslySetInnerHTML={{ __html: html }} />
          </pre>
        </div>
        {long && !open && <div aria-hidden className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-sunken to-transparent" />}
      </div>
      )}
      {long && !editing && (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center justify-center gap-1 border-t border-border py-1.5 text-xs font-medium text-fg-subtle hover:bg-hover hover:text-fg"
        >
          {open ? 'Show less' : `Show all ${lines.length} lines`}
          <IconChevronDown size={13} className={cn('transition-transform', open && 'rotate-180')} />
        </button>
      )}
    </div>
  )
}

function describeOutcome(r: SandboxResult, s: TaskScript): { accent: boolean; icon: Icon; title: string; body: string } | null {
  if (!r.ok) return null
  const then = s.then === 'run' ? 'Sentient would run the task' : 'you would get a notification'
  if (s.condition === 'alert') {
    const obj = r.result && typeof r.result === 'object' ? (r.result as Record<string, unknown>) : null
    if (obj?.alert === true) {
      const msg = typeof obj.message === 'string' ? `“${obj.message}” ` : ''
      return { accent: true, icon: IconBellRinging, title: 'This check would alert you', body: `${msg}So ${then}.` }
    }
    return { accent: false, icon: IconCircleCheck, title: 'No alert this time', body: 'Everything looks normal, so you wouldn’t hear from me.' }
  }
  if (s.last_result === null || s.last_result === undefined) {
    return { accent: false, icon: IconCircleCheck, title: 'This would be the first check', body: 'I’ll remember the result and tell you when a later check is different.' }
  }
  const same = JSON.stringify(r.result ?? null) === JSON.stringify(s.last_result)
  return same
    ? { accent: false, icon: IconCircleCheck, title: 'Same as last time', body: 'Nothing changed, so you wouldn’t hear from me.' }
    : { accent: true, icon: IconArrowsDiff, title: 'Something changed since the last check', body: `So ${then}.` }
}

function TestResult({ result, script, onClose }: { result: SandboxResult; script: TaskScript; onClose: () => void }) {
  const outcome = describeOutcome(result, script)
  const calls = Array.isArray(result.tool_calls) ? result.tool_calls.length : Number(result.tool_calls) || 0
  const hasResult = result.result !== null && result.result !== undefined

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 4 }}
      className={cn('overflow-hidden rounded-xl border bg-surface shadow-soft', result.ok ? 'border-success/25' : 'border-danger/30')}
    >
      <div className="flex items-center gap-2.5 border-b border-border px-4 py-2.5">
        {result.ok ? <IconCircleCheck size={17} className="text-success" /> : <IconAlertCircle size={17} className="text-danger" />}
        <span className="text-sm font-semibold text-fg">{result.ok ? 'Test finished' : 'The script hit a problem'}</span>
        <span className="text-xs text-fg-subtle">
          {formatDuration(result.duration_ms)} · {calls} tool {calls === 1 ? 'use' : 'uses'}
          {result.backend === 'docker' ? ' · in Docker' : ''}
        </span>
        <span className="flex-1" />
        <IconButton size="xs" label="Close test result" icon={<IconX size={13} />} onClick={onClose} />
      </div>
      <div className="space-y-3 p-4">
        {outcome && (
          <div className={cn('flex items-start gap-2.5 rounded-lg px-3 py-2.5 text-sm', outcome.accent ? 'bg-accent/10' : 'bg-active')}>
            <outcome.icon size={17} className={cn('mt-0.5 shrink-0', outcome.accent ? 'text-accent-text' : 'text-fg-muted')} />
            <div className="min-w-0">
              <div className="font-medium text-fg">{outcome.title}</div>
              <div className="mt-0.5 text-fg-muted">{outcome.body}</div>
            </div>
          </div>
        )}
        {result.error && (
          <Alert tone="danger" title="What went wrong">
            <span className="selectable whitespace-pre-wrap font-mono text-xs">{result.error}</span>
          </Alert>
        )}
        {hasResult && (
          <Block label="What it returned">
            {typeof result.result === 'object' ? <JsonView value={result.result} /> : <pre className="selectable whitespace-pre-wrap font-mono text-[12px] text-fg">{String(result.result)}</pre>}
          </Block>
        )}
        {result.stdout?.trim() && (
          <Block label="What it printed">
            <pre className="selectable max-h-48 overflow-auto whitespace-pre-wrap font-mono text-[12px] text-fg">{result.stdout}</pre>
          </Block>
        )}
        {result.stderr?.trim() && (
          <Block label="Warnings">
            <pre className="selectable max-h-48 overflow-auto whitespace-pre-wrap font-mono text-[12px] text-warning">{result.stderr}</pre>
          </Block>
        )}
        {result.files_created?.length > 0 && (
          <Block label="Files it made">
            <div className="flex flex-wrap gap-1.5">
              {result.files_created.map((f) => (
                <span key={f} className="rounded-md border border-border bg-elevated px-2 py-0.5 font-mono text-2xs text-fg-muted">
                  {f}
                </span>
              ))}
            </div>
          </Block>
        )}
        <p className="text-2xs text-fg-subtle">This was only a test. You weren’t notified and it wasn’t saved as the last result.</p>
      </div>
    </motion.div>
  )
}

function Block({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className="rounded-lg border border-border bg-sunken/60 px-3 py-2.5">{children}</div>
    </div>
  )
}
