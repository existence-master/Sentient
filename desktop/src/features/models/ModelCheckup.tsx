import { IconAlertCircle, IconAlertTriangle, IconChevronRight, IconCircleCheck, IconCircleDashed, IconStethoscope } from '@tabler/icons-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, Card, Spinner } from '@/components/ui'
import { useConfigEditor } from '@/hooks/config'
import { useModelCheckup, useOllamaPull, useSetRoles, type CheckupRow } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import { modelShortName, ROLE_META } from '@/lib/models'
import type { CheckupAction, CheckupCheck, CheckupStatus, RoleName } from '@/lib/types'
import { cn } from '@/lib/utils'

const STATUS: Record<CheckupStatus, { icon: typeof IconCircleCheck; className: string; text: string }> = {
  pass: { icon: IconCircleCheck, className: 'text-success', text: 'Works' },
  warn: { icon: IconAlertTriangle, className: 'text-warning', text: 'Works, with a warning' },
  fail: { icon: IconAlertCircle, className: 'text-danger', text: 'Needs a fix' },
  skip: { icon: IconCircleDashed, className: 'text-fg-faint', text: 'Not checked' }
}

/**
 * "Check my models": tries each role's model the way Sentient uses it and shows pass, warning or failure with a
 * plain fix. Informational: config only changes when the user presses a fix button.
 *
 * `roles` checks only these models (onboarding checks its picks before they are saved); `onUseModel` replaces the
 * default "switch this role" fix for the same reason.
 */
export function ModelCheckup({
  roles,
  onUseModel,
  autoStart,
  className
}: {
  roles?: Partial<Record<RoleName, string | null>>
  onUseModel?: (role: RoleName, model: string) => void
  /** Start checking as soon as it shows (the title bar's "Check my models"). */
  autoStart?: boolean
  className?: string
}) {
  const c = useModelCheckup()
  const autoStarted = useRef(false)
  useEffect(() => {
    if (!autoStart || autoStarted.current) return
    autoStarted.current = true
    void c.run(roles)
  }, [autoStart, c, roles])
  const started = c.rows.length > 0 || c.running || !!c.error
  const issues = c.rows.filter((r) => r.result && (r.result.status === 'warn' || r.result.status === 'fail')).length
  const own = c.rows.filter((r) => r.model)
  const shared = c.rows.filter((r) => !r.model).map((r) => ROLE_META[r.role].label)

  return (
    <Card className={cn('overflow-hidden', className)}>
      <div className="flex items-start gap-3 px-4 py-4">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated text-accent-text">
          <IconStethoscope size={18} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-fg">Check my models</span>
            {c.example && <Badge size="xs">Example result</Badge>}
          </div>
          <p className="mt-0.5 text-xs leading-relaxed text-fg-subtle">
            {c.running
              ? 'Trying each model in turn. Local models can take a minute the first time they load.'
              : c.status
                ? issues
                  ? `Found ${issues === 1 ? 'something' : `${issues} things`} to look at. Nothing changes unless you press a fix.`
                  : 'Every model passed.'
                : 'Tries each model the way Sentient uses it: a short reply, a tool call and, for local models, whether it fits on your graphics card.'}
          </p>
        </div>
        {c.running ? (
          <Button size="sm" variant="ghost" onClick={c.cancel}>
            Stop
          </Button>
        ) : (
          <Button size="sm" variant={started ? 'secondary' : 'primary'} onClick={() => void c.run(roles)}>
            {started ? 'Check again' : 'Check my models'}
          </Button>
        )}
      </div>

      {c.error && (
        <Alert tone="danger" title="Couldn't finish the check-up" className="mx-4 mb-4">
          {c.error}
        </Alert>
      )}

      {c.running && c.rows.length === 0 && (
        <div className="flex items-center gap-2 border-t border-border px-4 py-3 text-xs text-fg-muted">
          <Spinner size={14} /> Starting…
        </div>
      )}

      {c.rows.length > 0 && (
        <div className="divide-y divide-border border-t border-border">
          {own.map((row) => (
            <RoleRow key={row.role} row={row} running={c.running} onUseModel={onUseModel} />
          ))}
          {shared.length > 0 && (
            <div className="flex items-center gap-3 px-4 py-3 text-xs text-fg-subtle">
              <IconCircleDashed size={16} className="shrink-0 text-fg-faint" />
              {listText(shared)} {shared.length === 1 ? 'uses' : 'use'} the primary model, so {shared.length === 1 ? 'it shares' : 'they share'} its result.
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

function listText(items: string[]): string {
  return items.length < 2 ? (items[0] ?? '') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}

function RoleRow({ row, running, onUseModel }: { row: CheckupRow; running: boolean; onUseModel?: (role: RoleName, model: string) => void }) {
  const [open, setOpen] = useState(false)
  const meta = ROLE_META[row.role]
  const r = row.result
  const problems = r?.checks.filter((ch) => ch.status === 'warn' || ch.status === 'fail') ?? []

  let summary: React.ReactNode
  let icon: React.ReactNode
  if (!r) {
    icon = running ? <Spinner size={16} className="text-fg-muted" /> : <IconCircleDashed size={16} className="text-fg-faint" />
    summary = row.step ? `${row.step}…` : running ? 'Waiting' : 'Not checked'
  } else {
    const s = STATUS[r.status]
    icon = <s.icon size={16} className={s.className} />
    summary = r.status === 'pass' ? 'Everything works' : s.text
  }

  return (
    <div className="px-4 py-3">
      <div className="flex items-center gap-3">
        <span className="flex size-5 shrink-0 items-center justify-center">{icon}</span>
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-baseline gap-2">
            <span className="text-sm font-medium text-fg">{meta.label}</span>
            {row.model && <span className="truncate font-mono text-2xs text-fg-subtle">{modelShortName(row.model)}</span>}
          </div>
          <div className={cn('text-xs', r && r.status !== 'skip' ? STATUS[r.status].className : 'text-fg-subtle')}>{summary}</div>
        </div>
        {r && r.checks.length > 0 && (
          <button type="button" onClick={() => setOpen((o) => !o)} className="flex shrink-0 items-center gap-1 text-xs text-fg-muted hover:text-fg">
            {open ? 'Hide details' : 'Details'}
            <IconChevronRight size={13} className={cn('transition-transform', open && 'rotate-90')} />
          </button>
        )}
      </div>

      {(open ? r?.checks ?? [] : problems).length > 0 && (
        <ul className="mt-2.5 space-y-2 pl-8">
          {(open ? r?.checks ?? [] : problems).map((ch) => (
            <CheckLine key={ch.id} check={ch} role={row.role} onUseModel={onUseModel} />
          ))}
        </ul>
      )}
    </div>
  )
}

function CheckLine({ check, role, onUseModel }: { check: CheckupCheck; role: RoleName; onUseModel?: (role: RoleName, model: string) => void }) {
  const s = STATUS[check.status]
  return (
    <li className="flex items-start gap-2.5">
      <s.icon size={14} className={cn('mt-0.5 shrink-0', s.className)} />
      <div className="min-w-0 flex-1 text-xs leading-relaxed">
        <span className="font-medium text-fg">{check.label}.</span> <span className="text-fg-muted">{check.detail}</span>
        {check.fix && <div className="mt-0.5 text-fg">{check.fix}</div>}
      </div>
      {check.action && <FixButton action={check.action} role={role} onUseModel={onUseModel} />}
    </li>
  )
}

function FixButton({ action, role, onUseModel }: { action: CheckupAction; role: RoleName; onUseModel?: (role: RoleName, model: string) => void }) {
  const setRoles = useSetRoles()
  const { patch } = useConfigEditor()
  const pull = useOllamaPull()
  const [done, setDone] = useState(false)

  if (action.kind === 'pull_model') {
    if (pull.done) return <Badge tone="success">Downloaded</Badge>
    const label = pull.running ? (pull.progress !== null ? `Downloading ${Math.round(pull.progress * 100)}%` : 'Downloading') : action.label
    return (
      <div className="flex shrink-0 flex-col items-end gap-1">
        <Button size="xs" variant="secondary" loading={pull.running} onClick={() => void pull.pull(action.name)}>
          {label}
        </Button>
        {pull.error && <span className="max-w-48 truncate text-2xs text-danger">{pull.error}</span>}
      </div>
    )
  }

  if (done) return <Badge tone="success">Done. Check again to confirm</Badge>

  const apply = () => {
    if (action.kind === 'use_model') {
      if (onUseModel) {
        onUseModel(action.role, action.model)
        setDone(true)
        return
      }
      setRoles.mutate(
        { [action.role]: action.model },
        {
          onSuccess: () => {
            setDone(true)
            toast.success(`${ROLE_META[action.role].label} now uses ${modelShortName(action.model)}`)
          },
          onError: (e) => toast.error("Couldn't change the model", { description: errorMessage(e) })
        }
      )
    } else if (action.kind === 'set_reasoning') {
      patch({ models: { reasoning: { [action.role]: action.value } } }, { immediate: true })
      setDone(true)
    } else if (action.kind === 'set_context_length') {
      patch({ models: action.role ? { context_length_per_role: { [action.role]: action.value } } : { context_length: action.value } }, { immediate: true })
      setDone(true)
    }
  }

  return (
    <Button size="xs" variant="secondary" loading={setRoles.isPending} onClick={apply} aria-label={`${action.label} for ${ROLE_META[role].label}`}>
      {action.label}
    </Button>
  )
}
