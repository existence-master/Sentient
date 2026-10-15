import { IconAlertTriangle, IconCheck, IconShieldCheck, IconShieldExclamation, IconX } from '@tabler/icons-react'
import { useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import { Badge, Button, Switch, type Tone } from '@/components/ui'
import { useMcpActions } from '@/hooks/integrations'
import { errorMessage } from '@/lib/api'
import type { McpServer, McpToolReview, VetVerdict } from '@/lib/types'
import { cn } from '@/lib/utils'
import { RISK_LABEL, riskOf, toolLabel } from './meta'

export const VERDICT: Record<VetVerdict, { tone: Tone; label: string }> = {
  high: { tone: 'danger', label: 'High risk' },
  check: { tone: 'warning', label: 'Check this' },
  clear: { tone: 'success', label: 'Nothing found' }
}

const STATUS_LABEL: Record<McpToolReview['status'], { tone: Tone; label: string }> = {
  new: { tone: 'info', label: 'New' },
  changed: { tone: 'warning', label: 'Changed' },
  approved: { tone: 'success', label: 'Allowed' },
  declined: { tone: 'neutral', label: 'Declined' }
}

type Choice = Record<string, boolean>

function initialChoice(tools: McpToolReview[]): Choice {
  return Object.fromEntries(
    tools.map((t) => [t.mcp_name, t.status === 'new' || t.status === 'changed' ? t.suggested === 'approve' : t.status === 'approved'])
  )
}

/** The safety check of a server's tools: what each one says, its risk and findings, and Allow switches. Tools the
 * user hasn't answered for (new or changed) are not offered to Sentient. High-risk tools start switched off. */
export function McpReviewPanel({ server: s, onDone }: { server: McpServer; onDone?: () => void }) {
  const review = s.review
  const { review: save, setEnabled } = useMcpActions()
  const tools = useMemo(() => review?.tools ?? [], [review])
  const key = tools.map((t) => `${t.mcp_name}:${t.status}`).join('|')
  const [choice, setChoice] = useState<Choice>(() => initialChoice(tools))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => setChoice(initialChoice(tools)), [key])
  if (!review || !tools.length) return null

  const waiting = tools.filter((t) => t.status === 'new' || t.status === 'changed')
  const changed = tools.filter((t) => t.status === 'changed')
  const risky = tools.filter((t) => t.level === 'high')
  const allowed = tools.filter((t) => choice[t.mcp_name])
  const submit = (decisions: Record<string, 'approve' | 'decline'>, after?: () => void) =>
    save.mutate(
      { name: s.name, decisions },
      {
        onSuccess: (srv) => {
          const n = srv.tools.length
          toast.success(n ? `Sentient can use ${n} tool${n === 1 ? '' : 's'} from ${s.name}` : `Sentient won't use ${s.name}'s tools`)
          after?.()
          onDone?.()
        },
        onError: (e) => toast.error("Couldn't save your choices", { description: errorMessage(e) })
      }
    )
  const saveChoices = () => submit(Object.fromEntries(tools.map((t) => [t.mcp_name, choice[t.mcp_name] ? 'approve' : 'decline'])))
  const declineAll = () =>
    submit(Object.fromEntries(tools.map((t) => [t.mcp_name, 'decline'])), () =>
      setEnabled.mutate({ name: s.name, enabled: false }, { onError: (e) => toast.error(`Couldn't turn off ${s.name}`, { description: errorMessage(e) }) })
    )

  const title = changed.length
    ? `${s.name} changed ${changed.length === 1 ? 'a tool' : `${changed.length} tools`} since you approved ${changed.length === 1 ? 'it' : 'them'}`
    : waiting.length
      ? `Check ${s.name}'s tools before Sentient uses them`
      : `What Sentient may use from ${s.name}`
  const lead = changed.length
    ? 'A tool that changes what it says about itself can be trying to steer Sentient. Sentient stopped offering it until you look.'
    : waiting.length
      ? risky.length
        ? `Sentient's safety check found problems in ${risky.length === 1 ? 'one tool' : `${risky.length} tools`}. They start switched off.`
        : 'Sentient checked what each tool says about itself. Nothing is offered until you choose.'
      : 'Switch a tool off to stop Sentient from using it.'

  return (
    <div className={cn('border-t px-4 py-3', waiting.length ? (risky.length || changed.length ? 'border-danger/25 bg-danger/[0.04]' : 'border-warning/25 bg-warning/[0.04]') : 'border-border')}>
      <div className="flex items-start gap-2.5">
        {risky.length || changed.length ? <IconShieldExclamation size={18} className="mt-0.5 shrink-0 text-danger" /> : <IconShieldCheck size={18} className="mt-0.5 shrink-0 text-success" />}
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-fg">{title}</div>
          <p className="mt-0.5 text-xs text-fg-muted">{lead}</p>
        </div>
      </div>
      <ul className="mt-3 space-y-2">
        {tools.map((t) => (
          <ToolRow key={t.mcp_name} tool={t} on={!!choice[t.mcp_name]} onChange={(v) => setChoice((c) => ({ ...c, [t.mcp_name]: v }))} />
        ))}
      </ul>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <p className="mr-auto text-xs text-fg-subtle">
          {allowed.length} of {tools.length} allowed
        </p>
        <Button size="sm" variant="ghost" className="text-danger hover:text-danger" leftIcon={<IconX size={14} />} disabled={save.isPending} onClick={declineAll}>
          Decline all and turn off
        </Button>
        <Button size="sm" variant="primary" leftIcon={<IconCheck size={14} />} loading={save.isPending} onClick={saveChoices}>
          Save choices
        </Button>
      </div>
    </div>
  )
}

function ToolRow({ tool: t, on, onChange }: { tool: McpToolReview; on: boolean; onChange: (on: boolean) => void }) {
  const risk = RISK_LABEL[riskOf(t.risk)]
  const verdict = VERDICT[t.level]
  const status = STATUS_LABEL[t.status]
  return (
    <li className={cn('rounded-lg border bg-surface px-3 py-2.5', t.level === 'high' ? 'border-danger/30' : 'border-border')}>
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-sm font-medium text-fg">{toolLabel(t.mcp_name)}</span>
            <span className="font-mono text-2xs text-fg-subtle">{t.mcp_name}</span>
            <Badge size="xs" tone={status.tone}>
              {status.label}
            </Badge>
            <Badge size="xs" tone={risk.tone}>
              {risk.label}
            </Badge>
            <Badge size="xs" tone={verdict.tone} icon={t.level === 'high' ? <IconAlertTriangle /> : undefined}>
              {verdict.label}
            </Badge>
          </div>
          {t.status === 'changed' && t.previous_description !== undefined ? (
            <div className="mt-1.5 grid gap-1.5 text-xs sm:grid-cols-2">
              <div className="rounded-md border border-border bg-sunken/60 px-2 py-1.5">
                <div className="mb-0.5 text-2xs font-medium uppercase tracking-wide text-fg-subtle">Before</div>
                <div className="selectable break-words text-fg-muted">{t.previous_description || 'No description'}</div>
              </div>
              <div className="rounded-md border border-warning/30 bg-warning/[0.06] px-2 py-1.5">
                <div className="mb-0.5 text-2xs font-medium uppercase tracking-wide text-warning">Now</div>
                <div className="selectable break-words text-fg">{t.description}</div>
              </div>
            </div>
          ) : (
            <p className="selectable mt-1 break-words text-xs text-fg-muted">{t.description}</p>
          )}
          {t.findings.length > 0 && (
            <ul className="mt-1.5 space-y-0.5">
              {t.findings.map((f, i) => (
                <li key={i} className={cn('flex items-start gap-1.5 text-xs', f.level === 'error' ? 'text-danger' : 'text-warning')}>
                  <IconAlertTriangle size={12} className="mt-0.5 shrink-0" />
                  <span>{f.message}</span>
                </li>
              ))}
            </ul>
          )}
          {on && t.level === 'high' && <p className="mt-1.5 text-xs font-medium text-danger">Only allow this if you trust the server.</p>}
        </div>
        <label className="flex shrink-0 items-center gap-2 pt-0.5 text-xs text-fg-muted">
          {on ? 'Allow' : 'Decline'}
          <Switch size="sm" checked={on} onCheckedChange={onChange} aria-label={`Allow ${t.mcp_name}`} />
        </label>
      </div>
    </li>
  )
}
