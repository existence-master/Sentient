/** What Sentient checked its calls really did (#165): "Checked: file written, email sent", or what didn't happen. */
import { IconAlertTriangle, IconCircleCheck } from '@tabler/icons-react'
import { Alert, Tooltip } from '@/components/ui'
import type { OutcomeCheck } from '@/lib/types'
import { cn } from '@/lib/utils'

/** Failures a later call didn't settle. */
export function failedChecks(checks: OutcomeCheck[] | undefined): OutcomeCheck[] {
  return (checks ?? []).filter((c) => c.ok === false && !c.settled)
}

/**
 * Each tool call's failed check, by call id: live from its `tool_result`, else (a reloaded chat) the reply's checks
 * matched to its calls of the same tool in order.
 */
export function toolProblems(tools: Array<{ callId: string; name: string; check?: OutcomeCheck | null }>, checks?: OutcomeCheck[]): Record<string, string> {
  const left = [...(checks ?? [])]
  const out: Record<string, string> = {}
  for (const t of tools) {
    const i = left.findIndex((c) => c.tool === t.name)
    const c = (i >= 0 ? left[i] : undefined) ?? t.check
    if (i >= 0) left.splice(i, 1)
    if (c && c.ok === false && !c.settled) out[t.callId] = c.detail || 'It did not do what it said'
  }
  return out
}

function checkedLabels(checks: OutcomeCheck[]): string[] {
  return [...new Set(checks.filter((c) => c.ok).map((c) => c.action))]
}

export function OutcomeChecks({ checks, className }: { checks?: OutcomeCheck[]; className?: string }) {
  const failed = failedChecks(checks)
  const done = checkedLabels(checks ?? [])
  const readBack = (checks ?? []).some((c) => c.ok && c.how === 'read_back')
  if (failed.length) {
    return (
      <Alert tone="warning" icon={<IconAlertTriangle />} title="Needs your attention" className={className}>
        <ul className="space-y-1">
          {failed.map((c, i) => (
            <li key={`${c.tool}-${i}`} className="selectable break-words">
              {c.message || `${c.app}: ${c.detail}`}
            </li>
          ))}
        </ul>
        {done.length > 0 && <div className="mt-1.5 text-xs text-fg-subtle">Checked and fine: {done.join(', ')}.</div>}
      </Alert>
    )
  }
  if (!done.length) return null
  return (
    <Tooltip content={readBack ? 'Sentient read the result back itself, not just what the model said.' : 'Sentient checked what the app reported, not just what the model said.'}>
      <div className={cn('inline-flex items-center gap-1.5 text-xs text-fg-subtle', className)}>
        <IconCircleCheck size={13} className="shrink-0 text-success" />
        <span>Checked: {done.join(', ')}</span>
      </div>
    </Tooltip>
  )
}
