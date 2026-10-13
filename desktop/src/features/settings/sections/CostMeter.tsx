import { IconLock } from '@tabler/icons-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, Card, ProgressBar, SegmentedControl, Skeleton, type Tone } from '@/components/ui'
import { useKeepGoing, useUsageCost } from '@/hooks/core'
import { errorMessage } from '@/lib/api'
import { modelShortName, ROLE_META } from '@/lib/models'
import type { BudgetState, CostPeriod, CostRow, RoleName } from '@/lib/types'
import { formatNumber } from '@/lib/utils'
import { SchemaForm } from '../SchemaForm'

export const BUDGET_KEYS = ['models.monthly_budget_usd', 'models.budget_at_limit']

/** Dollars, with small amounts kept readable: $0.0042 shows as "$0.0042", not "$0.00". */
export function formatUsd(usd: number): string {
  if (usd === 0) return '$0.00'
  if (usd < 0.01) return `$${usd.toPrecision(2)}`
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' }).format(usd)
}

/**
 * The cost meter (#148): what paid cloud models cost today, this week and this month, by model and by role.
 * Models on the user's Claude or ChatGPT plan show as "On your plan", local ones as free. Then the monthly budget.
 */
export function CostMeter() {
  const cost = useUsageCost()
  const [period, setPeriod] = useState<CostPeriod['key']>('month')
  const data = cost.data
  const p = data?.periods.find((x) => x.key === period)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-fg">What your models cost</h3>
          <p className="text-xs text-fg-subtle">Paid cloud models in dollars. Your Claude or ChatGPT plan and this computer cost nothing extra.</p>
        </div>
        <SegmentedControl
          size="sm"
          className="shrink-0 whitespace-nowrap"
          value={period}
          onChange={setPeriod}
          aria-label="Period"
          options={[
            { value: 'today', label: 'Today' },
            { value: 'week', label: 'This week' },
            { value: 'month', label: 'This month' }
          ]}
        />
      </div>

      {cost.isLoading ? (
        <div className="grid grid-cols-3 gap-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-20 rounded-xl" />
          ))}
        </div>
      ) : cost.isError ? (
        <Alert tone="danger" title="Couldn't load costs">
          {errorMessage(cost.error)}
        </Alert>
      ) : data && p ? (
        <>
          {data.local_only && (
            <div className="flex items-center gap-2 rounded-lg border border-border bg-sunken/60 px-3 py-2 text-xs text-fg-muted">
              <IconLock size={14} className="shrink-0 text-fg-subtle" />
              Local only is on: nothing is sent to a model outside this computer. Change it in the model menu at the top.
            </div>
          )}
          <div className="grid grid-cols-3 gap-3">
            <Tile
              label="Spent"
              value={formatUsd(p.cost_usd)}
              sub={
                p.paid_calls
                  ? `${p.paid_calls} paid ${p.paid_calls === 1 ? 'call' : 'calls'}${p.unpriced_calls ? `, ${p.unpriced_calls} without a known price` : ''}`
                  : 'No paid calls'
              }
            />
            <Tile
              label="On your plan"
              value={p.plan_calls ? `${formatNumber(p.plan_tokens)} tokens` : 'None'}
              sub={p.plan_calls ? `${p.plan_calls} ${p.plan_calls === 1 ? 'call' : 'calls'}, no extra cost` : 'Claude Code or a ChatGPT plan'}
            />
            <Tile
              label="On this computer"
              value={p.local_calls ? `${formatNumber(p.local_tokens)} tokens` : 'None'}
              sub={p.local_calls ? `${p.local_calls} ${p.local_calls === 1 ? 'call' : 'calls'}, free` : 'Local models are free'}
            />
          </div>
          <BudgetCard budget={data.budget} />
          <div className="grid grid-cols-2 gap-3">
            <CostTable
              title="By model"
              empty={`No model calls ${p.label.toLowerCase()}.`}
              rows={p.by_model.map((m) => ({ key: m.model, label: modelShortName(m.model), title: m.model, row: m, plan: m.billing === 'plan', local: m.billing === 'local' }))}
            />
            <CostTable
              title="By role"
              empty={`No model calls ${p.label.toLowerCase()}.`}
              rows={p.by_role.map((r) => ({
                key: r.role,
                label: roleLabel(r.role),
                title: r.role,
                row: r,
                plan: r.plan_calls > 0,
                local: r.local_calls === r.calls
              }))}
            />
          </div>
        </>
      ) : null}

      <SchemaForm section="models" title="Monthly budget" include={BUDGET_KEYS} />
    </div>
  )
}

function roleLabel(role: string): string {
  return ROLE_META[role as RoleName]?.label ?? (role === 'other' || role === 'background' ? 'Background' : role)
}

function Tile({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <Card className="px-4 py-3.5">
      <div className="text-xs text-fg-subtle">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums tracking-tight text-fg">{value}</div>
      <div className="mt-0.5 truncate text-2xs text-fg-subtle" title={sub}>
        {sub}
      </div>
    </Card>
  )
}

const BUDGET_TONE: Record<BudgetState['state'], Tone> = { off: 'neutral', ok: 'accent', warning: 'warning', reached: 'danger' }

function BudgetCard({ budget }: { budget: BudgetState }) {
  const keep = useKeepGoing()
  if (budget.state === 'off') {
    return (
      <Card className="px-4 py-3 text-xs text-fg-subtle">
        No monthly budget yet. Set one below and Sentient warns you at 80% and pauses paid cloud models at 100%.
      </Card>
    )
  }
  const share = Math.min(1, budget.spent_usd / budget.monthly_usd)
  return (
    <Card className="space-y-2.5 px-4 py-3.5">
      <div className="flex items-baseline justify-between gap-3">
        <div className="text-sm font-medium text-fg">
          {formatUsd(budget.spent_usd)} <span className="font-normal text-fg-subtle">of {formatUsd(budget.monthly_usd)} this month</span>
        </div>
        <div className="flex items-center gap-2">
          {budget.kept_going && <Badge size="xs">Kept going</Badge>}
          <Badge size="xs" tone={BUDGET_TONE[budget.state]}>
            {budget.percent ?? 0}%
          </Badge>
        </div>
      </div>
      <ProgressBar value={share} tone={BUDGET_TONE[budget.state]} />
      {budget.message && (
        <div className="flex items-center justify-between gap-3">
          <p className="text-xs text-fg-muted">{budget.message}</p>
          {budget.state === 'reached' && budget.at_limit === 'ask' && !budget.kept_going && (
            <Button
              size="sm"
              variant="secondary"
              loading={keep.isPending}
              onClick={() =>
                keep.mutate(undefined, {
                  onSuccess: () => toast.success('Paid cloud models can be used again this month'),
                  onError: (e) => toast.error("Couldn't keep going", { description: errorMessage(e) })
                })
              }
            >
              Keep going this month
            </Button>
          )}
        </div>
      )}
    </Card>
  )
}

function CostTable({
  title,
  rows,
  empty
}: {
  title: string
  empty: string
  rows: Array<{ key: string; label: string; title: string; row: CostRow; plan: boolean; local: boolean }>
}) {
  return (
    <Card className="p-4">
      <h3 className="mb-3 text-sm font-semibold text-fg">{title}</h3>
      <div className="space-y-2">
        {rows.slice(0, 10).map((r) => (
          <div key={r.key} className="flex items-baseline justify-between gap-3 text-xs">
            <span className="min-w-0 truncate font-mono text-fg" title={r.title}>
              {r.label}
            </span>
            <span className="flex shrink-0 items-baseline gap-2 tabular-nums">
              <span className="text-fg-subtle">
                {formatNumber(r.row.prompt_tokens + r.row.completion_tokens)} tokens · {r.row.calls}
              </span>
              <span className="w-24 text-right font-medium text-fg">{costLabel(r)}</span>
            </span>
          </div>
        ))}
        {!rows.length && <p className="text-xs text-fg-subtle">{empty}</p>}
      </div>
    </Card>
  )
}

function costLabel(r: { row: CostRow; plan: boolean; local: boolean }): string {
  if (r.row.cost_usd !== null) return formatUsd(r.row.cost_usd) + (r.row.unpriced_calls ? '+' : '')
  if (r.row.unpriced_calls) return 'Price unknown'
  return r.plan ? 'On your plan' : 'Free'
}
