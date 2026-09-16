import { IconChartBar } from '@tabler/icons-react'
import { useMemo, useState } from 'react'
import { Card, EmptyState, SegmentedControl, Skeleton, Alert } from '@/components/ui'
import { useUsage } from '@/hooks/core'
import { errorMessage } from '@/lib/api'
import { modelShortName } from '@/lib/models'
import type { UsageReport } from '@/lib/types'
import { formatNumber } from '@/lib/utils'
import type { SectionProps } from '../SettingsPage'

export function UsageSection(_props: SectionProps) {
  const [days, setDays] = useState('30')
  const usage = useUsage(Number(days))
  const data = usage.data
  const total = data ? data.totals.prompt_tokens + data.totals.completion_tokens : 0
  const calls = data?.by_model.reduce((a, m) => a + m.calls, 0) ?? 0

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <p className="text-sm text-fg-muted">Tokens processed by every model, including background work.</p>
        <SegmentedControl
          size="sm"
          value={days}
          onChange={setDays}
          options={[
            { value: '7', label: '7 days' },
            { value: '30', label: '30 days' },
            { value: '90', label: '90 days' }
          ]}
        />
      </div>

      {usage.isLoading ? (
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-3">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-20 rounded-xl" />
            ))}
          </div>
          <Skeleton className="h-56 rounded-xl" />
        </div>
      ) : usage.isError ? (
        <Alert tone="danger" title="Couldn't load usage">
          {errorMessage(usage.error)}
        </Alert>
      ) : !data || !total ? (
        <Card>
          <EmptyState icon={<IconChartBar />} title="No usage yet" description="Start a conversation and token usage will show up here." />
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Input tokens" value={data.totals.prompt_tokens} />
            <Stat label="Output tokens" value={data.totals.completion_tokens} />
            <Stat label="Model calls" value={calls} />
          </div>
          <Card className="p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-fg">By day</h3>
              <div className="flex items-center gap-3 text-2xs text-fg-subtle">
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-sm bg-accent/45" /> Input
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-sm bg-accent" /> Output
                </span>
              </div>
            </div>
            <DailyChart data={data} days={Number(days)} />
          </Card>
          <div className="grid grid-cols-2 gap-3">
            <Breakdown title="By model" rows={data.by_model.map((m) => ({ key: m.model, label: modelShortName(m.model), value: m.prompt_tokens + m.completion_tokens, sub: `${m.calls} calls` }))} />
            <Breakdown title="By source" rows={data.by_source.map((s) => ({ key: s.source, label: s.source, value: s.prompt_tokens + s.completion_tokens, sub: `${s.calls} calls` }))} />
          </div>
        </>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <Card className="px-4 py-3.5">
      <div className="text-xs text-fg-subtle">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums tracking-tight text-fg">{formatNumber(value)}</div>
    </Card>
  )
}

function DailyChart({ data, days }: { data: UsageReport; days: number }) {
  const [hover, setHover] = useState<number | null>(null)
  const series = useMemo(() => {
    const byDay = new Map(data.by_day.map((d) => [d.day, d]))
    const out: Array<{ day: string; p: number; c: number }> = []
    const today = new Date()
    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(today)
      d.setDate(today.getDate() - i)
      const key = d.toISOString().slice(0, 10)
      const row = byDay.get(key)
      out.push({ day: key, p: row?.prompt_tokens ?? 0, c: row?.completion_tokens ?? 0 })
    }
    return out
  }, [data, days])

  const max = Math.max(1, ...series.map((s) => s.p + s.c))
  const W = 720
  const H = 180
  const gap = days > 60 ? 1 : 3
  const bw = (W - gap * (series.length - 1)) / series.length
  const h = hover !== null ? series[hover] : null

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H + 20}`} className="h-auto w-full" onMouseLeave={() => setHover(null)}>
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <line key={f} x1={0} x2={W} y1={H - H * f} y2={H - H * f} stroke="var(--border)" strokeDasharray="3 4" />
        ))}
        {series.map((s, i) => {
          const x = i * (bw + gap)
          const hp = (s.p / max) * H
          const hc = (s.c / max) * H
          return (
            <g key={s.day} onMouseEnter={() => setHover(i)}>
              <rect x={x} y={0} width={bw} height={H} fill="transparent" />
              <rect x={x} y={H - hp - hc} width={bw} height={hc} rx={Math.min(3, bw / 3)} fill="var(--accent)" opacity={hover === null || hover === i ? 1 : 0.5} />
              <rect x={x} y={H - hp} width={bw} height={hp} fill="var(--accent)" opacity={hover === null || hover === i ? 0.45 : 0.22} />
            </g>
          )
        })}
        {series.map((s, i) =>
          i % Math.ceil(series.length / 6) === 0 ? (
            <text key={s.day} x={i * (bw + gap)} y={H + 15} fontSize={10} fill="var(--fg-subtle)">
              {new Date(`${s.day}T00:00:00`).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
            </text>
          ) : null
        )}
      </svg>
      {h && (
        <div className="pointer-events-none absolute right-2 top-0 rounded-lg border border-border-strong bg-overlay px-2.5 py-1.5 text-xs shadow-pop">
          <div className="font-medium text-fg">{new Date(`${h.day}T00:00:00`).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })}</div>
          <div className="text-fg-muted">
            {formatNumber(h.p)} in · {formatNumber(h.c)} out
          </div>
        </div>
      )}
    </div>
  )
}

function Breakdown({ title, rows }: { title: string; rows: Array<{ key: string; label: string; value: number; sub: string }> }) {
  const max = Math.max(1, ...rows.map((r) => r.value))
  return (
    <Card className="p-4">
      <h3 className="mb-3 text-sm font-semibold text-fg">{title}</h3>
      <div className="space-y-2.5">
        {rows.slice(0, 8).map((r) => (
          <div key={r.key}>
            <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
              <span className="truncate font-mono text-fg" title={r.key}>
                {r.label}
              </span>
              <span className="shrink-0 tabular-nums text-fg-subtle">
                {formatNumber(r.value)} · {r.sub}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-active">
              <div className="h-full rounded-full bg-accent" style={{ width: `${(r.value / max) * 100}%` }} />
            </div>
          </div>
        ))}
        {!rows.length && <p className="text-xs text-fg-subtle">Nothing yet.</p>}
      </div>
    </Card>
  )
}
