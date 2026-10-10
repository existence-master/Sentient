import { ProgressBar, Tooltip } from '@/components/ui'
import type { ContextMeter } from '@/lib/types'
import { cn } from '@/lib/utils'

/** Start showing the meter in amber here; the engine's own warning starts at 85%. */
const BUSY_PERCENT = 70

/** "62% of context": how much of what the model reads at once the latest call used. Never blocks anything. */
export function ContextGauge({ meter, className }: { meter: ContextMeter; className?: string }) {
  const full = meter.percent >= BUSY_PERCENT
  return (
    <Tooltip content={`About ${meter.used.toLocaleString()} of the ${meter.length.toLocaleString()} tokens this model reads at once`}>
      <span className={cn('flex shrink-0 items-center gap-1.5 text-2xs tabular-nums', full ? 'text-warning' : 'text-fg-subtle', className)}>
        <ProgressBar value={Math.min(1, meter.percent / 100)} tone={full ? 'warning' : 'neutral'} size="sm" className="w-10" />
        {meter.percent}% of context
      </span>
    </Tooltip>
  )
}
