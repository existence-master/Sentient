/** Read only or Read and write for one connection (#141). */
import { IconEye, IconPencil } from '@tabler/icons-react'
import { SegmentedControl } from '@/components/ui'
import type { ConnectionAccess } from '@/lib/types'
import { cn } from '@/lib/utils'
import { riskOf } from './meta'

/** True when a connection has tools that change, send, delete or run something, so the choice matters. */
export function hasChangingTools(tools: Array<{ risk: string }>): boolean {
  return tools.some((t) => riskOf(t.risk) !== 'read')
}

export function accessHint(access: ConnectionAccess, app: string): string {
  return access === 'read'
    ? `Sentient can look things up in ${app} but can't change, send or delete anything there.`
    : `Sentient can also make changes, send and delete in ${app}, following your approval settings.`
}

export function AccessChoice({
  value,
  onChange,
  app,
  size = 'md',
  hint = true,
  className
}: {
  value: ConnectionAccess
  onChange: (access: ConnectionAccess) => void
  app: string
  size?: 'sm' | 'md'
  hint?: boolean
  className?: string
}) {
  return (
    <div className={cn('space-y-1.5', className)}>
      <SegmentedControl
        fullWidth={size === 'md'}
        size={size}
        aria-label={`What Sentient may do in ${app}`}
        value={value}
        onChange={onChange}
        options={[
          { value: 'read', label: 'Read only', icon: <IconEye size={14} /> },
          { value: 'read_write', label: 'Read and write', icon: <IconPencil size={14} /> }
        ]}
      />
      {hint && <p className="text-xs leading-relaxed text-fg-subtle">{accessHint(value, app)}</p>}
    </div>
  )
}
