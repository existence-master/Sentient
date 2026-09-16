import { IconArrowRight, IconBolt, IconSparkles } from '@tabler/icons-react'
import { motion } from 'motion/react'
import { Button, Spinner, StatusDot } from '@/components/ui'
import type { Integration } from '@/lib/types'
import { cn } from '@/lib/utils'
import { BrandIcon } from './BrandIcon'
import { statusMeta } from './meta'

export function IntegrationCard({
  integration: i,
  upgrades = 0,
  onOpen,
  onConnect
}: {
  integration: Integration
  /** Number of optional keyed providers that can replace this builtin. */
  upgrades?: number
  onOpen: () => void
  onConnect: () => void
}) {
  const status = statusMeta(i)
  const builtin = i.auth_type === 'builtin'
  const error = i.status === 'error'
  const connecting = i.status === 'connecting'
  const abilities = i.tools.length

  return (
    <motion.div
      layout="position"
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onOpen()
        }
      }}
      className={cn(
        'group flex min-h-[150px] cursor-pointer flex-col rounded-xl border bg-surface p-4 outline-none transition-[border-color,background-color,box-shadow] hover:border-border-strong hover:bg-elevated focus-visible:ring-3 focus-visible:ring-accent/25',
        error ? 'border-danger/30' : 'border-border'
      )}
    >
      <div className="flex items-start gap-3">
        <BrandIcon id={i.id} icon={i.icon} size={40} />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold text-fg">{i.display_name}</div>
          <div className="mt-0.5 flex min-w-0 items-center gap-1.5 text-xs">
            {connecting ? <Spinner size={11} className="text-warning" /> : <StatusDot tone={status.tone} />}
            <span className={cn('truncate', error ? 'text-danger' : i.connected || builtin ? 'text-fg-muted' : 'text-fg-subtle')}>
              {i.connected && i.account_label ? i.account_label : status.label}
            </span>
          </div>
        </div>
      </div>

      <p className={cn('mt-3 line-clamp-2 text-xs leading-relaxed', error ? 'text-danger/90' : 'text-fg-subtle')}>
        {error && i.error ? i.error : i.description}
      </p>
      <div className="flex-1" />

      <div className="@container/foot mt-3 flex min-w-0 items-center gap-2">
        <span className="flex min-w-0 flex-1 items-center gap-2.5 overflow-hidden whitespace-nowrap text-2xs text-fg-subtle">
          {abilities > 0 && (
            <span className="flex shrink-0 items-center gap-1 whitespace-nowrap">
              <IconSparkles size={12} className="shrink-0" />
              {abilities} {abilities === 1 ? 'ability' : 'abilities'}
            </span>
          )}
          {i.triggers.length > 0 && (
            <span className="flex shrink-0 items-center gap-1 whitespace-nowrap">
              <IconBolt size={12} className="shrink-0" />
              Triggers
            </span>
          )}
          {builtin && upgrades > 0 && (
            <span className="hidden min-w-0 truncate @[15rem]/foot:inline" title={`${upgrades} optional upgrade${upgrades === 1 ? '' : 's'}`}>
              {upgrades} optional upgrade{upgrades === 1 ? '' : 's'}
            </span>
          )}
          {i.alternative_for && <span className="min-w-0 truncate">Optional upgrade</span>}
        </span>
        {builtin ? (
          <span className="flex shrink-0 items-center gap-1 whitespace-nowrap text-xs text-fg-subtle transition-colors group-hover:text-fg-muted">
            Details <IconArrowRight size={13} />
          </span>
        ) : i.connected && !error ? (
          <span className="flex shrink-0 items-center gap-1 whitespace-nowrap text-xs text-fg-subtle transition-colors group-hover:text-fg-muted">
            Manage <IconArrowRight size={13} />
          </span>
        ) : (
          <Button
            size="xs"
            variant={error ? 'danger' : 'secondary'}
            className="px-2.5"
            onClick={(e) => {
              e.stopPropagation()
              onConnect()
            }}
          >
            {error ? 'Reconnect' : connecting ? 'Continue' : 'Connect'}
          </Button>
        )}
      </div>
    </motion.div>
  )
}
