import { IconDownload, IconExternalLink, IconKey } from '@tabler/icons-react'
import { Button, ProgressBar } from '@/components/ui'
import type { PullState } from '@/hooks/models'
import { getBridge } from '@/lib/bridge'
import { ROLE_META } from '@/lib/models'
import type { PresetMissing } from '@/lib/types'
import { cn, formatBytes } from '@/lib/utils'

/**
 * What a model preset still needs (a download, a key, Ollama itself), each with its fix right there. The caller
 * owns the download and the key dialog so they survive a menu closing.
 */
export function PresetFixes({
  missing,
  pull,
  onPull,
  onAddKey,
  className
}: {
  missing: PresetMissing[]
  pull: PullState
  onPull: (name: string) => void
  onAddKey: (provider: string) => void
  className?: string
}) {
  if (!missing.length) return null
  return (
    <div className={cn('space-y-2', className)}>
      {missing.map((m) => {
        const action = m.action
        const pulling = action?.kind === 'pull_model' && pull.name === action.name
        const done = pulling && pull.done
        return (
          <div key={`${m.kind}-${m.model ?? m.provider ?? ''}`} className="rounded-lg border border-border bg-sunken/50 px-3 py-2.5">
            <div className="flex items-start gap-2">
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium text-fg">{done ? `${action.name} is ready` : m.detail}</div>
                {!done && (
                  <div className="mt-0.5 text-2xs text-fg-subtle">
                    {m.fix} Used for {m.roles.map((r) => ROLE_META[r]?.label.toLowerCase() ?? r).join(', ')}.
                  </div>
                )}
              </div>
              {action?.kind === 'pull_model' && !done && (
                <Button size="xs" variant="secondary" leftIcon={<IconDownload size={12} />} loading={pulling && pull.running} disabled={pull.running} onClick={() => onPull(action.name)}>
                  {action.label}
                </Button>
              )}
              {action?.kind === 'add_key' && (
                <Button size="xs" variant="secondary" leftIcon={<IconKey size={12} />} onClick={() => onAddKey(action.provider)}>
                  {action.label}
                </Button>
              )}
              {m.kind === 'start_ollama' && (
                <Button size="xs" variant="ghost" leftIcon={<IconExternalLink size={12} />} onClick={() => void getBridge().openExternal('https://ollama.com/download')}>
                  Get Ollama
                </Button>
              )}
            </div>
            {pulling && (pull.running || pull.error) && (
              <div className="mt-2 space-y-1">
                <ProgressBar value={pull.progress} tone={pull.error ? 'danger' : 'accent'} />
                <div className="flex justify-between text-2xs text-fg-subtle">
                  <span className={cn('truncate', pull.error && 'text-danger')}>{pull.error ?? pull.status}</span>
                  {pull.total > 0 && (
                    <span className="tabular-nums">
                      {formatBytes(pull.completed)} / {formatBytes(pull.total)}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
