import { IconCircleCheck, IconDownload, IconX } from '@tabler/icons-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Button, IconButton, Input, ProgressBar } from '@/components/ui'
import { useOllamaPull } from '@/hooks/models'
import { SUGGESTED_PULLS } from '@/lib/models'
import { cn, formatBytes } from '@/lib/utils'

/** "Pull a model" box with suggestions and live download progress. */
export function OllamaPull({ className, installed = [], compact }: { className?: string; installed?: string[]; compact?: boolean }) {
  const [name, setName] = useState('')
  const pull = useOllamaPull()
  const isInstalled = (n: string) => installed.some((i) => i === n || i === `${n}:latest`)

  const start = (model: string) => {
    if (!model.trim()) return
    setName(model)
    void pull.pull(model.trim()).then(() => undefined)
  }

  return (
    <div className={cn('space-y-3', className)}>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault()
          start(name)
        }}
      >
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Model name, e.g. qwen3:8b"
          className="font-mono text-xs"
          disabled={pull.running}
        />
        <Button type="submit" variant="primary" leftIcon={<IconDownload size={15} />} disabled={!name.trim() || pull.running}>
          Pull
        </Button>
      </form>

      {!compact && (
        <div className="flex flex-wrap gap-1.5">
          {SUGGESTED_PULLS.map((s) => (
            <button
              key={s.name}
              type="button"
              disabled={pull.running || isInstalled(s.name)}
              onClick={() => start(s.name)}
              className="flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-xs text-fg-muted transition-colors hover:border-border-strong hover:text-fg disabled:opacity-50"
              title={s.note}
            >
              {isInstalled(s.name) && <IconCircleCheck size={12} className="text-success" />}
              <span className="font-mono">{s.name}</span>
              <span className="text-fg-faint">{s.size}</span>
            </button>
          ))}
        </div>
      )}

      {(pull.running || pull.error || pull.done) && (
        <div className="rounded-xl border border-border bg-sunken/50 p-3">
          <div className="mb-2 flex items-center gap-2 text-xs">
            <span className="font-mono text-fg">{pull.name}</span>
            <span className={cn('min-w-0 flex-1 truncate', pull.error ? 'text-danger' : pull.done ? 'text-success' : 'text-fg-subtle')}>
              {pull.error ?? pull.status}
            </span>
            {pull.total > 0 && !pull.done && (
              <span className="tabular-nums text-fg-subtle">
                {formatBytes(pull.completed)} / {formatBytes(pull.total)}
              </span>
            )}
            {pull.running ? (
              <IconButton size="xs" label="Cancel" icon={<IconX size={12} />} onClick={pull.cancel} />
            ) : (
              <IconButton
                size="xs"
                label="Dismiss"
                icon={<IconX size={12} />}
                onClick={() => {
                  if (pull.done) toast.success(`${pull.name} is ready`)
                  pull.reset()
                }}
              />
            )}
          </div>
          <ProgressBar value={pull.done ? 1 : pull.progress} tone={pull.error ? 'danger' : pull.done ? 'success' : 'accent'} />
        </div>
      )}
    </div>
  )
}
