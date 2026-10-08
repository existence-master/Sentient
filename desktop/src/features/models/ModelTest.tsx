import { IconAlertCircle, IconCircleCheck, IconPlayerPlay, IconTool } from '@tabler/icons-react'
import { Button, Tooltip } from '@/components/ui'
import { useTestEmbedding, useTestModel } from '@/hooks/models'
import { errorMessage } from '@/lib/api'
import type { RoleName } from '@/lib/types'
import { cn, formatDuration, truncate } from '@/lib/utils'

/** "Test" button with inline latency / tool support / error result. */
export function ModelTest({
  model,
  role,
  embedding,
  className,
  size = 'sm'
}: {
  model: string | null | undefined
  role?: RoleName
  embedding?: boolean
  className?: string
  size?: 'xs' | 'sm'
}) {
  const chat = useTestModel()
  const embed = useTestEmbedding()
  const m = embedding ? embed : chat
  const pending = m.isPending

  const run = () => {
    if (!model) return
    if (embedding) embed.mutate(model)
    else chat.mutate({ model, role })
  }

  let result: React.ReactNode = null
  if (m.error) {
    result = <Status ok={false} text={errorMessage(m.error)} />
  } else if (embedding && embed.data) {
    result = embed.data.ok ? <Status ok text={`Works · ${embed.data.dim} dimensions`} /> : <Status ok={false} text={embed.data.error ?? 'Failed'} />
  } else if (!embedding && chat.data) {
    const d = chat.data
    result = d.ok ? (
      <span className="flex min-w-0 items-center gap-2 text-xs">
        <IconCircleCheck size={14} className="shrink-0 text-success" />
        <span className="text-fg-muted">{formatDuration(d.latency_ms)}</span>
        <Tooltip content={d.supports_tools ? 'This model called a test tool correctly.' : "This model didn't call the test tool. Tools and tasks may not work well."}>
          <span className={cn('flex items-center gap-1', d.supports_tools ? 'text-success' : 'text-warning')}>
            <IconTool size={12} />
            {d.supports_tools ? 'Tools OK' : 'No tool use'}
          </span>
        </Tooltip>
      </span>
    ) : (
      <Status ok={false} text={d.error ?? 'Failed'} />
    )
  }

  return (
    <div className={cn('flex min-w-0 items-center gap-2.5', className)}>
      <Button size={size} variant="secondary" leftIcon={<IconPlayerPlay size={13} />} loading={pending} disabled={!model} onClick={run}>
        {pending ? 'Testing' : 'Test'}
      </Button>
      {!pending && result}
    </div>
  )
}

function Status({ ok, text }: { ok: boolean; text: string }) {
  return (
    <Tooltip content={text.length > 80 ? text : undefined}>
      <span className={cn('flex min-w-0 items-center gap-1.5 text-xs', ok ? 'text-success' : 'text-danger')}>
        {ok ? <IconCircleCheck size={14} className="shrink-0" /> : <IconAlertCircle size={14} className="shrink-0" />}
        <span className="truncate">{truncate(text, 80)}</span>
      </span>
    </Tooltip>
  )
}
