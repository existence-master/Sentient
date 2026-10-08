/** "Webhook: Shopify orders" line for triggered tasks that start from a webhook. */
import { IconWebhook } from '@tabler/icons-react'
import { useHooks } from '@/hooks/automations'
import { cn, relativeTime } from '@/lib/utils'

export function HookName({ id, className }: { id: string; className?: string }) {
  const hooks = useHooks()
  if (hooks.isLoading) return null
  const hook = hooks.data?.find((h) => h.id === id)
  return (
    <div className={cn('flex flex-wrap items-center gap-1.5 text-xs text-fg-muted', className)}>
      <IconWebhook size={13} className="text-accent-text" />
      {hook ? (
        <>
          Webhook <span className="font-medium text-fg">{hook.name}</span>
          <span className="text-fg-subtle">· {hook.last_called_at ? `last called ${relativeTime(hook.last_called_at)}` : 'not called yet'}</span>
        </>
      ) : hooks.isError ? (
        <span>A webhook</span>
      ) : (
        <span className="text-warning">This webhook no longer exists. Pick another one.</span>
      )}
    </div>
  )
}
