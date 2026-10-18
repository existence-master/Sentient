/**
 * "Allowed for now": the live answers wider than one call that Sentient remembers (#139, docs/API.md section 2).
 * "Allow for this task" lasts until the task is deleted; "Allow for 1 hour" runs out by itself. Each can be revoked.
 * "Always allow" answers are Allow rules and show in "Rules for apps and tools" instead.
 */
import { IconClockHour4, IconListCheck, IconX } from '@tabler/icons-react'
import { toast } from 'sonner'
import { Alert, Badge, Button, FormSection, Skeleton } from '@/components/ui'
import { useApprovalGrants, useRevokeGrant } from '@/hooks/core'
import { errorMessage } from '@/lib/api'
import type { ApprovalGrant } from '@/lib/types'
import { humanize, relativeTime } from '@/lib/utils'

const VIA: Record<string, string> = { desktop: 'in the app', telegram: 'on Telegram', discord: 'on Discord', whatsapp: 'on WhatsApp' }

function scopeLine(g: ApprovalGrant): string {
  const where = VIA[g.via] ?? `on ${humanize(g.via)}`
  const when = relativeTime(g.created_at)
  const said = `You allowed it ${where}${when ? ` ${when}` : ''}.`
  if (g.scope === 'task') return `Only in the task "${g.task_name ?? 'Untitled task'}". ${said}`
  const left = relativeTime(g.expires_at)
  return `In every chat and task${left ? `, ends ${left}` : ''}. ${said}`
}

export function ApprovalGrantsSection({ query }: { query: string }) {
  const grants = useApprovalGrants()
  const revoke = useRevokeGrant()
  const q = query.trim().toLowerCase()
  const list = (grants.data ?? []).filter((g) => !q || `${g.label} ${g.app ?? ''} ${g.task_name ?? ''}`.toLowerCase().includes(q))
  const description = 'Answers that let Sentient go ahead without asking for now. Revoke one to be asked again next time.'

  return (
    <FormSection title="Allowed for now" description={description}>
      {grants.isLoading ? (
        <div className="space-y-2 p-4">
          <Skeleton className="h-9" />
          <Skeleton className="h-9" />
        </div>
      ) : grants.isError ? (
        <div className="p-4">
          <Alert tone="danger" title="Couldn't load what you allowed">
            {errorMessage(grants.error)}
          </Alert>
        </div>
      ) : !list.length ? (
        <div className="px-4 py-6 text-center text-sm text-fg-subtle">
          {q ? 'Nothing you allowed matches your search.' : 'Nothing right now. When Sentient asks, you can allow a tool for a task or for an hour.'}
        </div>
      ) : (
        list.map((g) => (
          <div key={g.id} className="flex items-center gap-3 px-4 py-3">
            <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-success/12 text-success">
              {g.scope === 'task' ? <IconListCheck size={16} /> : <IconClockHour4 size={16} />}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 text-sm font-medium text-fg">
                <span className="truncate">{g.label}</span>
                <Badge size="xs" tone={g.scope === 'task' ? 'accent' : 'warning'}>
                  {g.scope === 'task' ? 'This task' : 'For a while'}
                </Badge>
              </div>
              <div className="truncate text-xs text-fg-subtle">{scopeLine(g)}</div>
            </div>
            <Button
              size="sm"
              variant="ghost"
              leftIcon={<IconX size={13} />}
              loading={revoke.isPending && revoke.variables === g.id}
              onClick={() =>
                revoke.mutate(g.id, {
                  onSuccess: () => toast.success('Revoked', { description: `Sentient will ask before using ${g.label} again.` }),
                  onError: (e) => toast.error("Couldn't revoke it", { description: errorMessage(e) })
                })
              }
            >
              Revoke
            </Button>
          </div>
        ))
      )}
    </FormSection>
  )
}
