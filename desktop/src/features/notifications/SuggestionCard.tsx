import { IconArrowRight, IconCheck, IconChevronDown, IconExternalLink, IconMail, IconSend, IconX } from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Badge, Button, Tooltip } from '@/components/ui'
import { BrandIcon } from '@/features/integrations/BrandIcon'
import { openExternal } from '@/features/integrations/InstructionsGuide'
import { qk } from '@/hooks/queryKeys'
import { useNotificationActions } from '@/hooks/notifications'
import { errorMessage, isApiError } from '@/lib/api'
import type { Notification, NotificationList, SuggestionPayload } from '@/lib/types'
import { cn, humanize } from '@/lib/utils'
import { SOURCE_LABEL, taskRoute } from './utils'

function patchPayload(qc: ReturnType<typeof useQueryClient>, id: string, patch: Partial<SuggestionPayload>, read = true) {
  qc.setQueryData<NotificationList>(qk.notifications, (old) =>
    old
      ? {
          unread: Math.max(0, old.unread - (read && old.notifications.some((n) => n.id === id && !n.read) ? 1 : 0)),
          notifications: old.notifications.map((n) => (n.id === id ? { ...n, read: read || n.read, payload: { ...n.payload, ...patch } } : n))
        }
      : old
  )
}

export function ConfidenceMeter({ value, className }: { value: number; className?: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100)
  const tone = value >= 0.8 ? 'bg-success' : value >= 0.6 ? 'bg-accent' : 'bg-warning'
  return (
    <Tooltip content="How sure Sentient is that this is worth doing">
      <span className={cn('inline-flex items-center gap-2', className)}>
        <span className="flex h-1.5 w-20 gap-0.5" aria-hidden>
          {[0, 1, 2, 3, 4].map((i) => (
            <span key={i} className="relative flex-1 overflow-hidden rounded-full bg-active">
              <span className={cn('absolute inset-y-0 left-0 rounded-full', tone)} style={{ width: `${Math.max(0, Math.min(1, value * 5 - i)) * 100}%` }} />
            </span>
          ))}
        </span>
        <span className="text-xs tabular-nums text-fg-muted">{pct}% sure</span>
      </span>
    </Tooltip>
  )
}

export function SuggestionCard({ n, onNavigate }: { n: Notification; onNavigate?: () => void }) {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const { respondToSuggestion } = useNotificationActions()
  const [why, setWhy] = useState(false)
  const s = n.payload.suggestion
  if (!s) return null

  const status = n.payload.status ?? 'pending'
  const taskId = n.payload.task_id ?? null
  const source = s.source_event?.source ?? 'sentient'
  const busy = respondToSuggestion.isPending
  const followUp = s.follow_up?.draft ? s.follow_up : null
  const isReply = followUp?.kind === 'waiting_on_you'

  const act = (action: 'approve' | 'dismiss') => {
    const previous = { status: n.payload.status ?? 'pending', task_id: n.payload.task_id ?? null } as Partial<SuggestionPayload>
    patchPayload(qc, n.id, { status: action === 'approve' ? 'approved' : 'dismissed' })
    respondToSuggestion.mutate(
      { id: n.id, action },
      {
        onSuccess: (res) => {
          if (action === 'approve') {
            patchPayload(qc, n.id, { status: 'approved', task_id: res.task_id ?? null })
            toast.success('On it', {
              description: followUp ? 'Sentient made a task to send your draft.' : 'Sentient created a task for this suggestion.',
              action: res.task_id
                ? {
                    label: 'View task',
                    onClick: () => {
                      onNavigate?.()
                      navigate(taskRoute(res.task_id as string))
                    }
                  }
                : undefined
            })
          } else {
            toast('Dismissed', {
              description: followUp ? 'Sentient will not bring this email up again.' : `Sentient will suggest “${humanize(s.suggestion_type)}” less often.`
            })
          }
        },
        onError: (e) => {
          if (isApiError(e) && e.status === 409) {
            void qc.invalidateQueries({ queryKey: qk.notifications })
            toast(e.detail)
            return
          }
          patchPayload(qc, n.id, previous, false)
          toast.error(action === 'approve' ? "Couldn't start that" : "Couldn't dismiss", { description: errorMessage(e) })
        }
      }
    )
  }

  return (
    <div className="space-y-2.5">
      <p className="text-md font-medium leading-snug text-fg">{s.description}</p>

      {followUp && (
        <div className="rounded-lg border border-border bg-elevated px-3 py-2.5">
          <div className="mb-1.5 flex items-center gap-1.5 text-2xs font-medium text-fg-subtle">
            <IconMail size={12} />
            <span className="min-w-0 truncate">
              {isReply ? 'Draft reply' : 'Draft nudge'} to {followUp.person}
            </span>
          </div>
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-fg-muted">{followUp.draft}</p>
        </div>
      )}

      {s.source_event?.summary && (
        <div className="flex items-center gap-2 rounded-lg border border-border bg-sunken/50 py-1.5 pl-2 pr-1.5">
          <span className="flex shrink-0 items-center gap-1.5 rounded-md border border-border bg-elevated px-1.5 py-0.5 text-2xs font-medium text-fg-muted">
            <BrandIcon id={source} size={12} bare />
            {SOURCE_LABEL[source] ?? humanize(source)}
          </span>
          <span className="min-w-0 flex-1 truncate text-xs text-fg-muted" title={s.source_event.summary}>
            {s.source_event.summary}
          </span>
          {s.source_event.url && (
            <Tooltip content={`Open in ${SOURCE_LABEL[source] ?? 'browser'}`}>
              <button
                type="button"
                aria-label="Open original"
                onClick={() => openExternal(s.source_event.url as string)}
                className="flex size-6 shrink-0 items-center justify-center rounded-md text-fg-subtle hover:bg-active hover:text-fg"
              >
                <IconExternalLink size={13} />
              </button>
            </Tooltip>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <ConfidenceMeter value={s.confidence ?? 0} />
        {s.reasoning && (
          <button type="button" onClick={() => setWhy((w) => !w)} aria-expanded={why} className="flex items-center gap-0.5 text-xs text-fg-subtle hover:text-fg">
            Why?
            <IconChevronDown size={12} className={cn('transition-transform', why && 'rotate-180')} />
          </button>
        )}
      </div>
      <AnimatePresence initial={false}>
        {why && s.reasoning && (
          <motion.p
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-l-2 border-accent/40 pl-3 text-xs leading-relaxed text-fg-muted"
          >
            {s.reasoning}
          </motion.p>
        )}
      </AnimatePresence>

      <div className="flex flex-wrap items-center gap-2 pt-0.5">
        {status === 'pending' ? (
          <>
            {followUp ? (
              <Tooltip content="Sentient makes a task to send this draft. It never sends on its own.">
                <Button size="sm" variant="primary" leftIcon={<IconSend size={14} />} disabled={busy} onClick={() => act('approve')}>
                  {isReply ? 'Send reply' : 'Send nudge'}
                </Button>
              </Tooltip>
            ) : (
              <Button size="sm" variant="primary" leftIcon={<IconCheck size={14} />} disabled={busy} onClick={() => act('approve')}>
                Do it
              </Button>
            )}
            <Button size="sm" variant="ghost" leftIcon={<IconX size={14} />} disabled={busy} onClick={() => act('dismiss')}>
              Dismiss
            </Button>
          </>
        ) : status === 'approved' ? (
          <>
            <Badge tone="success" icon={<IconCheck />}>
              Approved
            </Badge>
            {taskId ? (
              <button
                type="button"
                onClick={() => {
                  onNavigate?.()
                  navigate(taskRoute(taskId))
                }}
                className="flex items-center gap-1 text-xs font-medium text-accent-text hover:underline"
              >
                View task <IconArrowRight size={12} />
              </button>
            ) : (
              busy && <span className="text-xs text-fg-subtle">Creating a task…</span>
            )}
          </>
        ) : (
          <Badge tone="neutral">Dismissed</Badge>
        )}
      </div>
    </div>
  )
}
