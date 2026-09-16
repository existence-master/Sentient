/** Rendering for the newer notification flavours: script watchers, failed runs, helpers, dreams and skill fixes. */
import {
  IconAlertCircle,
  IconAlertTriangle,
  IconArrowRight,
  IconBellRinging,
  IconCircleCheck,
  IconFirstAidKit,
  IconMoonStars,
  IconPlayerPause,
  IconRepeat,
  IconUsersGroup,
  type Icon
} from '@tabler/icons-react'
import { useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Badge, Button, Markdown } from '@/components/ui'
import { dreamStatChips } from '@/features/usermodel/meta'
import { useNotificationActions } from '@/hooks/notifications'
import { useTaskActions } from '@/hooks/tasks'
import { errorMessage } from '@/lib/api'
import { useRetryRun } from '@/lib/leap/hooks-b'
import type { DreamStats, NotificationVariant } from '@/lib/leap/types-b'
import type { Notification } from '@/lib/types'
import { taskRoute } from './utils'

const VARIANT: Record<Exclude<NotificationVariant, null>, { icon: Icon; tone: string }> = {
  script_alert: { icon: IconBellRinging, tone: 'text-warning' },
  script_failed: { icon: IconAlertTriangle, tone: 'text-danger' },
  script_recovered: { icon: IconCircleCheck, tone: 'text-success' },
  run_failed: { icon: IconAlertCircle, tone: 'text-danger' },
  subagent: { icon: IconUsersGroup, tone: 'text-info' },
  dream: { icon: IconMoonStars, tone: 'text-[#a99cf5]' },
  skill_repair: { icon: IconFirstAidKit, tone: 'text-warning' }
}

export function variantIcon(v: NotificationVariant): { icon: Icon; tone: string } | null {
  return v ? VARIANT[v] : null
}

const str = (v: unknown) => (typeof v === 'string' && v ? v : null)

export function LeapNotificationBody({ n, variant, onNavigate }: { n: Notification; variant: Exclude<NotificationVariant, null>; onNavigate?: () => void }) {
  const navigate = useNavigate()
  const { markRead } = useNotificationActions()
  const tasks = useTaskActions()
  const retry = useRetryRun()
  const [stopped, setStopped] = useState(false)
  const [retried, setRetried] = useState(false)
  const p = (n.payload ?? {}) as Record<string, unknown>
  const taskId = n.task_id ?? str(p.task_id)
  const runId = str(p.run_id)

  const go = (route: string) => {
    if (!n.read) markRead.mutate(n.id)
    onNavigate?.()
    navigate(route)
  }

  let extra: ReactNode = null
  const actions: ReactNode[] = []

  switch (variant) {
    case 'script_alert':
    case 'script_failed':
    case 'script_recovered':
      if (variant === 'script_failed') extra = <p className="mt-1 text-xs text-fg-subtle">I’ll keep checking on schedule and let you know when it works again.</p>
      if (taskId) {
        actions.push(
          <Button key="open" size="xs" variant="secondary" rightIcon={<IconArrowRight size={12} />} onClick={() => go(taskRoute(taskId))}>
            Open watcher
          </Button>
        )
      }
      if (variant === 'script_alert' && taskId) {
        actions.push(
          stopped ? (
            <Badge key="stopped" size="sm">
              Paused
            </Badge>
          ) : (
            <Button
              key="stop"
              size="xs"
              variant="ghost"
              leftIcon={<IconPlayerPause size={12} />}
              loading={tasks.update.isPending}
              onClick={() =>
                tasks.update.mutate(
                  { id: taskId, patch: { enabled: false } },
                  {
                    onSuccess: () => {
                      setStopped(true)
                      if (!n.read) markRead.mutate(n.id)
                      toast.success('Stopped watching', { description: 'You can resume it from Tasks any time.' })
                    },
                    onError: (e) => toast.error('Couldn’t pause it', { description: errorMessage(e) })
                  }
                )
              }
            >
              Stop watching
            </Button>
          )
        )
      }
      break
    case 'run_failed':
      if (taskId && runId) {
        actions.push(
          retried ? (
            <Badge key="retried" size="sm" tone="info">
              Trying again
            </Badge>
          ) : (
            <Button
              key="retry"
              size="xs"
              variant="primary"
              leftIcon={<IconRepeat size={12} />}
              loading={retry.isPending}
              onClick={() =>
                retry.mutate(
                  { taskId, runId },
                  {
                    onSuccess: () => {
                      setRetried(true)
                      if (!n.read) markRead.mutate(n.id)
                      toast.success('Trying again', { description: 'Finished steps won’t be repeated.' })
                    },
                    onError: (e) => toast.error('Couldn’t retry this run', { description: errorMessage(e) })
                  }
                )
              }
            >
              Try again
            </Button>
          )
        )
      }
      if (taskId) {
        actions.push(
          <Button key="see" size="xs" variant="ghost" rightIcon={<IconArrowRight size={12} />} onClick={() => go(`${taskRoute(taskId)}?tab=runs${runId ? `&run=${encodeURIComponent(runId)}` : ''}`)}>
            See what happened
          </Button>
        )
      }
      break
    case 'subagent': {
      const goal = str(p.goal)
      if (goal) extra = <p className="mt-1 text-xs text-fg-subtle">You asked: {goal}</p>
      const sessionId = str(p.session_id)
      if (sessionId) {
        actions.push(
          <Button key="chat" size="xs" variant="secondary" rightIcon={<IconArrowRight size={12} />} onClick={() => go(`/chat/${encodeURIComponent(sessionId)}`)}>
            Open chat
          </Button>
        )
      }
      break
    }
    case 'dream': {
      const chips = dreamStatChips((p.stats ?? undefined) as Partial<DreamStats> | undefined)
      if (chips.length) {
        extra = (
          <div className="mt-2 flex flex-wrap gap-1">
            {chips.slice(0, 4).map((c) => (
              <span key={c.key} className="inline-flex h-5 items-center rounded-full border border-border bg-elevated px-2 text-2xs text-fg-muted">
                {c.text}
              </span>
            ))}
          </div>
        )
      }
      actions.push(
        <Button key="read" size="xs" variant="secondary" rightIcon={<IconArrowRight size={12} />} onClick={() => go('/about/dreams')}>
          Read the note
        </Button>
      )
      break
    }
    case 'skill_repair': {
      const skill = str(p.skill)
      actions.push(
        <Button key="review" size="xs" variant="secondary" rightIcon={<IconArrowRight size={12} />} onClick={() => go(`/skills?tab=pending${skill ? `&focus=${encodeURIComponent(skill)}` : ''}`)}>
          Review the fix
        </Button>
      )
      break
    }
  }

  return (
    <>
      <Markdown className="text-sm text-fg-muted [&_p]:my-0.5">{n.message}</Markdown>
      {extra}
      {actions.length > 0 && <div className="mt-2 flex flex-wrap items-center gap-2">{actions}</div>}
    </>
  )
}
