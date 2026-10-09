import {
  IconAlertTriangle,
  IconArrowRight,
  IconBolt,
  IconCheck,
  IconInfoCircle,
  IconListCheck,
  IconShieldCheck,
  IconShieldQuestion,
  IconSparkles,
  IconTrash,
  IconX,
  type Icon
} from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Badge, Button, IconButton, Markdown } from '@/components/ui'
import { qk } from '@/hooks/queryKeys'
import { useNotificationActions } from '@/hooks/notifications'
import { useTaskActions } from '@/hooks/tasks'
import { api, errorMessage } from '@/lib/api'
import type { ApprovalDecision, Notification, NotificationKind, NotificationList } from '@/lib/types'
import { cn, formatTime, relativeTime, truncate } from '@/lib/utils'
import { RISK_LABEL, riskOf, toolLabel } from '@/features/integrations/meta'
import { VariantBody, variantIcon } from './VariantBody'
import { SuggestionCard } from './SuggestionCard'
import { clickRoute, notificationVariant, taskRoute } from './utils'

const KIND: Record<NotificationKind, { icon: Icon; tone: string }> = {
  info: { icon: IconInfoCircle, tone: 'text-fg-muted' },
  task: { icon: IconListCheck, tone: 'text-info' },
  approval: { icon: IconShieldQuestion, tone: 'text-warning' },
  proactive: { icon: IconBolt, tone: 'text-accent-text' },
  skill: { icon: IconSparkles, tone: 'text-accent-text' },
  error: { icon: IconAlertTriangle, tone: 'text-danger' }
}

/** Ids already marked read by "mark read on view", shared by the panel and the page. */
const seen = new Set<string>()

function useMarkReadOnView(n: Notification, enabled: boolean) {
  const ref = useRef<HTMLLIElement>(null)
  const { markRead } = useNotificationActions()
  useEffect(() => {
    const el = ref.current
    if (!enabled || n.read || !el || seen.has(n.id) || typeof IntersectionObserver === 'undefined') return
    let timer: number | undefined
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && entry.intersectionRatio >= 0.6) {
          timer ??= window.setTimeout(() => {
            if (document.visibilityState !== 'visible' || seen.has(n.id)) return
            seen.add(n.id)
            markRead.mutate(n.id)
          }, 1800)
        } else if (timer !== undefined) {
          window.clearTimeout(timer)
          timer = undefined
        }
      },
      { threshold: [0, 0.6, 1] }
    )
    io.observe(el)
    return () => {
      io.disconnect()
      if (timer !== undefined) window.clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, n.id, n.read])
  return ref
}

export function NotificationItem({
  n,
  onNavigate,
  markOnView = true,
  dense = false
}: {
  n: Notification
  onNavigate?: () => void
  markOnView?: boolean
  dense?: boolean
}) {
  const navigate = useNavigate()
  const actions = useNotificationActions()
  const ref = useMarkReadOnView(n, markOnView)
  // Keep the "new" highlight while this view stays open, even after it's marked read.
  const [fresh] = useState(!n.read)
  const variant = notificationVariant(n)
  const kind = variantIcon(variant) ?? KIND[n.kind] ?? KIND.info
  const route = clickRoute(n)
  const proactive = n.kind === 'proactive' && !!n.payload?.suggestion

  const go = () => {
    if (!n.read) actions.markRead.mutate(n.id)
    if (!route) return
    onNavigate?.()
    navigate(route)
  }

  return (
    <motion.li
      ref={ref}
      layout="position"
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0, marginTop: 0 }}
      className={cn(
        'group relative list-none overflow-hidden rounded-xl border transition-colors',
        dense ? 'px-3.5 py-3' : 'px-4 py-3.5',
        fresh || !n.read ? 'border-accent/20 bg-accent/[0.04]' : 'border-border bg-surface hover:border-border-strong'
      )}
    >
      {(fresh || !n.read) && <span aria-label="Unread" className="absolute left-1.5 top-[22px] size-1.5 rounded-full bg-accent" />}
      <div className="flex gap-3">
        <span className={cn('flex size-8 shrink-0 items-center justify-center rounded-lg border border-border bg-elevated', kind.tone)}>
          <kind.icon size={16} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2 pr-12">
            {route ? (
              <button type="button" onClick={go} className="min-w-0 truncate text-left text-sm font-semibold text-fg hover:underline">
                {n.title || 'Sentient'}
              </button>
            ) : (
              <span className="min-w-0 truncate text-sm font-semibold text-fg">{n.title || 'Sentient'}</span>
            )}
            <span className="shrink-0 text-2xs text-fg-subtle" title={formatTime(n.created_at)}>
              {relativeTime(n.created_at)}
            </span>
          </div>

          <div className="mt-1.5">
            {proactive ? (
              <SuggestionCard n={n} onNavigate={onNavigate} />
            ) : variant ? (
              <VariantBody n={n} variant={variant} onNavigate={onNavigate} />
            ) : (
              <>
                <Markdown className="text-sm text-fg-muted [&_p]:my-0.5">{n.message}</Markdown>
                {n.kind === 'approval' && <ApprovalActions n={n} />}
                {n.kind === 'task' && n.payload?.event === 'approval_needed' && <PlanApprovalActions n={n} onNavigate={onNavigate} />}
                {n.kind === 'task' && n.payload?.event === 'question' && <TaskQuestionActions n={n} onNavigate={onNavigate} />}
                {n.kind === 'skill' && (
                  <Button
                    size="xs"
                    variant="secondary"
                    className="mt-2"
                    rightIcon={<IconArrowRight size={12} />}
                    onClick={() => {
                      if (!n.read) actions.markRead.mutate(n.id)
                      onNavigate?.()
                      navigate('/skills')
                    }}
                  >
                    Review skill
                  </Button>
                )}
              </>
            )}
          </div>
        </div>
      </div>
      <div className="absolute right-2 top-2.5 flex gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
        {!n.read && <IconButton size="xs" label="Mark read" icon={<IconCheck size={13} />} onClick={() => actions.markRead.mutate(n.id)} />}
        <IconButton size="xs" label="Delete" icon={<IconTrash size={13} />} onClick={() => actions.remove.mutate(n.id)} />
      </div>
    </motion.li>
  )
}

type ApprovalState = ApprovalDecision | 'expired'

function ApprovalActions({ n }: { n: Notification }) {
  const qc = useQueryClient()
  const { markRead } = useNotificationActions()
  const [busy, setBusy] = useState<ApprovalDecision | null>(null)
  const p = n.payload as Record<string, unknown>
  const approvalId = typeof p.approval_id === 'string' ? p.approval_id : null
  const state = (p.status as ApprovalState | undefined) ?? null
  const risk = RISK_LABEL[riskOf(p.risk as string)]
  const args = (p.arguments ?? {}) as Record<string, unknown>
  const argLine = Object.entries(args)
    .slice(0, 3)
    .map(([k, v]) => `${k}: ${typeof v === 'string' ? truncate(v, 48) : JSON.stringify(v)}`)
    .join(' · ')

  const setState = (status: ApprovalState) =>
    qc.setQueryData<NotificationList>(qk.notifications, (old) =>
      old
        ? {
            ...old,
            notifications: old.notifications.map((x) =>
              x.id === n.id ? { ...x, payload: { ...x.payload, status } as unknown as Notification['payload'] } : x
            )
          }
        : old
    )

  const respond = async (decision: ApprovalDecision) => {
    if (!approvalId) return
    setBusy(decision)
    try {
      const res = await api.approvals.respond(approvalId, decision)
      if (res.resolved) {
        setState(decision)
        toast.success(decision === 'deny' ? 'Denied' : 'Allowed', { description: decision === 'deny' ? 'Sentient will skip that step.' : 'Sentient is continuing.' })
      } else {
        setState('expired')
        toast('This request has already expired')
      }
      if (!n.read) markRead.mutate(n.id)
    } catch (e) {
      toast.error("Couldn't send your answer", { description: errorMessage(e) })
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="mt-2.5 space-y-2.5">
      {(typeof p.name === 'string' || argLine) && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-sunken/50 px-2.5 py-1.5 text-xs">
          {typeof p.name === 'string' && <span className="font-medium text-fg">{toolLabel(p.name)}</span>}
          <Badge size="xs" tone={risk.tone}>
            {risk.label}
          </Badge>
          {argLine && <span className="min-w-0 truncate font-mono text-2xs text-fg-subtle">{argLine}</span>}
        </div>
      )}
      {state ? (
        <div className="flex items-center gap-1.5 text-xs text-fg-subtle">
          {state === 'deny' ? <IconX size={13} className="text-danger" /> : state === 'expired' ? <IconInfoCircle size={13} /> : <IconShieldCheck size={13} className="text-success" />}
          {state === 'deny' ? 'You denied this' : state === 'expired' ? 'This request expired' : 'You allowed this'}
        </div>
      ) : approvalId ? (
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" variant="primary" leftIcon={<IconCheck size={14} />} loading={busy === 'allow'} disabled={!!busy} onClick={() => void respond('allow')}>
            Allow
          </Button>
          <Button size="sm" variant="ghost" leftIcon={<IconX size={14} />} loading={busy === 'deny'} disabled={!!busy} onClick={() => void respond('deny')}>
            Deny
          </Button>
        </div>
      ) : null}
    </div>
  )
}

/** A running task asked a question (`payload.event = "question"`): its options as buttons. */
function TaskQuestionActions({ n, onNavigate }: { n: Notification; onNavigate?: () => void }) {
  const navigate = useNavigate()
  const { markRead } = useNotificationActions()
  const { answerQuestion } = useTaskActions()
  const [answered, setAnswered] = useState<string | null>(null)
  const p = (n.payload ?? {}) as Record<string, unknown>
  const taskId = n.task_id ?? (typeof p.task_id === 'string' ? p.task_id : null)
  const runId = typeof p.run_id === 'string' ? p.run_id : null
  const options = Array.isArray(p.options) ? p.options.filter((o): o is string => typeof o === 'string' && !!o.trim()) : []
  // the engine marks the card when the question was answered or the run cancelled anywhere (task page, a chat app)
  const status = p.status === 'answered' || p.status === 'cancelled' ? p.status : null
  const shown = answered ?? (status === 'answered' && typeof p.answer === 'string' ? p.answer : null)
  if (!taskId || !runId) return null

  const send = (answer: string) =>
    answerQuestion.mutate(
      { id: taskId, runId, answer },
      {
        onSuccess: () => {
          setAnswered(answer)
          if (!n.read) markRead.mutate(n.id)
          toast.success('Thanks. The task is carrying on.')
        },
        onError: (e) => toast.error("Couldn't send your answer", { description: errorMessage(e) })
      }
    )

  return (
    <div className="mt-2.5 flex flex-wrap items-center gap-2">
      {shown !== null || status ? (
        <Badge tone={status === 'cancelled' && shown === null ? 'neutral' : 'success'}>
          {shown !== null ? `You answered: ${truncate(shown, 60)}` : status === 'cancelled' ? 'Run cancelled' : 'Answered'}
        </Badge>
      ) : (
        options.map((o) => (
          <Button key={o} size="sm" variant="secondary" loading={answerQuestion.isPending && answerQuestion.variables?.answer === o} disabled={answerQuestion.isPending} onClick={() => send(o)}>
            {o}
          </Button>
        ))
      )}
      <button
        type="button"
        onClick={() => {
          onNavigate?.()
          navigate(taskRoute(taskId))
        }}
        className="ml-auto flex items-center gap-1 text-xs font-medium text-fg-muted hover:text-fg"
      >
        {shown === null && !status ? (options.length ? 'Other answer' : 'Answer') : 'Open task'} <IconArrowRight size={12} />
      </button>
    </div>
  )
}

function PlanApprovalActions({ n, onNavigate }: { n: Notification; onNavigate?: () => void }) {
  const navigate = useNavigate()
  const { markRead } = useNotificationActions()
  const { approve, decline } = useTaskActions()
  const [clicked, setDone] = useState<'approved' | 'declined' | null>(null)
  // the engine marks the card when the plan is approved or declined anywhere (task page, chat, another window)
  const rawStatus = (n.payload as Record<string, unknown> | undefined)?.status
  const settled = rawStatus === 'approved' || rawStatus === 'declined' ? rawStatus : null
  const done = clicked ?? settled
  const taskId = n.task_id ?? (n.payload?.task_id as string | undefined)
  if (!taskId) return null

  const run = (which: 'approve' | 'decline') => {
    const m = which === 'approve' ? approve : decline
    m.mutate(taskId, {
      onSuccess: () => {
        setDone(which === 'approve' ? 'approved' : 'declined')
        if (!n.read) markRead.mutate(n.id)
        toast.success(which === 'approve' ? 'Plan approved' : 'Plan declined')
      },
      onError: (e) => toast.error(which === 'approve' ? "Couldn't approve the plan" : "Couldn't decline the plan", { description: errorMessage(e) })
    })
  }

  return (
    <div className="mt-2.5 flex flex-wrap items-center gap-2">
      {done ? (
        <Badge tone={done === 'approved' ? 'success' : 'neutral'}>{done === 'approved' ? 'Plan approved' : 'Plan declined'}</Badge>
      ) : (
        <>
          <Button size="sm" variant="primary" leftIcon={<IconCheck size={14} />} loading={approve.isPending} disabled={decline.isPending} onClick={() => run('approve')}>
            Approve plan
          </Button>
          <Button size="sm" variant="ghost" loading={decline.isPending} disabled={approve.isPending} onClick={() => run('decline')}>
            Decline
          </Button>
        </>
      )}
      <button
        type="button"
        onClick={() => {
          onNavigate?.()
          navigate(taskRoute(taskId))
        }}
        className="ml-auto flex items-center gap-1 text-xs font-medium text-fg-muted hover:text-fg"
      >
        Review plan <IconArrowRight size={12} />
      </button>
    </div>
  )
}
