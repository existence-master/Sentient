import {
  IconCheck,
  IconClockX,
  IconCreditCard,
  IconEdit,
  IconEye,
  IconSend,
  IconShieldCheck,
  IconShieldQuestion,
  IconTerminal2,
  IconTrash,
  IconWorldWww,
  IconX,
  type Icon
} from '@tabler/icons-react'
import { motion } from 'motion/react'
import { Badge, Button, JsonView, Markdown, type Tone } from '@/components/ui'
import type { ApprovalView } from '@/lib/chatFold'
import type { ApprovalDecision } from '@/lib/types'
import { cn, truncate } from '@/lib/utils'
import { hostOf, refLabel } from './cards/bits'
import { toolMeta } from './toolMeta'

interface EffectiveRisk {
  label: string
  /** "Sentient wants to ..." */
  verb: string
  icon: Icon
  tone: Tone
}

/**
 * The engine reports the effective risk (§10 `risk_fn`: a click on "Place order" becomes `send`).
 * `send` covers purchases, messages, posts and deletions, so read the reason to name it plainly.
 */
export function effectiveRisk(approval: Pick<ApprovalView, 'risk' | 'reason' | 'name' | 'arguments'>): EffectiveRisk {
  const text = `${approval.reason ?? ''} ${JSON.stringify(approval.arguments ?? {})}`.toLowerCase()
  if (approval.risk === 'exec') return { label: 'Runs code', verb: 'run code on your computer', icon: IconTerminal2, tone: 'danger' }
  if (approval.risk === 'send') {
    if (/(purchase|buy|order|checkout|pay\b|payment|card|subscribe|book(ing)? and pay)/.test(text))
      return { label: 'Purchase', verb: 'make a purchase', icon: IconCreditCard, tone: 'danger' }
    if (/(delete|remove|erase|unsubscribe|cancel (my|the|this) )/.test(text))
      return { label: 'Deletes', verb: 'delete something', icon: IconTrash, tone: 'danger' }
    if (/(post|publish|tweet|comment|review)/.test(text)) return { label: 'Posts publicly', verb: 'post something publicly', icon: IconSend, tone: 'danger' }
    if (/(submit|confirm|form)/.test(text)) return { label: 'Submits', verb: 'submit something for you', icon: IconSend, tone: 'danger' }
    return { label: 'Sends', verb: 'send something on your behalf', icon: IconSend, tone: 'danger' }
  }
  if (approval.risk === 'write') return { label: 'Changes things', verb: 'make a change', icon: IconEdit, tone: 'warning' }
  return { label: 'Reads', verb: 'look something up', icon: IconEye, tone: 'neutral' }
}

/** One plain line saying what exactly will happen, e.g. `Click "Place order" on amazon.in`. */
function actionLine(approval: ApprovalView): string | null {
  const a = approval.arguments ?? {}
  const quoted = approval.target ?? refLabel(a.ref) ?? /["“']([^"”']{2,60})["”']/.exec(approval.reason ?? '')?.[1]
  const url = typeof a.url === 'string' ? a.url : undefined
  switch (approval.name) {
    case 'browser_click':
      return quoted ? `Click “${quoted}”` : 'Click a button on the page'
    case 'browser_type':
      return `Type “${truncate(String(a.text ?? ''), 60)}”${a.submit ? ' and submit' : ''}`
    case 'browser_open':
      return url ? `Open ${hostOf(url)}` : null
    case 'browser_press':
      return `Press ${String(a.key ?? 'a key')}`
    default:
      return null
  }
}

export function ApprovalCard({
  approval,
  onRespond
}: {
  approval: ApprovalView
  onRespond?: (approvalId: string, decision: ApprovalDecision) => void
}) {
  const risk = effectiveRisk(approval)
  const meta = toolMeta(approval.name)

  if (approval.status !== 'pending') {
    const allowed = approval.status === 'allow' || approval.status === 'allow_session'
    return (
      <div className="flex items-center gap-2 px-1 text-xs text-fg-subtle">
        {approval.status === 'expired' ? (
          <IconClockX size={13} />
        ) : allowed ? (
          <IconShieldCheck size={13} className="text-success" />
        ) : (
          <IconX size={13} className="text-danger" />
        )}
        {approval.status === 'allow'
          ? 'You allowed this once'
          : approval.status === 'allow_session'
            ? 'You allowed this for the rest of this chat'
            : approval.status === 'deny'
              ? 'You denied this'
              : 'This request expired'}
      </div>
    )
  }

  const serious = risk.tone === 'danger'
  const action = actionLine(approval)
  const code = approval.name === 'execute_code' ? String((approval.arguments as { code?: string }).code ?? '') : ''
  const purpose = approval.name === 'execute_code' ? String((approval.arguments as { purpose?: string }).purpose ?? '') : ''
  const isBrowser = approval.name.startsWith('browser_')
  const args = approval.arguments ?? {}
  // A purchase or a send asks for a single, deliberate "yes": no blanket "allow for this chat".
  const allowSession = !(risk.label === 'Purchase' || risk.label === 'Deletes')

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn('overflow-hidden rounded-xl border shadow-soft', serious ? 'border-danger/30 bg-danger/[0.05]' : 'border-warning/30 bg-warning/[0.06]')}
    >
      <div className="flex items-start gap-3 px-4 pt-3.5">
        <div className={cn('flex size-8 shrink-0 items-center justify-center rounded-lg', serious ? 'bg-danger/12 text-danger' : 'bg-warning/15 text-warning')}>
          {serious ? <risk.icon size={17} /> : <IconShieldQuestion size={17} />}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-semibold text-fg">Sentient wants to {risk.verb}</span>
            <Badge size="xs" tone={risk.tone} icon={<risk.icon />}>
              {risk.label}
            </Badge>
          </div>
          {action && <div className="mt-1 text-md font-medium text-fg">{action}</div>}
          <p className="mt-0.5 text-sm text-fg-muted">
            {purpose ||
              approval.reason?.trim() || (
                <>
                  {meta.done} using <span className="font-mono text-xs">{approval.name}</span>
                </>
              )}
          </p>
          {isBrowser && typeof args.url === 'string' && (
            <div className="mt-1 flex items-center gap-1 text-xs text-fg-subtle">
              <IconWorldWww size={12} /> {hostOf(args.url)}
            </div>
          )}
        </div>
      </div>
      {code ? (
        <div className="mx-4 mt-3 max-h-56 overflow-auto [&_.md-code]:bg-surface">
          <Markdown>{'```python\n' + code.replace(/\s+$/, '') + '\n```'}</Markdown>
        </div>
      ) : (
        !isBrowser &&
        Object.keys(args).length > 0 && (
          <div className="mx-4 mt-3 max-h-48 overflow-auto rounded-lg border border-border bg-surface px-3 py-2">
            <JsonView value={args} collapsedDepth={2} />
          </div>
        )
      )}
      <div className="flex flex-wrap items-center gap-2 px-4 py-3">
        <Button size="sm" variant="primary" leftIcon={<IconCheck size={14} />} onClick={() => onRespond?.(approval.approvalId, 'allow')}>
          {risk.label === 'Purchase' ? 'Yes, buy it' : 'Allow once'}
        </Button>
        {allowSession && (
          <Button size="sm" variant="secondary" onClick={() => onRespond?.(approval.approvalId, 'allow_session')}>
            Allow for this chat
          </Button>
        )}
        <Button size="sm" variant="ghost" onClick={() => onRespond?.(approval.approvalId, 'deny')}>
          {risk.label === 'Purchase' ? "Don't buy" : 'Deny'}
        </Button>
      </div>
    </motion.div>
  )
}
