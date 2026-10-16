import {
  IconAlertTriangle,
  IconCamera,
  IconCheck,
  IconClick,
  IconClockX,
  IconCreditCard,
  IconEdit,
  IconEye,
  IconFolder,
  IconKeyboard,
  IconMail,
  IconMailOpened,
  IconSend,
  IconShieldCheck,
  IconShieldQuestion,
  IconTerminal2,
  IconTrash,
  IconLock,
  IconWorldWww,
  IconX,
  type Icon
} from '@tabler/icons-react'
import { motion } from 'motion/react'
import { Badge, Button, JsonView, Markdown, type Tone } from '@/components/ui'
import type { ApprovalView } from '@/lib/chatFold'
import type { ApprovalDecision } from '@/lib/types'
import { cn, humanize, truncate } from '@/lib/utils'
import { ActionPreview } from './ActionPreview'
import { hostOf, refLabel } from './cards/bits'
import { toolMeta } from './toolMeta'
import { resolveWording, type RiskLabel } from './approvalRisk'

const STATUS_TEXT: Partial<Record<ApprovalView['status'], string>> = {
  allow: 'You allowed this once',
  allow_session: 'You allowed this for the rest of this chat',
  allow_task: 'You allowed this for this task',
  allow_always: 'You always allow this now. Change it in Settings > Approvals & safety.',
  deny: 'You denied this',
  expired: 'This request expired'
}

/** "1 hour", "2 hours", "30 minutes" for `allow_timed` (the engine's `tools.approvals.grant_minutes`). */
export function duration(minutes?: number | null): string {
  const m = Math.max(1, Math.round(minutes ?? 60))
  if (m % 60 === 0) return `${m / 60} hour${m === 60 ? '' : 's'}`
  return `${m} minute${m === 1 ? '' : 's'}`
}

interface EffectiveRisk {
  label: string
  /** "Sentient wants to ..." */
  verb: string
  icon: Icon
  tone: Tone
  caution?: string
}

const RISK_ICONS: Record<RiskLabel, Icon> = {
  'Runs a command': IconTerminal2,
  'Home security': IconLock,
  'Runs code': IconTerminal2,
  Purchase: IconCreditCard,
  Deletes: IconTrash,
  'Posts publicly': IconSend,
  Submits: IconSend,
  'Sends an email': IconMail,
  Sends: IconSend,
  'Changes things': IconEdit,
  Reads: IconEye
}

/**
 * Icons for the engine's describe_fn action labels (#347). Keyed by label with a generic
 * fallback: the engine may send labels this table has never seen.
 */
const ENGINE_ICONS: Record<string, Icon> = {
  Purchase: IconCreditCard,
  Deletes: IconTrash,
  'Posts publicly': IconSend,
  'Sends an email': IconMail,
  Sends: IconSend,
  Submits: IconSend,
  'Runs a command': IconTerminal2,
  'Runs code': IconTerminal2,
  'Changes something': IconEdit,
  'Looks something up': IconEye,
  Clicks: IconClick,
  'Types into the page': IconKeyboard,
  'Presses a key': IconKeyboard,
  'Takes a photo': IconCamera,
  'Looks at the screen': IconEye,
  'Moves to the trash': IconTrash
}

/**
 * The engine reports the effective risk (§10 `risk_fn`: a click on "Place order" becomes `send`).
 * The wording comes from {@link resolveWording}: a trash-kind preview (#142) reads as a move to
 * the trash, not a loss, and is checked first; a browser tool's describe_fn label is trusted
 * as-is; otherwise the tool's name and declared risk come first (#346, #372), word guessing
 * only as a last resort and never from the arguments' content — except the slug keys of
 * multi-execute tools, which name the action (#344). When the rules return a generic label,
 * the engine's risk_label breaks the tie if a describe_fn actually ran (#347).
 */
export function effectiveRisk(
  approval: Pick<ApprovalView, 'risk' | 'reason' | 'name' | 'arguments'> & Pick<Partial<ApprovalView>, 'preview' | 'riskLabel'>
): EffectiveRisk {
  const wording = resolveWording(approval)
  const icon =
    wording.label in RISK_ICONS
      ? RISK_ICONS[wording.label as RiskLabel]
      : (ENGINE_ICONS[wording.label] ?? IconShieldQuestion)
  return { ...wording, icon }
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
    case 'home_assistant_unlock':
      return `Unlock “${approval.target ?? String(a.entity_id ?? 'a door')}”`
    case 'home_assistant_lock':
      return `Lock “${approval.target ?? String(a.entity_id ?? 'a door')}”`
    case 'home_assistant_open_cover':
      return `Open “${approval.target ?? String(a.entity_id ?? '')}”`
    case 'home_assistant_close_cover':
      return `Close “${approval.target ?? String(a.entity_id ?? '')}”`
    case 'home_assistant_set_alarm':
      return `${humanize(String(a.mode ?? 'change'))}: “${approval.target ?? String(a.entity_id ?? 'your alarm')}”`
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
    const allowed = approval.status !== 'deny' && approval.status !== 'expired'
    return (
      <div className="flex items-center gap-2 px-1 text-xs text-fg-subtle">
        {approval.status === 'expired' ? (
          <IconClockX size={13} />
        ) : allowed ? (
          <IconShieldCheck size={13} className="text-success" />
        ) : (
          <IconX size={13} className="text-danger" />
        )}
        {STATUS_TEXT[approval.status] ??
          (approval.status === 'allow_timed' ? `You allowed this for ${duration(approval.grantMinutes)}` : 'This request expired')}
      </div>
    )
  }

  const serious = risk.tone === 'danger'
  const action = actionLine(approval)
  const code = approval.name === 'execute_code' ? String((approval.arguments as { code?: string }).code ?? '') : ''
  const purpose = approval.name === 'execute_code' ? String((approval.arguments as { purpose?: string }).purpose ?? '') : ''
  const isBrowser = approval.name.startsWith('browser_')
  const args = approval.arguments ?? {}
  const command = approval.name === 'terminal_run' ? String(args.command ?? '') : ''
  // A purchase, a deletion or a command on this computer asks for a single, deliberate "yes" every time: no blanket
  // answer (the engine ignores it for commands too, ADR 0019). After outside content came in, every send asks
  // anyway, so the blanket answer would not help either. The engine says which answers it offers (#139).
  // A call a second look flagged (#143) asks for this one call only, too.
  const single =
    risk.label === 'Purchase' || risk.label === 'Deletes' || approval.name === 'terminal_run' || !!approval.untrusted || !!approval.warning
  const offered = (approval.choices?.length ? approval.choices : (['allow', 'allow_session', 'deny'] as ApprovalDecision[])).filter(
    (c) => c !== 'allow' && c !== 'deny' && c !== 'allow_task' && !single
  )

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
          {risk.caution && <div className="mt-1 text-sm font-medium text-danger">{risk.caution}</div>}
          {approval.warning && (
            <div
              role="alert"
              className="mt-2 flex items-start gap-2 rounded-lg border border-danger/30 bg-danger/[0.08] px-2.5 py-2 text-sm font-medium text-fg"
            >
              <IconAlertTriangle size={16} className="mt-px shrink-0 text-danger" />
              <span className="selectable min-w-0 break-words">{approval.warning}</span>
            </div>
          )}
          <p className="mt-0.5 text-sm text-fg-muted">
            {(command && 'It runs on this computer with your permissions, in this folder:') ||
              purpose ||
              approval.reason?.trim() || (
                <>
                  {meta.done} using <span className="font-mono text-xs">{approval.name}</span>
                </>
              )}
          </p>
          {command && (
            <div className="mt-1 flex items-center gap-1 text-xs text-fg-subtle">
              <IconFolder size={12} className="shrink-0" />
              <span className="selectable truncate font-mono">{approval.target || (typeof args.cwd === 'string' && args.cwd) || 'The default folder'}</span>
            </div>
          )}
          {approval.untrusted && (
            <div className="mt-1.5 flex items-start gap-1.5 text-xs text-fg-muted">
              <IconMailOpened size={13} className="mt-px shrink-0" /> {approval.untrusted}
            </div>
          )}
          {isBrowser && typeof args.url === 'string' && (
            <div className="mt-1 flex items-center gap-1 text-xs text-fg-subtle">
              <IconWorldWww size={12} /> {hostOf(args.url)}
            </div>
          )}
        </div>
      </div>
      {command ? (
        // plain text, never Markdown: the user must see exactly what will run
        <pre className="selectable mx-4 mt-3 max-h-56 overflow-auto whitespace-pre-wrap break-all rounded-lg border border-border bg-surface px-3 py-2 font-mono text-[12px] leading-relaxed text-fg">
          {command.replace(/\s+$/, '')}
        </pre>
      ) : code ? (
        <div className="mx-4 mt-3 max-h-56 overflow-auto [&_.md-code]:bg-surface">
          <Markdown>{'```python\n' + code.replace(/\s+$/, '') + '\n```'}</Markdown>
        </div>
      ) : approval.preview ? (
        // what will be sent, the way a person reads it (#142); the raw details stay one click away
        <div className="mx-4 mt-3 space-y-1.5">
          <ActionPreview preview={approval.preview} />
          <details className="group text-xs text-fg-subtle">
            <summary className="cursor-pointer select-none hover:text-fg-muted">Details</summary>
            <div className="mt-1.5 max-h-48 overflow-auto rounded-lg border border-border bg-surface px-3 py-2">
              <JsonView value={args} collapsedDepth={2} />
            </div>
          </details>
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
        {offered.map((c) => (
          <Button
            key={c}
            size="sm"
            variant="secondary"
            title={c === 'allow_always' ? 'Saves an Allow rule. Change it any time in Settings > Approvals & safety.' : undefined}
            onClick={() => onRespond?.(approval.approvalId, c)}
          >
            {c === 'allow_session' ? 'Allow for this chat' : c === 'allow_timed' ? `Allow for ${duration(approval.grantMinutes)}` : 'Always allow'}
          </Button>
        ))}
        <Button size="sm" variant="ghost" onClick={() => onRespond?.(approval.approvalId, 'deny')}>
          {risk.label === 'Purchase' ? "Don't buy" : 'Deny'}
        </Button>
      </div>
    </motion.div>
  )
}
