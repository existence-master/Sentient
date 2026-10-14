import {
  IconCreditCard,
  IconEdit,
  IconEye,
  IconSend,
  IconTerminal2,
  IconTrash,
  IconWorldWww,
  type Icon
} from '@tabler/icons-react'
import type { ApprovalView } from '@/lib/chatFold'
import type { Tone } from '@/components/ui'

export interface EffectiveRisk {
  label: string
  /** "Sentient wants to ..." */
  verb: string
  icon: Icon
  tone: Tone
}

/**
 * Engine `risk_label` (the tool's `describe_fn`, docs/API.md §10) → card wording. Keys are
 * lowercased engine labels. The engine names the action from what it is about to do: a
 * click on "Place order" arrives as "Purchase", a click on an ordinary button as "Clicks".
 */
const ENGINE_LABEL_WORDING: Record<string, EffectiveRisk> = {
  purchase: { label: 'Purchase', verb: 'make a purchase', icon: IconCreditCard, tone: 'danger' },
  deletes: { label: 'Deletes', verb: 'delete something', icon: IconTrash, tone: 'danger' },
  'posts publicly': { label: 'Posts publicly', verb: 'post something publicly', icon: IconSend, tone: 'danger' },
  sends: { label: 'Sends', verb: 'send something on your behalf', icon: IconSend, tone: 'danger' },
  'runs code': { label: 'Runs code', verb: 'run code on your computer', icon: IconTerminal2, tone: 'danger' },
  'runs a command': { label: 'Runs a command', verb: 'run a command on this computer', icon: IconTerminal2, tone: 'danger' },
  clicks: { label: 'Clicks', verb: 'click something on the page', icon: IconWorldWww, tone: 'warning' },
  'types into the page': { label: 'Types into the page', verb: 'type into the page', icon: IconWorldWww, tone: 'warning' },
  'presses a key': { label: 'Presses a key', verb: 'press a key on the page', icon: IconWorldWww, tone: 'warning' },
  'takes a photo': { label: 'Takes a photo', verb: 'take a photo', icon: IconEye, tone: 'warning' },
  'looks at the screen': { label: 'Looks at the screen', verb: 'look at the screen', icon: IconEye, tone: 'neutral' },
  'looks something up': { label: 'Looks something up', verb: 'look something up', icon: IconEye, tone: 'neutral' },
  'changes something': { label: 'Changes something', verb: 'make a change', icon: IconEdit, tone: 'warning' }
}

/**
 * Plain wording for the approval card: what kind of action a tool call is, in plain words.
 *
 * The engine reports the effective risk (`risk_fn`: a click on "Place order" becomes `send`).
 * `send` covers purchases, messages, posts and deletions, so the tool's own wording
 * (`risk_label` from its `describe_fn`) names it first: it sees the actual element, while
 * guessing from the tool name and reason mislabels every send-risk `browser_click` as a
 * purchase, because the tool's description mentions "buy, pay" (#347).
 *
 * When the engine sends no label (older engines), the tool-name and reason rules below are
 * the fallback: word guessing is a last resort and only ever looks at the tool name and
 * the reason (the tool's description): never at the arguments' content, so an email body
 * mentioning "post", "publish" or "tweet" cannot read as posting publicly.
 *
 * One deliberate exception: multi-execute tools (Composio's COMPOSIO_MULTI_EXECUTE_TOOL)
 * carry the real action in slug keys (`tool_slug`, `action_slug`, `slug` — the same
 * SLUG_KEYS as sentient/integrations/mcp.py). Slugs name the action, not the user's
 * content, so they are treated like tool names: only those keys are read, never the
 * arguments around them.
 */
export function effectiveRisk(
  approval: Pick<ApprovalView, 'risk' | 'reason' | 'name' | 'arguments' | 'riskLabel'> & Pick<Partial<ApprovalView>, 'preview'>
): EffectiveRisk {
  const text = `${approval.reason ?? ''} ${JSON.stringify(approval.arguments ?? {})}`.toLowerCase()
  // a delete for good became its app's trash version (#142): it can come back, so it reads as a change, not a loss
  if (approval.preview?.kind === 'trash' && approval.risk !== 'exec')
    return { label: 'Moves to the trash', verb: 'move something to the trash', icon: IconTrash, tone: 'warning' }
  // Prefer the engine's own label (#347). It is a plain string on the event, so it can be
  // tested here without the UI imports.
  const engineLabel = typeof approval.riskLabel === 'string' ? approval.riskLabel.trim() : ''
  if (engineLabel) {
    const wording = ENGINE_LABEL_WORDING[engineLabel.toLowerCase()]
    if (wording) return wording
    // A label this app does not know: show it verbatim in the badge; the verb, icon and
    // tone still come from the name and reason rules below, so the card never looks broken.
    return { ...effectiveRisk({ ...approval, riskLabel: null }), label: engineLabel }
  }
  if (approval.name === 'terminal_run')
    return { label: 'Runs a command', verb: 'run a command on this computer', icon: IconTerminal2, tone: 'danger' }
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
