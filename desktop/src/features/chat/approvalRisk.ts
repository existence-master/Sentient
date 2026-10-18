/**
 * Pure wording for the approval card: what kind of action a tool call is, in plain words.
 *
 * Split out of ApprovalCard.tsx so it can be unit-tested without the UI imports (#344).
 * It has no runtime imports: `node --test tests/approvalRisk.test.ts` runs it directly.
 *
 * The rule (#344, refined by #347): the heading comes from the tool's NAME and its DECLARED RISK first
 * (e.g. `send_email`, `gmail_send` means "send an email"; Home Assistant's locks read as "Home security").
 * Two exceptions: a click/type/press tool's describe_fn always runs and reads the actual element, so its risk_label
 * names the action better than the rules (a "Send feedback" button reads as Sends, not Purchase); and when
 * the rules return a generic label, the engine's risk_label breaks the tie if a describe_fn actually ran.
 * When the engine's label has no known verb, the heading falls back to what the rules would have said.
 * Word guessing is a last resort and only ever looks at the tool name and the reason (the tool's description,
 * e.g. "mcp_mailer: Send an email to one address"): never at the arguments' content, so an email body
 * mentioning "post", "publish" or "tweet" cannot read as posting publicly.
 *
 * One deliberate exception: multi-execute tools (Composio's COMPOSIO_MULTI_EXECUTE_TOOL)
 * carry the real action in slug keys (`tool_slug`, `action_slug`, `slug` — the same
 * SLUG_KEYS as sentient/integrations/mcp.py). Slugs name the action, not the user's
 * content, so they are treated like tool names: only those keys are read, never the
 * arguments around them.
 */

export type RiskLabel =
  | 'Runs a command'
  | 'Home security'
  | 'Runs code'
  | 'Purchase'
  | 'Signs up'
  | 'Deletes'
  | 'Posts publicly'
  | 'Submits'
  | 'Sends an email'
  | 'Sends'
  | 'Changes things'
  | 'Reads'

export type RiskTone = 'danger' | 'warning' | 'neutral'

export interface RiskWording {
  label: RiskLabel
  /** "Sentient wants to ..." */
  verb: string
  tone: RiskTone
  /** One plain sentence on what could go wrong, for actions on the home's locks, doors and alarm (#175). */
  caution?: string
}

/**
 * Home Assistant's locks, alarm, garage doors and gates (#175), named from the tool name. The engine makes them
 * ask every time (`send`); a cover only when it is a garage door, gate or door (its risk is raised to `send`).
 */
const HOME_SECURITY: Record<string, { verb: string; caution?: string }> = {
  home_assistant_unlock: { verb: 'unlock a door', caution: 'Anyone at that door can walk in once it is unlocked.' },
  home_assistant_lock: { verb: 'lock a door' },
  home_assistant_open_cover: { verb: 'open a garage door or gate', caution: 'Anyone outside can get in while it is open.' },
  home_assistant_close_cover: { verb: 'close a garage door or gate', caution: 'Make sure nothing and nobody is in the way.' },
  home_assistant_set_alarm: { verb: 'change your alarm', caution: 'With the alarm off, nothing warns you of a break-in.' },
  home_assistant_turn_on: { verb: 'switch on something that opens or unlocks', caution: 'It sounds like a door, lock or alarm.' },
  home_assistant_turn_off: { verb: 'switch off something that guards your home', caution: 'It sounds like a door, lock or alarm.' },
  home_assistant_run_script: { verb: 'run a script that opens or unlocks', caution: 'It sounds like a door, lock or alarm.' }
}

/** Tool names that send email: send_email, gmail_send, mailer, ... — checked before any word guessing. */
function isEmailTool(name: string): boolean {
  const tokens = name.toLowerCase().split(/[^a-z0-9]+/)
  // Match known email tokens only: a bare endsWith('mail') also catches voicemail, blackmail, ...
  // ('voicemail' even ends with 'email', so exclude known non-email compounds explicitly).
  const emailish = (t: string) =>
    !/(voice|black)mail$/.test(t) &&
    (t === 'email' || t === 'gmail' || t === 'mailer' || t === 'mail' || t.endsWith('email'))
  if (!tokens.some(emailish)) return false
  // A delete/list/search of mail is not sending one; those keep the word-guessing path.
  return !tokens.some((t) => /^(delete|deletes|remove|trash|list|search|read|get|fetch)/.test(t))
}

/** Slug keys that name the action a multi-execute tool runs — mirrors SLUG_KEYS in sentient/integrations/mcp.py. */
const SLUG_KEYS = new Set(['tool_slug', 'action_slug', 'slug'])

/**
 * Collect the action slugs from a tool's arguments (mirrors _slugs_in in
 * sentient/integrations/mcp.py). Only the slug keys are read: their values name
 * the action (e.g. GMAIL_DELETE_MESSAGE), never the user's content.
 */
function actionSlugs(args: Record<string, unknown> | undefined): string[] {
  const slugs: string[] = []
  const visit = (v: unknown): void => {
    if (Array.isArray(v)) {
      v.forEach(visit)
      return
    }
    if (v && typeof v === 'object') {
      for (const [k, val] of Object.entries(v)) {
        if (SLUG_KEYS.has(k) && typeof val === 'string') slugs.push(val)
        else visit(val)
      }
    }
  }
  visit(args)
  return slugs
}

export function approvalRisk(approval: {
  risk: 'read' | 'write' | 'send' | 'exec'
  reason?: string | null
  name: string
  /**
   * Mostly unread: only the slug keys (tool_slug/action_slug/slug) are collected —
   * they name the action, not the user's content. Everything else is the user's
   * data and never steers the wording.
   */
  arguments?: Record<string, unknown>
  /** The engine's own label for the call (§10 `describe_fn`), e.g. "Signs up" for a free newsletter button (#378). */
  riskLabel?: string | null
}): RiskWording {
  // A multi-execute tool's real action lives in its slugs; treat them like tool names.
  const name = [approval.name, ...actionSlugs(approval.arguments)].join(' ')
  if (approval.name === 'terminal_run')
    return { label: 'Runs a command', verb: 'run a command on this computer', tone: 'danger' }
  const home = HOME_SECURITY[approval.name]
  if (home && (approval.risk === 'send' || approval.risk === 'exec'))
    return { label: 'Home security', verb: home.verb, tone: 'danger', ...(home.caution ? { caution: home.caution } : {}) }
  if (approval.risk === 'exec') return { label: 'Runs code', verb: 'run code on your computer', tone: 'danger' }
  // A free subscribe, follow or "notify me" (#378): the engine saw no price or plan on or near the button.
  if (approval.riskLabel === 'Signs up' && approval.risk !== 'read')
    return { label: 'Signs up', verb: 'sign you up for something', tone: approval.risk === 'send' ? 'danger' : 'warning' }
  if (approval.risk === 'send') {
    if (isEmailTool(name))
      return { label: 'Sends an email', verb: 'send an email', tone: 'danger' }
    // Last resort: guess from the tool name and the reason (the tool's description).
    // The arguments are deliberately not read: their content is the user's data,
    // not what the tool does.
    const text = `${name} ${approval.reason ?? ''}`.toLowerCase()
    if (/(\bpurchase\b|\bbuy\b|\border\b|\bcheckout\b|\bpay\b|\bpayment\b|\bcard\b|\bsubscribe\b|\bbook(ing)? and pay\b)/.test(text))
      return { label: 'Purchase', verb: 'make a purchase', tone: 'danger' }
    if (/(delete|remove|erase|unsubscribe|cancel (my|the|this) )/.test(text))
      return { label: 'Deletes', verb: 'delete something', tone: 'danger' }
    // Narrow email fallback: an unrecognized tool whose description explicitly says it sends
    // email (e.g. "Send an email to one address"). Kept after the delete check so a tool like
    // "Delete all emails" still reads as deletion. Never reads the arguments.
    if (/(send|sends|sending|sent).{0,40}?\b(e-?mail|gmail)\b/.test(text))
      return { label: 'Sends an email', verb: 'send an email', tone: 'danger' }
    if (/(post|publish|tweet|comment|review)/.test(text))
      return { label: 'Posts publicly', verb: 'post something publicly', tone: 'danger' }
    if (/(submit|confirm|form)/.test(text)) return { label: 'Submits', verb: 'submit something for you', tone: 'danger' }
    return { label: 'Sends', verb: 'send something on your behalf', tone: 'danger' }
  }
  if (approval.risk === 'write') {
    // MCP tools without annotations default to `write` (see MCPManager._risk_for), so a
    // send-like email tool (mcp_mailer_send_email) or a mail delete (mcp_gmail_gmail_delete)
    // can arrive here. Name them from the tool name first; drafts and other write-only
    // mail tools keep the generic wording.
    if (isEmailSendingTool(name))
      return { label: 'Sends an email', verb: 'send an email', tone: 'danger' }
    const text = `${name} ${approval.reason ?? ''}`.toLowerCase()
    if (/(delete|remove|erase|unsubscribe|cancel (my|the|this) )/.test(text))
      return { label: 'Deletes', verb: 'delete something', tone: 'danger' }
    return { label: 'Changes things', verb: 'make a change', tone: 'warning' }
  }
  return { label: 'Reads', verb: 'look something up', tone: 'neutral' }
}

/**
 * An email tool whose name says it sends: send_email, gmail_send, mcp_mailer_send_email.
 * `mailer` stays an email-name cue via isEmailTool but is NOT a send verb — otherwise
 * mcp_mailer_create_draft would read as sending an email. Likewise a draft cue
 * (mcp_mailer_create_reply_draft) vetoes the send reading: creating a draft is not
 * sending. Draft creators keep the generic wording.
 */
function isEmailSendingTool(name: string): boolean {
  const tokens = name.toLowerCase().split(/[^a-z0-9]+/)
  if (tokens.some((t) => /^(draft|drafts)$/.test(t))) return false
  return isEmailTool(name) && tokens.some((t) => /^(send|sends|sending|sent|reply|forward)$/.test(t))
}

/**
 * The engine's generic per-risk fallback labels (sentient/tools/base.py RISK_LABELS):
 * the engine always fills risk_label, so receiving one of these means no tool
 * describe_fn ran and the card should keep the tool-name rules below (#346).
 */
const ENGINE_FALLBACK: Record<'read' | 'write' | 'send' | 'exec', string> = {
  read: 'Looks something up',
  write: 'Changes something',
  send: 'Sends',
  exec: 'Runs code'
}

/**
 * Plain verbs for the engine's describe_fn action labels (sentient/browser/safety.py,
 * terminal, nodes, home assistant): "Sentient wants to ..." + verb.
 */
const ENGINE_VERBS: Record<string, string> = {
  Purchase: 'make a purchase',
  Deletes: 'delete something',
  'Posts publicly': 'post something publicly',
  'Sends an email': 'send an email',
  Sends: 'send something on your behalf',
  Submits: 'submit something for you',
  'Signs up': 'sign you up for something',
  'Runs a command': 'run a command on this computer',
  'Runs code': 'run code on your computer',
  'Changes something': 'make a change',
  'Looks something up': 'look something up',
  Clicks: 'click',
  'Types into the page': 'type into the page',
  'Presses a key': 'press a key',
  'Takes a photo': 'take a photo',
  'Looks at the screen': 'look at the screen'
}

/**
 * The browser tools whose describe_fn reads the actual element (sentient/browser/tools.py):
 * only these get a risk_kind, so only their risk_label is trusted as-is. The other
 * browser_* tools (e.g. browser_select) keep the tool-name rules.
 */
const BROWSER_DESCRIBE_TOOLS: ReadonlySet<string> = new Set(['browser_click', 'browser_type', 'browser_press'])

const ENGINE_TONE: Record<'read' | 'write' | 'send' | 'exec', RiskTone> = {
  read: 'neutral',
  write: 'warning',
  send: 'danger',
  exec: 'danger'
}

export interface EngineWording {
  label: string
  /** "Sentient wants to ..." */
  verb: string
  tone: RiskTone
}

/**
 * Wording from the engine's own risk_label, or null to keep the tool-name rules (#346).
 *
 * A tool's describe_fn names the actual action (a browser click's describe_fn reads the
 * clicked element: "Place order" -> "Purchase", an ordinary button -> "Clicks"), so when
 * the engine's label is specific it names the card more accurately than our rules (#347).
 * The engine always fills risk_label, so a label that is just the generic per-risk
 * fallback means no describe_fn ran: return null and let approvalRisk name it.
 */
export function engineWording(
  riskLabel: string | null | undefined,
  risk: 'read' | 'write' | 'send' | 'exec',
  /**
   * What the tool-name rules said: used for the heading when the engine's label
   * has no known verb, so the heading stays a real sentence (#347 review).
   */
  fallbackVerb = 'do this'
): EngineWording | null {
  const label = (riskLabel ?? '').trim()
  if (!label || label === ENGINE_FALLBACK[risk]) return null
  return { label, verb: ENGINE_VERBS[label] ?? fallbackVerb, tone: ENGINE_TONE[risk] }
}

/**
 * Labels approvalRisk returns when the tool-name rules could not name the action.
 * Only for these does the engine's risk_label get the tiebreaker (#347).
 */
export const GENERIC_RISK_LABELS: ReadonlySet<string> = new Set(['Sends', 'Changes things', 'Reads'])

export interface CardWording {
  label: string
  /** "Sentient wants to ..." */
  verb: string
  tone: RiskTone
  /** One plain sentence on what could go wrong (#175). */
  caution?: string
}

/**
 * The full heading precedence for an approval card, icon aside (#347):
 *
 * 1. A trash-kind preview (#142) reads as a move to the trash, not a loss.
 * 2. A click/type/press tool's describe_fn always runs and reads the actual element, so
 *    its risk_label is trusted as-is: "Send feedback" reads as Sends, not Purchase.
 *    Other browser_* tools have no describe_fn and keep the tool-name rules.
 * 3. The tool-name rules (#346, #372) come first otherwise: terminal_run, Home security
 *    and the rest already name the action with their caution and icon.
 * 4. When the rules return a generic label, the engine's risk_label breaks the tie if
 *    a describe_fn actually ran.
 */
export function resolveWording(approval: {
  risk: 'read' | 'write' | 'send' | 'exec'
  reason?: string | null
  name: string
  arguments?: Record<string, unknown>
  preview?: { kind?: string | null } | null
  riskLabel?: string | null
}): CardWording {
  if (approval.preview?.kind === 'trash' && approval.risk !== 'exec')
    return { label: 'Moves to the trash', verb: 'move something to the trash', tone: 'warning' }
  if (BROWSER_DESCRIBE_TOOLS.has(approval.name)) {
    const label = (approval.riskLabel ?? '').trim()
    // The verb falls back to what the rules would have said, so an engine label
    // with no known verb (e.g. a future "Signs up") still reads as a sentence.
    if (label) return { label, verb: ENGINE_VERBS[label] ?? approvalRisk(approval).verb, tone: ENGINE_TONE[approval.risk] }
  }
  const wording = approvalRisk(approval)
  if (!GENERIC_RISK_LABELS.has(wording.label)) return wording
  return engineWording(approval.riskLabel, approval.risk, wording.verb) ?? wording
}

/**
 * Pick the small-print reason for an approval card (#409).
 * Skips the reason when it matches the badge label after trimming and ignoring case,
 * avoiding duplicate lines such as a [Signs up] badge followed by "Signs up" in small print.
 */
export function approvalCardReason(reason?: string | null, badgeLabel?: string | null): string | null {
  const trimmed = reason?.trim()
  if (!trimmed) return null
  if (badgeLabel && trimmed.toLowerCase() === badgeLabel.trim().toLowerCase()) {
    return null
  }
  return trimmed
}

/**
 * The small print of a waiting approval card that has no reason of its own. Nothing has happened yet, so never
 * past tense and never the tool's developer name: nothing when the action line above already says what will happen
 * ("Click “Subscribe”"), else the tool's plain "would" wording ("Would write file", "Would use Gmail: send email").
 */
export function waitingSmallPrint(actionLine?: string | null, would?: string | null): string | null {
  if (actionLine?.trim()) return null
  return would?.trim() || null
}
