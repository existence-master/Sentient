/**
 * Pure wording for the approval card: what kind of action a tool call is, in plain words.
 *
 * Split out of ApprovalCard.tsx so it can be unit-tested without the UI imports (#344).
 * It has no runtime imports: `node --test tests/approvalRisk.test.ts` runs it directly.
 *
 * The rule (#344): the heading comes from the tool's NAME and its DECLARED RISK first
 * (e.g. `send_email`, `gmail_send` means "send an email"). Word guessing is a last resort
 * and only ever looks at the tool name and the reason (the tool's description, e.g.
 * "mcp_mailer: Send an email to one address"): never at the arguments' content, so an
 * email body mentioning "post", "publish" or "tweet" cannot read as posting publicly.
 *
 * One deliberate exception: multi-execute tools (Composio's COMPOSIO_MULTI_EXECUTE_TOOL)
 * carry the real action in slug keys (`tool_slug`, `action_slug`, `slug` — the same
 * SLUG_KEYS as sentient/integrations/mcp.py). Slugs name the action, not the user's
 * content, so they are treated like tool names: only those keys are read, never the
 * arguments around them.
 */

export type RiskLabel =
  | 'Runs a command'
  | 'Runs code'
  | 'Purchase'
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
}): RiskWording {
  // A multi-execute tool's real action lives in its slugs; treat them like tool names.
  const name = [approval.name, ...actionSlugs(approval.arguments)].join(' ')
  if (approval.name === 'terminal_run')
    return { label: 'Runs a command', verb: 'run a command on this computer', tone: 'danger' }
  if (approval.risk === 'exec') return { label: 'Runs code', verb: 'run code on your computer', tone: 'danger' }
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
