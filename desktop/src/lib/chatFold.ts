/**
 * Chat timeline model shared by history and live streaming, so both render identically.
 *
 * - `foldTranscript` turns raw transcript rows into timeline items: each run of
 *   assistant rows (with tool_calls) + their tool rows becomes ONE assistant turn.
 * - `applyAgentEvent` reduces live events into the same `AssistantTurnView`.
 */
import type { AgentEvent, ApprovalDecision, Risk, TranscriptMessage } from './types'
import { safeJsonParse } from './utils'

export type ToolStatus = 'running' | 'awaiting_approval' | 'done' | 'error' | 'denied'

/** Live `tool_progress` accumulated per call (§10). Not persisted: history shows the final result only. */
export interface ToolProgressView {
  stdout: string
  stderr: string
  status?: string
  frame?: { image: string; url?: string; title?: string }
  subagent?: { subagentId?: string; messages: string[] }
}

export interface ToolCallView {
  callId: string
  name: string
  arguments: Record<string, unknown>
  result?: unknown
  isError?: boolean
  durationMs?: number | null
  status: ToolStatus
  progress?: ToolProgressView
}

const MAX_STREAM_CHARS = 24_000

export type ApprovalStatus = 'pending' | ApprovalDecision | 'expired'

export interface ApprovalView {
  approvalId: string
  callId: string
  name: string
  arguments: Record<string, unknown>
  risk: Risk
  reason: string
  status: ApprovalStatus
  /** Engine label of what is acted on ("Place order"), live only. */
  target?: string | null
  riskLabel?: string | null
}

export type TurnSegment =
  | { kind: 'thinking'; text: string }
  | { kind: 'text'; text: string }
  | { kind: 'tool'; callId: string }
  /** Something the user added while the reply was running (`chat.steer`). */
  | { kind: 'interjection'; text: string; id: string }

export type TurnStatus = 'streaming' | 'done' | 'error' | 'cancelled'

export interface AssistantTurnView {
  kind: 'assistant'
  id: string
  segments: TurnSegment[]
  tools: Record<string, ToolCallView>
  approvals: ApprovalView[]
  status: TurnStatus
  error?: { message: string; recoverable: boolean }
  messageId?: string | null
  usage?: { model: string; prompt_tokens: number; completion_tokens: number }
  createdAt: string
  /** True while the turn lives in the chat store (not yet reloaded from history). */
  live?: boolean
}

export interface AttachmentView {
  name: string
  size?: number
  mime?: string
  /** Object URL for a local image preview. */
  previewUrl?: string
}

export interface UserMessageView {
  kind: 'user'
  id: string
  text: string
  attachments: AttachmentView[]
  createdAt: string
  pending?: boolean
}

export type TimelineItem = UserMessageView | AssistantTurnView

export function emptyTurn(id: string): AssistantTurnView {
  return { kind: 'assistant', id, segments: [], tools: {}, approvals: [], status: 'streaming', createdAt: new Date().toISOString(), live: true }
}

function parseToolResult(content: string | null): unknown {
  if (content === null || content === undefined) return undefined
  const trimmed = content.trim()
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) return safeJsonParse(trimmed, content)
  return content
}

function resultLooksLikeError(result: unknown): boolean {
  if (typeof result === 'string') return /^(error|tool error|failed)\b/i.test(result.trim())
  if (result && typeof result === 'object' && !Array.isArray(result)) {
    const r = result as Record<string, unknown>
    return typeof r.error === 'string' && r.error.length > 0 && r.ok !== true
  }
  return false
}

const DECLINED_RE = /user declined|declined this action|denied by (the )?user|user denied|not approved/i

/**
 * The user said no to an approval, so the call did not happen. The engine returns
 * `{error: "NOT DONE. The user declined...", declined: true}`; older transcripts only have the error text.
 */
export function wasDenied(result: unknown): boolean {
  if (result && typeof result === 'object' && !Array.isArray(result)) {
    const r = result as { declined?: unknown; error?: unknown }
    if (r.declined === true) return true
    return typeof r.error === 'string' && (DECLINED_RE.test(r.error) || /\bdeclined\b/i.test(r.error))
  }
  if (typeof result !== 'string') return false
  const s = result.trim()
  if (s.startsWith('{')) {
    try {
      return wasDenied(JSON.parse(s))
    } catch {
      /* plain text */
    }
  }
  return DECLINED_RE.test(s) || (/^(error|not done|tool error)\b/i.test(s) && /\bdeclined\b/i.test(s))
}

export function foldTranscript(rows: TranscriptMessage[]): TimelineItem[] {
  const items: TimelineItem[] = []
  let turn: AssistantTurnView | null = null
  let prevRole: string | null = null

  for (const row of rows) {
    const flagged = (row as { interjection?: boolean }).interjection === true
    // A steer is persisted as a user row in the middle of the agent loop: right after a tool
    // result and before the assistant continues. Keep it inside the running turn.
    if (row.role === 'user' && turn && (flagged || prevRole === 'tool')) {
      turn.segments.push({ kind: 'interjection', text: row.content ?? '', id: row.id })
      prevRole = row.role
      continue
    }
    prevRole = row.role
    if (row.role === 'user') {
      turn = null
      items.push({
        kind: 'user',
        id: row.id,
        text: row.content ?? '',
        attachments: (row.attachments ?? []).map((name) => ({ name })),
        createdAt: row.created_at
      })
    } else if (row.role === 'assistant') {
      if (!turn) {
        turn = { kind: 'assistant', id: row.id, segments: [], tools: {}, approvals: [], status: 'done', createdAt: row.created_at }
        items.push(turn)
      }
      if (row.thinking?.trim()) turn.segments.push({ kind: 'thinking', text: row.thinking })
      if (row.content?.trim()) turn.segments.push({ kind: 'text', text: row.content })
      for (const tc of row.tool_calls ?? []) {
        turn.segments.push({ kind: 'tool', callId: tc.id })
        turn.tools[tc.id] = {
          callId: tc.id,
          name: tc.function?.name ?? 'tool',
          arguments: safeJsonParse<Record<string, unknown>>(tc.function?.arguments, {}),
          status: 'done'
        }
      }
      if (!row.tool_calls?.length) turn.messageId = row.id
    } else if (row.role === 'tool') {
      const callId = row.tool_call_id ?? ''
      const target = turn?.tools[callId]
      const result = parseToolResult(row.content)
      if (target) {
        target.result = result
        target.isError = resultLooksLikeError(result)
        target.status = wasDenied(result) ? 'denied' : target.isError ? 'error' : 'done'
      } else if (turn) {
        // orphan tool row: show it anyway
        turn.segments.push({ kind: 'tool', callId: callId || row.id })
        turn.tools[callId || row.id] = { callId: callId || row.id, name: row.name ?? 'tool', arguments: {}, result, status: 'done' }
      }
    }
  }
  return items
}

function appendText(segments: TurnSegment[], kind: 'thinking' | 'text', text: string): TurnSegment[] {
  const last = segments[segments.length - 1]
  if (last && last.kind === kind) {
    return [...segments.slice(0, -1), { kind, text: last.text + text }]
  }
  return [...segments, { kind, text }]
}

/** Pure reducer: live agent event -> updated turn. Unknown events return the turn unchanged. */
export function applyAgentEvent(turn: AssistantTurnView, ev: AgentEvent): AssistantTurnView {
  switch (ev.type) {
    case 'thinking_delta':
      return { ...turn, segments: appendText(turn.segments, 'thinking', ev.text) }
    case 'text_delta':
      return { ...turn, segments: appendText(turn.segments, 'text', ev.text) }
    case 'user_interjection':
      return { ...turn, segments: [...turn.segments, { kind: 'interjection', text: ev.text, id: `i-${turn.segments.length}-${ev.text.length}` }] }
    case 'tool_progress': {
      const prev = turn.tools[ev.call_id] ?? { callId: ev.call_id, name: ev.name, arguments: {}, status: 'running' as const }
      const p: ToolProgressView = prev.progress ?? { stdout: '', stderr: '' }
      let next: ToolProgressView = p
      if (ev.kind === 'stdout' || ev.kind === 'stderr') {
        const joined = (p[ev.kind] + (ev.text ?? '')).slice(-MAX_STREAM_CHARS)
        next = { ...p, [ev.kind]: joined }
      } else if (ev.kind === 'status') {
        next = { ...p, status: ev.text ?? p.status }
      } else if (ev.kind === 'frame' && ev.image) {
        const d = ev.data ?? {}
        next = { ...p, frame: { image: ev.image, url: typeof d.url === 'string' ? d.url : p.frame?.url, title: typeof d.title === 'string' ? d.title : p.frame?.title } }
      } else if (ev.kind === 'subagent') {
        const d = ev.data ?? {}
        const message = typeof d.message === 'string' ? d.message : ev.text
        next = {
          ...p,
          subagent: {
            subagentId: typeof d.subagent_id === 'string' ? d.subagent_id : p.subagent?.subagentId,
            messages: message ? [...(p.subagent?.messages ?? []), message].slice(-40) : (p.subagent?.messages ?? [])
          }
        }
      }
      const segments = turn.tools[ev.call_id] ? turn.segments : [...turn.segments, { kind: 'tool' as const, callId: ev.call_id }]
      return { ...turn, segments, tools: { ...turn.tools, [ev.call_id]: { ...prev, progress: next } } }
    }
    case 'tool_call': {
      if (turn.tools[ev.call_id]) return turn
      return {
        ...turn,
        segments: [...turn.segments, { kind: 'tool', callId: ev.call_id }],
        tools: { ...turn.tools, [ev.call_id]: { callId: ev.call_id, name: ev.name, arguments: ev.arguments ?? {}, status: 'running' } }
      }
    }
    case 'approval_request': {
      const existing = turn.tools[ev.call_id]
      const tools = {
        ...turn.tools,
        [ev.call_id]: { ...(existing ?? { callId: ev.call_id, name: ev.name, arguments: ev.arguments ?? {} }), status: 'awaiting_approval' as const }
      }
      const segments = existing ? turn.segments : [...turn.segments, { kind: 'tool' as const, callId: ev.call_id }]
      return {
        ...turn,
        tools,
        segments,
        approvals: [
          ...turn.approvals.filter((a) => a.approvalId !== ev.approval_id),
          {
            approvalId: ev.approval_id,
            callId: ev.call_id,
            name: ev.name,
            arguments: ev.arguments ?? {},
            risk: ev.risk,
            reason: ev.reason,
            target: ev.target ?? null,
            riskLabel: ev.risk_label ?? null,
            status: 'pending'
          }
        ]
      }
    }
    case 'tool_result': {
      const prev = turn.tools[ev.call_id] ?? { callId: ev.call_id, name: ev.name, arguments: {}, status: 'running' as const }
      const approval = turn.approvals.find((a) => a.callId === ev.call_id)
      const denied = approval?.status === 'deny' || wasDenied(ev.result)
      const segments = turn.tools[ev.call_id] ? turn.segments : [...turn.segments, { kind: 'tool' as const, callId: ev.call_id }]
      return {
        ...turn,
        segments,
        tools: {
          ...turn.tools,
          [ev.call_id]: {
            ...prev,
            result: ev.result,
            isError: ev.is_error,
            durationMs: ev.duration_ms,
            status: denied ? 'denied' : ev.is_error ? 'error' : 'done'
          }
        },
        approvals: turn.approvals.map((a) => (a.callId === ev.call_id && a.status === 'pending' ? { ...a, status: 'expired' } : a))
      }
    }
    case 'usage': {
      const u = turn.usage
      return {
        ...turn,
        usage: {
          model: ev.model,
          prompt_tokens: (u?.prompt_tokens ?? 0) + (ev.prompt_tokens ?? 0),
          completion_tokens: (u?.completion_tokens ?? 0) + (ev.completion_tokens ?? 0)
        }
      }
    }
    case 'error':
      return { ...turn, error: { message: ev.message, recoverable: ev.recoverable ?? true }, status: turn.status === 'streaming' ? 'error' : turn.status }
    case 'done': {
      if (ev.cancelled) {
        const error = turn.error?.message === 'Stopped.' ? undefined : turn.error
        return { ...turn, status: 'cancelled', error, messageId: ev.message_id }
      }
      const hasText = turn.segments.some((s) => s.kind === 'text' && s.text.trim())
      const segments = !hasText && ev.content?.trim() ? [...turn.segments, { kind: 'text' as const, text: ev.content }] : turn.segments
      const tools = Object.fromEntries(
        Object.entries(turn.tools).map(([k, t]) => [k, t.status === 'running' || t.status === 'awaiting_approval' ? { ...t, status: 'done' as const } : t])
      )
      return { ...turn, segments, tools, status: turn.error && !hasText && !ev.content ? 'error' : 'done', messageId: ev.message_id }
    }
    default:
      return turn
  }
}

/** Plain text of a turn (for copy). */
export function turnText(turn: AssistantTurnView): string {
  return turn.segments
    .filter((s): s is { kind: 'text'; text: string } => s.kind === 'text')
    .map((s) => s.text)
    .join('\n\n')
    .trim()
}
