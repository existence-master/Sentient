/**
 * Live chat turns started from this window.
 *
 * History comes from React Query (`useMessages`) and is folded with `foldTranscript`.
 * While a turn streams, the optimistic user message and the assistant turn live here,
 * keyed by session id (or `pending:<client_id>` until the engine assigns one).
 * When the turn finishes the transcript is refetched and the live items are dropped,
 * so the timeline switches seamlessly to persisted history.
 */
import { toast } from 'sonner'
import { create } from 'zustand'
import { api, errorMessage } from '@/lib/api'
import { applyAgentEvent, emptyTurn, meterFrom, type AssistantTurnView, type AttachmentView, type TimelineItem, type UserMessageView } from '@/lib/chatFold'
import { queryClient } from '@/lib/queryClient'
import type { AgentEvent, ApprovalDecision, ChatEvent, ContextMeter, Session, TranscriptMessage } from '@/lib/types'
import { uid } from '@/lib/utils'
import { live } from '@/lib/ws'
import { qk } from '@/hooks/queryKeys'

export interface ChatRequestDraft {
  text: string
  attachments?: AttachmentView[]
  model?: string
}

export interface LiveSession {
  key: string
  sessionId: string | null
  clientId: string
  user: UserMessageView
  turn: AssistantTurnView
  streaming: boolean
  /** Number of transcript rows that existed before this turn (history shown above the live items). */
  baseCount: number
  request: ChatRequestDraft
  transport: 'ws' | 'http'
  abort?: AbortController
  /** Messages sent with `chat.steer` that the model hasn't picked up yet. */
  steers: Array<{ id: string; text: string }>
  /** Queued messages the engine dropped when the reply was stopped (never sent). */
  unsent?: Array<{ id: string; text: string }>
  /** Items of the previous live turn, kept visible when a late steer started a new turn. */
  carry?: TimelineItem[]
}

interface ChatState {
  live: Record<string, LiveSession>
  /** client_id -> session_id, so a "new chat" view can navigate once the engine assigns an id. */
  resolved: Record<string, string>
  /** session_id -> how full the model's context was after its latest call (#131). Kept after the turn ends. */
  context: Record<string, ContextMeter>
  send: (sessionId: string | null, draft: ChatRequestDraft) => Promise<string>
  cancel: (sessionId: string) => void
  retry: (sessionId: string) => Promise<void>
  respondApproval: (sessionId: string, approvalId: string, decision: ApprovalDecision) => Promise<void>
  handleEvent: (ev: ChatEvent) => void
  dropLive: (key: string) => void
}

const pendingKey = (clientId: string) => `pending:${clientId}`

function patchLive(state: ChatState, key: string, fn: (l: LiveSession) => LiveSession): Partial<ChatState> {
  const cur = state.live[key]
  return cur ? { live: { ...state.live, [key]: fn(cur) } } : {}
}

export function liveItems(l: LiveSession | undefined): TimelineItem[] {
  return l ? [...(l.carry ?? []), l.user, l.turn] : []
}

async function finishTurn(key: string, sessionId: string | null) {
  const store = useChat.getState()
  void queryClient.invalidateQueries({ queryKey: qk.sessions })
  if (!sessionId) return
  let rows: TranscriptMessage[]
  try {
    rows = await queryClient.fetchQuery({ queryKey: qk.messages(sessionId), queryFn: () => api.sessions.messages(sessionId), staleTime: 0 })
  } catch {
    return // keep the live items visible if history can't be loaded
  }
  const l = useChat.getState().live[key]
  if (!l || l.streaming) return
  // Cancelled/failed turns aren't persisted by the engine: keep showing "Stopped" / the error
  // on top of the history the turn started from, until the next send or retry in this chat
  // (or an app restart).
  const persistedReply = rows.length > l.baseCount && rows[rows.length - 1]?.role !== 'user'
  if (!persistedReply && (l.turn.status === 'cancelled' || l.turn.status === 'error')) return
  if (l.unsent?.length || l.user.notSent) return // keep "your queued message wasn't sent" visible until the next send
  store.dropLive(key)
}

export const useChat = create<ChatState>((set, get) => ({
  live: {},
  resolved: {},
  context: {},

  send: async (sessionId, draft) => {
    const clientId = uid()
    const key = sessionId ?? pendingKey(clientId)
    const existing = sessionId ? get().live[sessionId] : undefined
    if (existing?.streaming && sessionId) {
      // Steering: the text reaches the model at its next round (§10). Attachments wait for the next turn.
      const text = draft.text.trim()
      if (!text) return key
      if (existing.transport === 'ws' && live.send({ type: 'chat.steer', session_id: sessionId, text })) {
        set(patchLive(get(), sessionId, (l) => ({ ...l, steers: [...l.steers, { id: uid(), text }] })))
      } else {
        toast.message('Sentient is still replying', { description: 'Stop the current reply or wait for it to finish.' })
      }
      return key
    }
    const rows = sessionId ? queryClient.getQueryData<TranscriptMessage[]>(qk.messages(sessionId)) : undefined
    const now = new Date().toISOString()
    const entry: LiveSession = {
      key,
      sessionId,
      clientId,
      user: { kind: 'user', id: `u-${clientId}`, text: draft.text, attachments: draft.attachments ?? [], createdAt: now, pending: true },
      turn: emptyTurn(`a-${clientId}`),
      streaming: true,
      baseCount: rows?.length ?? 0,
      request: draft,
      transport: 'ws',
      steers: []
    }
    set({ live: { ...get().live, [key]: entry } })

    const attachments = (draft.attachments ?? []).map((a) => a.name)
    const wsOk = live.isOpen || (await live.waitOpen(1500))
    if (wsOk && live.send({ type: 'chat.send', session_id: sessionId ?? undefined, text: draft.text, attachments, model: draft.model, client_id: clientId, channel: 'desktop' })) {
      return key
    }

    // NDJSON fallback when the socket is down.
    const abort = new AbortController()
    set(patchLive(get(), key, (l) => ({ ...l, transport: 'http', abort })))
    void (async () => {
      let currentKey = key
      try {
        for await (const ev of api.chat.stream({ text: draft.text, session_id: sessionId ?? undefined, attachments, model: draft.model, channel: 'desktop' }, abort.signal)) {
          if (ev.type === 'session') {
            get().handleEvent({ ...ev, client_id: clientId })
            currentKey = ev.session_id
          } else {
            get().handleEvent({ ...ev, session_id: ev.session_id ?? get().live[currentKey]?.sessionId ?? undefined } as ChatEvent)
          }
        }
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') {
          get().handleEvent({ type: 'done', content: '', message_id: null, cancelled: true, session_id: get().live[currentKey]?.sessionId })
        } else {
          set(patchLive(get(), currentKey, (l) => ({
            ...l,
            streaming: false,
            turn: { ...l.turn, status: 'error', error: { message: errorMessage(err), recoverable: true } }
          })))
        }
      }
    })()
    return key
  },

  cancel: (sessionId) => {
    const l = get().live[sessionId]
    if (!l?.streaming) return
    if (l.transport === 'http') l.abort?.abort()
    else if (l.sessionId) live.send({ type: 'chat.cancel', session_id: l.sessionId })
  },

  retry: async (sessionId) => {
    const l = get().live[sessionId]
    const draft = l?.request ?? lastUserDraft(sessionId)
    if (!draft) return
    if (l) get().dropLive(sessionId)
    await get().send(sessionId, draft)
  },

  respondApproval: async (sessionId, approvalId, decision) => {
    set(patchLive(get(), sessionId, (l) => ({
      ...l,
      turn: { ...l.turn, approvals: l.turn.approvals.map((a) => (a.approvalId === approvalId ? { ...a, status: decision } : a)) }
    })))
    if (live.send({ type: 'approval.respond', approval_id: approvalId, decision })) return
    try {
      await api.approvals.respond(approvalId, decision)
    } catch (err) {
      toast.error("Couldn't send your answer", { description: errorMessage(err) })
    }
  },

  handleEvent: (ev) => {
    const state = get()
    switch (ev.type) {
      case 'hello':
      case 'pong':
        return
      case 'session': {
        const pk = ev.client_id ? pendingKey(ev.client_id) : null
        const pending = pk ? state.live[pk] : undefined
        if (pending && pk) {
          const { [pk]: _moved, ...rest } = state.live
          void _moved
          set({
            live: { ...rest, [ev.session_id]: { ...pending, key: ev.session_id, sessionId: ev.session_id, user: { ...pending.user, pending: false } } },
            resolved: { ...state.resolved, [pending.clientId]: ev.session_id }
          })
          // Show the new chat in the sidebar right away.
          queryClient.setQueryData<Session[]>(qk.sessions, (old) =>
            old?.some((s) => s.id === ev.session_id)
              ? old
              : [{ id: ev.session_id, title: null, channel: 'desktop', created_at: new Date().toISOString(), updated_at: new Date().toISOString() }, ...(old ?? [])]
          )
        } else if (state.live[ev.session_id]) {
          set(patchLive(state, ev.session_id, (l) => ({ ...l, user: { ...l.user, pending: false } })))
        }
        return
      }
      case 'steer_ack': {
        const l = state.live[ev.session_id]
        if (!l || ev.queued) return
        // No reply was running any more: the engine starts a new turn with the oldest queued text.
        const [first, ...rest] = l.steers
        if (!first) return
        const clientId = uid()
        set({
          live: {
            ...state.live,
            [ev.session_id]: {
              ...l,
              clientId,
              carry: l.streaming ? l.carry : [...(l.carry ?? []), l.user, l.turn],
              user: { kind: 'user', id: `u-${clientId}`, text: first.text, attachments: [], createdAt: new Date().toISOString() },
              turn: emptyTurn(`a-${clientId}`),
              streaming: true,
              request: { text: first.text },
              steers: rest
            }
          }
        })
        return
      }
      case 'approval.ack': {
        if (ev.resolved) return
        for (const l of Object.values(state.live)) {
          if (l.turn.approvals.some((a) => a.approvalId === ev.approval_id)) {
            set(patchLive(get(), l.key, (x) => ({
              ...x,
              turn: { ...x.turn, approvals: x.turn.approvals.map((a) => (a.approvalId === ev.approval_id ? { ...a, status: 'expired' } : a)) }
            })))
            toast.message('That request already expired.')
          }
        }
        return
      }
      default: {
        // §17 a queued message dropped by Stop everything: settle the entry that sent it (matched by client_id),
        // never whatever turn is newest in that chat.
        if ((ev.type === 'done' && ev.dropped) || (ev.type === 'error' && ev.dropped)) {
          if (ev.type === 'error') return // the done that follows carries the text
          const owner = Object.values(state.live).find((l) => ev.client_id && l.clientId === ev.client_id)
          if (owner) {
            set(patchLive(state, owner.key, (l) => ({
              ...l,
              streaming: false,
              user: { ...l.user, pending: false, notSent: true },
              turn: { ...l.turn, status: 'cancelled' }
            })))
            return
          }
          const sidKey = ev.session_id && state.live[ev.session_id] ? ev.session_id : null
          if (sidKey) {
            // its entry was already replaced: show the text as not sent without touching the current turn
            set(patchLive(state, sidKey, (l) => ({ ...l, unsent: [...(l.unsent ?? []), ...(ev.dropped ?? []).map((text) => ({ id: uid(), text }))] })))
          } else {
            toast("Stopped. Your queued message wasn't sent.", { description: (ev.dropped ?? []).join(' / ').slice(0, 200) })
          }
          return
        }
        const sid = ev.session_id
        if (ev.type === 'usage' && sid) {
          // the latest call's model decides: one with an unknown context length clears the meter
          const { [sid]: _old, ...others } = get().context
          void _old
          const meter = meterFrom(ev)
          set({ context: meter ? { ...others, [sid]: meter } : others })
        }
        const key = sid && state.live[sid] ? sid : null
        if (!key) {
          if (ev.type === 'error' && !sid) toast.error(ev.message)
          return
        }
        const next = applyAgentEvent(state.live[key].turn, ev as AgentEvent)
        const finished = ev.type === 'done'
        if (ev.type === 'user_interjection') {
          const steers = state.live[key].steers
          const idx = Math.max(0, steers.findIndex((s) => s.text.trim() === ev.text.trim()))
          set(patchLive(state, key, (l) => ({ ...l, turn: next, steers: steers.filter((_, i) => i !== idx) })))
          return
        }
        // "A reply is already in progress." arrives without a turn: settle the optimistic entry.
        const orphanError = ev.type === 'error' && !ev.turn_id && !state.live[key].turn.segments.length
        // A stopped reply drops what was queued behind it (steers, §17 dropped turns): show them as not sent.
        const stopped = ev.type === 'done' && ev.cancelled
        set(patchLive(state, key, (l) => ({
          ...l,
          turn: orphanError ? { ...next, status: 'error' } : next,
          streaming: finished || orphanError ? false : l.streaming,
          user: { ...l.user, pending: false },
          steers: stopped ? [] : l.steers,
          unsent: stopped ? [...(l.unsent ?? []), ...l.steers] : l.unsent
        })))
        if (finished) void finishTurn(key, get().live[key]?.sessionId ?? null)
      }
    }
  },

  dropLive: (key) => {
    const { [key]: removed, ...rest } = get().live
    if (!removed) return
    removed.user.attachments.forEach((a) => a.previewUrl && URL.revokeObjectURL(a.previewUrl))
    set({ live: rest })
  }
}))

function lastUserDraft(sessionId: string): ChatRequestDraft | null {
  const rows = queryClient.getQueryData<TranscriptMessage[]>(qk.messages(sessionId)) ?? []
  const lastUser = [...rows].reverse().find((r) => r.role === 'user')
  return lastUser ? { text: lastUser.content ?? '', attachments: (lastUser.attachments ?? []).map((name) => ({ name })) } : null
}

/** Route every chat event from the socket into the store (call once at startup). */
export function installChatEvents(): () => void {
  return live.onChat((ev) => useChat.getState().handleEvent(ev))
}
