import { IconArrowDown, IconFileUpload, IconMessageOff, IconSpy, IconWorldWww } from '@tabler/icons-react'
import { ChatNote, OtherMessage, OutsideContentBar } from './OutsideContent'
import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router'
import { toast } from 'sonner'
import { Button, EmptyState, IconButton, Skeleton } from '@/components/ui'
import { useBrowserView } from '@/features/browser/state'
import { ChannelBadge } from '@/features/channels/meta'
import { HelpersTray } from './HelpersTray'
import { useBootstrap, useConfig, useMessages, useRuleProposals, useSessions, useSetChatModel, useSetChatScope } from '@/hooks/core'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage, isApiError } from '@/lib/api'
import { foldTranscript, type AttachmentView, type TimelineItem } from '@/lib/chatFold'
import { chatModelLabel } from '@/lib/models'
import { queryClient } from '@/lib/queryClient'
import type { ApprovalDecision, Session } from '@/lib/types'
import { cn } from '@/lib/utils'
import { liveItems, useChat } from '@/stores/chat'
import { AssistantTurn } from './AssistantTurn'
import { scopeLabel } from '@/features/memory/bits'
import { ChatScopeMenu } from './ChatScopeMenu'
import { Composer } from './Composer'
import { useMemorySaveToasts } from './memoryToasts'
import { RuleProposalCard } from './RuleProposalCard'
import { takeScreenShare } from './screenShare'
import { EmptyChat, type Suggestion } from './EmptyChat'
import { IncognitoBadge, IncognitoIntro } from './Incognito'
import { useAttachments } from './useAttachments'
import { UserMessage } from './UserMessage'

export function ChatPage() {
  const { sessionId } = useParams()
  const location = useLocation()
  // `pendingKey`: a message already sent from outside the chat view (push to talk) whose new chat this view shows.
  // `incognito` (or `#/chat?incognito=1`): a new incognito chat (#154).
  const state = location.state as { fresh?: number; prompt?: { text: string; autoSend?: boolean }; pendingKey?: string; incognito?: boolean } | null
  const newIncognito = !sessionId && (!!state?.incognito || /(^|[?&])incognito=1\b/.test(location.search))
  return (
    <ChatView
      key={sessionId ?? `new-${state?.fresh ?? ''}${newIncognito ? '-incognito' : ''}`}
      sessionId={sessionId}
      initialPrompt={sessionId ? undefined : state?.prompt}
      initialPendingKey={sessionId ? undefined : state?.pendingKey}
      newIncognito={newIncognito}
    />
  )
}

function ChatView({
  sessionId,
  initialPrompt,
  initialPendingKey,
  newIncognito = false
}: {
  sessionId?: string
  initialPrompt?: { text: string; autoSend?: boolean }
  initialPendingKey?: string
  newIncognito?: boolean
}) {
  const navigate = useNavigate()
  const [pendingKey, setPendingKey] = useState<string | null>(initialPendingKey ?? null)
  const liveKey = sessionId ?? pendingKey
  const liveSession = useChat((s) => (liveKey ? s.live[liveKey] : undefined))
  const resolvedId = useChat((s) => (pendingKey ? s.resolved[pendingKey.replace('pending:', '')] : undefined))
  // "Remembered: ... Undo" for memories saved from this chat (#234), also during a new chat's first reply
  useMemorySaveToasts(sessionId ?? resolvedId)
  const context = useChat((s) => (sessionId ? s.context[sessionId] : undefined))
  const send = useChat((s) => s.send)
  const cancel = useChat((s) => s.cancel)
  const retry = useChat((s) => s.retry)
  const respondApproval = useChat((s) => s.respondApproval)
  const browserOpen = useBrowserView((s) => s.open)
  const toggleBrowser = useBrowserView((s) => s.toggle)

  const bootstrap = useBootstrap()
  const config = useConfig()
  const sessions = useSessions()
  const messages = useMessages(sessionId)
  const proposals = useRuleProposals(sessionId)
  const attachments = useAttachments()
  const [inject, setInject] = useState<{ text: string; nonce: number } | null>(null)
  const [dragging, setDragging] = useState(false)
  const dragDepth = useRef(0)

  const assistantName = bootstrap.data?.assistant.name ?? 'Sentient'
  const showThinking = config.data?.chat.show_thinking ?? true
  const session = sessions.data?.find((s) => s.id === sessionId)
  const knownIncognito = useChat((s) => (sessionId ? !!s.incognito[sessionId] : false))
  const incognito = newIncognito || knownIncognito || !!session?.incognito || !!liveSession?.request.incognito
  const usesMemory = config.data?.chat.incognito_uses_memory ?? true

  // The chat's own model (#312). A new chat keeps the pick here until its first message creates it.
  const [newChatModel, setNewChatModel] = useState<string | null>(null)
  const [creating, setCreating] = useState(false) // a new chat with its own model is being created
  const setModel = useSetChatModel()
  const chatModel = sessionId ? ((session?.model as string | null | undefined) ?? null) : newChatModel
  const onChatModelChange = useCallback(
    (model: string | null) => {
      if (!sessionId) {
        setNewChatModel(model)
        return
      }
      setModel.mutate(
        { id: sessionId, model },
        {
          onSuccess: () => toast.success(model ? `This chat now uses ${chatModelLabel(model)}` : 'This chat uses the default model again'),
          onError: (e) => toast.error("Couldn't change this chat's model", { description: errorMessage(e) })
        }
      )
    },
    [sessionId, setModel]
  )

  // The memories this chat uses (#155). A new chat keeps the pick here until its first message creates it.
  const [newChatScope, setNewChatScope] = useState<string>('personal')
  const setScope = useSetChatScope()
  const chatScope = sessionId ? ((session?.memory_scope as string | undefined) ?? 'personal') : newChatScope
  const onChatScopeChange = useCallback(
    (scope: string) => {
      if (!sessionId) {
        setNewChatScope(scope)
        return
      }
      setScope.mutate(
        { id: sessionId, scope },
        {
          onSuccess: () => toast.success(`This chat now uses your ${scopeLabel(scope)} memories`),
          onError: (e) => toast.error("Couldn't change this chat's memories", { description: errorMessage(e) })
        }
      )
    },
    [sessionId, setScope]
  )
  const newChatOwnScope = newChatScope !== 'personal' ? newChatScope : null

  // A brand-new chat navigates to its session once the engine assigns an id.
  useEffect(() => {
    if (!sessionId && resolvedId) navigate(`/chat/${resolvedId}`, { replace: true })
  }, [sessionId, resolvedId, navigate])

  const items: TimelineItem[] = useMemo(() => {
    const rows = messages.data ?? []
    const history = foldTranscript(liveSession ? rows.slice(0, liveSession.baseCount) : rows)
    return [...history, ...liveItems(liveSession)]
  }, [messages.data, liveSession])

  const streaming = !!liveSession?.streaming
  const lastAssistantIndex = items.map((i) => i.kind).lastIndexOf('assistant')

  const onSend = useCallback(
    async (text: string, files: AttachmentView[]) => {
      if (!sessionId && (newChatModel || newChatOwnScope)) {
        // a new chat with its own model (#312) or memories (#155): create it first so even its first reply uses them
        if (creating) return
        let id: string
        setCreating(true) // until this view moves to the new chat, so a second message can't start another one
        try {
          id = (await api.sessions.create({ incognito: newIncognito, model: newChatModel, memory_scope: newChatOwnScope })).session_id
        } catch (err) {
          setCreating(false)
          setInject({ text, nonce: Date.now() }) // the message box was cleared: put the text back
          toast.error("Couldn't start the chat", { description: errorMessage(err) })
          return
        }
        useChat.getState().adopt(id, { incognito: newIncognito })
        const now = new Date().toISOString()
        const row: Session = {
          id,
          title: null,
          channel: 'desktop',
          created_at: now,
          updated_at: now,
          incognito: newIncognito,
          model: newChatModel,
          memory_scope: newChatScope
        }
        queryClient.setQueryData<Session[]>(qk.sessions, (old) => [row, ...(old ?? []).filter((s) => s.id !== id)])
        await send(id, { text, attachments: files })
        navigate(`/chat/${id}`, { replace: true })
        return
      }
      const key = await send(sessionId ?? null, { text, attachments: files, incognito: !sessionId && newIncognito })
      if (!sessionId) setPendingKey(key)
    },
    [send, sessionId, newIncognito, newChatModel, newChatOwnScope, newChatScope, navigate, creating]
  )

  const onPick = (s: Pick<Suggestion, 'prompt' | 'autoSend'>) => {
    if (s.autoSend) void onSend(s.prompt, [])
    else setInject({ text: s.prompt, nonce: Date.now() })
  }

  // A picture from Share this window / Share a region (#172): attach it to this new chat and wait for the question.
  useEffect(() => {
    if (sessionId) return
    const shot = takeScreenShare()
    if (shot) attachments.add([shot], { source: 'screen' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Prompt handed over from onboarding ("Try: Plan my week").
  const handedOver = useRef(false)
  useEffect(() => {
    if (!initialPrompt || handedOver.current) return
    handedOver.current = true
    onPick({ prompt: initialPrompt.text, autoSend: initialPrompt.autoSend })
    window.history.replaceState({ ...window.history.state, usr: null }, '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPrompt])

  const onRetry = useCallback(() => {
    if (sessionId) void retry(sessionId)
  }, [retry, sessionId])

  const onApprove = useCallback(
    (approvalId: string, decision: ApprovalDecision) => {
      if (liveSession) void respondApproval(liveSession.key, approvalId, decision)
    },
    [respondApproval, liveSession]
  )

  const empty = !sessionId && !liveSession
  const notFound = sessionId && isApiError(messages.error) && messages.error.status === 404
  const loadingHistory = !!sessionId && messages.isLoading && !liveSession

  return (
    <div
      className="relative flex h-full flex-col"
      onDragEnter={(e) => {
        if (!e.dataTransfer.types.includes('Files')) return
        dragDepth.current++
        setDragging(true)
      }}
      onDragOver={(e) => e.dataTransfer.types.includes('Files') && e.preventDefault()}
      onDragLeave={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1)
        if (!dragDepth.current) setDragging(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        dragDepth.current = 0
        setDragging(false)
        if (e.dataTransfer.files.length) attachments.add(e.dataTransfer.files)
      }}
    >
      {!empty && (
        <div
          className={cn(
            'flex h-12 shrink-0 items-center gap-3 border-b px-6',
            incognito ? 'border-incognito/25 bg-incognito/8' : 'border-border'
          )}
        >
          {incognito && <IconSpy size={17} className="shrink-0 text-incognito" aria-hidden />}
          <h2 className="min-w-0 truncate text-sm font-medium text-fg">
            {session?.title?.trim() || (incognito ? 'Incognito chat' : streaming ? 'New chat' : 'Chat')}
          </h2>
          {incognito ? <IncognitoBadge /> : <ChannelBadge channel={session?.channel} withLabel />}
          <span className="flex-1" />
          {streaming && <span className="text-xs text-fg-subtle">{assistantName} is replying…</span>}
          {sessionId && <ChatScopeMenu value={chatScope} onChange={onChatScopeChange} />}
          <HelpersTray sessionId={sessionId} />
          <IconButton size="sm" label="Browser live view" icon={<IconWorldWww size={16} />} onClick={toggleBrowser} active={browserOpen} />
        </div>
      )}

      {empty ? (
        <div className="flex min-h-0 flex-1 flex-col items-center overflow-y-auto px-6 py-10">
          <div className="my-auto w-full max-w-[720px]">
            {newIncognito ? (
              <IncognitoIntro usesMemory={usesMemory} />
            ) : (
              <EmptyChat userName={bootstrap.data?.assistant.user_name} onPick={onPick} />
            )}
            <div className="mt-8">
              <div className="mb-1.5 flex items-center justify-end gap-1.5 text-xs text-fg-subtle">
                <span>Memories</span>
                <ChatScopeMenu value={newChatScope} onChange={onChatScopeChange} />
              </div>
              <Composer
                draftKey="new"
                assistantName={assistantName}
                streaming={false}
                busy={creating}
                attachments={attachments}
                onSend={onSend}
                onStop={() => undefined}
                autoFocus
                inject={inject}
                chatModel={chatModel}
                onChatModelChange={onChatModelChange}
              />
            </div>
          </div>
        </div>
      ) : notFound ? (
        <div className="flex flex-1 items-center justify-center">
          <EmptyState
            icon={<IconMessageOff />}
            title="This chat doesn't exist anymore"
            description="It may have been deleted."
            action={
              <Button variant="primary" onClick={() => navigate('/chat')}>
                Start a new chat
              </Button>
            }
          />
        </div>
      ) : (
        <>
          <Timeline items={items} streaming={streaming} loading={loadingHistory}>
            {items.map((item, i) =>
              item.kind === 'other' ? (
                <OtherMessage key={item.id} message={item} />
              ) : item.kind === 'note' ? (
                <ChatNote key={item.id} note={item} />
              ) : item.kind === 'user' ? (
                <UserMessage key={item.id} message={item} onRestore={(text) => setInject({ text, nonce: Date.now() })} />
              ) : (
                <AssistantTurn
                  key={item.id}
                  turn={item}
                  assistantName={assistantName}
                  showThinking={showThinking}
                  isLast={i === lastAssistantIndex && i === items.length - 1}
                  onRetry={onRetry}
                  onApprove={onApprove}
                  steers={liveSession && item.id === liveSession.turn.id ? liveSession.steers : undefined}
                  unsent={liveSession && item.id === liveSession.turn.id ? liveSession.unsent : undefined}
                  onRestore={(text) => setInject({ text, nonce: Date.now() })}
                />
              )
            )}
            {/* "Make this a rule?" for something the user said in this chat (#130) */}
            {!loadingHistory && proposals.data?.map((p) => <RuleProposalCard key={p.id} proposal={p} />)}
          </Timeline>
          <div className="shrink-0 px-6 pb-4 pt-2">
            <div className="mx-auto w-full max-w-[760px]">
              {sessionId && session?.untrusted ? <OutsideContentBar sessionId={sessionId} source={session.untrusted} /> : null}
              {incognito && (
                <p className="mb-2 flex items-center justify-center gap-1.5 text-xs text-fg-subtle">
                  <IconSpy size={13} className="text-incognito" aria-hidden />
                  Incognito: nothing here is remembered, and the chat is deleted when you leave it.
                </p>
              )}
              <Composer
                draftKey={sessionId ?? 'pending'}
                assistantName={assistantName}
                streaming={streaming}
                context={context}
                attachments={attachments}
                onSend={onSend}
                onStop={() => liveSession && cancel(liveSession.key)}
                autoFocus
                inject={inject}
                chatModel={chatModel}
                onChatModelChange={sessionId ? onChatModelChange : undefined}
              />
            </div>
          </div>
        </>
      )}

      <AnimatePresence>
        {dragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="pointer-events-none absolute inset-3 z-30 flex items-center justify-center rounded-2xl border-2 border-dashed border-accent/60 bg-surface/85 backdrop-blur-sm"
          >
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="flex size-14 items-center justify-center rounded-2xl bg-accent/12 text-accent-text">
                <IconFileUpload size={26} />
              </div>
              <div className="text-md font-medium text-fg">Drop to attach</div>
              <div className="text-sm text-fg-subtle">Images, PDFs, documents, spreadsheets. Up to 50 MB each.</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function Timeline({ items, streaming, loading, children }: { items: TimelineItem[]; streaming: boolean; loading: boolean; children: React.ReactNode }) {
  const scroller = useRef<HTMLDivElement>(null)
  const content = useRef<HTMLDivElement>(null)
  const [atBottom, setAtBottom] = useState(true)
  const stick = useRef(true)

  const scrollToBottom = useCallback((smooth = false) => {
    const el = scroller.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' })
  }, [])

  // `?at=top` keeps the first message in view (deep links from search, screenshots).
  const atTop = useRef(/(^|[?&])at=top\b/.test(useLocation().search))
  if (atTop.current) stick.current = false

  useLayoutEffect(() => {
    if (!loading && !atTop.current) scrollToBottom()
  }, [loading, scrollToBottom])

  // Keep pinned to the bottom while content grows (streaming, images loading).
  useEffect(() => {
    const el = content.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      if (stick.current) scrollToBottom()
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [scrollToBottom])

  // A new message from the user always scrolls down.
  const count = items.length
  const historyCount = useRef<number | null>(null)
  useEffect(() => {
    if (atTop.current) {
      // Stay at the top while history loads; the first new message scrolls down as usual.
      if (count === 0 || historyCount.current === null || count === historyCount.current) {
        if (count > 0 && historyCount.current === null) historyCount.current = count
        return
      }
      atTop.current = false
    }
    stick.current = true
    scrollToBottom(true)
  }, [count, scrollToBottom])

  return (
    <div className="relative min-h-0 flex-1">
      <div
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget
          const bottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
          stick.current = bottom
          setAtBottom(bottom)
        }}
        className="h-full overflow-y-auto"
      >
        <div ref={content} className="mx-auto w-full max-w-[760px] space-y-7 px-6 pb-6 pt-7">
          {loading ? (
            <div className="space-y-8">
              <div className="flex justify-end">
                <Skeleton className="h-10 w-2/5 rounded-2xl" />
              </div>
              <div className="flex gap-3.5">
                <Skeleton className="size-6 rounded-full" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-4/5" />
                </div>
              </div>
            </div>
          ) : (
            children
          )}
        </div>
      </div>
      <AnimatePresence>
        {!atBottom && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6 }}
            className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center"
          >
            <button
              type="button"
              onClick={() => {
                stick.current = true
                scrollToBottom(true)
              }}
              className={cn(
                'pointer-events-auto flex h-8 items-center gap-1.5 rounded-full border border-border-strong bg-overlay px-3 text-xs font-medium text-fg shadow-pop hover:bg-elevated'
              )}
            >
              <IconArrowDown size={14} />
              {streaming ? 'Jump to latest' : 'Latest'}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
