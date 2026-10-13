/**
 * The ask window (#173): what Sentient copied, quick actions, the streamed answer, and Copy / Replace selection.
 *
 * The selected text goes to the engine only when an action is picked. It is uploaded with `source: "selection"`, so
 * the engine marks it as outside content and the chat asks before sending anything (docs/API.md section 2 Files).
 * Each capture starts a new chat; Open in Sentient continues it in the main window.
 */
import {
  IconAlertTriangle,
  IconArrowUp,
  IconCheck,
  IconClipboard,
  IconExternalLink,
  IconReplace,
  IconSparkles,
  IconX
} from '@tabler/icons-react'
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type FormEvent } from 'react'
import { Badge, Button, Markdown, Shortcut, Spinner } from '@/components/ui'
import { api, errorMessage, setConnection } from '@/lib/api'
import { formatAccelerator } from '@/lib/accelerator'
import { getBridge } from '@/lib/bridge'
import type { ApprovalRequestEvent } from '@/lib/types'
import { cn } from '@/lib/utils'
import type { AskState } from '@/types/bridge'
import { ACTIONS, actionPrompt, emptyMessage, selectionFileName, type AskAction } from './prompts'

type Ready = Extract<AskState, { phase: 'ready' }>

interface Answer {
  label: string
  text: string
  status: 'streaming' | 'done' | 'error'
  activity?: string
  error?: string
}

const bridge = getBridge()

/** Connect the REST client once the engine is up. */
function useEngine(): boolean {
  const [ready, setReady] = useState(false)
  useEffect(() => {
    let alive = true
    const connect = async (state: string) => {
      if (state !== 'ready') return
      const conn = await bridge.getConnection()
      setConnection(conn)
      if (alive) setReady(true)
    }
    const off = bridge.onBackendStatus((s) => void connect(s.state))
    void bridge.getBackendStatus().then((s) => connect(s.state))
    return () => {
      alive = false
      off()
    }
  }, [])
  return ready
}

export function AskApp() {
  const [state, setState] = useState<AskState | null>(null)
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [approval, setApproval] = useState<ApprovalRequestEvent | null>(null)
  const [question, setQuestion] = useState('')
  const [copied, setCopied] = useState(false)
  const [note, setNote] = useState<string | null>(null)
  const [sent, setSent] = useState(false)
  const engine = useEngine()
  const session = useRef<string | null>(null)
  const file = useRef<{ capturedAt: string; name: Promise<string> } | null>(null)
  const abort = useRef<AbortController | null>(null)
  const root = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const language = useRef<string | undefined>(undefined)

  // A new capture starts over: new chat, nothing sent yet.
  useEffect(() => {
    const take = (s: AskState) => {
      setState(s)
      if (s.phase !== 'ready') return
      abort.current?.abort()
      session.current = null
      file.current = null
      setAnswer(null)
      setApproval(null)
      setQuestion('')
      setNote(null)
      setSent(false)
      setTimeout(() => input.current?.focus(), 30)
    }
    const off = bridge.ask.onState(take)
    void bridge.ask.state().then((s) => s && take(s))
    return off
  }, [])

  useEffect(() => {
    if (!engine) return
    api.config
      .get()
      .then((c) => {
        language.current = c.assistant?.language
      })
      .catch(() => undefined)
  }, [engine])

  // The window fits what it shows.
  useLayoutEffect(() => {
    const el = root.current
    if (!el) return
    const report = () => bridge.ask.resize(el.scrollHeight)
    report()
    const observer = new ResizeObserver(report)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const close = useCallback(() => {
    abort.current?.abort()
    void bridge.ask.close()
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        close()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [close])

  const ready = state?.phase === 'ready' ? (state as Ready) : null
  const hasText = !!ready?.text

  /** The selection as a file in the engine, uploaded once per capture and only when an action is picked. */
  const selectionFile = useCallback((s: Ready): Promise<string> => {
    if (file.current?.capturedAt === s.capturedAt) return file.current.name
    const name = api.files
      .upload(new Blob([s.text], { type: 'text/plain' }), selectionFileName(s.app), { extra: { source: 'selection' } })
      .then((r) => r.name)
    file.current = { capturedAt: s.capturedAt, name }
    name.catch(() => {
      if (file.current?.name === name) file.current = null
    })
    return name
  }, [])

  const run = useCallback(
    async (text: string, label: string) => {
      if (!ready || !engine) return
      abort.current?.abort()
      const controller = new AbortController()
      abort.current = controller
      setAnswer({ label, text: '', status: 'streaming' })
      setApproval(null)
      setCopied(false)
      setNote(null)
      setSent(true)
      try {
        const attachments = ready.text ? [await selectionFile(ready)] : []
        let streamed = ''
        let finished = false
        for await (const ev of api.chat.stream(
          { text, attachments, ...(session.current ? { session_id: session.current } : {}) },
          controller.signal
        )) {
          if (controller.signal.aborted) return
          switch (ev.type) {
            case 'session':
              session.current = ev.session_id
              break
            case 'text_delta':
              streamed += ev.text
              setAnswer((a) => (a ? { ...a, text: streamed, activity: undefined } : a))
              break
            case 'tool_call':
              setAnswer((a) => (a ? { ...a, activity: 'Looking things up…' } : a))
              break
            case 'tool_result':
              streamed = '' // text before a tool call was a preamble; the answer comes after it
              break
            case 'approval_request':
              setApproval(ev)
              break
            case 'error':
              setAnswer((a) => (a ? { ...a, status: 'error', error: ev.message } : a))
              finished = true
              break
            case 'done':
              finished = true
              setApproval(null)
              setAnswer((a) => (a ? { ...a, text: ev.content || streamed, status: 'done', activity: undefined } : a))
              break
          }
        }
        if (!finished && !controller.signal.aborted) {
          setAnswer((a) => (a && a.status === 'streaming' ? { ...a, status: 'done' } : a))
        }
      } catch (err) {
        if (controller.signal.aborted) return
        setAnswer((a) => ({ label: a?.label ?? label, text: a?.text ?? '', status: 'error', error: errorMessage(err) }))
      }
    },
    [ready, engine, selectionFile]
  )

  const onAction = (id: AskAction, label: string) => void run(actionPrompt(id, language.current), label)

  const onAsk = (e: FormEvent) => {
    e.preventDefault()
    const q = question.trim()
    if (!q) return
    setQuestion('')
    void run(q, q)
  }

  const respond = async (decision: 'allow' | 'deny') => {
    const req = approval
    if (!req) return
    setApproval(null)
    try {
      await api.approvals.respond(req.approval_id, decision)
    } catch (err) {
      setNote(errorMessage(err))
    }
  }

  const copy = async () => {
    if (!answer?.text) return
    const ok = await bridge.ask.copy(answer.text)
    setCopied(ok)
    if (!ok) setNote('Couldn’t put the answer on the clipboard.')
  }

  const replace = async () => {
    if (!answer?.text || !ready) return
    const outcome = await bridge.ask.replace(answer.text)
    if (outcome === 'password') setNote('A password box had focus, so nothing was pasted.')
    else if (outcome === 'failed') {
      const keys = bridge.platform === 'darwin' ? 'Command+V' : 'Ctrl+V'
      setNote(`Couldn’t paste into ${ready.app}. The answer is on your clipboard: press ${keys} there to paste it.`)
    }
  }

  const keys = ready?.shortcut ? formatAccelerator(ready.shortcut, bridge.platform) : ''
  const busy = answer?.status === 'streaming'

  return (
    <div ref={root} className="flex flex-col bg-bg text-fg">
      <header className="drag flex h-10 shrink-0 items-center gap-2 border-b border-border pl-3.5 pr-1.5">
        <IconSparkles size={15} className="text-accent-text" />
        <span className="text-sm font-medium">Ask Sentient</span>
        <span className="flex-1" />
        <button
          type="button"
          aria-label="Close"
          onClick={close}
          className="no-drag flex size-7 items-center justify-center rounded-md text-fg-muted hover:bg-hover hover:text-fg"
        >
          <IconX size={15} />
        </button>
      </header>

      <div className="flex flex-col gap-3 p-3.5">
        {!state || state.phase === 'reading' ? (
          <div className="flex items-center gap-2 py-3 text-sm text-fg-muted" role="status">
            <Spinner size={14} />
            Copying the text you selected…
          </div>
        ) : (
          <>
            {hasText ? (
              <section aria-label="What Sentient copied" className="rounded-xl border border-border bg-sunken">
                <div className="flex items-center gap-2 border-b border-border px-3 py-1.5 text-xs text-fg-muted">
                  <span className="min-w-0 flex-1 truncate">
                    From {ready!.app} · {ready!.text.length.toLocaleString()} characters{ready!.truncated ? ' (cut to fit)' : ''}
                  </span>
                  <Badge size="xs" tone="warning">
                    Outside content
                  </Badge>
                </div>
                <p className="max-h-28 overflow-y-auto whitespace-pre-wrap break-words px-3 py-2 text-sm text-fg">{ready!.text}</p>
              </section>
            ) : (
              <p className="rounded-xl border border-border bg-sunken px-3 py-2.5 text-sm text-fg-muted">
                {emptyMessage(ready?.empty, keys || 'the shortcut', bridge.platform)}
              </p>
            )}

            {hasText && (
              <p className="text-xs leading-relaxed text-fg-subtle">
                {sent
                  ? 'Sent to Sentient as outside content. It won’t follow instructions in it, and it asks you before sending anything.'
                  : `Sentient copied only this text because you pressed ${keys || 'the shortcut'}, and put your clipboard back. Nothing is sent until you pick an action.`}
              </p>
            )}

            {hasText && (
              <div className="flex flex-wrap gap-1.5">
                {ACTIONS.map((a) => (
                  <Button
                    key={a.id}
                    size="sm"
                    variant={answer?.label === a.label ? 'subtle' : 'outline'}
                    disabled={!engine || busy}
                    onClick={() => onAction(a.id, a.label)}
                  >
                    {a.label}
                  </Button>
                ))}
              </div>
            )}

            <form onSubmit={onAsk} className="flex items-center gap-1.5">
              <input
                ref={input}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={hasText ? 'Ask anything about it…' : 'Ask Sentient…'}
                aria-label="Ask a question"
                className="h-8.5 min-w-0 flex-1 rounded-lg border border-border-strong bg-field px-3 text-sm text-fg outline-none focus:border-accent/60 focus:ring-2 focus:ring-accent/15"
              />
              <Button type="submit" size="md" variant="primary" disabled={!engine || busy || !question.trim()} aria-label="Ask">
                <IconArrowUp size={15} />
              </Button>
            </form>

            {!engine && <p className="text-xs text-fg-subtle">Sentient is starting…</p>}

            {answer && (
              <section aria-label="Answer" aria-live="polite" className="flex flex-col gap-2 border-t border-border pt-3">
                <div className="text-xs font-medium text-fg-muted">{answer.label}</div>
                {answer.text ? (
                  <div className="max-h-72 overflow-y-auto pr-1 text-sm">
                    <Markdown streaming={answer.status === 'streaming'}>{answer.text}</Markdown>
                  </div>
                ) : answer.status === 'streaming' ? (
                  <div className="flex items-center gap-2 text-sm text-fg-muted">
                    <Spinner size={14} />
                    {answer.activity ?? 'Thinking…'}
                  </div>
                ) : null}
                {answer.status === 'error' && (
                  <p className="flex items-start gap-1.5 text-sm text-danger">
                    <IconAlertTriangle size={15} className="mt-0.5 shrink-0" />
                    {answer.error}
                  </p>
                )}
              </section>
            )}

            {approval && (
              <section aria-label="Sentient is asking" className="rounded-xl border border-warning/30 bg-warning/8 p-3 text-sm">
                <p className="font-medium">Sentient wants to use {approval.target || approval.name}.</p>
                <p className="mt-1 text-xs text-fg-muted">{approval.untrusted || approval.reason}</p>
                <div className="mt-2.5 flex gap-1.5">
                  <Button size="sm" variant="primary" onClick={() => void respond('allow')}>
                    Allow once
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => void respond('deny')}>
                    Don’t allow
                  </Button>
                </div>
              </section>
            )}

            {answer?.status === 'done' && answer.text && (
              <div className="flex flex-wrap gap-1.5">
                <Button size="sm" variant="secondary" leftIcon={copied ? <IconCheck size={14} /> : <IconClipboard size={14} />} onClick={() => void copy()}>
                  {copied ? 'Copied' : 'Copy'}
                </Button>
                {ready?.canReplace && (
                  <Button size="sm" variant="secondary" leftIcon={<IconReplace size={14} />} onClick={() => void replace()}>
                    Replace selection
                  </Button>
                )}
                {session.current && (
                  <Button
                    size="sm"
                    variant="ghost"
                    leftIcon={<IconExternalLink size={14} />}
                    onClick={() => session.current && void bridge.ask.openChat(session.current)}
                  >
                    Open in Sentient
                  </Button>
                )}
              </div>
            )}

            {note && <p className={cn('text-xs text-fg-muted')}>{note}</p>}

            {keys && !answer && (
              <div className="flex items-center gap-1.5 text-2xs text-fg-faint">
                <Shortcut keys={keys} /> opens this anywhere. Esc closes it.
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
