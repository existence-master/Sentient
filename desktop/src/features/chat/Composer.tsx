import {
  IconAlertCircle,
  IconAlertTriangle,
  IconArrowUp,
  IconMicrophone,
  IconPaperclip,
  IconPlayerStopFilled,
  IconRefresh,
  IconWaveSine,
  IconX
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { IconButton, Spinner, Textarea, Tooltip } from '@/components/ui'
import { openBrowserView } from '@/features/browser/state'
import { useMemoryActions } from '@/hooks/memory'
import { useTaskActions } from '@/hooks/tasks'
import { useVoiceStatus } from '@/hooks/voice'
import { errorMessage, isNotImplemented } from '@/lib/api'
import type { AttachmentView } from '@/lib/chatFold'
import type { ContextMeter } from '@/lib/types'
import { cn } from '@/lib/utils'
import { ContextGauge } from './ContextMeter'
import { ModelOverride } from './ModelOverride'
import { filterCommands, parseSlash, slashQuery, SlashMenu, type SlashCommand } from './slash'
import type { useAttachments } from './useAttachments'
import { useDictation } from './useDictation'

const drafts = new Map<string, string>()

export interface ComposerProps {
  draftKey: string
  assistantName: string
  /** While a reply streams the composer stays usable: sending adds to the running reply (§10 steering). */
  streaming: boolean
  attachments: ReturnType<typeof useAttachments>
  onSend: (text: string, attachments: AttachmentView[], model: string | undefined) => void
  onStop: () => void
  autoFocus?: boolean
  /** Imperative text injection (suggestions). */
  inject?: { text: string; nonce: number } | null
  /** How full the model's context was after its latest call in this chat (#131). */
  context?: ContextMeter | null
  className?: string
}

export function Composer({ draftKey, assistantName, streaming, attachments, onSend, onStop, autoFocus, inject, context, className }: ComposerProps) {
  const [text, setText] = useState(() => drafts.get(draftKey) ?? '')
  const [model, setModel] = useState<string | undefined>()
  const [menuIndex, setMenuIndex] = useState(0)
  const [menuDismissed, setMenuDismissed] = useState(false)
  const textarea = useRef<HTMLTextAreaElement>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()
  const voiceStatus = useVoiceStatus()
  const tasks = useTaskActions()
  const memory = useMemoryActions()

  const dictation = useDictation((t) => {
    setText((cur) => (cur.trim() ? `${cur.trimEnd()} ${t}` : t))
    textarea.current?.focus()
  })
  const dictationMissing = dictation.unavailable || (voiceStatus.isError && isNotImplemented(voiceStatus.error))

  const query = slashQuery(text)
  const commands = query === null || menuDismissed ? [] : filterCommands(query)
  const menuOpen = commands.length > 0

  useEffect(() => {
    drafts.set(draftKey, text)
  }, [draftKey, text])

  useEffect(() => {
    setMenuIndex(0)
    if (query === null) setMenuDismissed(false)
  }, [query])

  useEffect(() => {
    if (autoFocus) textarea.current?.focus()
  }, [autoFocus])

  useEffect(() => {
    if (!inject) return
    setText(inject.text)
    requestAnimationFrame(() => {
      const el = textarea.current
      if (!el) return
      el.focus()
      el.setSelectionRange(inject.text.length, inject.text.length)
    })
  }, [inject])

  useEffect(() => {
    const focus = () => textarea.current?.focus()
    window.addEventListener('sentient:focus-composer', focus)
    return () => window.removeEventListener('sentient:focus-composer', focus)
  }, [])

  const hasText = text.trim().length > 0
  // While streaming only text can be added; attachments wait for the next message.
  const canSend = !attachments.uploading && (hasText || (!streaming && attachments.ready > 0))

  const clear = () => {
    setText('')
    drafts.delete(draftKey)
  }

  const runCommand = (command: SlashCommand, arg: string): boolean => {
    switch (command.name) {
      case 'new':
        clear()
        navigate('/chat', { state: { fresh: Date.now() } })
        return true
      case 'voice':
        clear()
        navigate('/voice')
        return true
      case 'browser':
        clear()
        openBrowserView()
        return true
      case 'task':
        if (!arg) {
          toast.message('What should the task be?', { description: 'Type it after /task, for example: /task send me the weather every morning at 8' })
          return false
        }
        clear()
        tasks.create.mutate(
          { prompt: arg },
          {
            onSuccess: (task) =>
              toast.success('Task created', {
                description: task.name || arg,
                action: { label: 'View', onClick: () => navigate(`/tasks?task=${encodeURIComponent(task.task_id)}`) }
              }),
            onError: (e) => toast.error("Couldn't create the task", { description: errorMessage(e) })
          }
        )
        return true
      case 'remember':
        if (!arg) {
          toast.message('What should Sentient remember?', { description: 'Type it after /remember, for example: /remember my sister is called Asha' })
          return false
        }
        clear()
        memory.create.mutate(
          { content: arg, source: 'manual' },
          {
            onSuccess: (r) => toast.success(r.action === 'SKIP' ? 'Sentient already knew that' : 'Saved to memory', { description: arg }),
            onError: (e) => toast.error("Couldn't save that", { description: errorMessage(e) })
          }
        )
        return true
    }
  }

  const pickCommand = (command: SlashCommand) => {
    if (command.arg) {
      setText(`/${command.name} `)
      requestAnimationFrame(() => textarea.current?.focus())
    } else {
      runCommand(command, '')
    }
  }

  const send = () => {
    const slash = parseSlash(text)
    if (slash) {
      runCommand(slash.command, slash.arg)
      return
    }
    if (!canSend) return
    onSend(text.trim(), streaming ? [] : attachments.take(), model)
    clear()
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (menuOpen) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault()
        setMenuIndex((i) => (i + (e.key === 'ArrowDown' ? 1 : commands.length - 1)) % commands.length)
        return
      }
      if ((e.key === 'Enter' && !e.shiftKey) || e.key === 'Tab') {
        e.preventDefault()
        pickCommand(commands[Math.min(menuIndex, commands.length - 1)])
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        setMenuDismissed(true)
        return
      }
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      send()
    }
    if (e.key === 'Escape' && streaming && !hasText) onStop()
  }

  const placeholder =
    dictation.state === 'recording' ? 'Listening…' : streaming ? `Add something while ${assistantName} works…` : `Message ${assistantName}…`

  return (
    <div className={cn('relative w-full', className)}>
      <AnimatePresence>{menuOpen && <SlashMenu items={commands} active={menuIndex} onHover={setMenuIndex} onPick={pickCommand} />}</AnimatePresence>
      {context?.warning && (
        <p role="status" className="mb-2 flex items-start gap-1.5 px-1 text-xs leading-relaxed text-warning">
          <IconAlertTriangle size={14} className="mt-0.5 shrink-0" />
          {context.warning}
        </p>
      )}
      <div className="rounded-2xl border border-border-strong bg-elevated shadow-pop transition-[border-color,box-shadow] focus-within:border-accent/45 focus-within:shadow-glow">
        <AnimatePresence initial={false}>
          {attachments.items.length > 0 && (
            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
              <div className="flex flex-wrap gap-2 px-3 pt-3">
                {attachments.items.map((a) => (
                  <div
                    key={a.id}
                    className={cn(
                      'group relative flex h-12 max-w-56 items-center gap-2.5 overflow-hidden rounded-xl border bg-surface pl-1.5 pr-7',
                      a.status === 'error' ? 'border-danger/40' : 'border-border'
                    )}
                  >
                    {a.previewUrl ? (
                      <img src={a.previewUrl} alt="" className="size-9 rounded-lg object-cover" />
                    ) : (
                      <div className="flex size-9 items-center justify-center rounded-lg bg-accent/10 text-2xs font-semibold uppercase text-accent-text">
                        {(a.file.name.split('.').pop() ?? 'file').slice(0, 4)}
                      </div>
                    )}
                    <div className="min-w-0">
                      <div className="truncate text-xs font-medium text-fg">{a.file.name || 'Pasted image'}</div>
                      <div className={cn('text-2xs', a.status === 'error' ? 'text-danger' : 'text-fg-subtle')}>
                        {a.status === 'uploading'
                          ? `Uploading ${Math.round(a.progress * 100)}%`
                          : a.status === 'error'
                            ? 'Upload failed'
                            : streaming
                              ? 'Sends with your next message'
                              : 'Ready'}
                      </div>
                    </div>
                    {a.status === 'uploading' && (
                      <div className="absolute inset-x-0 bottom-0 h-0.5 bg-active">
                        <div className="h-full bg-accent transition-[width]" style={{ width: `${a.progress * 100}%` }} />
                      </div>
                    )}
                    <div className="absolute right-1 top-1 flex flex-col gap-0.5">
                      <button type="button" aria-label="Remove attachment" onClick={() => attachments.remove(a.id)} className="flex size-5 items-center justify-center rounded-md text-fg-subtle hover:bg-active hover:text-fg">
                        <IconX size={12} />
                      </button>
                      {a.status === 'error' && (
                        <Tooltip content={a.error ?? 'Retry'}>
                          <button type="button" aria-label="Retry upload" onClick={() => attachments.retry(a.id)} className="flex size-5 items-center justify-center rounded-md text-danger hover:bg-active">
                            <IconRefresh size={12} />
                          </button>
                        </Tooltip>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <Textarea
          ref={textarea}
          autoGrow
          rows={1}
          maxHeight={260}
          minHeight={48}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          onBlur={() => setMenuDismissed(true)}
          onFocus={() => setMenuDismissed(false)}
          onPaste={(e) => {
            const files = Array.from(e.clipboardData.files)
            if (files.length) {
              if (!e.clipboardData.getData('text')) e.preventDefault()
              attachments.add(files)
            }
          }}
          placeholder={placeholder}
          aria-label="Message"
          aria-expanded={menuOpen}
          className="border-0 bg-transparent px-4 pb-1 pt-3.5 text-md shadow-none hover:border-0 focus:border-0 focus:ring-0"
        />

        <div className="flex items-center gap-1 px-2 pb-2">
          <IconButton size="md" label="Attach files" icon={<IconPaperclip size={17} />} onClick={() => fileInput.current?.click()} />
          <input
            ref={fileInput}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files?.length) attachments.add(e.target.files)
              e.target.value = ''
            }}
          />
          <ModelOverride value={model} onChange={setModel} />

          <div className="flex-1" />
          {context && <ContextGauge meter={context} className="mr-1.5" />}

          {dictation.state === 'recording' ? (
            <button
              type="button"
              onClick={dictation.stop}
              className="flex h-8 items-center gap-2 rounded-lg bg-danger/12 px-2.5 text-xs font-medium text-danger hover:bg-danger/18"
            >
              <span className="size-2 animate-pulse rounded-full bg-danger" />
              {Math.floor(dictation.elapsed / 60)}:{String(dictation.elapsed % 60).padStart(2, '0')}
              <span className="text-danger/80">Stop</span>
            </button>
          ) : (
            <IconButton
              size="md"
              label={dictationMissing ? 'Dictation needs the voice engine (coming soon)' : dictation.supported ? 'Dictate' : 'Microphone not available'}
              disabled={!dictation.supported || dictationMissing || dictation.state === 'transcribing'}
              icon={dictation.state === 'transcribing' ? <Spinner size={15} /> : <IconMicrophone size={17} />}
              onClick={() => void dictation.start()}
            />
          )}
          <IconButton size="md" label="Voice mode" icon={<IconWaveSine size={17} />} onClick={() => navigate('/voice')} />

          {streaming && (
            <Tooltip content="Stop the reply" shortcut={hasText ? undefined : 'Esc'}>
              <button
                type="button"
                aria-label="Stop"
                onClick={onStop}
                className={cn(
                  'ml-1 flex size-8.5 items-center justify-center rounded-full transition-transform hover:scale-105 active:scale-95',
                  hasText ? 'border border-border-strong bg-surface text-fg' : 'bg-fg text-bg'
                )}
              >
                <IconPlayerStopFilled size={14} />
              </button>
            </Tooltip>
          )}
          {(!streaming || hasText) && (
            <Tooltip content={attachments.uploading ? 'Waiting for uploads…' : streaming ? 'Add to the reply' : 'Send'} shortcut="Enter">
              <button
                type="button"
                aria-label={streaming ? 'Add to the reply' : 'Send'}
                disabled={!canSend && !parseSlash(text)}
                onClick={send}
                className="ml-1 flex size-8.5 items-center justify-center rounded-full bg-accent text-accent-fg shadow-soft transition-[transform,background-color,color] hover:scale-105 hover:bg-accent-hover active:scale-95 disabled:scale-100 disabled:bg-active disabled:text-fg-faint disabled:shadow-none"
              >
                <IconArrowUp size={17} stroke={2.4} />
              </button>
            </Tooltip>
          )}
        </div>
      </div>
      <div className="mt-2 flex items-center justify-center gap-1.5 text-2xs text-fg-faint">
        {attachments.items.some((a) => a.status === 'error') ? (
          <span className="flex items-center gap-1 text-danger">
            <IconAlertCircle size={11} /> Some files failed to upload
          </span>
        ) : streaming ? (
          <span>{assistantName} reads what you add before its next step · Esc to stop</span>
        ) : (
          <span>Enter to send · Shift+Enter for a new line · type / for commands</span>
        )}
      </div>
    </div>
  )
}
