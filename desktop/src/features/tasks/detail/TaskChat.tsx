/** Task conversation: change requests that send the task back to planning (v2 TaskChatSection). */
import { IconArrowUp, IconMessages, IconSparkles } from '@tabler/icons-react'
import { useEffect, useRef, useState } from 'react'
import { Avatar, EmptyState, IconButton, Spinner, Textarea } from '@/components/ui'
import { useBootstrap } from '@/hooks/core'
import type { Task } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'

export function TaskChat({ task, sending, onSend }: { task: Task; sending: boolean; onSend: (message: string) => void }) {
  const [text, setText] = useState('')
  const end = useRef<HTMLDivElement>(null)
  const { data: boot } = useBootstrap()
  const history = task.chat_history ?? []

  useEffect(() => {
    end.current?.scrollIntoView({ block: 'nearest' })
  }, [history.length])

  const blocked =
    task.status === 'processing'
      ? 'This task is running. Cancel the run before asking for changes.'
      : task.task_type === 'swarm'
        ? "Change requests aren't available for swarm tasks. Re-run it with a new goal instead."
        : null
  const replanning = task.status === 'planning' && history.length > 0 && history[history.length - 1]?.role === 'user'

  const send = () => {
    const msg = text.trim()
    if (!msg || blocked || sending) return
    onSend(msg)
    setText('')
  }

  return (
    <section className="flex flex-col gap-3">
      {history.length === 0 ? (
        <EmptyState
          compact
          icon={<IconMessages />}
          title="Ask for changes"
          description="Tell Sentient what to change: another source, a different format, extra steps. It replans and asks you to approve again."
        />
      ) : (
        <div className="space-y-3">
          {history.map((m, i) => {
            const mine = m.role === 'user'
            return (
              <div key={i} className={cn('flex items-end gap-2', mine && 'flex-row-reverse')}>
                {mine ? (
                  <Avatar name={boot?.assistant.user_name || 'You'} size={24} />
                ) : (
                  <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-accent/15 text-accent-text">
                    <IconSparkles size={13} />
                  </span>
                )}
                <div className={cn('max-w-[80%]', mine && 'text-right')}>
                  <div
                    className={cn(
                      'selectable inline-block whitespace-pre-wrap rounded-2xl px-3 py-2 text-left text-sm leading-relaxed',
                      mine ? 'rounded-br-md bg-accent/14 text-fg' : 'rounded-bl-md border border-border bg-elevated text-fg'
                    )}
                  >
                    {m.content}
                  </div>
                  <div className="mt-0.5 px-1 text-2xs text-fg-faint">{relativeTime(m.timestamp)}</div>
                </div>
              </div>
            )
          })}
          {replanning && (
            <div className="flex items-center gap-2 pl-8 text-xs text-fg-subtle">
              <Spinner size={12} className="text-info" /> Replanning with your change…
            </div>
          )}
          <div ref={end} />
        </div>
      )}

      <div className={cn('rounded-xl border border-border bg-elevated p-1.5 focus-within:border-accent/45', blocked && 'opacity-70')}>
        <div className="flex items-end gap-1.5">
          <Textarea
            autoGrow
            rows={1}
            maxHeight={140}
            value={text}
            disabled={!!blocked}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
            placeholder={blocked ?? 'Ask for a change, e.g. “also include the calendar invite link”'}
            aria-label="Request a change"
            className="border-0 bg-transparent px-2 shadow-none hover:border-0 focus:border-0 focus:ring-0"
          />
          <IconButton variant="primary" size="md" label="Send change request" icon={<IconArrowUp size={16} />} loading={sending} disabled={!text.trim() || !!blocked} onClick={send} />
        </div>
      </div>
      {!blocked && <p className="text-2xs text-fg-subtle">Sending a change sends the task back to planning. You'll approve the new plan before it runs.</p>}
    </section>
  )
}
