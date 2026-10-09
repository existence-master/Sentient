/** A question a running task asked (`ask_user`): pick an option or type an answer, and the run carries on. */
import { IconMessageQuestion, IconSend } from '@tabler/icons-react'
import { useState } from 'react'
import { Button, Textarea } from '@/components/ui'
import type { Run, Task } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'
import { useTaskOps } from '../useTaskOps'

export function RunQuestion({ task, run, className }: { task: Task; run: Run; className?: string }) {
  const ops = useTaskOps()
  const [text, setText] = useState('')
  const [picked, setPicked] = useState<string | null>(null)
  const q = run.pending_question
  if (!q?.question) return null
  const id = task.task_id
  const sending = ops.isBusy('answerQuestion', id)
  const send = (answer: string) => {
    const a = answer.trim()
    if (!a || sending) return
    setPicked(a)
    void ops.answerQuestion(id, run.run_id, a).then((res) => {
      if (res) setText('')
      else setPicked(null)
    })
  }

  return (
    <form
      className={cn('overflow-hidden rounded-xl border border-warning/30 bg-warning/5', className)}
      onSubmit={(e) => {
        e.preventDefault()
        send(text)
      }}
    >
      <div className="flex items-start gap-3 border-b border-warning/20 px-4 py-3">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-warning/15 text-warning">
          <IconMessageQuestion size={17} />
        </span>
        <div className="min-w-0">
          <div className="text-sm font-semibold text-fg">Sentient needs your answer to carry on</div>
          <div className="text-xs text-fg-muted">
            The task is paused{q.asked_at ? ` since ${relativeTime(q.asked_at)}` : ''}. Your answer picks it up right where it stopped.
          </div>
        </div>
      </div>
      <div className="space-y-3 px-4 py-4">
        <p className="whitespace-pre-wrap text-sm font-medium text-fg">{q.question}</p>
        {q.options.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {q.options.map((o) => (
              <Button key={o} type="button" size="sm" variant="secondary" loading={sending && picked === o} disabled={sending} onClick={() => send(o)}>
                {o}
              </Button>
            ))}
          </div>
        )}
        <Textarea
          autoGrow
          rows={1}
          minHeight={38}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send(text)
            }
          }}
          placeholder={q.options.length ? 'Or type your own answer' : 'Your answer'}
          aria-label="Your answer"
          className="bg-surface"
        />
      </div>
      <div className="flex items-center gap-3 border-t border-warning/20 px-4 py-2.5">
        <span className="text-xs text-fg-subtle">You can also answer from the notification or a paired Telegram or Discord chat.</span>
        <span className="flex-1" />
        <Button type="submit" size="sm" variant="primary" leftIcon={<IconSend size={14} />} disabled={!text.trim() || sending} loading={sending && picked === text.trim()}>
          Send answer
        </Button>
      </div>
    </form>
  )
}
