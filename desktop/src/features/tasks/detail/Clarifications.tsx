/** Clarifying questions: a form while the task waits for answers, a Q&A list afterwards. */
import { IconMessageQuestion } from '@tabler/icons-react'
import { useState } from 'react'
import { Button, Textarea } from '@/components/ui'
import type { ClarificationAnswer, ClarifyingQuestion } from '@/lib/types'
import { cn } from '@/lib/utils'

export function Clarifications({
  questions,
  pending,
  submitting,
  onSubmit
}: {
  questions: ClarifyingQuestion[]
  pending: boolean
  submitting: boolean
  onSubmit: (answers: ClarificationAnswer[]) => void
}) {
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [touched, setTouched] = useState(false)
  const open = questions.filter((q) => !q.answer)
  const missing = open.filter((q) => !answers[q.question_id]?.trim())

  if (!pending) {
    const answered = questions.filter((q) => q.answer)
    if (!answered.length) return null
    return (
      <details className="group rounded-xl border border-border bg-surface">
        <summary className="flex cursor-pointer list-none items-center gap-2 px-3.5 py-2.5 text-sm font-medium text-fg">
          <IconMessageQuestion size={16} className="text-fg-subtle" />
          Your answers
          <span className="rounded-full bg-active px-1.5 text-2xs text-fg-muted">{answered.length}</span>
          <span className="flex-1" />
          <span className="text-xs text-fg-subtle group-open:hidden">Show</span>
        </summary>
        <dl className="space-y-2.5 border-t border-border px-3.5 py-3">
          {answered.map((q) => (
            <div key={q.question_id}>
              <dt className="text-xs text-fg-subtle">{q.text}</dt>
              <dd className="mt-0.5 whitespace-pre-wrap text-sm text-fg">{q.answer}</dd>
            </div>
          ))}
        </dl>
      </details>
    )
  }

  return (
    <form
      className="overflow-hidden rounded-xl border border-warning/30 bg-warning/5"
      onSubmit={(e) => {
        e.preventDefault()
        setTouched(true)
        if (missing.length) return
        onSubmit(open.map((q) => ({ question_id: q.question_id, answer_text: answers[q.question_id].trim() })))
      }}
    >
      <div className="flex items-start gap-3 border-b border-warning/20 px-4 py-3">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-warning/15 text-warning">
          <IconMessageQuestion size={17} />
        </span>
        <div>
          <div className="text-sm font-semibold text-fg">A few questions before Sentient plans this</div>
          <div className="text-xs text-fg-muted">Answer them and planning picks up right where it left off.</div>
        </div>
      </div>
      <div className="space-y-4 px-4 py-4">
        {questions.map((q, i) => (
          <div key={q.question_id} className="space-y-1.5">
            <label htmlFor={`q-${q.question_id}`} className="flex gap-2 text-sm font-medium text-fg">
              <span className="text-fg-subtle tabular-nums">{i + 1}.</span>
              {q.text}
            </label>
            {q.answer ? (
              <p className="ml-5 whitespace-pre-wrap text-sm text-fg-muted">{q.answer}</p>
            ) : (
              <Textarea
                id={`q-${q.question_id}`}
                autoGrow
                rows={1}
                minHeight={38}
                value={answers[q.question_id] ?? ''}
                invalid={touched && !answers[q.question_id]?.trim()}
                onChange={(e) => setAnswers((a) => ({ ...a, [q.question_id]: e.target.value }))}
                placeholder="Your answer"
                className={cn('ml-5 w-[calc(100%-1.25rem)] bg-surface')}
              />
            )}
          </div>
        ))}
      </div>
      <div className="flex items-center gap-3 border-t border-warning/20 px-4 py-2.5">
        <span className="text-xs text-fg-subtle">{touched && missing.length ? `Answer ${missing.length} more question${missing.length === 1 ? '' : 's'}` : `${open.length - missing.length} of ${open.length} answered`}</span>
        <span className="flex-1" />
        <Button type="submit" size="sm" variant="primary" loading={submitting}>
          Submit answers
        </Button>
      </div>
    </form>
  )
}
