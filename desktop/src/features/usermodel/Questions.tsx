/** Open questions Sentient has about you, as friendly cards you can answer or set aside. */
import { IconMessageCircleQuestion, IconSend } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Button, Textarea } from '@/components/ui'
import { errorMessage } from '@/lib/api'
import { useUserModelActions } from '@/hooks/userModel'
import type { Insight, UserModelQuestion } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'
import { dimensionMeta, tint } from './meta'

export function OpenQuestions({ questions, insights }: { questions: UserModelQuestion[]; insights: Insight[] }) {
  if (!questions.length) return null
  return (
    <section aria-label="Open questions" className="space-y-3">
      <div>
        <h2 className="text-md font-semibold text-fg">A few things I’m not sure about</h2>
        <p className="mt-0.5 text-sm text-fg-subtle">Answer if you like. Skipping is fine, I won’t ask again.</p>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <AnimatePresence initial={false}>
          {questions.map((q) => (
            <QuestionCard key={q.id} q={q} related={insights.find((i) => i.id === q.insight_id)} />
          ))}
        </AnimatePresence>
      </div>
    </section>
  )
}

function QuestionCard({ q, related }: { q: UserModelQuestion; related?: Insight }) {
  const { answer, dismiss } = useUserModelActions()
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const color = related ? dimensionMeta(related.dimension).color : 'var(--accent)'

  const send = () => {
    const a = text.trim()
    if (!a) return
    answer.mutate(
      { id: q.id, answer: a },
      {
        onSuccess: () => toast.success('Thanks, that helps', { description: 'I’ll fold it into how I understand you.' }),
        onError: (e) => toast.error('Couldn’t save your answer', { description: errorMessage(e) })
      }
    )
  }

  return (
    <motion.div
      layout="position"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.97 }}
      className="relative flex flex-col overflow-hidden rounded-2xl border border-border bg-elevated/60 p-4 shadow-soft"
    >
      <div aria-hidden className="pointer-events-none absolute -right-10 -top-12 size-36 rounded-full opacity-60 blur-2xl" style={{ background: tint(color, 22) }} />
      <div className="relative flex gap-3">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-xl" style={{ background: tint(color, 16), color }}>
          <IconMessageCircleQuestion size={17} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium leading-relaxed text-fg">{q.question}</p>
          <div className="mt-1 text-2xs text-fg-subtle">
            {related ? `About ${dimensionMeta(related.dimension).label.toLowerCase()}` : 'A question for you'} · {relativeTime(q.created_at)}
          </div>
        </div>
      </div>
      <div className={cn('relative mt-3', open ? 'space-y-2' : 'flex items-center justify-end gap-1.5')}>
        {open ? (
          <>
            <Textarea
              autoFocus
              autoGrow
              minHeight={56}
              maxHeight={160}
              value={text}
              placeholder="In your own words…"
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
                if (e.key === 'Escape') setOpen(false)
              }}
            />
            <div className="flex justify-end gap-1.5">
              <Button size="sm" variant="ghost" onClick={() => setOpen(false)}>
                Cancel
              </Button>
              <Button size="sm" variant="primary" leftIcon={<IconSend size={14} />} disabled={!text.trim()} loading={answer.isPending} onClick={send}>
                Send answer
              </Button>
            </div>
          </>
        ) : (
          <>
            <Button size="sm" variant="ghost" loading={dismiss.isPending} onClick={() => dismiss.mutate(q.id, { onError: (e) => toast.error('Couldn’t set it aside', { description: errorMessage(e) }) })}>
              Not now
            </Button>
            <Button size="sm" variant="secondary" onClick={() => setOpen(true)}>
              Answer
            </Button>
          </>
        )}
      </div>
    </motion.div>
  )
}
