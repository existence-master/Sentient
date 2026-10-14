import { IconArrowBackUp, IconArrowUpRight, IconCheck, IconChecks, IconInbox, IconLanguage, IconPencil, IconShieldCheck, IconX } from '@tabler/icons-react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Alert, Badge, Button, EmptyState, Skeleton, Textarea } from '@/components/ui'
import { useMemoryReview, useMemoryReviewActions } from '@/hooks/memory'
import { errorMessage } from '@/lib/api'
import type { MemoryReviewItem, MergedMemory } from '@/lib/types'
import { parseDate, relativeTime } from '@/lib/utils'

/** Memories Sentient holds until the user approves them (ADR 0021): from emails, web pages, imports, unprompted work. */
export function ReviewTab() {
  const review = useMemoryReview()
  const actions = useMemoryReviewActions()

  const groups = useMemo(() => {
    const by = new Map<string, MemoryReviewItem[]>()
    // instructions get their own group, whatever they came from (#234)
    for (const item of review.data?.items ?? []) {
      const key = item.reason === 'instruction' ? '#instructions' : item.from
      by.set(key, [...(by.get(key) ?? []), item])
    }
    return [...by.entries()]
  }, [review.data])

  if (review.isLoading) {
    return (
      <div className="space-y-3">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-28 rounded-xl" />
        ))}
      </div>
    )
  }
  if (review.isError) {
    return (
      <Alert tone="danger" title="Couldn't load memories waiting for review" action={<Button size="sm" onClick={() => void review.refetch()}>Retry</Button>}>
        {errorMessage(review.error)}
      </Alert>
    )
  }
  const merged = review.data?.merged ?? []
  if (!review.data?.items.length) {
    return (
      <div className="space-y-5">
        <EmptyState
          icon={<IconInbox />}
          title="Nothing to review"
          description="When Sentient learns something from an email, a web page, an import or work it did on its own, it waits here for your OK before using it."
        />
        {merged.length > 0 && <MergedHistory items={merged} />}
      </div>
    )
  }

  const approveAll = (from: string) =>
    actions.approveAll.mutate(from, {
      onSuccess: (r) => toast.success(`Approved ${r.approved} from ${from}`),
      onError: (e) => toast.error('Couldn’t approve them', { description: errorMessage(e) })
    })

  return (
    <div className="space-y-5">
      <Alert tone="info" icon={<IconShieldCheck />} title="Sentient wants to remember these">
        They came from an email, a web page, an import or work Sentient did on its own, or they tell Sentient how to act, so it waits for your
        OK before using them. Anything you don’t review is let go after {review.data.expire_days} days.
      </Alert>
      {groups.map(([from, items]) => (
        <section key={from} className="space-y-2.5">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-medium text-fg">{items[0]?.reason === 'instruction' ? 'Instructions for Sentient' : `From ${from}`}</h3>
            <span className="rounded-full bg-active px-1.5 text-2xs tabular-nums text-fg-muted">{items.length}</span>
            <div className="flex-1" />
            {items.length > 1 && from !== '#instructions' && (
              <Button size="sm" variant="ghost" leftIcon={<IconChecks size={14} />} loading={actions.approveAll.isPending && actions.approveAll.variables === from} onClick={() => approveAll(from)}>
                Approve all from {from}
              </Button>
            )}
          </div>
          <ul className="space-y-2.5">
            {items.map((item) => (
              <ReviewCard key={`${item.kind}:${item.id}`} item={item} />
            ))}
          </ul>
        </section>
      ))}
      {merged.length > 0 && <MergedHistory items={merged} />}
    </div>
  )
}

/** Review history (#353): memories the nightly tidy-up kept just once because they said the same thing in two
 * languages. Undo brings the merged one back, and the tidy-up never merges that pair again. */
function MergedHistory({ items }: { items: MergedMemory[] }) {
  const { undoMerge } = useMemoryReviewActions()
  const undo = (id: number) =>
    undoMerge.mutate(id, {
      onSuccess: () => toast.success('Brought back as its own memory'),
      onError: (e) => toast.error('Couldn’t undo it', { description: errorMessage(e) })
    })
  return (
    <section className="space-y-2.5">
      <div>
        <h3 className="text-sm font-medium text-fg">Tidied up overnight</h3>
        <p className="mt-0.5 text-xs text-fg-subtle">These said the same thing as another memory in a different language, so Sentient kept one of them.</p>
      </div>
      <ul className="space-y-2.5">
        {items.map((m) => (
          <li key={m.id} className="rounded-xl border border-border bg-surface p-4">
            <div className="flex flex-wrap items-center gap-2 text-xs text-fg-subtle">
              <Badge size="xs" tone="neutral">
                <IconLanguage size={12} /> Merged
              </Badge>
              <span>{relativeTime(m.merged_at)}</span>
              <span className="text-fg-faint">· {m.by}</span>
              <div className="flex-1" />
              <Button size="xs" variant="ghost" leftIcon={<IconArrowBackUp size={13} />} loading={undoMerge.isPending && undoMerge.variables === m.id} onClick={() => undo(m.id)}>
                Undo
              </Button>
            </div>
            <p className="selectable mt-2.5 text-sm font-medium leading-relaxed text-fg">{m.text}</p>
            <p className="selectable mt-1.5 text-xs leading-relaxed text-fg-muted">
              {m.merged_into ? (
                <>
                  <span className="text-fg-subtle">Merged into: </span>
                  {m.merged_into.text}
                </>
              ) : (
                'The memory it was merged into has since been deleted.'
              )}
            </p>
          </li>
        ))}
      </ul>
    </section>
  )
}

/** Where in an export it was: "In “10K training plan” · 16 Sept 2026 · chatgpt-export.zip". */
function ImportPlace({ item }: { item: MemoryReviewItem }) {
  const day = item.dated ? new Date(`${item.dated}T12:00:00`) : null
  const parts = [
    item.title ? `In “${item.title}”` : null,
    day && !Number.isNaN(day.getTime()) ? day.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : null,
    item.export
  ].filter(Boolean)
  return (
    <p className="mt-1.5 truncate pl-3.5 text-2xs text-fg-subtle" title={parts.join(' · ')}>
      {parts.join(' · ')}
    </p>
  )
}

function ReviewCard({ item }: { item: MemoryReviewItem }) {
  const navigate = useNavigate()
  const { approve, discard } = useMemoryReviewActions()
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState(item.text)
  const expires = parseDate(item.expires_at)
  const daysLeft = expires ? Math.max(0, Math.ceil((expires.getTime() - Date.now()) / 86_400_000)) : null

  const doApprove = (content?: string) =>
    approve.mutate(
      { kind: item.kind, id: item.id, content },
      {
        onSuccess: () => toast.success(item.kind === 'fact' ? 'Sentient will remember this' : 'Added to what Sentient knows about you'),
        onError: (e) => toast.error('Couldn’t approve it', { description: errorMessage(e) })
      }
    )
  const doDiscard = () =>
    discard.mutate(
      { kind: item.kind, id: item.id },
      { onSuccess: () => toast.success('Not saved'), onError: (e) => toast.error('Couldn’t remove it', { description: errorMessage(e) }) }
    )

  return (
    <li className="rounded-xl border border-border bg-surface p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs text-fg-subtle">
        <Badge size="xs" tone="neutral">
          {item.kind === 'fact' ? 'Memory' : 'About you'}
        </Badge>
        {item.reason === 'instruction' && (
          <Badge size="xs" tone="accent">
            Instruction
          </Badge>
        )}
        <span>{relativeTime(item.created_at)}</span>
        {daysLeft !== null && (
          <span className="text-fg-faint">· {daysLeft > 1 ? `let go in ${daysLeft} days` : 'let go soon'} if not reviewed</span>
        )}
        <div className="flex-1" />
        {item.session_id && (
          <Button size="xs" variant="ghost" rightIcon={<IconArrowUpRight size={12} />} onClick={() => navigate(`/chat/${item.session_id}`)}>
            Open chat
          </Button>
        )}
      </div>

      {editing ? (
        <div className="mt-3 space-y-2">
          <Textarea autoGrow value={text} onChange={(e) => setText(e.target.value)} aria-label="Memory text" />
          <div className="flex justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => (setEditing(false), setText(item.text))}>
              Cancel
            </Button>
            <Button size="sm" variant="primary" leftIcon={<IconCheck size={14} />} disabled={!text.trim()} loading={approve.isPending} onClick={() => doApprove(text.trim())}>
              Save and approve
            </Button>
          </div>
        </div>
      ) : (
        <p className="selectable mt-2.5 text-sm font-medium leading-relaxed text-fg">{item.text}</p>
      )}

      {item.snippet && (
        <p className="selectable mt-2.5 border-l-2 border-border-strong pl-3 text-xs leading-relaxed text-fg-muted">
          <span className="text-fg-subtle">{item.reason === 'instruction' ? 'You said: ' : 'It came from: '}</span>
          {item.snippet}
        </p>
      )}
      {(item.title || item.export) && <ImportPlace item={item} />}

      {!editing && (
        <div className="mt-3 flex flex-wrap gap-2">
          <Button size="sm" variant="primary" leftIcon={<IconCheck size={14} />} loading={approve.isPending} onClick={() => doApprove()}>
            Approve
          </Button>
          <Button size="sm" variant="secondary" leftIcon={<IconPencil size={14} />} onClick={() => setEditing(true)}>
            Edit
          </Button>
          <Button size="sm" variant="ghost" leftIcon={<IconX size={14} />} loading={discard.isPending} onClick={doDiscard}>
            Discard
          </Button>
        </div>
      )}
    </li>
  )
}
