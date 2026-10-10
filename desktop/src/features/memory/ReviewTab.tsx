import { IconArrowUpRight, IconCheck, IconChecks, IconInbox, IconPencil, IconShieldCheck, IconX } from '@tabler/icons-react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Alert, Badge, Button, EmptyState, Skeleton, Textarea } from '@/components/ui'
import { useMemoryReview, useMemoryReviewActions } from '@/hooks/memory'
import { errorMessage } from '@/lib/api'
import type { MemoryReviewItem } from '@/lib/types'
import { parseDate, relativeTime } from '@/lib/utils'

/** Memories Sentient holds until the user approves them (ADR 0021): from emails, web pages, imports, unprompted work. */
export function ReviewTab() {
  const review = useMemoryReview()
  const actions = useMemoryReviewActions()

  const groups = useMemo(() => {
    const by = new Map<string, MemoryReviewItem[]>()
    for (const item of review.data?.items ?? []) by.set(item.from, [...(by.get(item.from) ?? []), item])
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
  if (!review.data?.items.length) {
    return (
      <EmptyState
        icon={<IconInbox />}
        title="Nothing to review"
        description="When Sentient learns something from an email, a web page, an import or work it did on its own, it waits here for your OK before using it."
      />
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
        They came from an email, a web page, an import or work Sentient did on its own, so it waits for your OK before using them. Anything you
        don’t review is let go after {review.data.expire_days} days.
      </Alert>
      {groups.map(([from, items]) => (
        <section key={from} className="space-y-2.5">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-medium text-fg">From {from}</h3>
            <span className="rounded-full bg-active px-1.5 text-2xs tabular-nums text-fg-muted">{items.length}</span>
            <div className="flex-1" />
            {items.length > 1 && (
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
    </div>
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
          <span className="text-fg-subtle">It came from: </span>
          {item.snippet}
        </p>
      )}

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
