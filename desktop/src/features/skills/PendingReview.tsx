import { IconArrowUpRight, IconCheck, IconCode, IconEyeCheck, IconFileDescription, IconFilePencil, IconFirstAidKit, IconPencil, IconSparkles, IconX } from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Alert, Button, ConfirmDialog, EmptyState, SegmentedControl, Skeleton } from '@/components/ui'
import { qk } from '@/hooks/queryKeys'
import { useEvolutionLog, useSkillActions, useSkillDiff } from '@/hooks/skills'
import { api, errorMessage } from '@/lib/api'
import type { EvolutionLogEntry, Skill, SkillsList } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'
import { ReviewNowButton } from './actions'
import { DiffView } from './DiffView'
import { AuthorBadge } from './SkillCard'
import { SkillBody } from './SkillDetail'
import { SkillEditor, splitList, type SkillDraft } from './SkillEditor'
import { isRepairProposal, splitFrontmatter } from './meta'

function proposalFor(log: EvolutionLogEntry[] | undefined, name: string) {
  return log?.find(
    (e) => e.detail?.name === name && (e.kind === 'skill_repair_proposed' || ((e.kind === 'skill_created' || e.kind === 'skill_patched') && e.detail?.pending === true))
  )
}

type ProposalKind = 'new' | 'update' | 'repair'

const KIND_FILTERS: Array<{ id: 'all' | ProposalKind; label: string }> = [
  { id: 'all', label: 'Everything' },
  { id: 'repair', label: 'Fixes after a problem' },
  { id: 'update', label: 'Improvements' },
  { id: 'new', label: 'New skills' }
]

export function PendingReview({ list, focus, onOpenSkill }: { list: SkillsList; focus: string | null; onOpenSkill: (name: string) => void }) {
  const log = useEvolutionLog(300)
  const [kindFilter, setKindFilter] = useState<'all' | ProposalKind>('all')
  if (!list.pending.length) {
    return (
      <EmptyState
        icon={<IconEyeCheck />}
        title="Nothing waiting for review"
        description="When Sentient notices a procedure worth repeating, or a better way to run an existing skill, it drafts it here. Nothing changes until you approve."
        action={<ReviewNowButton />}
      />
    )
  }
  const classify = (p: Skill): ProposalKind =>
    isRepairProposal(p) || proposalFor(log.data, p.name)?.kind === 'skill_repair_proposed' ? 'repair' : list.active.some((a) => a.name === p.name) ? 'update' : 'new'
  const kinds = new Map(list.pending.map((p) => [p.name, classify(p)]))
  const count = (k: 'all' | ProposalKind) => (k === 'all' ? list.pending.length : [...kinds.values()].filter((x) => x === k).length)
  const visible = [...list.pending]
    .filter((p) => kindFilter === 'all' || kinds.get(p.name) === kindFilter)
    .sort((a, b) => Number(b.name === focus) - Number(a.name === focus) || Number(kinds.get(b.name) === 'repair') - Number(kinds.get(a.name) === 'repair'))

  return (
    <div className="space-y-5">
      {list.pending.length > 1 && (
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter proposals">
          {KIND_FILTERS.filter((f) => f.id === 'all' || count(f.id) > 0).map((f) => {
            const active = kindFilter === f.id
            return (
              <button
                key={f.id}
                type="button"
                aria-pressed={active}
                onClick={() => setKindFilter(active && f.id !== 'all' ? 'all' : f.id)}
                className={cn(
                  'inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-xs font-medium transition-colors',
                  active
                    ? f.id === 'repair'
                      ? 'border-warning/40 bg-warning/12 text-fg'
                      : 'border-border-strong bg-active text-fg'
                    : 'border-border bg-surface text-fg-muted hover:border-border-strong hover:text-fg'
                )}
              >
                {f.id === 'repair' && <IconFirstAidKit size={13} className="text-warning" />}
                {f.label}
                <span className={active ? '' : 'text-fg-subtle'}>{count(f.id)}</span>
              </button>
            )
          })}
        </div>
      )}
      <AnimatePresence initial={false}>
        {visible.map((p) => (
          <ReviewCard
            key={p.name}
            pending={p}
            isUpdate={list.active.some((a) => a.name === p.name)}
            repair={kinds.get(p.name) === 'repair'}
            entry={proposalFor(log.data, p.name)}
            focused={focus === p.name}
            onOpenSkill={onOpenSkill}
          />
        ))}
      </AnimatePresence>
    </div>
  )
}

function ReviewCard({
  pending,
  isUpdate,
  repair,
  entry,
  focused,
  onOpenSkill
}: {
  pending: Skill
  isUpdate: boolean
  repair: boolean
  entry: EvolutionLogEntry | undefined
  focused: boolean
  onOpenSkill: (name: string) => void
}) {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const ref = useRef<HTMLDivElement>(null)
  const diff = useSkillDiff(pending.name)
  const actions = useSkillActions()
  const [mode, setMode] = useState<'rendered' | 'raw'>('rendered')
  const [draft, setDraft] = useState<SkillDraft | null>(null)
  const [busy, setBusy] = useState<'approve' | 'reject' | null>(null)
  const [confirmReject, setConfirmReject] = useState(false)

  useEffect(() => {
    if (focused) ref.current?.scrollIntoView({ block: 'start', behavior: window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'instant' })
  }, [focused])

  const proposedBody = useMemo(() => (diff.data ? splitFrontmatter(diff.data.proposed).body : ''), [diff.data])
  const reason = pending.reason || (typeof entry?.detail?.reason === 'string' ? entry.detail.reason : '')
  const origin = pending.origin && typeof pending.origin === 'object' ? pending.origin : null
  const sessionId = typeof entry?.detail?.session_id === 'string' ? entry.detail.session_id : (origin?.session_id ?? null)
  const taskId = typeof entry?.detail?.task_id === 'string' ? entry.detail.task_id : (origin?.task_id ?? null)

  const dropFromPending = () => {
    const prev = qc.getQueryData<SkillsList>(qk.skills.all)
    if (prev) qc.setQueryData<SkillsList>(qk.skills.all, { ...prev, pending: prev.pending.filter((s) => s.name !== pending.name) })
    return prev
  }

  const approve = async (edited?: SkillDraft) => {
    setBusy('approve')
    const prev = dropFromPending()
    try {
      const body = edited
        ? { description: edited.description, body: edited.body, tags: splitList(edited.tags), requires_tools: splitList(edited.requires_tools) }
        : null
      // edit the proposal itself (never the active copy), then approve it once
      if (body) await api.skills.update(pending.name, { ...body, target: 'pending' })
      await actions.approve.mutateAsync(pending.name)
      void qc.invalidateQueries({ queryKey: qk.skills.all })
      toast.success(isUpdate ? `Updated ${pending.name}` : `${pending.name} is now active`, {
        description: 'Sentient will use it the next time it applies.',
        action: { label: 'View', onClick: () => onOpenSkill(pending.name) }
      })
    } catch (e) {
      if (prev) qc.setQueryData(qk.skills.all, prev)
      toast.error("Couldn't approve", { description: errorMessage(e) })
      setBusy(null)
    }
  }

  const reject = async () => {
    setBusy('reject')
    const prev = dropFromPending()
    try {
      await actions.reject.mutateAsync(pending.name)
      toast.success(isUpdate ? `Kept the current ${pending.name}` : `Discarded ${pending.name}`)
    } catch (e) {
      if (prev) qc.setQueryData(qk.skills.all, prev)
      toast.error("Couldn't reject", { description: errorMessage(e) })
      setBusy(null)
    }
  }

  return (
    <motion.div
      ref={ref}
      layout="position"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0, marginTop: 0 }}
      className={cn('scroll-mt-4 overflow-hidden rounded-2xl border bg-surface shadow-soft', focused ? 'border-accent/45 ring-3 ring-accent/10' : repair ? 'border-warning/30' : 'border-border')}
    >
      {/* header */}
      <div className="relative overflow-hidden border-b border-border px-5 py-4">
        <div aria-hidden className="pointer-events-none absolute -right-20 -top-24 size-64 rounded-full opacity-[0.10] blur-3xl" style={{ background: `radial-gradient(circle, var(${repair ? '--warning' : isUpdate ? '--info' : '--accent'}), transparent 70%)` }} />
        <div className="relative flex items-start gap-3.5">
          <div
            className={cn(
              'flex size-10 shrink-0 items-center justify-center rounded-xl border',
              repair ? 'border-warning/30 bg-warning/10 text-warning' : isUpdate ? 'border-info/25 bg-info/10 text-info' : 'border-accent/25 bg-accent/10 text-accent-text'
            )}
          >
            {repair ? <IconFirstAidKit size={20} /> : isUpdate ? <IconFilePencil size={20} /> : <IconSparkles size={20} />}
          </div>
          <div className="min-w-0 flex-1">
            <div className={cn('text-xs font-medium uppercase tracking-wide', repair ? 'text-warning' : 'text-fg-subtle')}>
              {repair ? 'Fix proposed after a problem' : isUpdate ? 'Proposed improvement' : 'Sentient learned a new skill'}
              {entry && <span className="normal-case tracking-normal"> · {relativeTime(entry.ts)}</span>}
            </div>
            <h3 className="mt-0.5 flex flex-wrap items-center gap-2 text-md font-semibold text-fg">
              <span className="font-mono">{pending.name}</span>
              {isUpdate && (
                <button type="button" onClick={() => onOpenSkill(pending.name)} className="text-xs font-normal text-fg-subtle hover:text-accent-text">
                  current version →
                </button>
              )}
            </h3>
            <p className="mt-1 text-sm text-fg-muted">{pending.description}</p>
          </div>
          <AuthorBadge author={pending.author} />
        </div>
      </div>

      <div className="space-y-4 p-5">
        {/* why */}
        <div className={cn('rounded-xl border px-4 py-3', repair ? 'border-warning/25 bg-warning/[0.06]' : 'border-border bg-sunken/50')}>
          <div className={cn('text-xs font-medium', repair ? 'text-warning' : 'text-fg-muted')}>{repair ? 'What went wrong' : 'Why Sentient proposed this'}</div>
          <p className="mt-1 text-sm leading-relaxed text-fg">
            {reason || (repair ? 'Something went wrong the last time Sentient used this skill, so it drafted a fix.' : isUpdate ? 'Sentient found a better way to run this skill while working for you.' : 'Sentient noticed a multi-step procedure in your recent work that is likely to come up again.')}
          </p>
          {(sessionId || taskId) && (
            <button
              type="button"
              onClick={() => navigate(sessionId ? `/chat/${sessionId}` : `/tasks?task=${encodeURIComponent(taskId as string)}`)}
              className="mt-1.5 inline-flex items-center gap-0.5 text-xs text-fg-subtle hover:text-accent-text"
            >
              {sessionId ? 'Open the chat it came from' : 'Open the task it came from'} <IconArrowUpRight size={12} />
            </button>
          )}
        </div>

        {/* content */}
        {draft ? (
          <SkillEditor draft={draft} onChange={setDraft} minHeight={360} />
        ) : diff.isLoading ? (
          <Skeleton className="h-64 rounded-xl" />
        ) : diff.isError ? (
          <Alert tone="danger" title="Couldn't load the proposal">
            {errorMessage(diff.error)}
          </Alert>
        ) : diff.data && isUpdate ? (
          <DiffView current={diff.data.current} proposed={diff.data.proposed} />
        ) : diff.data ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-fg-muted">Proposed SKILL.md</span>
              <div className="flex-1" />
              <SegmentedControl
                size="sm"
                aria-label="Proposal view"
                value={mode}
                onChange={setMode}
                options={[
                  { value: 'rendered', label: 'Readable', icon: <IconFileDescription size={12} /> },
                  { value: 'raw', label: 'Raw', icon: <IconCode size={12} /> }
                ]}
              />
            </div>
            {mode === 'rendered' ? (
              <SkillBody body={proposedBody} />
            ) : (
              <pre className="selectable overflow-x-auto rounded-xl border border-border bg-sunken px-4 py-3 font-mono text-[12px] leading-relaxed text-fg">{diff.data.proposed}</pre>
            )}
          </div>
        ) : null}
      </div>

      {/* actions */}
      <div className="flex flex-wrap items-center gap-2 border-t border-border bg-elevated/40 px-5 py-3">
        <p className="mr-auto text-xs text-fg-subtle">{repair ? 'Approving updates the skill so the same problem doesn’t happen again.' : isUpdate ? 'Approving replaces the current version.' : 'Nothing is used until you approve.'}</p>
        {draft ? (
          <>
            <Button variant="ghost" leftIcon={<IconX size={15} />} onClick={() => setDraft(null)}>
              Cancel edit
            </Button>
            <Button variant="primary" leftIcon={<IconCheck size={15} />} loading={busy === 'approve'} disabled={!draft.body.trim()} onClick={() => void approve(draft)}>
              Save and approve
            </Button>
          </>
        ) : (
          <>
            <Button variant="ghost" className="text-danger hover:text-danger" leftIcon={<IconX size={15} />} disabled={!!busy} loading={busy === 'reject'} onClick={() => setConfirmReject(true)}>
              Reject
            </Button>
            <Button
              variant="secondary"
              leftIcon={<IconPencil size={15} />}
              disabled={!diff.data || !!busy}
              onClick={() => setDraft({ name: pending.name, description: pending.description, body: proposedBody, tags: pending.tags.join(', '), requires_tools: pending.requires_tools.join(', ') })}
            >
              Edit
            </Button>
            <Button variant="primary" leftIcon={<IconCheck size={15} />} disabled={!!busy} loading={busy === 'approve'} onClick={() => void approve()}>
              {repair ? 'Approve fix' : isUpdate ? 'Approve update' : 'Approve skill'}
            </Button>
          </>
        )}
      </div>
      <ConfirmDialog
        open={confirmReject}
        onOpenChange={setConfirmReject}
        title={isUpdate ? `Keep the current ${pending.name}?` : `Discard ${pending.name}?`}
        description={isUpdate ? 'The proposed change is discarded and the current version stays active.' : 'Sentient will not use this skill. It may propose something similar again later.'}
        confirmLabel={isUpdate ? 'Reject update' : 'Discard skill'}
        onConfirm={reject}
      />
    </motion.div>
  )
}
