import {
  IconAlertTriangle,
  IconArchive,
  IconBulb,
  IconChartBar,
  IconCheck,
  IconChecklist,
  IconClock,
  IconEye,
  IconFilePencil,
  IconListNumbers,
  IconPencil,
  IconRestore,
  IconTool,
  IconTrash,
  IconX,
  IconZzz
} from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, ConfirmDialog, Markdown, Sheet, Skeleton } from '@/components/ui'
import { qk } from '@/hooks/queryKeys'
import { useSkill, useSkillActions } from '@/hooks/skills'
import { errorMessage } from '@/lib/api'
import type { SkillDetail as SkillDetailType, SkillsList } from '@/lib/types'
import { formatDateTime, relativeTime } from '@/lib/utils'
import { AuthorBadge, SkillStat } from './SkillCard'
import { SkillEditor, splitList, type SkillDraft } from './SkillEditor'
import { parseSections } from './meta'

const SECTION_ICON: Record<string, ReactNode> = {
  'when to use': <IconBulb size={15} />,
  procedure: <IconListNumbers size={15} />,
  pitfalls: <IconAlertTriangle size={15} />,
  verification: <IconChecklist size={15} />
}

export function toDraft(s: Pick<SkillDetailType, 'name' | 'description' | 'body' | 'tags' | 'requires_tools'>): SkillDraft {
  return { name: s.name, description: s.description, body: s.body, tags: s.tags.join(', '), requires_tools: s.requires_tools.join(', ') }
}

/** Rendered SKILL.md split into its standard sections. */
export function SkillBody({ body }: { body: string }) {
  const parsed = useMemo(() => parseSections(body), [body])
  if (!parsed.sections.length) return <Markdown className="!text-sm">{body}</Markdown>
  return (
    <div className="space-y-3">
      {parsed.title && <h2 className="text-lg font-semibold tracking-tight text-fg">{parsed.title}</h2>}
      {parsed.intro && <Markdown className="!text-sm">{parsed.intro}</Markdown>}
      {parsed.sections.map((s) => (
        <section key={s.title} className="rounded-xl border border-border bg-surface">
          <h3 className="flex items-center gap-2 border-b border-border px-4 py-2.5 text-sm font-semibold text-fg">
            <span className="text-accent-text">{SECTION_ICON[s.title.toLowerCase()] ?? <IconFilePencil size={15} />}</span>
            {s.title}
          </h3>
          <div className="px-4 py-3">
            <Markdown className="!text-sm [&_li]:my-1">{s.content || '_Empty_'}</Markdown>
          </div>
        </section>
      ))}
    </div>
  )
}

export function SkillDetailSheet({
  name,
  list,
  editing,
  onEditingChange,
  onOpenChange,
  onReview
}: {
  name: string | null
  list: SkillsList | undefined
  editing: boolean
  onEditingChange: (editing: boolean) => void
  onOpenChange: (open: boolean) => void
  onReview: (name: string) => void
}) {
  const qc = useQueryClient()
  const skill = useSkill(name ?? undefined)
  const actions = useSkillActions()
  const [draft, setDraft] = useState<SkillDraft | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const inActive = !!list?.active.some((s) => s.name === name)
  const inArchived = !!list?.archived.some((s) => s.name === name)
  const hasPendingUpdate = inActive && !!list?.pending.some((s) => s.name === name)

  useEffect(() => {
    if (editing && skill.data) setDraft(toDraft(skill.data))
    if (!editing) setDraft(null)
  }, [editing, skill.data])

  const dirty = !!draft && !!skill.data && JSON.stringify(draft) !== JSON.stringify(toDraft(skill.data))

  /** Optimistically move a skill between lists. */
  const move = (to: 'active' | 'archived' | null) => {
    const prev = qc.getQueryData<SkillsList>(qk.skills.all)
    if (!prev || !name) return prev
    const item = [...prev.active, ...prev.archived].find((s) => s.name === name)
    const next: SkillsList = {
      active: prev.active.filter((s) => s.name !== name),
      archived: prev.archived.filter((s) => s.name !== name),
      pending: prev.pending
    }
    if (item && to) next[to] = [...next[to], { ...item, state: to }].sort((a, b) => a.name.localeCompare(b.name))
    qc.setQueryData(qk.skills.all, next)
    return prev
  }

  const run = async (label: string, to: 'active' | 'archived' | null, fn: () => Promise<unknown>, done: string) => {
    const prev = move(to)
    try {
      await fn()
      toast.success(done)
    } catch (e) {
      if (prev) qc.setQueryData(qk.skills.all, prev)
      toast.error(`Couldn't ${label}`, { description: errorMessage(e) })
    }
  }

  const save = () => {
    if (!draft || !name) return
    actions.update.mutate(
      { name, body: { description: draft.description, body: draft.body, tags: splitList(draft.tags), requires_tools: splitList(draft.requires_tools) } },
      {
        onSuccess: () => {
          toast.success(`Saved ${name}`, { description: inActive ? 'Version bumped. Sentient uses the new procedure from now on.' : undefined })
          onEditingChange(false)
        },
        onError: (e) => toast.error("Couldn't save skill", { description: errorMessage(e) })
      }
    )
  }

  const s = skill.data
  return (
    <Sheet
      open={!!name}
      onOpenChange={onOpenChange}
      width={editing ? 980 : 720}
      title={<span className="font-mono">{name}</span>}
      description={s ? `Version ${s.version} · ${inArchived && !inActive ? 'archived' : s.state === 'stale' ? 'stale' : 'active'}` : undefined}
      actions={
        s &&
        (editing ? (
          <>
            <Button size="sm" variant="ghost" leftIcon={<IconX size={14} />} onClick={() => onEditingChange(false)}>
              Cancel
            </Button>
            <Button size="sm" variant="primary" leftIcon={<IconCheck size={14} />} disabled={!dirty} loading={actions.update.isPending} onClick={save}>
              Save
            </Button>
          </>
        ) : (
          <>
            <Button size="sm" variant="ghost" leftIcon={<IconPencil size={14} />} onClick={() => onEditingChange(true)}>
              Edit
            </Button>
            {inActive ? (
              <Button size="sm" variant="ghost" leftIcon={<IconArchive size={14} />} onClick={() => void run('archive', 'archived', () => actions.archive.mutateAsync(s.name), `Archived ${s.name}`)}>
                Archive
              </Button>
            ) : inArchived ? (
              <Button size="sm" variant="ghost" leftIcon={<IconRestore size={14} />} onClick={() => void run('restore', 'active', () => actions.restore.mutateAsync(s.name), `Restored ${s.name}`)}>
                Restore
              </Button>
            ) : null}
            <Button size="sm" variant="ghost" className="text-danger hover:text-danger" leftIcon={<IconTrash size={14} />} onClick={() => setConfirmDelete(true)}>
              Delete
            </Button>
          </>
        ))
      }
    >
      {skill.isLoading ? (
        <div className="space-y-3 p-5">
          <Skeleton className="h-6 w-2/3" />
          <Skeleton className="h-24 rounded-xl" />
          <Skeleton className="h-48 rounded-xl" />
        </div>
      ) : skill.isError ? (
        <div className="p-5">
          <Alert tone="danger" title="Couldn't load this skill">
            {errorMessage(skill.error)}
          </Alert>
        </div>
      ) : s ? (
        <div className="space-y-5 p-5">
          {editing && draft ? (
            <SkillEditor draft={draft} onChange={setDraft} minHeight={420} />
          ) : (
            <>
              <div className="space-y-3">
                <p className="text-md leading-relaxed text-fg">{s.description}</p>
                <div className="flex flex-wrap items-center gap-2">
                  <AuthorBadge author={s.author} />
                  {s.state === 'stale' && (
                    <Badge tone="neutral" icon={<IconZzz />}>
                      Stale
                    </Badge>
                  )}
                  {inArchived && !inActive && <Badge tone="neutral" icon={<IconArchive />}>Archived</Badge>}
                  {s.created_by_review && <Badge tone="accent">Learned from your work</Badge>}
                  {s.tags.map((t) => (
                    <span key={t} className="rounded-md bg-active px-1.5 py-0.5 text-xs text-fg-muted">
                      #{t}
                    </span>
                  ))}
                </div>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 rounded-xl border border-border bg-sunken/40 px-3.5 py-2.5 text-sm text-fg-muted">
                  <SkillStat icon={<IconChartBar size={14} />} value={<>{s.use_count} uses</>} label="Times used" />
                  <SkillStat icon={<IconEye size={14} />} value={<>{s.view_count} views</>} label="Times read by Sentient" />
                  <SkillStat icon={<IconFilePencil size={14} />} value={<>{s.patch_count} edits</>} label="Times improved" />
                  <span title={formatDateTime(s.last_used_at)} className="inline-flex items-center gap-1">
                    <IconClock size={14} /> {s.last_used_at ? `last used ${relativeTime(s.last_used_at)}` : 'never used'}
                  </span>
                  {s.requires_tools.length > 0 && (
                    <span className="inline-flex items-center gap-1">
                      <IconTool size={14} /> needs <span className="font-mono text-xs">{s.requires_tools.join(', ')}</span>
                    </span>
                  )}
                </div>
              </div>

              {hasPendingUpdate && (
                <Alert
                  tone="info"
                  icon={<IconFilePencil />}
                  title="Sentient proposed an improvement to this skill"
                  action={
                    <Button size="sm" variant="secondary" onClick={() => onReview(s.name)}>
                      Review
                    </Button>
                  }
                >
                  Compare the change side by side before it replaces this version.
                </Alert>
              )}
              {s.state === 'stale' && (
                <Alert tone="neutral" icon={<IconZzz />} title="Not used recently">
                  The curator marks skills stale when they go unused and archives them if they stay that way. Using it again makes it active.
                </Alert>
              )}

              <SkillBody body={s.body} />
            </>
          )}
        </div>
      ) : null}
      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title={`Delete ${name}?`}
        description="The skill folder is removed, including any pending update or archived copy. Archive instead if you might want it back."
        confirmLabel="Delete skill"
        onConfirm={async () => {
          if (!name) return
          await run('delete', null, () => actions.remove.mutateAsync(name), `Deleted ${name}`)
          onOpenChange(false)
        }}
      />
    </Sheet>
  )
}
