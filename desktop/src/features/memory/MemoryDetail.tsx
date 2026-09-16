import { IconCalendar, IconCheck, IconClock, IconHistory, IconHourglassHigh, IconPencil, IconTrash, IconX } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { Alert, Button, ConfirmDialog, Sheet, Textarea } from '@/components/ui'
import { useHotkey } from '@/hooks/useHotkey'
import { formatDateTime } from '@/lib/utils'
import { ExpiryBadge, SourceChip, TopicPill } from './bits'
import { useNow, useOptimisticMemoryActions } from './hooks'
import { countdown, expiresInMs, recentRelative, sourceMeta, type MemoryWithHistory } from './meta'

export function MemoryDetail({
  memory,
  open,
  onOpenChange,
  onForgetSource
}: {
  memory: MemoryWithHistory | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onForgetSource: (source: string) => void
}) {
  const { update, remove } = useOptimisticMemoryActions()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [confirm, setConfirm] = useState(false)
  const now = useNow()

  useEffect(() => {
    setEditing(false)
    setDraft(memory?.content ?? '')
  }, [memory?.id, memory?.content])

  const dirty = !!memory && draft.trim() !== memory.content && draft.trim().length > 0
  const save = () => {
    if (!memory || !dirty) return
    update.mutate({ id: memory.id, content: draft.trim() })
    setEditing(false)
  }
  useHotkey('mod+enter', save, { enabled: editing && dirty })

  const ms = memory ? expiresInMs(memory, now) : null
  const src = memory ? sourceMeta(memory.source) : null

  return (
    <Sheet
      open={open && !!memory}
      onOpenChange={onOpenChange}
      width={440}
      title="Memory"
      description={memory ? `#${memory.id} · ${memory.memory_type}` : undefined}
      actions={
        memory &&
        !editing && (
          <>
            <Button size="sm" variant="ghost" leftIcon={<IconPencil size={14} />} onClick={() => setEditing(true)}>
              Edit
            </Button>
            <Button size="sm" variant="ghost" className="text-danger hover:text-danger" leftIcon={<IconTrash size={14} />} onClick={() => setConfirm(true)}>
              Forget
            </Button>
          </>
        )
      }
    >
      {memory && src && (
        <div className="space-y-6 p-5">
          {editing ? (
            <div className="space-y-2">
              <Textarea autoFocus autoGrow minHeight={110} value={draft} onChange={(e) => setDraft(e.target.value)} className="text-md" />
              <p className="text-xs text-fg-subtle">Saving re-analyzes topics and whether this is short-term. Ctrl+Enter to save.</p>
              <div className="flex justify-end gap-2">
                <Button size="sm" variant="ghost" leftIcon={<IconX size={14} />} onClick={() => (setEditing(false), setDraft(memory.content))}>
                  Cancel
                </Button>
                <Button size="sm" variant="primary" leftIcon={<IconCheck size={14} />} disabled={!dirty} onClick={save}>
                  Save
                </Button>
              </div>
            </div>
          ) : (
            <button type="button" onClick={() => setEditing(true)} className="selectable -m-2 block w-[calc(100%+16px)] rounded-lg p-2 text-left text-lg leading-relaxed text-fg transition-colors hover:bg-hover" title="Click to edit">
              {memory.content}
            </button>
          )}

          <div className="flex flex-wrap gap-1.5 pt-2">
            {memory.topics.map((t) => (
              <TopicPill key={t} topic={t} />
            ))}
            <ExpiryBadge memory={memory} now={now} />
          </div>

          {memory.previous_content && memory.previous_content !== memory.content && (
            <div className="rounded-xl border border-border bg-sunken/60 p-3.5">
              <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-fg-muted">
                <IconHistory size={13} /> Previously
              </div>
              <p className="text-sm text-fg-subtle line-through decoration-fg-faint">{memory.previous_content}</p>
            </div>
          )}

          <dl className="divide-y divide-border rounded-xl border border-border bg-surface text-sm">
            <Row icon={<src.icon size={14} />} label="Source">
              <div className="flex items-center justify-between gap-2">
                <SourceChip source={memory.source} className="text-sm text-fg" />
                <button type="button" onClick={() => onForgetSource(memory.source)} className="text-xs text-fg-subtle underline-offset-2 hover:text-danger hover:underline">
                  Forget all from this source
                </button>
              </div>
              <div className="text-xs text-fg-subtle">{src.description}</div>
            </Row>
            <Row icon={<IconCalendar size={14} />} label="Learned">
              {formatDateTime(memory.created_at)} {recentRelative(memory.created_at) && <span className="text-fg-subtle">· {recentRelative(memory.created_at)}</span>}
            </Row>
            {memory.updated_at > memory.created_at && (
              <Row icon={<IconClock size={14} />} label="Updated">
                {formatDateTime(memory.updated_at)} {recentRelative(memory.updated_at) && <span className="text-fg-subtle">· {recentRelative(memory.updated_at)}</span>}
              </Row>
            )}
            <Row icon={<IconHourglassHigh size={14} />} label="Kept">
              {memory.memory_type === 'short-term' && ms !== null ? (
                <>
                  Until {formatDateTime(memory.expires_at)} <span className="text-fg-subtle">· {countdown(ms)} left</span>
                </>
              ) : (
                <>Long-term, until you edit or forget it</>
              )}
            </Row>
          </dl>

          {memory.memory_type === 'short-term' && (
            <Alert tone="info" icon={<IconHourglassHigh />}>
              Short-term memories cover plans and temporary situations. Sentient forgets them automatically when they expire.
            </Alert>
          )}
        </div>
      )}
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title="Forget this memory?"
        description="Sentient will no longer recall it in conversations or tasks. This can't be undone."
        confirmLabel="Forget"
        onConfirm={() => {
          if (!memory) return
          remove.mutate(memory.id)
          onOpenChange(false)
        }}
      />
    </Sheet>
  )
}

function Row({ icon, label, children }: { icon: React.ReactNode; label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 px-3.5 py-3">
      <dt className="flex w-20 shrink-0 items-center gap-1.5 self-start pt-px text-xs text-fg-subtle">
        {icon}
        {label}
      </dt>
      <dd className="min-w-0 flex-1 text-fg">{children}</dd>
    </div>
  )
}
