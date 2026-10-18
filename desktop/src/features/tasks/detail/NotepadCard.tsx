/** Recurring tasks (docs/API.md §4 "Recurring tasks remember"): the notes its runs keep for the next run. */
import { IconNotes, IconPencil, IconTrash } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { Button, Textarea } from '@/components/ui'
import type { Task } from '@/lib/types'

const MAX_CHARS = 1500

export function NotepadCard({ task, onSave }: { task: Task; onSave: (notepad: string | null) => Promise<unknown> }) {
  const notepad = task.notepad ?? ''
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(notepad)
  const [saving, setSaving] = useState(false)
  useEffect(() => {
    if (!editing) setDraft(notepad)
  }, [notepad, editing])

  const save = async (value: string | null) => {
    setSaving(true)
    try {
      await onSave(value)
      setEditing(false)
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="group space-y-2">
      <div className="flex items-center gap-2">
        <IconNotes size={16} className="text-fg-subtle" />
        <h3 className="text-sm font-semibold text-fg">Notes between runs</h3>
        <span className="flex-1" />
        {editing ? (
          <div className="flex items-center gap-1.5">
            <Button size="xs" variant="ghost" onClick={() => (setDraft(notepad), setEditing(false))} disabled={saving}>
              Cancel
            </Button>
            <Button size="xs" variant="primary" loading={saving} onClick={() => void save(draft.trim() || null)}>
              Save
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
            {notepad && (
              <Button size="xs" variant="ghost" leftIcon={<IconTrash size={13} />} loading={saving} onClick={() => void save(null)}>
                Clear
              </Button>
            )}
            <Button size="xs" variant="ghost" leftIcon={<IconPencil size={13} />} onClick={() => setEditing(true)}>
              Edit
            </Button>
          </div>
        )}
      </div>
      {editing ? (
        <Textarea autoFocus autoGrow minHeight={80} maxLength={MAX_CHARS} value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => e.key === 'Escape' && setEditing(false)} />
      ) : (
        <div className="rounded-xl border border-border bg-sunken/40 px-4 py-3">
          {notepad ? (
            <p className="selectable whitespace-pre-wrap text-sm leading-relaxed text-fg-muted">{notepad}</p>
          ) : (
            <p className="text-sm text-fg-subtle">Nothing yet. Each run leaves a short note here for the next one.</p>
          )}
          <p className="mt-2 text-2xs text-fg-subtle">Every run reads these notes and what the last few runs found, so it tells you only what’s new.</p>
        </div>
      )}
    </section>
  )
}
