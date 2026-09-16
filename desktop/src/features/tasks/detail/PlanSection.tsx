/** Plan steps with tool icons; editable (drag reorder, add, remove) while the plan awaits approval. */
import { IconGripVertical, IconListNumbers, IconPencil, IconPlus, IconX } from '@tabler/icons-react'
import { Reorder, useDragControls } from 'motion/react'
import { useState } from 'react'
import { Button, IconButton, Input, Select } from '@/components/ui'
import type { PlanStep } from '@/lib/types'
import { cn, uid } from '@/lib/utils'
import { SectionHeading } from '../parts'
import { ToolTile, pluginIdFor, toolIdentity, useToolNames } from '../tools'

interface DraftStep extends PlanStep {
  id: string
}

export function PlanSection({
  plan,
  editable,
  saving,
  onSave,
  title = 'Plan',
  emptyText = 'No plan yet.'
}: {
  plan: PlanStep[]
  editable: boolean
  saving?: boolean
  onSave: (plan: PlanStep[]) => Promise<unknown> | void
  title?: string
  emptyText?: string
}) {
  const [draft, setDraft] = useState<DraftStep[] | null>(null)
  const { names, options } = useToolNames()

  const start = () => setDraft(plan.map((s) => ({ ...s, id: uid() })))
  const save = async () => {
    if (!draft) return
    const clean = draft.filter((s) => s.description.trim() || s.tool).map(({ tool, description }) => ({ tool, description: description.trim() }))
    await onSave(clean)
    setDraft(null)
  }

  return (
    <section className="space-y-2.5">
      <SectionHeading
        icon={<IconListNumbers />}
        title={title}
        count={plan.length || undefined}
        description={editable && !draft ? 'Review the steps, then approve' : undefined}
        actions={
          editable &&
          (draft ? (
            <div className="flex items-center gap-1.5">
              <Button size="xs" variant="ghost" onClick={() => setDraft(null)} disabled={saving}>
                Cancel
              </Button>
              <Button size="xs" variant="primary" onClick={() => void save()} loading={saving}>
                Save plan
              </Button>
            </div>
          ) : (
            <Button size="xs" variant="ghost" leftIcon={<IconPencil size={13} />} onClick={start} disabled={!plan.length && !editable}>
              Edit steps
            </Button>
          ))
        }
      />

      {draft ? (
        <div className="space-y-2">
          <Reorder.Group axis="y" values={draft} onReorder={setDraft} className="space-y-2">
            {draft.map((step, i) => (
              <EditableStep
                key={step.id}
                step={step}
                index={i}
                options={options}
                onChange={(patch) => setDraft((d) => d?.map((s) => (s.id === step.id ? { ...s, ...patch } : s)) ?? d)}
                onRemove={() => setDraft((d) => d?.filter((s) => s.id !== step.id) ?? d)}
              />
            ))}
          </Reorder.Group>
          <button
            type="button"
            onClick={() => setDraft((d) => [...(d ?? []), { id: uid(), tool: '', description: '' }])}
            className="flex h-9 w-full items-center justify-center gap-1.5 rounded-xl border border-dashed border-border-strong text-sm text-fg-muted transition-colors hover:border-accent/50 hover:text-accent-text"
          >
            <IconPlus size={15} /> Add step
          </button>
        </div>
      ) : plan.length ? (
        <ol className="relative space-y-0">
          {plan.map((step, i) => (
            <li key={i} className="relative flex gap-3 pb-3 last:pb-0">
              {i < plan.length - 1 && <span aria-hidden className="absolute bottom-0 left-[13.5px] top-8 w-px bg-border" />}
              <ToolTile name={step.tool} />
              <div className="min-w-0 flex-1 rounded-xl border border-border bg-surface px-3 py-2">
                <div className="flex items-center gap-2 text-2xs font-medium uppercase tracking-wide text-fg-subtle">
                  <span className="tabular-nums">Step {i + 1}</span>
                  <span className="text-fg-faint">·</span>
                  <span className="normal-case tracking-normal">{toolIdentity(step.tool, names).label}</span>
                </div>
                <p className="mt-0.5 text-sm leading-relaxed text-fg">{step.description || <span className="text-fg-subtle">No description</span>}</p>
              </div>
            </li>
          ))}
        </ol>
      ) : (
        <p className="rounded-xl border border-dashed border-border px-4 py-5 text-center text-sm text-fg-subtle">{emptyText}</p>
      )}
    </section>
  )
}

function EditableStep({
  step,
  index,
  options,
  onChange,
  onRemove
}: {
  step: DraftStep
  index: number
  options: Array<{ value: string; label: string }>
  onChange: (patch: Partial<PlanStep>) => void
  onRemove: () => void
}) {
  const controls = useDragControls()
  const [dragging, setDragging] = useState(false)
  const toolId = pluginIdFor(step.tool)
  const toolOptions = options.some((o) => o.value === toolId) || !toolId ? options : [{ value: toolId, label: toolIdentity(toolId).label }, ...options]
  return (
    <Reorder.Item
      value={step}
      dragListener={false}
      dragControls={controls}
      onDragStart={() => setDragging(true)}
      onDragEnd={() => setDragging(false)}
      className={cn('relative flex items-center gap-2 rounded-xl border bg-surface p-2', dragging ? 'z-10 border-accent/50 shadow-pop' : 'border-border')}
    >
      <button
        type="button"
        aria-label={`Drag step ${index + 1}`}
        onPointerDown={(e) => controls.start(e)}
        className="flex h-8 w-5 shrink-0 cursor-grab touch-none items-center justify-center rounded text-fg-faint hover:text-fg-muted active:cursor-grabbing"
      >
        <IconGripVertical size={15} />
      </button>
      <span className="w-4 shrink-0 text-center text-xs tabular-nums text-fg-subtle">{index + 1}</span>
      <Select
        size="sm"
        className="w-44 shrink-0"
        aria-label="Tool"
        placeholder="Choose a tool"
        value={toolId || undefined}
        onValueChange={(v) => onChange({ tool: v })}
        options={toolOptions.map((o) => ({ ...o, icon: <ToolTile name={o.value} size="sm" className="border-0 bg-transparent" /> }))}
      />
      <Input size="sm" value={step.description} onChange={(e) => onChange({ description: e.target.value })} placeholder="What should this step do?" aria-label="Step description" />
      <IconButton size="sm" label="Remove step" icon={<IconX size={14} />} onClick={onRemove} className="hover:text-danger" />
    </Reorder.Item>
  )
}
