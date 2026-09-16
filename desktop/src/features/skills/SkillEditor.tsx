import { IconColumns2, IconEye, IconPencil } from '@tabler/icons-react'
import { useState } from 'react'
import { Field, Input, Markdown, SegmentedControl, Textarea } from '@/components/ui'
import { cn } from '@/lib/utils'

export interface SkillDraft {
  name: string
  description: string
  body: string
  tags: string
  requires_tools: string
}

export const splitList = (s: string) =>
  s
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean)

/** Metadata fields + split markdown editor with live preview. */
export function SkillEditor({
  draft,
  onChange,
  nameEditable,
  nameError,
  minHeight = 380
}: {
  draft: SkillDraft
  onChange: (d: SkillDraft) => void
  nameEditable?: boolean
  nameError?: string
  minHeight?: number
}) {
  const [layout, setLayout] = useState<'split' | 'edit' | 'preview'>('split')
  const set = (patch: Partial<SkillDraft>) => onChange({ ...draft, ...patch })

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-2">
        {nameEditable && (
          <Field label="Name" htmlFor="skill-name" error={nameError} description="Lowercase letters, digits and dashes.">
            <Input id="skill-name" className="font-mono" placeholder="weekly-review" value={draft.name} onChange={(e) => set({ name: e.target.value })} />
          </Field>
        )}
        <Field label="Description" htmlFor="skill-desc" description="One sentence. Sentient reads this to decide when the skill applies." className={nameEditable ? '' : 'md:col-span-2'}>
          <Input id="skill-desc" placeholder="What this skill does and when" value={draft.description} onChange={(e) => set({ description: e.target.value })} />
        </Field>
        <Field label="Tags" htmlFor="skill-tags" optional description="Comma separated.">
          <Input id="skill-tags" placeholder="productivity, email" value={draft.tags} onChange={(e) => set({ tags: e.target.value })} />
        </Field>
        <Field label="Requires tools" htmlFor="skill-tools" optional description="Integration ids, e.g. gmail, gcalendar. Hidden when missing.">
          <Input id="skill-tools" className="font-mono" placeholder="gmail" value={draft.requires_tools} onChange={(e) => set({ requires_tools: e.target.value })} />
        </Field>
      </div>

      <div className="overflow-hidden rounded-xl border border-border bg-surface">
        <div className="flex h-10 items-center gap-2 border-b border-border px-3">
          <span className="font-mono text-xs text-fg-muted">SKILL.md</span>
          <div className="flex-1" />
          <SegmentedControl
            size="sm"
            aria-label="Editor layout"
            value={layout}
            onChange={setLayout}
            options={[
              { value: 'edit', label: 'Write', icon: <IconPencil size={12} /> },
              { value: 'split', label: 'Split', icon: <IconColumns2 size={12} /> },
              { value: 'preview', label: 'Preview', icon: <IconEye size={12} /> }
            ]}
          />
        </div>
        <div className={cn('grid', layout === 'split' ? 'grid-cols-2' : 'grid-cols-1')} style={{ minHeight }}>
          {layout !== 'preview' && (
            <Textarea
              value={draft.body}
              spellCheck={false}
              onChange={(e) => set({ body: e.target.value })}
              style={{ minHeight }}
              className="h-full rounded-none border-0 bg-transparent px-4 py-3.5 font-mono text-[12.5px] leading-relaxed shadow-none hover:border-0 focus:border-0 focus:ring-0"
            />
          )}
          {layout !== 'edit' && (
            <div className={cn('overflow-y-auto bg-sunken/30 px-5 py-4', layout === 'split' && 'border-l border-border')} style={{ maxHeight: Math.max(minHeight, 560) }}>
              {draft.body.trim() ? <Markdown className="!text-sm">{draft.body}</Markdown> : <p className="text-sm text-fg-subtle">Nothing to preview yet.</p>}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
