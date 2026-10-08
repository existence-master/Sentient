/** One thing Sentient believes about you: statement, gentle certainty, evidence and corrections. */
import { IconArrowBackUp, IconCheck, IconDots, IconPencil, IconQuote, IconThumbDown, IconTrash, IconX } from '@tabler/icons-react'
import { motion } from 'motion/react'
import { useState } from 'react'
import {
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  IconButton,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Textarea,
  Tooltip
} from '@/components/ui'
import type { Insight } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'
import { certainty, dimensionMeta, evidenceMeta, tint } from './meta'

export interface InsightActions {
  onConfirm: (i: Insight) => void
  onRetire: (i: Insight) => void
  onRestore: (i: Insight) => void
  onEdit: (i: Insight, statement: string) => Promise<unknown> | void
  onDelete: (i: Insight) => void
}

export function CertaintyMeter({ level, color, className }: { level: number; color: string; className?: string }) {
  return (
    <span aria-hidden className={cn('inline-flex items-end gap-[3px]', className)}>
      {[1, 2, 3, 4].map((n) => (
        <span
          key={n}
          className="w-[3px] rounded-full transition-colors"
          style={{ height: 4 + n * 2, background: n <= level ? color : 'var(--active)' }}
        />
      ))}
    </span>
  )
}

export function InsightCard({ insight, actions, retired = false }: { insight: Insight; actions: InsightActions; retired?: boolean }) {
  const meta = dimensionMeta(insight.dimension)
  const c = certainty(insight)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(insight.statement)
  const [saving, setSaving] = useState(false)
  const disputed = insight.status === 'disputed'
  const canConfirm = insight.source !== 'user' && insight.status !== 'confirmed'

  const save = async () => {
    const s = draft.trim()
    if (!s || s === insight.statement) return setEditing(false)
    setSaving(true)
    try {
      await actions.onEdit(insight, s)
      setEditing(false)
    } finally {
      setSaving(false)
    }
  }

  return (
    <motion.li
      layout="position"
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98 }}
      className={cn(
        'group relative list-none rounded-xl border bg-surface px-4 py-3.5 transition-colors',
        disputed ? 'border-warning/25 bg-warning/[0.03]' : 'border-border hover:border-border-strong',
        retired && 'bg-transparent opacity-80'
      )}
    >
      {editing ? (
        <div className="space-y-2">
          <Textarea
            autoFocus
            autoGrow
            minHeight={64}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void save()
              if (e.key === 'Escape') setEditing(false)
            }}
            aria-label="Edit what Sentient believes"
          />
          <div className="flex items-center justify-between gap-2">
            <span className="text-2xs text-fg-subtle">Say it the way you would. I will treat your words as confirmed.</span>
            <div className="flex gap-1.5">
              <Button size="xs" variant="ghost" onClick={() => (setDraft(insight.statement), setEditing(false))}>
                Cancel
              </Button>
              <Button size="xs" variant="primary" loading={saving} disabled={!draft.trim()} onClick={() => void save()}>
                Save
              </Button>
            </div>
          </div>
        </div>
      ) : (
        <>
          <p className={cn('pr-16 text-sm leading-relaxed', retired ? 'text-fg-subtle line-through decoration-fg-faint' : 'text-fg')}>{insight.statement}</p>
          <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs">
            <Tooltip content={c.hint}>
              <span className={cn('inline-flex cursor-default items-center gap-1.5', disputed ? 'text-warning' : 'text-fg-muted')}>
                <CertaintyMeter level={c.level} color={disputed ? 'var(--warning)' : meta.color} />
                {c.word}
              </span>
            </Tooltip>
            <EvidencePopover insight={insight} color={meta.color} />
            <span className="text-fg-faint">{relativeTime(insight.updated_at)}</span>
          </div>
        </>
      )}

      {!editing && (
        <div className="absolute right-2 top-2 flex items-center gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
          {retired ? (
            <Tooltip content="Actually, this is right">
              <Button size="xs" variant="ghost" leftIcon={<IconArrowBackUp size={13} />} onClick={() => actions.onRestore(insight)}>
                Restore
              </Button>
            </Tooltip>
          ) : (
            <>
              {canConfirm && <IconButton size="xs" label="That’s right" icon={<IconCheck size={14} />} onClick={() => actions.onConfirm(insight)} />}
              <IconButton size="xs" label="This is wrong" icon={<IconThumbDown size={14} />} onClick={() => actions.onRetire(insight)} />
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <IconButton size="xs" label="More" tooltip={false} icon={<IconDots size={14} />} />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-48">
                  <DropdownMenuItem icon={<IconPencil />} onSelect={() => (setDraft(insight.statement), setEditing(true))}>
                    Edit wording
                  </DropdownMenuItem>
                  <DropdownMenuItem icon={<IconTrash />} danger onSelect={() => actions.onDelete(insight)}>
                    Forget this completely
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </>
          )}
        </div>
      )}
      {insight.status === 'confirmed' && !retired && (
        <span className="pointer-events-none absolute right-3 top-3.5 flex size-4 items-center justify-center rounded-full transition-opacity group-hover:opacity-0" style={{ background: tint(meta.color, 18), color: meta.color }}>
          <IconCheck size={11} stroke={2.5} />
        </span>
      )}
    </motion.li>
  )
}

function EvidencePopover({ insight, color }: { insight: Insight; color: string }) {
  const ev = insight.evidence ?? []
  const label = ev.length ? 'Why I think this' : insight.source === 'user' ? 'You added this' : 'Why I think this'
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button type="button" className="inline-flex items-center gap-1 rounded-md px-1 py-0.5 text-fg-subtle transition-colors hover:bg-hover hover:text-fg">
          <IconQuote size={12} />
          {label}
          {ev.length > 0 && <span className="tabular-nums text-fg-faint">{ev.length}</span>}
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[380px] p-0">
        <div className="border-b border-border px-4 py-3">
          <div className="text-sm font-semibold text-fg">Why I think this</div>
          <div className="mt-0.5 text-xs text-fg-subtle">{ev.length ? 'The moments this came from. Only you and Sentient can see them.' : 'Nothing to show here.'}</div>
        </div>
        {ev.length ? (
          <ol className="max-h-80 space-y-3 overflow-y-auto px-4 py-3">
            {ev.map((e, idx) => {
              const m = evidenceMeta(e.kind)
              return (
                <li key={`${e.ref ?? idx}`} className="flex gap-2.5">
                  <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md" style={{ background: tint(color, 14), color }}>
                    <m.icon size={13} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="selectable border-l-2 pl-2.5 text-sm leading-relaxed text-fg" style={{ borderColor: tint(color, 45) }}>
                      {e.quote}
                    </p>
                    <div className="mt-1 text-2xs text-fg-subtle">
                      {m.label}
                      {e.at && ` · ${relativeTime(e.at)}`}
                    </div>
                  </div>
                </li>
              )
            })}
          </ol>
        ) : (
          <p className="px-4 py-3 text-sm text-fg-muted">{insight.source === 'user' ? 'You told me this yourself, so I treat it as true.' : 'I no longer have the moments this came from.'}</p>
        )}
      </PopoverContent>
    </Popover>
  )
}

export function RetireUndo({ onUndo }: { onUndo: () => void }) {
  return (
    <button type="button" onClick={onUndo} className="inline-flex items-center gap-1 text-xs font-medium">
      <IconX size={12} /> Undo
    </button>
  )
}
