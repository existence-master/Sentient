import { useMemo, useState } from 'react'
import { SegmentedControl } from '@/components/ui'
import { cn } from '@/lib/utils'
import { diffLines, diffStats, inlineChange, type DiffRow } from './meta'

/** Side-by-side (or unified) line diff of two SKILL.md files. Unchanged runs are folded. */
export function DiffView({ current, proposed, className }: { current: string; proposed: string; className?: string }) {
  const rows = useMemo(() => diffLines(current, proposed), [current, proposed])
  const stats = useMemo(() => diffStats(rows), [rows])
  const [mode, setMode] = useState<'split' | 'unified'>('split')
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  // fold runs of >6 unchanged lines, keeping 2 lines of context on each side
  const blocks = useMemo(() => {
    const out: Array<{ kind: 'rows'; rows: DiffRow[] } | { kind: 'fold'; id: number; rows: DiffRow[] }> = []
    let run: DiffRow[] = []
    const flush = (atEnd: boolean) => {
      if (run.length > 6) {
        const head = out.length ? run.slice(0, 2) : []
        const tail = atEnd ? [] : run.slice(-2)
        if (head.length) out.push({ kind: 'rows', rows: head })
        out.push({ kind: 'fold', id: run[0].type === 'same' ? run[0].left : 0, rows: run.slice(head.length, run.length - tail.length) })
        if (tail.length) out.push({ kind: 'rows', rows: tail })
      } else if (run.length) out.push({ kind: 'rows', rows: run })
      run = []
    }
    for (const r of rows) {
      if (r.type === 'same') run.push(r)
      else {
        flush(false)
        out.push({ kind: 'rows', rows: [r] })
      }
    }
    flush(true)
    return out
  }, [rows])

  return (
    <div className={cn('overflow-hidden rounded-xl border border-border bg-sunken/50', className)}>
      <div className="flex items-center gap-3 border-b border-border px-3 py-2">
        <span className="text-xs font-medium text-fg-muted">Changes</span>
        <span className="font-mono text-xs text-success">+{stats.added}</span>
        <span className="font-mono text-xs text-danger">−{stats.removed}</span>
        <div className="flex-1" />
        <SegmentedControl
          size="sm"
          aria-label="Diff layout"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'split', label: 'Side by side' },
            { value: 'unified', label: 'Unified' }
          ]}
        />
      </div>
      {mode === 'split' && (
        <div className="grid grid-cols-2 border-b border-border text-2xs font-medium uppercase tracking-wide text-fg-subtle">
          <div className="px-3 py-1.5">Current</div>
          <div className="border-l border-border px-3 py-1.5">Proposed</div>
        </div>
      )}
      <div className="selectable overflow-x-auto font-mono text-[12px] leading-[1.65]">
        {blocks.map((b, bi) =>
          b.kind === 'fold' && !expanded.has(bi) ? (
            <button
              key={bi}
              type="button"
              onClick={() => setExpanded((s) => new Set(s).add(bi))}
              className="block w-full bg-hover px-3 py-1 text-left font-sans text-xs text-fg-subtle hover:bg-active hover:text-fg"
            >
              ⋯ {b.rows.length} unchanged lines
            </button>
          ) : (
            b.rows.map((r, i) => (mode === 'split' ? <SplitRow key={`${bi}-${i}`} row={r} /> : <UnifiedRow key={`${bi}-${i}`} row={r} />))
          )
        )}
      </div>
    </div>
  )
}

const num = 'w-9 shrink-0 select-none pr-2 text-right text-fg-faint'
const del = 'bg-danger/10'
const add = 'bg-success/10'

function Cell({ n, text, tone, children }: { n?: number; text?: string; tone?: string; children?: React.ReactNode }) {
  return (
    <div className={cn('flex min-w-0', tone)}>
      <span className={num}>{n ?? ''}</span>
      <span className="min-w-0 flex-1 whitespace-pre-wrap break-words pr-3 text-fg">{children ?? (text || ' ')}</span>
    </div>
  )
}

function SplitRow({ row }: { row: DiffRow }) {
  if (row.type === 'same')
    return (
      <div className="grid grid-cols-2">
        <Cell n={row.left} text={row.text} />
        <div className="border-l border-border">
          <Cell n={row.right} text={row.text} />
        </div>
      </div>
    )
  if (row.type === 'change') {
    const c = inlineChange(row.from, row.to)
    return (
      <div className="grid grid-cols-2">
        <Cell n={row.left} tone={del}>
          {c.pre}
          <mark className="rounded-sm bg-danger/25 text-fg">{c.a}</mark>
          {c.post}
        </Cell>
        <div className="border-l border-border">
          <Cell n={row.right} tone={add}>
            {c.pre}
            <mark className="rounded-sm bg-success/25 text-fg">{c.b}</mark>
            {c.post}
          </Cell>
        </div>
      </div>
    )
  }
  return (
    <div className="grid grid-cols-2">
      {row.type === 'del' ? <Cell n={row.left} text={row.text} tone={del} /> : <div className="bg-hover/60" />}
      <div className="border-l border-border">{row.type === 'add' ? <Cell n={row.right} text={row.text} tone={add} /> : <div className="h-full bg-hover/60" />}</div>
    </div>
  )
}

function UnifiedRow({ row }: { row: DiffRow }) {
  const line = (sign: string, text: string, tone?: string, n?: number) => (
    <div className={cn('flex', tone)}>
      <span className={num}>{n ?? ''}</span>
      <span className={cn('w-4 shrink-0 select-none', sign === '+' ? 'text-success' : sign === '−' ? 'text-danger' : 'text-fg-faint')}>{sign}</span>
      <span className="flex-1 whitespace-pre-wrap break-words pr-3 text-fg">{text || ' '}</span>
    </div>
  )
  if (row.type === 'same') return line(' ', row.text, undefined, row.right)
  if (row.type === 'del') return line('−', row.text, del, row.left)
  if (row.type === 'add') return line('+', row.text, add, row.right)
  return (
    <>
      {line('−', row.from, del, row.left)}
      {line('+', row.to, add, row.right)}
    </>
  )
}
