/** Search and filters (status, type, source). */
import { IconSearch, IconX } from '@tabler/icons-react'
import { useEffect, useRef, useState } from 'react'
import { Button, Input, Kbd, Select } from '@/components/ui'
import { isEditableTarget } from '@/lib/utils'
import { STATUS_FILTERS, type TaskFilterState } from './hooks'
import { KIND_META, SOURCE_META, type TaskKind, type TaskSource } from './meta'

const KIND_OPTIONS = [
  { value: 'all', label: 'Any type' },
  ...(Object.keys(KIND_META) as TaskKind[]).map((k) => {
    const I = KIND_META[k].icon
    return { value: k, label: KIND_META[k].label, icon: <I size={14} /> }
  })
]

const SOURCE_OPTIONS = [
  { value: 'all', label: 'Any source' },
  ...(Object.keys(SOURCE_META) as TaskSource[]).map((s) => {
    const I = SOURCE_META[s].icon
    return { value: s, label: s === 'you' ? 'Created by you' : s === 'chat' ? 'From chat' : 'Proactive', icon: <I size={14} /> }
  })
]

export function TaskToolbar({
  filters,
  onChange,
  active,
  resultCount
}: {
  filters: TaskFilterState
  onChange: (patch: Partial<Record<keyof TaskFilterState, string>>) => void
  active: boolean
  resultCount: number
}) {
  const [q, setQ] = useState(filters.q)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => setQ(filters.q), [filters.q])

  useEffect(() => {
    const t = window.setTimeout(() => {
      if (q !== filters.q) onChange({ q })
    }, 180)
    return () => window.clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === '/' && !isEditableTarget(e.target) && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        input.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Input
        ref={input}
        size="sm"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => e.key === 'Escape' && (setQ(''), input.current?.blur())}
        placeholder="Search tasks"
        aria-label="Search tasks"
        leftIcon={<IconSearch />}
        wrapperClassName="w-full max-w-64 min-w-40 flex-1"
        rightSlot={q ? <button type="button" aria-label="Clear search" onClick={() => setQ('')} className="flex size-5 items-center justify-center rounded text-fg-subtle hover:text-fg"><IconX size={12} /></button> : <Kbd className="mr-0.5">/</Kbd>}
      />
      <Select size="sm" aria-label="Filter by status" value={filters.status} onValueChange={(v) => onChange({ status: v })} options={STATUS_FILTERS} className="w-40" />
      <Select size="sm" aria-label="Filter by type" value={filters.kind} onValueChange={(v) => onChange({ kind: v })} options={KIND_OPTIONS} className="w-34" />
      <Select size="sm" aria-label="Filter by source" value={filters.source} onValueChange={(v) => onChange({ source: v })} options={SOURCE_OPTIONS} className="w-38" />
      {active && (
        <>
          <Button size="sm" variant="ghost" leftIcon={<IconX size={14} />} onClick={() => (setQ(''), onChange({ q: '', status: 'all', kind: 'all', source: 'all' }))}>
            Clear
          </Button>
          <span className="text-xs text-fg-subtle">
            {resultCount} result{resultCount === 1 ? '' : 's'}
          </span>
        </>
      )}
    </div>
  )
}
