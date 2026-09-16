import { IconChevronRight } from '@tabler/icons-react'
import { useState, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

export interface JsonViewProps {
  value: unknown
  /** Nodes deeper than this start collapsed. */
  collapsedDepth?: number
  className?: string
  maxStringLength?: number
}

/** Collapsible JSON tree for tool arguments/results and the raw config viewer. */
export function JsonView({ value, collapsedDepth = 2, className, maxStringLength = 400 }: JsonViewProps) {
  return (
    <div className={cn('selectable overflow-x-auto font-mono text-[12px] leading-[1.6]', className)}>
      <Node value={value} depth={0} collapsedDepth={collapsedDepth} maxStringLength={maxStringLength} />
    </div>
  )
}

function Primitive({ value, maxStringLength }: { value: unknown; maxStringLength: number }) {
  const [expanded, setExpanded] = useState(false)
  if (value === null) return <span className="text-fg-subtle">null</span>
  if (value === undefined) return <span className="text-fg-subtle">undefined</span>
  if (typeof value === 'boolean') return <span className="text-info">{String(value)}</span>
  if (typeof value === 'number') return <span className="text-warning">{value}</span>
  const s = String(value)
  const long = s.length > maxStringLength
  return (
    <span className="whitespace-pre-wrap break-words text-success">
      "{long && !expanded ? `${s.slice(0, maxStringLength)}…` : s}"
      {long && (
        <button type="button" onClick={() => setExpanded((e) => !e)} className="ml-1 text-2xs text-fg-subtle underline">
          {expanded ? 'less' : `+${s.length - maxStringLength}`}
        </button>
      )}
    </span>
  )
}

function Node({
  name,
  value,
  depth,
  collapsedDepth,
  maxStringLength,
  last = true
}: {
  name?: string
  value: unknown
  depth: number
  collapsedDepth: number
  maxStringLength: number
  last?: boolean
}) {
  const isArray = Array.isArray(value)
  const isObject = value !== null && typeof value === 'object'
  const [open, setOpen] = useState(depth < collapsedDepth)
  const label: ReactNode = name !== undefined && <span className="text-fg-muted">{JSON.stringify(name)}: </span>
  const comma = last ? '' : ','

  if (!isObject) {
    return (
      <div style={{ paddingLeft: depth ? 14 : 0 }}>
        {label}
        <Primitive value={value} maxStringLength={maxStringLength} />
        <span className="text-fg-faint">{comma}</span>
      </div>
    )
  }

  const entries = isArray ? (value as unknown[]).map((v, i) => [String(i), v] as const) : Object.entries(value as object)
  const [openBr, closeBr] = isArray ? ['[', ']'] : ['{', '}']

  if (!entries.length) {
    return (
      <div style={{ paddingLeft: depth ? 14 : 0 }}>
        {label}
        <span className="text-fg-subtle">
          {openBr}
          {closeBr}
        </span>
        {comma}
      </div>
    )
  }

  return (
    <div style={{ paddingLeft: depth ? 14 : 0 }}>
      <button type="button" onClick={() => setOpen((o) => !o)} className="-ml-3.5 inline-flex items-center text-left hover:text-fg">
        <IconChevronRight size={11} className={cn('mr-0.5 text-fg-subtle transition-transform', open && 'rotate-90')} />
        {label}
        <span className="text-fg-subtle">{openBr}</span>
        {!open && (
          <span className="text-fg-subtle">
            {' '}
            {entries.length} {isArray ? (entries.length === 1 ? 'item' : 'items') : entries.length === 1 ? 'key' : 'keys'} {closeBr}
            {comma}
          </span>
        )}
      </button>
      {open && (
        <>
          {entries.map(([k, v], i) => (
            <Node
              key={k}
              name={isArray ? undefined : k}
              value={v}
              depth={depth + 1}
              collapsedDepth={collapsedDepth}
              maxStringLength={maxStringLength}
              last={i === entries.length - 1}
            />
          ))}
          <div className="text-fg-subtle">
            {closeBr}
            {comma}
          </div>
        </>
      )}
    </div>
  )
}
