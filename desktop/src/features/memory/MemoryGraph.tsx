/**
 * Interactive 2D memory graph (lazy-loaded: `React.lazy(() => import('./MemoryGraph'))`).
 * Nodes are colored by primary topic and gently clustered by topic; links are similarity edges.
 */
import { IconFocusCentered, IconMinus, IconPlus } from '@tabler/icons-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D, { type ForceGraphMethods, type LinkObject, type NodeObject } from 'react-force-graph-2d'
import { IconButton } from '@/components/ui'
import { resolveTheme } from '@/lib/theme'
import type { MemoryGraph as GraphData, MemoryGraphNode } from '@/lib/types'
import { truncate } from '@/lib/utils'
import { useUI } from '@/stores/ui'
import { linkEnd } from './hooks'
import { primaryTopic, TOPIC_META, TOPIC_ORDER, topicMeta } from './meta'

type GNode = NodeObject<MemoryGraphNode & { degree: number; topic: string; color: string }>
type GLink = LinkObject<GNode, { value: number }>

const EMPTY_GRAPH: { nodes: GNode[]; links: GLink[] } = { nodes: [], links: [] }

export interface MemoryGraphProps {
  data: GraphData
  /** Node ids matching the current filters/search; `null` = everything matches. */
  matches: Set<number> | null
  selectedId: number | null
  onSelect: (id: number | null) => void
}

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(() => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false)
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (!mq) return
    const on = () => setReduced(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return reduced
}

export default function MemoryGraph({ data, matches, selectedId, onSelect }: MemoryGraphProps) {
  const container = useRef<HTMLDivElement>(null)
  const fg = useRef<ForceGraphMethods<GNode, GLink> | undefined>(undefined)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [hover, setHover] = useState<GNode | null>(null)
  const reduced = usePrefersReducedMotion()
  const dark = resolveTheme(useUI((s) => s.theme)) === 'dark'
  const fitted = useRef(false)
  const settled = useRef(false)
  /** Zoom level right after fitting; labels for every node appear only well beyond it. */
  const baseScale = useRef<number | null>(null)
  const dur = reduced ? 0 : 500

  useEffect(() => {
    const el = container.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ width: el.clientWidth, height: el.clientHeight }))
    ro.observe(el)
    setSize({ width: el.clientWidth, height: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  // Keep node objects stable across refetches so positions don't jump.
  const cache = useRef(new Map<number, GNode>())
  const graph = useMemo(() => {
    const degree = new Map<number, number>()
    for (const l of data.links) {
      degree.set(linkEnd(l.source), (degree.get(linkEnd(l.source)) ?? 0) + 1)
      degree.set(linkEnd(l.target), (degree.get(linkEnd(l.target)) ?? 0) + 1)
    }
    const next = new Map<number, GNode>()
    const nodes = data.nodes.map((n) => {
      const topic = primaryTopic(n.topics)
      const prev = cache.current.get(n.id)
      const node = Object.assign(prev ?? {}, n, { degree: degree.get(n.id) ?? 0, topic, color: topicMeta(topic).color }) as GNode
      next.set(n.id, node)
      return node
    })
    cache.current = next
    const links: GLink[] = data.links
      .filter((l) => next.has(linkEnd(l.source)) && next.has(linkEnd(l.target)))
      .map((l) => ({ source: linkEnd(l.source), target: linkEnd(l.target), value: l.value }))
    return { nodes, links }
  }, [data])

  const neighbors = useMemo(() => {
    const map = new Map<number, Set<number>>()
    for (const l of graph.links) {
      const a = linkEnd(l.source)
      const b = linkEnd(l.target)
      if (!map.has(a)) map.set(a, new Set())
      if (!map.has(b)) map.set(b, new Set())
      map.get(a)!.add(b)
      map.get(b)!.add(a)
    }
    return map
  }, [graph.links])

  // Forces: gentle topic clustering around a ring + softer charge. Configured as soon as the
  // canvas exists and before any data is passed, so warmup ticks already use them.
  const [forcesReady, setForcesReady] = useState(false)
  const mounted = size.width > 0
  useEffect(() => {
    const g = fg.current
    if (!mounted || !g) return
    let nodes: GNode[] = []
    const ring = 72
    const anchor = (topic: string) => {
      const i = Math.max(0, TOPIC_ORDER.indexOf(topic))
      const a = (i / TOPIC_ORDER.length) * Math.PI * 2 - Math.PI / 2
      return { x: Math.cos(a) * ring, y: Math.sin(a) * ring }
    }
    const cluster = Object.assign(
      (alpha: number) => {
        for (const n of nodes) {
          const t = anchor(n.topic)
          n.vx = (n.vx ?? 0) + (t.x - (n.x ?? 0)) * 0.11 * alpha
          n.vy = (n.vy ?? 0) + (t.y - (n.y ?? 0)) * 0.11 * alpha
        }
      },
      { initialize: (ns: GNode[]) => void (nodes = ns) }
    )
    g.d3Force('cluster', cluster)
    const charge = g.d3Force('charge')
    charge?.strength?.(-14)
    const link = g.d3Force('link')
    link?.distance?.((l: GLink) => 24 + (1 - l.value) * 40)
    setForcesReady(true)
  }, [mounted])

  // Fit once the first layout exists (warmup ticks run synchronously when data arrives).
  useEffect(() => {
    if (!forcesReady || fitted.current || !graph.nodes.length) return
    const t = window.setTimeout(() => {
      fg.current?.zoomToFit(0, 56)
      baseScale.current = fg.current?.zoom() ?? null
      fitted.current = true
      setLayoutReady(true)
    }, 40)
    return () => window.clearTimeout(t)
  }, [forcesReady, graph.nodes.length])

  // Focus the selected node (also when it was preselected before the first layout existed).
  const [layoutReady, setLayoutReady] = useState(false)
  useEffect(() => {
    if (selectedId === null || !layoutReady) return
    const node = cache.current.get(selectedId)
    const g = fg.current
    if (!node || !g || node.x === undefined) return
    g.centerAt(node.x, node.y, dur)
    const target = (baseScale.current ?? g.zoom()) * 1.8
    if (g.zoom() < target) g.zoom(target, dur)
  }, [selectedId, dur, layoutReady])

  const focusSet = useMemo(() => {
    const id = hover?.id ?? selectedId
    if (id === null || id === undefined) return null
    const s = new Set<number>([Number(id)])
    neighbors.get(Number(id))?.forEach((n) => s.add(n))
    return s
  }, [hover, selectedId, neighbors])

  const drawNode = useCallback(
    (node: GNode, ctx: CanvasRenderingContext2D, scale: number) => {
      const id = Number(node.id)
      const x = node.x ?? 0
      const y = node.y ?? 0
      // sizes are in screen pixels so the graph reads the same at every zoom level
      const px = 1 / scale
      const r = (4 + Math.sqrt(node.degree) * 1.3) * px
      const matched = matches === null || matches.has(id)
      const focused = focusSet === null || focusSet.has(id)
      const isSelected = id === selectedId
      const isHover = hover?.id === node.id
      const alpha = !matched ? 0.12 : focused ? 1 : 0.28

      ctx.globalAlpha = alpha
      if (isSelected || isHover) {
        ctx.beginPath()
        ctx.arc(x, y, r + 6 * px, 0, Math.PI * 2)
        ctx.fillStyle = `${node.color}33`
        ctx.fill()
      }
      ctx.beginPath()
      ctx.arc(x, y, r, 0, Math.PI * 2)
      ctx.fillStyle = node.color
      ctx.fill()
      ctx.lineWidth = 1.2 / scale
      ctx.strokeStyle = dark ? 'rgba(11,11,15,0.9)' : 'rgba(255,255,255,0.95)'
      ctx.stroke()
      if (node.memory_type === 'short-term') {
        ctx.beginPath()
        ctx.setLineDash([2 * px, 2 * px])
        ctx.arc(x, y, r + 2.6 * px, 0, Math.PI * 2)
        ctx.strokeStyle = node.color
        ctx.lineWidth = 1.1 / scale
        ctx.stroke()
        ctx.setLineDash([])
      }
      if (isSelected) {
        ctx.beginPath()
        ctx.arc(x, y, r + 3 * px, 0, Math.PI * 2)
        ctx.strokeStyle = dark ? '#ffffff' : '#17171c'
        ctx.lineWidth = 1.6 / scale
        ctx.stroke()
      }

      const zoomedIn = baseScale.current !== null && scale > baseScale.current * 2.4
      const showLabel = matched && (isHover || isSelected || (focusSet !== null && focusSet.has(id)) || zoomedIn)
      if (showLabel) {
        const strong = isHover || isSelected
        const fontSize = (strong ? 12 : 10.5) / scale
        ctx.font = `${strong ? 600 : 500} ${fontSize}px Inter Variable, Inter, system-ui, sans-serif`
        const text = truncate(node.content, strong ? 64 : 34)
        const tw = ctx.measureText(text).width
        const padX = 5 / scale
        const h = fontSize + 6 / scale
        const lx = x - tw / 2 - padX
        const ly = y + r + 5 / scale
        ctx.globalAlpha = Math.min(1, alpha + 0.1)
        ctx.fillStyle = dark ? 'rgba(23,23,31,0.92)' : 'rgba(255,255,255,0.95)'
        ctx.strokeStyle = dark ? 'rgba(255,255,255,0.12)' : 'rgba(15,15,25,0.12)'
        ctx.lineWidth = 1 / scale
        ctx.beginPath()
        ctx.roundRect(lx, ly, tw + padX * 2, h, 4 / scale)
        ctx.fill()
        ctx.stroke()
        ctx.fillStyle = dark ? '#ececf0' : '#17171c'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(text, x, ly + h / 2)
      }
      ctx.globalAlpha = 1
    },
    [matches, focusSet, selectedId, hover, dark]
  )

  /** Faint topic names above each cluster. */
  const drawClusterLabels = useCallback(
    (ctx: CanvasRenderingContext2D, scale: number) => {
      const acc = new Map<string, { x: number; minY: number; n: number; matched: number }>()
      for (const n of graph.nodes) {
        if (n.x === undefined || n.y === undefined) continue
        const a = acc.get(n.topic) ?? { x: 0, minY: Infinity, n: 0, matched: 0 }
        a.x += n.x
        a.minY = Math.min(a.minY, n.y)
        a.n += 1
        if (matches === null || matches.has(Number(n.id))) a.matched += 1
        acc.set(n.topic, a)
      }
      ctx.textAlign = 'center'
      ctx.textBaseline = 'bottom'
      ctx.font = `600 ${11 / scale}px Inter Variable, Inter, system-ui, sans-serif`
      for (const [topic, a] of acc) {
        if (a.n < 2) continue
        ctx.globalAlpha = a.matched ? (focusSet ? 0.35 : 0.75) : 0.15
        ctx.fillStyle = topicMeta(topic).color
        ctx.fillText(topicMeta(topic).short.toUpperCase(), a.x / a.n, a.minY - 12 / scale)
      }
      ctx.globalAlpha = 1
    },
    [graph.nodes, matches, focusSet]
  )

  const linkColor = useCallback(
    (l: GLink) => {
      const a = linkEnd(l.source)
      const b = linkEnd(l.target)
      const dim = matches !== null && (!matches.has(a) || !matches.has(b))
      if (focusSet && focusSet.has(a) && focusSet.has(b) && (a === (hover?.id ?? selectedId) || b === (hover?.id ?? selectedId))) {
        const src = cache.current.get(a)
        return `${src?.color ?? '#888888'}cc`
      }
      if (dim || focusSet) return dark ? 'rgba(255,255,255,0.05)' : 'rgba(15,15,25,0.06)'
      const src = cache.current.get(a)
      return `${src?.color ?? '#888888'}${dark ? '55' : '66'}`
    },
    [matches, focusSet, hover, selectedId, dark]
  )

  const visibleTopics = useMemo(() => {
    const counts = new Map<string, number>()
    for (const n of graph.nodes) counts.set(n.topic, (counts.get(n.topic) ?? 0) + 1)
    return TOPIC_ORDER.filter((t) => counts.has(t)).map((t) => ({ topic: t, count: counts.get(t)! }))
  }, [graph.nodes])

  return (
    <div
      ref={container}
      className="relative size-full overflow-hidden"
      style={{
        backgroundImage: `radial-gradient(circle at 50% 45%, color-mix(in oklab, var(--accent) 6%, transparent), transparent 60%), radial-gradient(var(--border) 1px, transparent 1px)`,
        backgroundSize: '100% 100%, 22px 22px'
      }}
    >
      {size.width > 0 && (
        <ForceGraph2D<GNode, GLink>
          ref={fg}
          width={size.width}
          height={size.height}
          graphData={forcesReady ? graph : EMPTY_GRAPH}
          backgroundColor="rgba(0,0,0,0)"
          nodeId="id"
          nodeRelSize={4}
          nodeVal={(n) => 1 + n.degree}
          nodeLabel={() => ''}
          nodeCanvasObject={drawNode}
          onRenderFramePost={drawClusterLabels}
          nodePointerAreaPaint={(node, color, ctx, scale) => {
            ctx.fillStyle = color
            ctx.beginPath()
            ctx.arc(node.x ?? 0, node.y ?? 0, (9 + Math.sqrt(node.degree) * 1.3) / scale, 0, Math.PI * 2)
            ctx.fill()
          }}
          linkColor={linkColor}
          linkWidth={(l) => {
            const a = linkEnd(l.source)
            const b = linkEnd(l.target)
            const id = hover?.id ?? selectedId
            return id !== null && id !== undefined && (a === Number(id) || b === Number(id)) ? 2.2 : 1 + Math.max(0, l.value - 0.8) * 5
          }}
          warmupTicks={reduced ? 320 : 240}
          cooldownTicks={reduced ? 0 : undefined}
          cooldownTime={reduced ? 0 : 1800}
          d3VelocityDecay={0.35}
          minZoom={0.3}
          maxZoom={8}
          enableNodeDrag={!reduced}
          onEngineStop={() => {
            if (!settled.current) {
              settled.current = true
              fg.current?.zoomToFit(dur, 56)
              window.setTimeout(() => (baseScale.current = fg.current?.zoom() ?? baseScale.current), dur + 30)
            }
          }}
          onNodeHover={(n) => {
            setHover(n ?? null)
            if (container.current) container.current.style.cursor = n ? 'pointer' : 'grab'
          }}
          onNodeClick={(n) => onSelect(Number(n.id))}
          onBackgroundClick={() => onSelect(null)}
        />
      )}

      {/* hover card */}
      {hover && (
        <div className="pointer-events-none absolute left-1/2 top-3 z-10 w-max max-w-[min(480px,80%)] -translate-x-1/2 rounded-xl border border-border-strong bg-overlay/95 px-3.5 py-2.5 shadow-pop backdrop-blur">
          <div className="flex items-center gap-1.5 text-2xs font-medium uppercase tracking-wide" style={{ color: hover.color }}>
            <span className="size-1.5 rounded-full" style={{ background: hover.color }} />
            {hover.topics.join(' · ') || hover.topic}
            {hover.memory_type === 'short-term' && <span className="text-fg-subtle">· short-term</span>}
          </div>
          <div className="mt-1 text-sm leading-snug text-fg">{hover.content}</div>
        </div>
      )}

      {/* legend */}
      <div className="absolute bottom-3 left-3 z-10 max-w-[calc(100%-140px)] rounded-xl border border-border bg-surface/85 px-3 py-2.5 shadow-soft backdrop-blur">
        <div className="flex flex-wrap gap-x-3.5 gap-y-1.5">
          {visibleTopics.map(({ topic, count }) => (
            <span key={topic} className="flex items-center gap-1.5 text-xs text-fg-muted">
              <span className="size-2 rounded-full" style={{ background: TOPIC_META[topic]?.color }} />
              {TOPIC_META[topic]?.short ?? topic}
              <span className="text-fg-faint">{count}</span>
            </span>
          ))}
        </div>
        <div className="mt-1.5 flex items-center gap-3 border-t border-border pt-1.5 text-2xs text-fg-subtle">
          <span className="flex items-center gap-1.5">
            <span className="size-2.5 rounded-full border border-dashed border-fg-muted" /> short-term
          </span>
          <span>line = similar memories</span>
          <span className="hidden sm:inline">scroll to zoom · drag to pan</span>
        </div>
      </div>

      {/* zoom controls */}
      <div className="absolute bottom-3 right-3 z-10 flex flex-col gap-1 rounded-xl border border-border bg-surface/85 p-1 shadow-soft backdrop-blur">
        <IconButton size="sm" label="Zoom in" side="left" icon={<IconPlus size={15} />} onClick={() => fg.current?.zoom((fg.current?.zoom() ?? 1) * 1.4, dur)} />
        <IconButton size="sm" label="Zoom out" side="left" icon={<IconMinus size={15} />} onClick={() => fg.current?.zoom((fg.current?.zoom() ?? 1) / 1.4, dur)} />
        <IconButton size="sm" label="Fit to screen" side="left" icon={<IconFocusCentered size={15} />} onClick={() => fg.current?.zoomToFit(dur, 48)} />
      </div>
    </div>
  )
}
