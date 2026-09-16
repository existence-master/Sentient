import { IconArrowUpRight, IconTimeline } from '@tabler/icons-react'
import { useMemo, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { Alert, EmptyState, Skeleton } from '@/components/ui'
import { dayKey, dayLabel } from '@/features/memory/meta'
import { useEvolutionLog } from '@/hooks/skills'
import { errorMessage } from '@/lib/api'
import type { EvolutionLogEntry } from '@/lib/types'
import { cn, formatDateTime, formatTime, parseDate } from '@/lib/utils'
import { EVOLUTION_META, evolutionMeta } from './meta'

const str = (v: unknown) => (typeof v === 'string' ? v : v === null || v === undefined ? '' : String(v))
const list = (v: unknown) => (Array.isArray(v) ? v.map(str) : [])
const num = (v: unknown) => (typeof v === 'number' ? v : Number(v) || 0)

export function EvolutionLog({ onOpenSkill }: { onOpenSkill: (name: string, pending?: boolean) => void }) {
  const log = useEvolutionLog(300)
  const [kind, setKind] = useState<string>('all')

  const counts = useMemo(() => {
    const m = new Map<string, number>()
    log.data?.forEach((e) => m.set(e.kind, (m.get(e.kind) ?? 0) + 1))
    return m
  }, [log.data])

  const days = useMemo(() => {
    const groups = new Map<string, EvolutionLogEntry[]>()
    const sorted = [...(log.data ?? [])].sort((a, b) => (parseDate(b.ts)?.getTime() ?? 0) - (parseDate(a.ts)?.getTime() ?? 0))
    for (const e of sorted) {
      if (kind !== 'all' && e.kind !== kind) continue
      const k = dayKey(e.ts)
      if (!groups.has(k)) groups.set(k, [])
      groups.get(k)!.push(e)
    }
    return [...groups.entries()]
  }, [log.data, kind])

  if (log.isLoading) {
    return (
      <div className="space-y-3">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-16 rounded-xl" />
        ))}
      </div>
    )
  }
  if (log.isError) {
    return (
      <Alert tone="danger" title="Couldn't load the evolution log">
        {errorMessage(log.error)}
      </Alert>
    )
  }
  if (!log.data?.length) {
    return (
      <EmptyState
        icon={<IconTimeline />}
        title="Nothing has changed yet"
        description="As Sentient learns skills, tidies them up and refreshes your profile, every change is recorded here."
      />
    )
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-1.5">
        <FilterChip active={kind === 'all'} onClick={() => setKind('all')} label="Everything" count={log.data.length} />
        {Object.entries(EVOLUTION_META)
          .filter(([k]) => counts.has(k))
          .map(([k, meta]) => (
            <FilterChip key={k} active={kind === k} onClick={() => setKind(kind === k ? 'all' : k)} label={meta.label} count={counts.get(k) ?? 0} color={meta.color} icon={<meta.icon size={13} />} />
          ))}
      </div>

      <div className="space-y-6">
        {days.map(([day, entries]) => (
          <section key={day}>
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-fg-subtle">{dayLabel(day)}</h3>
            <ol className="relative">
              <span aria-hidden className="absolute bottom-4 left-[15px] top-4 w-px bg-border" />
              {entries.map((e, i) => (
                <Entry key={`${e.ts}-${i}`} entry={e} onOpenSkill={onOpenSkill} />
              ))}
            </ol>
          </section>
        ))}
      </div>
    </div>
  )
}

function FilterChip({ active, onClick, label, count, color, icon }: { active: boolean; onClick: () => void; label: string; count: number; color?: string; icon?: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        'inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-xs font-medium transition-colors',
        active ? 'border-border-strong bg-active text-fg' : 'border-border bg-surface text-fg-muted hover:border-border-strong hover:text-fg'
      )}
    >
      {icon && <span style={{ color }}>{icon}</span>}
      {label}
      <span className="text-fg-subtle">{count}</span>
    </button>
  )
}

function SkillLink({ name, pending, onOpenSkill }: { name: string; pending?: boolean; onOpenSkill: (name: string, pending?: boolean) => void }) {
  return (
    <button type="button" onClick={() => onOpenSkill(name, pending)} className="rounded bg-active px-1 font-mono text-[0.92em] text-fg hover:bg-accent/15 hover:text-accent-text">
      {name}
    </button>
  )
}

function Entry({ entry, onOpenSkill }: { entry: EvolutionLogEntry; onOpenSkill: (name: string, pending?: boolean) => void }) {
  const navigate = useNavigate()
  const meta = evolutionMeta(entry.kind)
  const d = entry.detail ?? {}
  const name = str(d.name)
  const pending = d.pending === true
  let title: ReactNode = meta.label
  let body: ReactNode = null
  const origin: ReactNode =
    d.session_id ? (
      <OriginLink onClick={() => navigate(`/chat/${str(d.session_id)}`)}>From a chat</OriginLink>
    ) : d.task_id ? (
      <OriginLink onClick={() => navigate(`/tasks?task=${encodeURIComponent(str(d.task_id))}`)}>From a task run</OriginLink>
    ) : null

  switch (entry.kind) {
    case 'skill_created':
      title = pending ? (
        <>Sentient learned a new skill, <SkillLink name={name} pending onOpenSkill={onOpenSkill} />, and asked for your review</>
      ) : d.approved ? (
        <>You approved <SkillLink name={name} onOpenSkill={onOpenSkill} /></>
      ) : (
        <>You created <SkillLink name={name} onOpenSkill={onOpenSkill} /></>
      )
      break
    case 'skill_patched':
      title = pending ? (
        <>Sentient proposed an improvement to <SkillLink name={name} pending onOpenSkill={onOpenSkill} /></>
      ) : (
        <>Improvement to <SkillLink name={name} onOpenSkill={onOpenSkill} /> {d.approved ? 'approved' : 'applied'}</>
      )
      break
    case 'skill_archived':
      title =
        str(d.by) === 'curator' ? (
          <>The curator archived <SkillLink name={name} onOpenSkill={onOpenSkill} />{d.idle_days ? ` after ${num(d.idle_days)} days unused` : ''}</>
        ) : (
          <>You archived <SkillLink name={name} onOpenSkill={onOpenSkill} /></>
        )
      break
    case 'profile_updated': {
      const appended = num(d.learned_appended)
      title = 'Sentient refreshed your profile'
      body = (
        <ul className="flex flex-wrap gap-1.5">
          <Pill>{appended ? `${appended} new ${appended === 1 ? 'fact' : 'facts'} added to USER.md` : 'USER.md unchanged'}</Pill>
          {num(d.memory_chars) > 0 && <Pill>MEMORY.md rewritten ({num(d.memory_chars).toLocaleString()} characters)</Pill>}
          <Pill>
            considered {num(d.facts_considered)} {num(d.facts_considered) === 1 ? 'fact' : 'facts'} and {num(d.summaries_considered)} conversation{' '}
            {num(d.summaries_considered) === 1 ? 'summary' : 'summaries'}
          </Pill>
        </ul>
      )
      break
    }
    case 'curator_run': {
      const staled = list(d.staled)
      const archived = list(d.archived)
      const merges = list(d.merge_proposals)
      title = 'The curator reviewed your skills'
      body =
        staled.length || archived.length || merges.length ? (
          <div className="space-y-1 text-fg-muted">
            {staled.length > 0 && (
              <div>
                Marked stale: {staled.map((n) => <SkillLink key={n} name={n} onOpenSkill={onOpenSkill} />).reduce<ReactNode[]>((a, c, i) => (i ? [...a, ', ', c] : [c]), [])}
              </div>
            )}
            {archived.length > 0 && (
              <div>
                Archived: {archived.map((n) => <SkillLink key={n} name={n} onOpenSkill={onOpenSkill} />).reduce<ReactNode[]>((a, c, i) => (i ? [...a, ', ', c] : [c]), [])}
              </div>
            )}
            {merges.length > 0 && <div>Proposed merges: {merges.join(', ')}</div>}
          </div>
        ) : (
          <span className="text-fg-subtle">Everything was in good shape. Nothing to change.</span>
        )
      break
    }
    case 'summary_created':
      title = 'Sentient remembered a conversation'
      body = d.start_at ? (
        <span className="text-fg-subtle">
          {formatDateTime(str(d.start_at))} – {formatTime(str(d.end_at))}
        </span>
      ) : null
      break
  }

  return (
    <li className="relative flex gap-3 py-2">
      <span className="relative z-10 mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full border border-border bg-surface" style={{ color: meta.color }}>
        <meta.icon size={15} />
      </span>
      <div className="min-w-0 flex-1 rounded-xl border border-border bg-surface px-3.5 py-2.5">
        <div className="flex items-start gap-3">
          <div className="min-w-0 flex-1 text-sm leading-relaxed text-fg">{title}</div>
          <time className="shrink-0 text-xs tabular-nums text-fg-subtle" title={formatDateTime(entry.ts)}>
            {formatTime(entry.ts)}
          </time>
        </div>
        {body && <div className="mt-1.5 text-sm">{body}</div>}
        {typeof d.reason === 'string' && d.reason && <p className="mt-2 border-l-2 border-accent/30 pl-2.5 text-sm text-fg-muted">{d.reason}</p>}
        {origin && <div className="mt-1.5">{origin}</div>}
      </div>
    </li>
  )
}

function Pill({ children }: { children: ReactNode }) {
  return <li className="rounded-md bg-active px-2 py-0.5 text-xs text-fg-muted">{children}</li>
}

function OriginLink({ onClick, children }: { onClick: () => void; children: ReactNode }) {
  return (
    <button type="button" onClick={onClick} className="inline-flex items-center gap-0.5 text-xs text-fg-subtle hover:text-accent-text">
      {children}
      <IconArrowUpRight size={12} />
    </button>
  )
}
