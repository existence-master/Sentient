/** "Coming from Hermes?": preview a Hermes home folder, pick what to bring over, import it (docs/API.md §19). */
import {
  IconAlertTriangle,
  IconArrowRight,
  IconBrain,
  IconCalendarRepeat,
  IconCheck,
  IconChevronDown,
  IconFolderOpen,
  IconMoodSmile,
  IconPlugConnected,
  IconSparkles,
  IconTransfer,
  type Icon
} from '@tabler/icons-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { Alert, Badge, Button, Dialog, Input, Skeleton, Switch } from '@/components/ui'
import { DiffView } from '@/features/skills/DiffView'
import { scheduleSentence } from '@/features/tasks/schedule'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import type { HermesJobItem, HermesMcpItem, HermesMemoryItem, HermesPart, HermesPersona, HermesPreview, HermesResult, HermesSkillItem } from '@/lib/types'
import { cn } from '@/lib/utils'

type AnyItem = HermesSkillItem | HermesMemoryItem | HermesPersona | HermesJobItem | HermesMcpItem

const PARTS: Array<{ id: HermesPart; label: string; icon: Icon; empty: string }> = [
  { id: 'skills', label: 'Skills', icon: IconSparkles, empty: 'No skills of your own to bring over.' },
  { id: 'memory', label: 'Memory and profile', icon: IconBrain, empty: 'No memories to bring over.' },
  { id: 'persona', label: 'Personality', icon: IconMoodSmile, empty: 'No SOUL.md, or it matches the one you have.' },
  { id: 'jobs', label: 'Scheduled jobs', icon: IconCalendarRepeat, empty: 'No scheduled jobs to bring over.' },
  { id: 'mcp', label: 'MCP servers', icon: IconPlugConnected, empty: 'No MCP servers to bring over.' }
]

/** A card that opens the import dialog. Used in onboarding and Settings. */
export function HermesImportCard({ className }: { className?: string }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={cn(
          'group flex w-full items-center gap-3 rounded-xl border border-border bg-surface/70 px-4 py-3 text-left transition-colors hover:border-border-strong hover:bg-elevated',
          className
        )}
      >
        <IconTransfer size={18} className="shrink-0 text-accent-text" />
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium text-fg">Coming from Hermes?</span>
          <span className="block text-xs text-fg-subtle">Bring your skills, memory, personality, scheduled jobs and MCP servers.</span>
        </span>
        <IconArrowRight size={15} className="text-fg-faint transition-transform group-hover:translate-x-0.5 group-hover:text-fg-muted" />
      </button>
      <HermesImportDialog open={open} onOpenChange={setOpen} />
    </>
  )
}

export function HermesImportDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const qc = useQueryClient()
  const info = useQuery({ queryKey: ['import', 'hermes'], queryFn: api.imports.hermes.info, enabled: open })
  const [path, setPath] = useState('')
  const [picked, setPicked] = useState<Set<HermesPart>>(new Set())
  const [result, setResult] = useState<HermesResult | null>(null)

  useEffect(() => {
    if (info.data && !path) setPath(info.data.path)
  }, [info.data, path])

  const preview = useMutation({ mutationFn: (p: string) => api.imports.hermes.preview(p) })
  const look = (p: string) =>
    preview.mutate(p, {
      // persona is never picked for the user: replacing the personality needs an explicit yes
      onSuccess: (plan) => setPicked(new Set(PARTS.filter((x) => x.id !== 'persona' && plan.counts[x.id] > 0).map((x) => x.id)))
    })
  const apply = useMutation({
    mutationFn: () => api.imports.hermes.apply({ path: preview.data?.path, parts: [...picked] }),
    onSuccess: (r) => {
      setResult(r)
      for (const key of [qk.skills.all, qk.memories.all, qk.userModel, qk.tasks.all, qk.integrations.mcp]) void qc.invalidateQueries({ queryKey: key })
    },
    onError: (e) => toast.error("Couldn't import", { description: errorMessage(e) })
  })
  const undoMemories = useMutation({
    mutationFn: api.imports.hermes.removeMemories,
    onSuccess: (r) => {
      toast.success(`Removed ${r.facts + r.insights} imported memories`)
      void qc.invalidateQueries({ queryKey: qk.memories.all })
      void qc.invalidateQueries({ queryKey: qk.userModel })
    },
    onError: (e) => toast.error("Couldn't remove them", { description: errorMessage(e) })
  })

  const close = (next: boolean) => {
    onOpenChange(next)
    if (!next) {
      preview.reset()
      setResult(null)
    }
  }
  const choose = async () => {
    const [dir] = await getBridge().pickFiles({ directory: true, multiple: false })
    if (dir) {
      setPath(dir)
      preview.reset()
    }
  }
  const toggle = (id: HermesPart, on: boolean) =>
    setPicked((s) => {
      const next = new Set(s)
      if (on) next.add(id)
      else next.delete(id)
      return next
    })

  const plan = preview.data
  const footer = result ? (
    <Button variant="primary" onClick={() => close(false)}>
      Done
    </Button>
  ) : plan ? (
    <>
      <Button variant="ghost" onClick={() => preview.reset()}>
        Back
      </Button>
      <Button variant="primary" leftIcon={<IconCheck size={15} />} disabled={!picked.size} loading={apply.isPending} onClick={() => apply.mutate()}>
        Import
      </Button>
    </>
  ) : (
    <Button variant="primary" rightIcon={<IconArrowRight size={15} />} disabled={!path.trim()} loading={preview.isPending} onClick={() => look(path.trim())}>
      Look inside
    </Button>
  )

  return (
    <Dialog
      open={open}
      onOpenChange={close}
      size="lg"
      modalLock
      title="Bring your things over from Hermes"
      description={result ? undefined : 'Nothing is imported until you choose Import. Keys, tokens and sign-ins are never copied.'}
      footer={footer}
    >
      {result ? (
        <ImportSummary result={result} onUndoMemories={() => undoMemories.mutate()} undoing={undoMemories.isPending} />
      ) : plan ? (
        <div className="max-h-[58vh] space-y-3 overflow-y-auto pr-1">
          <p className="truncate font-mono text-xs text-fg-subtle" title={plan.path}>
            {plan.path}
          </p>
          {PARTS.map((part) => (
            <PartCard key={part.id} part={part} plan={plan} on={picked.has(part.id)} onChange={(v) => toggle(part.id, v)} />
          ))}
          <Suggestions plan={plan} />
        </div>
      ) : (
        <div className="space-y-3">
          <p className="text-sm text-fg-muted">
            Sentient reads your Hermes folder and shows what it found. You choose what to bring over: skills (to review first), memory, your
            personality, scheduled jobs (paused until you turn them on) and MCP servers (off until you sign in again).
          </p>
          <div className="flex gap-2">
            {info.isLoading ? (
              <Skeleton className="h-9 flex-1 rounded-lg" />
            ) : (
              <Input className="flex-1 font-mono" value={path} onChange={(e) => setPath(e.target.value)} placeholder="Your Hermes folder" aria-label="Hermes folder" />
            )}
            <Button variant="secondary" leftIcon={<IconFolderOpen size={15} />} onClick={() => void choose()}>
              Choose folder
            </Button>
          </div>
          {info.data && !info.data.exists && path === info.data.path && (
            <p className="text-xs text-fg-subtle">There&apos;s no Hermes folder in the usual place. Choose the folder that has config.yaml in it.</p>
          )}
          {preview.isError && (
            <Alert tone="danger" icon={<IconAlertTriangle />}>
              {errorMessage(preview.error)}
            </Alert>
          )}
        </div>
      )}
    </Dialog>
  )
}

function itemsOf(plan: HermesPreview, id: HermesPart): AnyItem[] {
  if (id === 'persona') return plan.persona ? [plan.persona] : []
  return plan[id]
}

function PartCard({ part, plan, on, onChange }: { part: (typeof PARTS)[number]; plan: HermesPreview; on: boolean; onChange: (on: boolean) => void }) {
  const [open, setOpen] = useState(part.id === 'persona')
  const items = itemsOf(plan, part.id)
  const count = plan.counts[part.id]
  const skipped = items.length - count
  return (
    <section className="rounded-xl border border-border bg-surface">
      <div className="flex items-center gap-3 px-4 py-3">
        <part.icon size={17} className="shrink-0 text-fg-subtle" />
        <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left" onClick={() => setOpen((o) => !o)} disabled={!items.length}>
          <span className="text-sm font-medium text-fg">{part.label}</span>
          {count > 0 && <Badge size="xs">{part.id === 'persona' ? 'New personality' : `${count} to import`}</Badge>}
          {skipped > 0 && <span className="text-xs text-fg-subtle">{skipped} left out</span>}
          {items.length > 0 && <IconChevronDown size={14} className={cn('text-fg-subtle transition-transform', open && 'rotate-180')} />}
        </button>
        {count > 0 ? (
          <Switch checked={on} onCheckedChange={onChange} aria-label={`Import ${part.label}`} />
        ) : (
          <span className="text-xs text-fg-subtle">Nothing to import</span>
        )}
      </div>
      {!items.length && <p className="border-t border-border px-4 py-2.5 text-xs text-fg-subtle">{part.empty}</p>}
      {open && items.length > 0 && (
        <div className="border-t border-border px-4 py-3">
          {part.id === 'persona' && plan.persona ? (
            <div className="space-y-2">
              <p className="text-xs text-fg-muted">{plan.persona.note}</p>
              {plan.persona.action === 'import' && <DiffView current={plan.persona.current} proposed={plan.persona.proposed} />}
            </div>
          ) : (
            <ul className="space-y-2">
              {items.map((item) => (
                <ItemRow key={item.key} part={part.id} item={item} />
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  )
}

function ItemRow({ part, item }: { part: HermesPart; item: AnyItem }) {
  const [showCode, setShowCode] = useState(false)
  const skip = item.action === 'skip'
  let title: ReactNode = item.key
  let detail: ReactNode = null
  let code: string | null = null
  if (part === 'skills' && 'folder' in item) {
    title = item.target && item.target !== item.name ? `${item.name} (as ${item.target})` : item.name
    detail = item.description || null
  } else if (part === 'memory' && 'text' in item) {
    title = item.text
    detail = item.kind === 'fact' ? 'Memory' : 'About you'
  } else if (part === 'jobs' && 'schedule_text' in item) {
    title = item.name
    detail = item.schedule ? scheduleSentence(item.schedule) : item.schedule_text ? `Hermes schedule: ${item.schedule_text}` : null
    code = item.script?.code ?? null
  } else if (part === 'mcp' && 'transport' in item) {
    title = item.name
    detail = item.transport === 'http' ? item.url : [item.command, ...(item.args ?? [])].filter(Boolean).join(' ')
  }
  return (
    <li className={cn('text-sm', skip && 'opacity-60')}>
      <div className="flex items-start gap-2">
        {skip ? <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-fg-faint" /> : <IconCheck size={14} className="mt-0.5 shrink-0 text-success" />}
        <div className="min-w-0 flex-1">
          <div className="break-words text-fg">{title}</div>
          {detail && <div className="truncate font-mono text-2xs text-fg-subtle">{detail}</div>}
          <div className="text-xs text-fg-muted">{item.note}</div>
          {code && (
            <button type="button" className="mt-1 text-xs font-medium text-accent-text" onClick={() => setShowCode((s) => !s)}>
              {showCode ? 'Hide the script' : 'Show the script'}
            </button>
          )}
          {code && showCode && <pre className="selectable mt-1.5 max-h-48 overflow-auto rounded-lg bg-sunken p-2.5 font-mono text-2xs text-fg">{code}</pre>}
        </div>
      </div>
    </li>
  )
}

function Suggestions({ plan }: { plan: HermesPreview }) {
  const { wake_word, tts_voice } = plan.suggestions
  if (!wake_word && !tts_voice) return null
  const parts = [wake_word && `the wake phrase "${wake_word}"`, tts_voice && `the voice ${tts_voice}`].filter(Boolean)
  return (
    <Alert tone="info" title="Voice settings">
      Hermes used {parts.join(' and ')}. These aren&apos;t copied; you can pick your own in Settings &gt; Voice.
    </Alert>
  )
}

function ImportSummary({ result, onUndoMemories, undoing }: { result: HermesResult; onUndoMemories: () => void; undoing: boolean }) {
  const lines: Array<{ text: string; action?: ReactNode }> = []
  if (result.skills) lines.push({ text: plural(result.skills.imported.length, 'skill is', 'skills are') + ' waiting for your review in Skills.' })
  if (result.memory) {
    const n = result.memory.facts + result.memory.insights
    lines.push({
      text: `${plural(result.memory.facts, 'memory', 'memories')} and ${plural(result.memory.insights, 'thing', 'things')} about you are waiting for your review in Memory.`,
      action:
        n > 0 ? (
          <Button size="xs" variant="ghost" loading={undoing} onClick={onUndoMemories}>
            Remove them
          </Button>
        ) : undefined
    })
  }
  if (result.persona) lines.push({ text: result.persona.updated ? 'Your personality now comes from Hermes.' : 'Your personality was not changed.' })
  if (result.jobs) lines.push({ text: `${plural(result.jobs.created.length, 'task was', 'tasks were')} added, paused. Resume one in Tasks and Sentient plans it for you to approve.` })
  if (result.mcp) lines.push({ text: `${plural(result.mcp.added.length, 'MCP server was', 'MCP servers were')} added, turned off. Turn them on in Integrations.` })
  const skipped = [...(result.skills?.skipped ?? []), ...(result.jobs?.skipped ?? []), ...(result.mcp?.skipped ?? [])]
  return (
    <div className="space-y-3">
      <ul className="space-y-2">
        {lines.map((l) => (
          <li key={l.text} className="flex items-center gap-2 text-sm text-fg">
            <IconCheck size={15} className="shrink-0 text-success" />
            <span className="flex-1">{l.text}</span>
            {l.action}
          </li>
        ))}
      </ul>
      {skipped.length > 0 && (
        <details className="rounded-lg border border-border px-3 py-2 text-xs text-fg-muted">
          <summary className="cursor-pointer text-fg-subtle">{plural(skipped.length, 'thing was', 'things were')} left out</summary>
          <ul className="mt-2 space-y-1">
            {skipped.map((s) => (
              <li key={s.key}>
                <span className="text-fg">{s.name}</span>: {s.note}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}

function plural(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`
}
