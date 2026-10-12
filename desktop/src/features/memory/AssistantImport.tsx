/** "From another assistant": read a ChatGPT or Claude export or an OpenClaw workspace, pick what to bring over and
 *  import it in the background. Everything lands in Memory > Review (docs/API.md §19). */
import {
  IconAlertTriangle,
  IconArrowRight,
  IconCheck,
  IconChevronDown,
  IconFileZip,
  IconFolderOpen,
  IconInbox,
  IconLock,
  IconTransfer
} from '@tabler/icons-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Alert, Badge, Button, Dialog, Input, ProgressBar, Switch } from '@/components/ui'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import type { MemoryImportApp, MemoryImportJob, MemoryImportPart, MemoryImportPreview } from '@/lib/types'
import { cn } from '@/lib/utils'

const APPS: Array<{ id: MemoryImportApp; name: string; pick: 'file' | 'folder'; how: string }> = [
  {
    id: 'chatgpt',
    name: 'ChatGPT',
    pick: 'file',
    how: 'In ChatGPT, open Settings > Data controls > Export data. You get an email with a zip file; download it and choose it here.'
  },
  {
    id: 'claude',
    name: 'Claude',
    pick: 'file',
    how: 'In Claude, open Settings > Privacy > Export data. You get an email with a zip file; download it and choose it here.'
  },
  {
    id: 'openclaw',
    name: 'OpenClaw',
    pick: 'folder',
    how: 'Choose your OpenClaw workspace folder, the one with MEMORY.md in it. It is usually .openclaw/workspace in your home folder.'
  }
]

const isApp = (v: string | null | undefined): v is MemoryImportApp => !!v && APPS.some((a) => a.id === v)

/** A card that opens the dialog (Settings > Advanced, next to the Hermes one). */
export function AssistantImportCard({ className }: { className?: string }) {
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
          <span className="block text-sm font-medium text-fg">Coming from ChatGPT, Claude or OpenClaw?</span>
          <span className="block text-xs text-fg-subtle">Bring what they remember about you into Sentient&apos;s memory.</span>
        </span>
        <IconArrowRight size={15} className="text-fg-faint transition-transform group-hover:translate-x-0.5 group-hover:text-fg-muted" />
      </button>
      <AssistantImportDialog open={open} onOpenChange={setOpen} />
    </>
  )
}

export function AssistantImportDialog({
  open,
  onOpenChange,
  initialApp,
  initialPath
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  initialApp?: string | null
  initialPath?: string | null
}) {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const [app, setApp] = useState<MemoryImportApp>(isApp(initialApp) ? initialApp : 'chatgpt')
  const [path, setPath] = useState(initialPath ?? '')
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [watching, setWatching] = useState(false)
  const auto = useRef(false)

  const info = useQuery({ queryKey: ['import', app], queryFn: () => api.imports.memory.info(app), enabled: open && app === 'openclaw' })
  const job = useQuery({
    queryKey: qk.memoryImportJob,
    queryFn: api.imports.memory.job,
    enabled: open,
    refetchInterval: (q) => (q.state.data?.status === 'running' ? 2000 : false)
  })
  const preview = useMutation({
    mutationFn: (v: { app: MemoryImportApp; path: string }) => api.imports.memory.preview(v.app, v.path),
    onSuccess: (plan) => setPicked(new Set(plan.parts.filter((p) => p.count > 0).map((p) => p.id)))
  })
  const apply = useMutation({
    mutationFn: () => api.imports.memory.apply(app, { path: preview.data?.path ?? path, parts: [...picked] }),
    onSuccess: (j) => {
      qc.setQueryData(qk.memoryImportJob, j)
      setWatching(true)
    },
    onError: (e) => toast.error("Couldn't start the import", { description: errorMessage(e) })
  })
  const cancel = useMutation({
    mutationFn: api.imports.memory.cancel,
    onSuccess: (j) => qc.setQueryData(qk.memoryImportJob, j)
  })
  const undo = useMutation({
    mutationFn: (a: MemoryImportApp) => api.imports.memory.remove(a),
    onSuccess: (r, a) => {
      toast.success(`Removed ${r.facts + r.insights} memories imported from ${APPS.find((x) => x.id === a)?.name ?? a}`)
      void qc.invalidateQueries({ queryKey: qk.memories.all })
      void qc.invalidateQueries({ queryKey: qk.userModel })
    },
    onError: (e) => toast.error("Couldn't remove them", { description: errorMessage(e) })
  })

  // OpenClaw lives in a known folder: start there
  useEffect(() => {
    if (app === 'openclaw' && info.data?.path && !path) setPath(info.data.path)
  }, [app, info.data, path])
  // a link with ?app=&path= (and the smoke screenshots) opens straight on the preview
  useEffect(() => {
    if (open && !auto.current && isApp(initialApp) && initialPath) {
      auto.current = true
      preview.mutate({ app: initialApp, path: initialPath })
    }
  }, [open, initialApp, initialPath, preview])
  // an import that is still running shows its progress when the dialog opens again
  useEffect(() => {
    if (open && job.data?.status === 'running') setWatching(true)
  }, [open, job.data?.status])

  const close = (next: boolean) => {
    onOpenChange(next)
    if (!next) {
      preview.reset()
      setWatching(false)
      auto.current = true
    }
  }
  const choose = async () => {
    const meta = APPS.find((a) => a.id === app)!
    const [p] = await getBridge().pickFiles(
      meta.pick === 'folder' ? { directory: true, multiple: false } : { multiple: false, filters: [{ name: `${meta.name} export`, extensions: ['zip', 'json'] }] }
    )
    if (p) {
      setPath(p)
      preview.reset()
    }
  }
  const switchApp = (a: MemoryImportApp) => {
    setApp(a)
    setPath('')
    preview.reset()
  }

  const plan = preview.data
  const current = job.data
  const showJob = watching && current

  const footer = showJob ? (
    current.status === 'running' ? (
      <>
        <Button variant="ghost" onClick={() => close(false)}>
          Keep it running
        </Button>
        <Button variant="secondary" loading={cancel.isPending} onClick={() => cancel.mutate()}>
          Stop
        </Button>
      </>
    ) : (
      <>
        <Button variant="ghost" onClick={() => close(false)}>
          Done
        </Button>
        <Button variant="primary" leftIcon={<IconInbox size={15} />} onClick={() => (close(false), navigate('/memory?tab=review'))}>
          Review them
        </Button>
      </>
    )
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
    <Button variant="primary" rightIcon={<IconArrowRight size={15} />} disabled={!path.trim()} loading={preview.isPending} onClick={() => preview.mutate({ app, path: path.trim() })}>
      Look inside
    </Button>
  )

  return (
    <Dialog
      open={open}
      onOpenChange={close}
      size="lg"
      modalLock={!!plan || apply.isPending}
      title={showJob ? `Importing from ${current.label}` : plan ? `What's in your ${plan.label} export` : 'Bring your memory from another assistant'}
      description={showJob || plan ? undefined : 'Nothing is imported until you choose Import, and everything waits for your OK in Memory > Review.'}
      footer={footer}
    >
      {showJob ? (
        <JobView job={current} onUndo={() => undo.mutate(current.app)} undoing={undo.isPending} />
      ) : plan ? (
        <PreviewView plan={plan} picked={picked} onToggle={(id, on) => setPicked((s) => toggled(s, id, on))} />
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-2" role="radiogroup" aria-label="Which assistant">
            {APPS.map((a) => (
              <button
                key={a.id}
                type="button"
                role="radio"
                aria-checked={app === a.id}
                onClick={() => switchApp(a.id)}
                className={cn(
                  'rounded-xl border px-3 py-2.5 text-left text-sm transition-colors',
                  app === a.id ? 'border-accent bg-accent/8 text-fg' : 'border-border bg-surface text-fg-muted hover:border-border-strong hover:text-fg'
                )}
              >
                <span className="block font-medium">{a.name}</span>
                <span className="block text-xs text-fg-subtle">{a.pick === 'file' ? 'Data export (zip)' : 'Workspace folder'}</span>
              </button>
            ))}
          </div>
          <p className="text-sm text-fg-muted">{APPS.find((a) => a.id === app)!.how}</p>
          <div className="flex gap-2">
            <Input
              className="flex-1 font-mono"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder={app === 'openclaw' ? 'Your OpenClaw workspace folder' : `Your ${APPS.find((a) => a.id === app)!.name} export zip`}
              aria-label="Export file or folder"
            />
            <Button variant="secondary" leftIcon={app === 'openclaw' ? <IconFolderOpen size={15} /> : <IconFileZip size={15} />} onClick={() => void choose()}>
              {app === 'openclaw' ? 'Choose folder' : 'Choose file'}
            </Button>
          </div>
          <p className="flex items-start gap-2 text-xs text-fg-subtle">
            <IconLock size={14} className="mt-px shrink-0" />
            Sentient reads it on this computer. Nothing is uploaded, and the export is never changed.
          </p>
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

function toggled(s: Set<string>, id: string, on: boolean): Set<string> {
  const next = new Set(s)
  if (on) next.add(id)
  else next.delete(id)
  return next
}

function PreviewView({ plan, picked, onToggle }: { plan: MemoryImportPreview; picked: Set<string>; onToggle: (id: string, on: boolean) => void }) {
  const batches = plan.parts.filter((p) => picked.has(p.id)).reduce((n, p) => n + p.batches, 0)
  return (
    <div className="max-h-[58vh] space-y-3 overflow-y-auto pr-1">
      <p className="truncate font-mono text-xs text-fg-subtle" title={plan.path}>
        {plan.path}
      </p>
      <Stats plan={plan} />
      {plan.parts.map((part) => (
        <PartCard key={part.id} part={part} on={picked.has(part.id)} onChange={(v) => onToggle(part.id, v)} />
      ))}
      {batches > 0 && (
        <Alert tone="info" title={`Your fast AI model reads this in ${plural(batches, 'batch', 'batches')}`}>
          With a local model that can take a while. It runs in the background, so you can keep using Sentient, and you can stop it any time.
        </Alert>
      )}
    </div>
  )
}

function Stats({ plan }: { plan: MemoryImportPreview }) {
  const s = plan.stats
  const bits =
    plan.app === 'openclaw'
      ? [plural(s.memories ?? 0, 'memory', 'memories'), plural(s.notes ?? 0, 'daily note', 'daily notes')]
      : [
          plural(s.conversations ?? 0, 'conversation', 'conversations'),
          plural(s.messages ?? 0, 'message from you', 'messages from you'),
          ...(plan.app === 'chatgpt' ? [plural(s.saved_memories ?? 0, 'saved memory', 'saved memories')] : []),
          ...(plan.app === 'claude' ? [plural(s.projects ?? 0, 'project', 'projects')] : [])
        ]
  return <p className="text-sm text-fg-muted">Found {bits.join(', ')}.</p>
}

function PartCard({ part, on, onChange }: { part: MemoryImportPart; on: boolean; onChange: (on: boolean) => void }) {
  const [open, setOpen] = useState(part.items.length > 0)
  const rows = part.items.length + part.sources_total
  const skipped = part.items.filter((i) => i.action === 'skip').length + part.sources.filter((x) => x.action === 'skip').length
  const shownSources = part.sources.slice(0, 6)
  return (
    <section className="rounded-xl border border-border bg-surface">
      <div className="flex items-center gap-3 px-4 py-3">
        <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left" onClick={() => setOpen((o) => !o)} disabled={!rows}>
          <span className="text-sm font-medium text-fg">{part.label}</span>
          {part.count > 0 && <Badge size="xs">{part.count} to import</Badge>}
          {skipped > 0 && <span className="text-xs text-fg-subtle">{skipped} left out</span>}
          {rows > 0 && <IconChevronDown size={14} className={cn('text-fg-subtle transition-transform', open && 'rotate-180')} />}
        </button>
        {part.count > 0 ? <Switch checked={on} onCheckedChange={onChange} aria-label={`Import ${part.label}`} /> : <span className="text-xs text-fg-subtle">Nothing to import</span>}
      </div>
      {open && rows > 0 && (
        <div className="space-y-2 border-t border-border px-4 py-3">
          {part.items.length > 0 && (
            <ul className="space-y-2">
              {part.items.map((item) => (
                <li key={item.key} className={cn('flex items-start gap-2 text-sm', item.action === 'skip' && 'opacity-60')}>
                  {item.action === 'skip' ? <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-fg-faint" /> : <IconCheck size={14} className="mt-0.5 shrink-0 text-success" />}
                  <div className="min-w-0 flex-1">
                    <div className="break-words text-fg">{item.text}</div>
                    <div className="text-xs text-fg-subtle">
                      {item.kind === 'insight' ? 'About you' : 'Memory'}
                      {item.dated && ` · ${shortDate(item.dated)}`}
                      {item.action === 'skip' && ` · ${item.note}`}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
          {shownSources.length > 0 && (
            <ul className="space-y-2">
              {shownSources.map((src) => (
                <li key={src.key} className={cn('text-sm', src.action === 'skip' && 'opacity-60')}>
                  <div className="flex items-baseline gap-2">
                    <span className="min-w-0 flex-1 truncate text-fg">{src.title}</span>
                    <span className="shrink-0 text-xs tabular-nums text-fg-subtle">
                      {src.dated ? `${shortDate(src.dated)} · ` : ''}
                      {plural(src.messages, part.id === 'conversations' ? 'message' : 'line', part.id === 'conversations' ? 'messages' : 'lines')}
                    </span>
                  </div>
                  {src.action === 'skip' ? (
                    <div className="text-xs text-fg-subtle">{src.note}</div>
                  ) : (
                    src.snippet && <div className="truncate text-xs text-fg-muted">“{src.snippet}”</div>
                  )}
                </li>
              ))}
            </ul>
          )}
          {part.sources_total > shownSources.length && <p className="text-xs text-fg-subtle">and {part.sources_total - shownSources.length} more</p>}
        </div>
      )}
    </section>
  )
}

function JobView({ job, onUndo, undoing }: { job: MemoryImportJob; onUndo: () => void; undoing: boolean }) {
  const running = job.status === 'running'
  const found = job.facts + job.insights
  return (
    <div className="space-y-4">
      <p className="truncate font-mono text-xs text-fg-subtle" title={job.path}>
        {job.export}
      </p>
      {job.total > 0 && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-fg-muted">
            <span className="truncate">{job.message}</span>
            <span className="shrink-0 tabular-nums">
              {job.done} of {job.total}
            </span>
          </div>
          <ProgressBar value={job.total ? job.done / job.total : null} tone={job.status === 'error' ? 'danger' : 'accent'} />
        </div>
      )}
      <div className="grid grid-cols-3 gap-2">
        <Stat label="To review" value={found} tone="text-success" />
        <Stat label="Filled in" value={job.merged} tone="text-fg" />
        <Stat label="Already known" value={job.skipped} tone="text-fg-muted" />
      </div>
      {!running && (
        <Alert
          tone={job.status === 'error' ? 'danger' : job.failed ? 'warning' : 'success'}
          icon={job.status === 'error' || job.failed ? <IconAlertTriangle /> : <IconCheck />}
          title={job.status === 'cancelled' ? 'Stopped' : job.status === 'error' ? 'The import stopped' : 'Done'}
          action={
            found > 0 ? (
              <Button size="xs" variant="ghost" loading={undoing} onClick={onUndo}>
                Remove everything from {job.label}
              </Button>
            ) : undefined
          }
        >
          {job.message}
        </Alert>
      )}
    </div>
  )
}

function Stat({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface px-3 py-2.5 text-center">
      <div className={cn('text-xl font-semibold tabular-nums', tone)}>{value}</div>
      <div className="text-xs text-fg-subtle">{label}</div>
    </div>
  )
}

function shortDate(day: string): string {
  const d = new Date(`${day}T12:00:00`)
  return Number.isNaN(d.getTime()) ? day : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

function plural(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`
}
