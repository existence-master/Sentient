import {
  IconAlertTriangle,
  IconCircleCheck,
  IconCopy,
  IconFileImport,
  IconFileText,
  IconPlus,
  IconRefresh,
  IconUpload
} from '@tabler/icons-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useRef, useState, type DragEvent } from 'react'
import { toast } from 'sonner'
import { Alert, Button, Dialog, Input, ProgressBar, Textarea } from '@/components/ui'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import type { MemoryImportResult, MemoryWriteResult } from '@/lib/types'
import { cn, formatBytes } from '@/lib/utils'
import { sourceMeta } from './meta'

// ---------------------------------------------------------------------------- add
const ACTION_COPY: Record<MemoryWriteResult['action'], { title: string; body: string; icon: React.ReactNode; tone: 'success' | 'info' | 'warning' }> = {
  ADD: { title: 'Added a new memory', body: 'Sentient will recall this when it is relevant.', icon: <IconCircleCheck />, tone: 'success' },
  UPDATE: { title: 'Updated an existing memory', body: 'This replaced something Sentient already knew.', icon: <IconRefresh />, tone: 'info' },
  DELETE: { title: 'Removed an outdated memory', body: 'This contradicted something Sentient knew, so that memory was removed.', icon: <IconRefresh />, tone: 'info' },
  SKIP: { title: 'Already remembered', body: 'Sentient already knows this, so nothing was added.', icon: <IconCopy />, tone: 'warning' }
}

export function AddMemoryDialog({ open, onOpenChange, onShow }: { open: boolean; onOpenChange: (o: boolean) => void; onShow: (id: number) => void }) {
  const qc = useQueryClient()
  const [content, setContent] = useState('')
  const [result, setResult] = useState<MemoryWriteResult | null>(null)
  const create = useMutation({
    mutationFn: (text: string) => api.memories.create(text, 'manual'),
    onSuccess: (r) => {
      setResult(r)
      if (r.action !== 'SKIP') setContent('')
      void qc.invalidateQueries({ queryKey: qk.memories.all })
    },
    onError: (e) => toast.error("Couldn't add memory", { description: errorMessage(e) })
  })
  const close = (o: boolean) => {
    onOpenChange(o)
    if (!o) {
      setResult(null)
      create.reset()
    }
  }
  const copy = result ? ACTION_COPY[result.action] : null

  return (
    <Dialog
      open={open}
      onOpenChange={close}
      title="Add a memory"
      description="Tell Sentient something about you. It checks for duplicates and updates what it already knows."
      modalLock={!!content}
      footer={
        <>
          <Button variant="ghost" onClick={() => close(false)}>
            {result ? 'Done' : 'Cancel'}
          </Button>
          <Button variant="primary" leftIcon={<IconPlus size={15} />} loading={create.isPending} disabled={!content.trim()} onClick={() => create.mutate(content.trim())}>
            {result ? 'Add another' : 'Add memory'}
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        <Textarea
          autoFocus
          autoGrow
          minHeight={96}
          placeholder="e.g. I'm allergic to peanuts, or My sister's name is Aditi"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && content.trim()) create.mutate(content.trim())
          }}
        />
        {create.isPending && <p className="text-xs text-fg-subtle">Checking what Sentient already knows…</p>}
        {result && copy && (
          <Alert
            tone={copy.tone}
            icon={copy.icon}
            title={copy.title}
            action={
              result.id !== null && result.action !== 'DELETE' ? (
                <Button size="xs" variant="ghost" onClick={() => (close(false), onShow(result.id as number))}>
                  View
                </Button>
              ) : undefined
            }
          >
            {copy.body}
            {result.content && <div className="mt-1 text-fg">“{result.content}”</div>}
          </Alert>
        )}
      </div>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------- import
const ACCEPT = ['.pdf', '.txt', '.md', '.markdown', '.docx']

export function ImportDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const qc = useQueryClient()
  const input = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [progress, setProgress] = useState<number | null>(null)
  const [phase, setPhase] = useState<'idle' | 'uploading' | 'learning' | 'done' | 'error'>('idle')
  const [result, setResult] = useState<MemoryImportResult | null>(null)
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)
  const abort = useRef<AbortController | null>(null)

  const reset = () => {
    setFile(null)
    setProgress(null)
    setPhase('idle')
    setResult(null)
    setError('')
  }
  const close = (o: boolean) => {
    if (!o && (phase === 'uploading' || phase === 'learning')) abort.current?.abort()
    onOpenChange(o)
    if (!o) reset()
  }

  const pick = (f: File | undefined | null) => {
    if (!f) return
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
    if (!ACCEPT.includes(ext)) {
      setError(`${f.name} isn't supported. Use a PDF, Word, Markdown or text file.`)
      setPhase('error')
      return
    }
    setError('')
    setPhase('idle')
    setResult(null)
    setFile(f)
  }

  const start = async () => {
    if (!file) return
    abort.current = new AbortController()
    setPhase('uploading')
    setProgress(0)
    try {
      const r = await api.memories.import(file, {
        signal: abort.current.signal,
        onProgress: (f) => {
          setProgress(f)
          if (f >= 1) setPhase('learning')
        }
      })
      setResult(r)
      setPhase('done')
      void qc.invalidateQueries({ queryKey: qk.memories.all })
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return
      setError(errorMessage(e))
      setPhase('error')
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDrag(false)
    pick(e.dataTransfer.files?.[0])
  }
  const busy = phase === 'uploading' || phase === 'learning'

  return (
    <Dialog
      open={open}
      onOpenChange={close}
      title="Import a document"
      description="Sentient reads the document and learns facts about you from it. Your existing memories are kept."
      modalLock={busy}
      footer={
        <>
          <Button variant="ghost" onClick={() => close(false)}>
            {phase === 'done' ? 'Done' : 'Cancel'}
          </Button>
          {phase === 'done' ? (
            <Button variant="secondary" onClick={reset}>
              Import another
            </Button>
          ) : (
            <Button variant="primary" leftIcon={<IconFileImport size={15} />} disabled={!file || busy} loading={busy} onClick={() => void start()}>
              Learn from file
            </Button>
          )}
        </>
      }
    >
      <div className="space-y-3">
        <input ref={input} type="file" accept={ACCEPT.join(',')} className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
        {phase !== 'done' && (
          <div
            role="button"
            tabIndex={0}
            onClick={() => !busy && input.current?.click()}
            onKeyDown={(e) => e.key === 'Enter' && !busy && input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDrag(true)
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
            className={cn(
              'flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-8 text-center transition-colors',
              drag ? 'border-accent bg-accent/8' : 'border-border-strong bg-sunken/40 hover:border-fg-faint hover:bg-hover',
              busy && 'pointer-events-none opacity-70'
            )}
          >
            <div className="flex size-11 items-center justify-center rounded-xl border border-border bg-elevated text-accent-text shadow-soft">
              {file ? <IconFileText size={20} /> : <IconUpload size={20} />}
            </div>
            {file ? (
              <>
                <div className="text-sm font-medium text-fg">{file.name}</div>
                <div className="text-xs text-fg-subtle">{formatBytes(file.size)} · click to choose a different file</div>
              </>
            ) : (
              <>
                <div className="text-sm font-medium text-fg">Drop a file here, or click to browse</div>
                <div className="text-xs text-fg-subtle">PDF, Word (.docx), Markdown or text · a resume, notes, a bio</div>
              </>
            )}
          </div>
        )}

        {busy && (
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs text-fg-muted">
              <span>{phase === 'uploading' ? 'Uploading…' : 'Reading and learning facts… this can take a minute for long documents.'}</span>
              {phase === 'uploading' && progress !== null && <span className="tabular-nums">{Math.round(progress * 100)}%</span>}
            </div>
            <ProgressBar value={phase === 'uploading' ? progress : null} />
          </div>
        )}

        {phase === 'error' && error && (
          <Alert tone="danger" icon={<IconAlertTriangle />} title="Import failed">
            {error}
          </Alert>
        )}

        {phase === 'done' && result && (
          <div className="space-y-3">
            <Alert tone="success" icon={<IconCircleCheck />} title={`Learned from ${sourceMeta(result.source).label}`}>
              {result.added + result.updated === 0 ? 'Nothing new: Sentient already knew everything in this document.' : 'New memories are tagged with this file as their source.'}
            </Alert>
            <div className="grid grid-cols-3 gap-2">
              <Stat label="Added" value={result.added} tone="text-success" />
              <Stat label="Updated" value={result.updated} tone="text-info" />
              <Stat label="Skipped" value={result.skipped} tone="text-fg-muted" hint="duplicates" />
            </div>
          </div>
        )}
      </div>
    </Dialog>
  )
}

function Stat({ label, value, tone, hint }: { label: string; value: number; tone: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface px-3 py-2.5 text-center">
      <div className={cn('text-xl font-semibold tabular-nums', tone)}>{value}</div>
      <div className="text-xs text-fg-subtle">
        {label}
        {hint && <span className="text-fg-faint"> · {hint}</span>}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------- forget by source
export function ForgetSourceDialog({
  open,
  onOpenChange,
  sources,
  initial
}: {
  open: boolean
  onOpenChange: (o: boolean) => void
  sources: Array<{ source: string; count: number }>
  initial?: string
}) {
  const qc = useQueryClient()
  const [source, setSource] = useState(initial ?? '')
  const [typed, setTyped] = useState('')
  const chosen = sources.find((s) => s.source === (source || initial))
  const label = chosen ? sourceMeta(chosen.source).label : ''
  const remove = useMutation({
    mutationFn: (s: string) => api.memories.deleteBySource(s),
    onSuccess: (r) => {
      toast.success(`Forgot ${r.deleted} ${r.deleted === 1 ? 'memory' : 'memories'}`, { description: `Everything learned from ${label}.` })
      void qc.invalidateQueries({ queryKey: qk.memories.all })
      close(false)
    },
    onError: (e) => toast.error("Couldn't forget that source", { description: errorMessage(e) })
  })
  const close = (o: boolean) => {
    onOpenChange(o)
    if (!o) {
      setTyped('')
      setSource('')
    }
  }
  const confirmWord = 'forget'
  const armed = !!chosen && typed.trim().toLowerCase() === confirmWord

  return (
    <Dialog
      open={open}
      onOpenChange={close}
      size="md"
      title="Forget everything from a source"
      description="Removes every memory learned from one place, such as an imported file or onboarding. Conversations and files themselves are not deleted."
      footer={
        <>
          <Button variant="ghost" onClick={() => close(false)}>
            Cancel
          </Button>
          <Button variant="danger" disabled={!armed} loading={remove.isPending} onClick={() => chosen && remove.mutate(chosen.source)}>
            {chosen ? `Forget ${chosen.count} ${chosen.count === 1 ? 'memory' : 'memories'}` : 'Forget'}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <div className="max-h-56 space-y-1 overflow-y-auto">
          {sources.map((s) => {
            const meta = sourceMeta(s.source)
            const active = chosen?.source === s.source
            return (
              <button
                key={s.source}
                type="button"
                onClick={() => setSource(s.source)}
                className={cn(
                  'flex w-full items-center gap-2.5 rounded-lg border px-3 py-2 text-left text-sm transition-colors',
                  active ? 'border-danger/40 bg-danger/8 text-fg' : 'border-transparent text-fg-muted hover:bg-hover hover:text-fg'
                )}
              >
                <meta.icon size={15} className={active ? 'text-danger' : 'text-fg-subtle'} />
                <span className="min-w-0 flex-1 truncate">{meta.label}</span>
                <span className="text-xs tabular-nums text-fg-subtle">{s.count}</span>
              </button>
            )
          })}
        </div>
        {chosen && (
          <Alert tone="danger" icon={<IconAlertTriangle />} title={`This permanently forgets ${chosen.count} ${chosen.count === 1 ? 'memory' : 'memories'} from ${label}`}>
            <span>
              Type <b className="font-mono text-fg">{confirmWord}</b> to confirm.
            </span>
            <Input className="mt-2" value={typed} onChange={(e) => setTyped(e.target.value)} placeholder={confirmWord} aria-label="Type forget to confirm" />
          </Alert>
        )}
      </div>
    </Dialog>
  )
}
