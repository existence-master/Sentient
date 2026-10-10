/**
 * "Used 3 memories" under a reply or a task result: what Sentient had in mind when it answered or did the task
 * (docs/API.md §2 "Memory sources", §4 Run).
 * The list is what the engine put in front of the model or what a memory look-up returned, never the model's own
 * claim. Each item can be fixed or forgotten in place, with the same calls the Memory and About you pages use.
 */
import { IconArrowUpRight, IconBrain, IconCheck, IconChevronRight, IconTrash, IconUserHeart } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Alert, Button, IconButton, Popover, PopoverContent, PopoverTrigger, Textarea } from '@/components/ui'
import { useOptimisticMemoryActions } from '@/features/memory/hooks'
import { sourceMeta } from '@/features/memory/meta'
import { useUserModelActions } from '@/hooks/userModel'
import { errorMessage } from '@/lib/api'
import type { MemorySource } from '@/lib/types'
import { cn } from '@/lib/utils'

type Change = { kind: 'fixed'; text: string } | { kind: 'forgotten' }

const keyOf = (s: MemorySource) => `${s.kind}:${s.id}`

function sourceLabel(s: MemorySource): string {
  if (s.kind === 'insight') return s.source === 'user' ? 'Something you told it about you' : 'Something it learned about you'
  return sourceMeta(s.source || 'conversation').label
}

export function MemorySources({ sources, assistantName, task = false }: { sources: MemorySource[]; assistantName: string; task?: boolean }) {
  const [open, setOpen] = useState(false)
  const [changes, setChanges] = useState<Record<string, Change>>({})
  if (!sources.length) return null
  const n = sources.length

  return (
    <div className="text-sm">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="-ml-1 flex h-7 items-center gap-1.5 rounded-md px-1 text-xs text-fg-subtle transition-colors hover:text-fg"
      >
        <IconBrain size={13} />
        Used {n} {n === 1 ? 'memory' : 'memories'}
        <IconChevronRight size={12} className={cn('transition-transform', open && 'rotate-90')} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-1 rounded-xl border border-border bg-sunken/35">
              <p className="border-b border-border px-3.5 py-2 text-xs text-fg-subtle">
                {task
                  ? `${assistantName} had these in mind while doing this task. If one is wrong, fix it and the next run will use the change.`
                  : `${assistantName} had these in mind when it replied. If one is wrong, fix it and the next reply will use the change.`}
              </p>
              <ul className="divide-y divide-border">
                {sources.map((s) => (
                  <SourceRow
                    key={keyOf(s)}
                    source={s}
                    task={task}
                    change={changes[keyOf(s)]}
                    onChange={(c) => setChanges((prev) => ({ ...prev, [keyOf(s)]: c }))}
                  />
                ))}
              </ul>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function SourceRow({ source, task, change, onChange }: { source: MemorySource; task: boolean; change?: Change; onChange: (c: Change) => void }) {
  const navigate = useNavigate()
  const [fixing, setFixing] = useState(false)
  const forgotten = change?.kind === 'forgotten'
  const text = change?.kind === 'fixed' ? change.text : source.text
  const Icon = source.kind === 'insight' ? IconUserHeart : IconBrain

  return (
    <li className="group/src flex items-start gap-2.5 px-3.5 py-2.5">
      <Icon size={14} className="mt-0.5 shrink-0 text-fg-faint" />
      <div className="min-w-0 flex-1">
        <p className={cn('selectable break-words leading-snug', forgotten ? 'text-fg-subtle line-through decoration-fg-faint' : 'text-fg')}>{text}</p>
        <p className="mt-0.5 text-2xs text-fg-subtle">
          {forgotten ? 'Forgotten' : change?.kind === 'fixed' ? 'Fixed' : sourceLabel(source)}
          {!change && source.via === 'tool' && (task ? ' · Looked up while working' : ' · Looked up while replying')}
        </p>
      </div>
      {!forgotten && (
        <div className="flex shrink-0 items-center gap-0.5">
          <Popover open={fixing} onOpenChange={setFixing}>
            <PopoverTrigger asChild>
              <Button size="xs" variant="ghost" className="text-fg-subtle hover:text-fg">
                This is wrong
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-80">
              <FixPanel
                source={source}
                text={text}
                onDone={(c) => {
                  onChange(c)
                  setFixing(false)
                }}
              />
            </PopoverContent>
          </Popover>
          <IconButton
            size="sm"
            label={source.kind === 'insight' ? 'Open in About you' : 'Open in Memory'}
            icon={<IconArrowUpRight size={14} />}
            onClick={() => navigate(source.kind === 'insight' ? '/about' : `/memory?m=${source.id}`)}
          />
        </div>
      )}
    </li>
  )
}

/** Fix the wording or forget it, with the same calls the Memory and About you pages use. */
function FixPanel({ source, text, onDone }: { source: MemorySource; text: string; onDone: (c: Change) => void }) {
  const memory = useOptimisticMemoryActions()
  const insights = useUserModelActions()
  const [draft, setDraft] = useState(text)
  const [busy, setBusy] = useState<'save' | 'forget' | null>(null)
  const [confirmForget, setConfirmForget] = useState(false)
  const [error, setError] = useState('')
  const fact = source.kind === 'fact'
  const changed = draft.trim() && draft.trim() !== text

  const run = async (what: 'save' | 'forget') => {
    setBusy(what)
    setError('')
    try {
      if (what === 'save') {
        const next = draft.trim()
        if (fact) await memory.update.mutateAsync({ id: Number(source.id), content: next })
        else {
          await insights.patch.mutateAsync({ id: String(source.id), patch: { statement: next } })
          toast.success('Fixed', { description: 'Sentient will use your wording from now on.' })
        }
        onDone({ kind: 'fixed', text: next })
      } else {
        if (fact) await memory.remove.mutateAsync(Number(source.id))
        else {
          await insights.remove.mutateAsync(String(source.id))
          toast.success('Forgotten')
        }
        onDone({ kind: 'forgotten' })
      }
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="space-y-2.5">
      <div className="text-sm font-medium text-fg">What should it say?</div>
      <Textarea
        autoFocus
        autoGrow
        minHeight={64}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && changed && !busy) void run('save')
        }}
        aria-label="Correct this memory"
      />
      {error && (
        <Alert tone="danger" title="That didn't work">
          {error}
        </Alert>
      )}
      <div className="flex items-center justify-between gap-2">
        <Button
          size="xs"
          variant="ghost"
          className="text-danger hover:text-danger"
          leftIcon={<IconTrash size={13} />}
          loading={busy === 'forget'}
          disabled={!!busy}
          onClick={() => (confirmForget ? void run('forget') : setConfirmForget(true))}
        >
          {confirmForget ? 'Yes, forget it' : 'Forget it'}
        </Button>
        <Button size="xs" variant="primary" leftIcon={<IconCheck size={13} />} loading={busy === 'save'} disabled={!changed || !!busy} onClick={() => void run('save')}>
          Save fix
        </Button>
      </div>
      {confirmForget && !busy && <p className="text-2xs text-fg-subtle">Sentient will stop using this in chats and tasks. This can't be undone.</p>}
    </div>
  )
}
