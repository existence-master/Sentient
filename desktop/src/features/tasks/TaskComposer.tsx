/** "New task" composer: natural-language prompt, swarm toggle, model override and schedule preview. */
import {
  IconAlertTriangle,
  IconArrowUp,
  IconCalendarSearch,
  IconCheck,
  IconCpu,
  IconRadar,
  IconRefresh,
  IconSparkles,
  IconUsersGroup,
  IconX
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Badge, Button, Input, Popover, PopoverContent, PopoverTrigger, Spinner, Textarea, Tooltip } from '@/components/ui'
import { useConfig } from '@/hooks/core'
import { useLocalModels } from '@/hooks/models'
import { useTaskPreview } from '@/hooks/tasks'
import { errorMessage, isApiError } from '@/lib/api'
import { localModelValue, looksLikeEmbedding, modelShortName } from '@/lib/models'
import type { Task, TaskPreview, TaskPreviewWithScript } from '@/lib/types'
import { cn, modKey } from '@/lib/utils'
import { KIND_META, priorityMeta, taskKind } from './meta'
import { describeSchedule } from './schedule'
import { useTaskOps } from './useTaskOps'

const EXAMPLES = [
  'Every weekday at 9am, summarise my unread email and flag anything urgent',
  'When an invoice email arrives, save the PDF to Drive',
  'Tell me when the Sony WH-1000XM6 drops below ₹25,000 on Amazon',
  'Every Friday at 5pm, post a recap of merged PRs to Slack'
]

const SWARM_EXAMPLES = ['Research these 5 CRMs and compare pricing, integrations and reviews', 'Summarise each of the 8 papers in my reading list']

export function TaskComposer({ initialPrompt = '', autoFocus, className }: { initialPrompt?: string; autoFocus?: boolean; className?: string }) {
  const [prompt, setPrompt] = useState(initialPrompt)
  const [swarm, setSwarm] = useState(false)
  const [model, setModel] = useState<string | undefined>()
  const [focused, setFocused] = useState(false)
  const [previewFor, setPreviewFor] = useState<string | null>(null)
  const ref = useRef<HTMLTextAreaElement>(null)
  const preview = useTaskPreview()
  const ops = useTaskOps()
  const navigate = useNavigate()
  const creating = ops.isBusy('create')

  useEffect(() => {
    if (autoFocus) ref.current?.focus()
  }, [autoFocus])

  useEffect(() => {
    const focus = (e: Event) => {
      const text = (e as CustomEvent<string | undefined>).detail
      if (typeof text === 'string') setPrompt(text)
      ref.current?.focus()
    }
    window.addEventListener('sentient:new-task', focus)
    return () => window.removeEventListener('sentient:new-task', focus)
  }, [])

  const trimmed = prompt.trim()
  const stale = previewFor !== null && previewFor !== trimmed

  const runPreview = () => {
    if (!trimmed || swarm) return
    setPreviewFor(trimmed)
    preview.mutate(trimmed)
  }

  const clearPreview = () => {
    setPreviewFor(null)
    preview.reset()
  }

  const submit = async () => {
    if (!trimmed || creating) return
    const task = await ops.create({ prompt: trimmed, is_swarm: swarm, model })
    if (!task) return
    setPrompt('')
    clearPreview()
    toast.success(swarm ? 'Swarm started' : 'Sentient is planning your task', {
      description: swarm ? 'Agents are splitting up the work.' : "You'll be asked to approve the plan before it runs.",
      action: { label: 'Open', onClick: () => navigate(`/tasks?task=${encodeURIComponent(task.task_id)}`) }
    })
  }

  const examples = swarm ? SWARM_EXAMPLES : EXAMPLES
  const showExamples = focused && !trimmed

  return (
    <div
      className={cn(
        'group/composer relative rounded-2xl border border-border-strong bg-elevated shadow-soft transition-[border-color,box-shadow] duration-200 focus-within:border-accent/45 focus-within:shadow-glow',
        className
      )}
    >
      <div className="flex items-start gap-3 px-4 pt-3.5">
        <span
          className={cn(
            'mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg transition-colors',
            swarm ? 'bg-info/12 text-info' : 'bg-accent/12 text-accent-text'
          )}
        >
          {swarm ? <IconUsersGroup size={16} /> : <IconSparkles size={16} />}
        </span>
        <Textarea
          ref={ref}
          autoGrow
          maxHeight={180}
          rows={1}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setTimeout(() => setFocused(false), 120)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              void submit()
            }
            if (e.key === 'Escape') ref.current?.blur()
          }}
          placeholder={swarm ? 'Describe a big job to split across agents…' : 'Describe a task. “Every morning at 8, summarise my inbox”'}
          aria-label="Describe a new task"
          className="min-h-8 border-0 bg-transparent px-0 py-1 text-md shadow-none hover:border-0 focus:border-0 focus:ring-0"
        />
      </div>

      <AnimatePresence initial={false}>
        {showExamples && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="flex flex-wrap gap-1.5 px-4 pb-1 pl-14 pt-2">
              {examples.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => setPrompt(ex)}
                  className="rounded-full border border-border bg-surface px-2.5 py-1 text-xs text-fg-muted transition-colors hover:border-border-strong hover:text-fg"
                >
                  {ex}
                </button>
              ))}
            </div>
          </motion.div>
        )}
        {previewFor !== null && !swarm && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="px-4 pb-1 pl-14 pt-2.5">
              <PreviewCard
                loading={preview.isPending}
                error={preview.error}
                data={preview.data}
                stale={stale}
                onRefresh={runPreview}
                onClose={clearPreview}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex items-center gap-1.5 px-3 pb-2.5 pt-1.5 @xl:pl-13">
        <Tooltip content={swarm ? 'Single agent' : 'Run in parallel with multiple agents'}>
          <button
            type="button"
            aria-pressed={swarm}
            onClick={() => {
              setSwarm((s) => !s)
              clearPreview()
            }}
            className={cn(
              'flex h-7 items-center gap-1.5 rounded-lg border px-2 text-xs font-medium transition-colors',
              swarm ? 'border-info/30 bg-info/12 text-info' : 'border-transparent text-fg-subtle hover:bg-hover hover:text-fg'
            )}
          >
            <IconUsersGroup size={15} />
            <span className="hidden @xl:inline">Parallel agents</span>
            {swarm && <IconCheck size={12} />}
          </button>
        </Tooltip>
        <TaskModelPicker value={model} onChange={setModel} />
        <span className="flex-1" />
        <span className="hidden text-2xs text-fg-faint @3xl:inline">Enter to create · Shift+Enter for a new line</span>
        {!swarm && (
          <Tooltip content="See how Sentient reads the name and schedule">
            <Button size="sm" variant="ghost" leftIcon={<IconCalendarSearch size={15} />} disabled={!trimmed || preview.isPending} onClick={runPreview} aria-label="Preview">
              <span className="hidden @xl:inline">Preview</span>
            </Button>
          </Tooltip>
        )}
        <Button size="sm" variant="primary" loading={creating} disabled={!trimmed} onClick={() => void submit()} rightIcon={<IconArrowUp size={15} />}>
          {swarm ? 'Start swarm' : 'Create task'}
        </Button>
      </div>
    </div>
  )
}

function PreviewCard({
  loading,
  error,
  data,
  stale,
  onRefresh,
  onClose
}: {
  loading: boolean
  error: unknown
  data: TaskPreview | undefined
  stale: boolean
  onRefresh: () => void
  onClose: () => void
}) {
  let body: ReactNode
  if (loading) {
    body = (
      <div className="flex items-center gap-2.5 text-sm text-fg-muted">
        <Spinner size={14} className="text-accent-text" />
        Reading your request…
      </div>
    )
  } else if (error) {
    const unavailable = isApiError(error) && error.status === 503
    body = (
      <div className="flex items-start gap-2.5 text-sm">
        <IconAlertTriangle size={16} className="mt-0.5 shrink-0 text-warning" />
        <div className="min-w-0">
          <div className="font-medium text-fg">{unavailable ? 'Preview needs a working model' : "Couldn't preview this task"}</div>
          <div className="mt-0.5 line-clamp-2 text-xs text-fg-subtle">{errorMessage(error)}. You can still create the task.</div>
        </div>
      </div>
    )
  } else if (data) {
    const pb = data as TaskPreviewWithScript
    const script = pb.task_type === 'script' || !!pb.script
    const kind = taskKind({ task_type: (script ? 'script' : 'single') as Task['task_type'], schedule: data.schedule })
    const KindIcon = KIND_META[kind].icon
    const sched = describeSchedule(data.schedule)
    const pr = priorityMeta(data.priority)
    body = (
      <div className="flex items-start gap-3">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border bg-surface text-accent-text">
          <KindIcon size={16} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="truncate text-sm font-medium text-fg">{data.name}</span>
            <Badge size="xs">{KIND_META[kind].label}</Badge>
            {data.priority !== 1 && <Badge size="xs" tone={pr.tone}>{pr.label} priority</Badge>}
          </div>
          <div className="mt-0.5 text-sm text-fg-muted">
            {sched.text}
            {sched.detail && <span className="text-fg-subtle"> · matching {sched.detail}</span>}
            {sched.zone && <span className="text-fg-subtle"> ({sched.zone})</span>}
          </div>
          {script && (
            <div className="mt-2 flex items-start gap-2 rounded-lg border border-info/20 bg-info/[0.06] px-2.5 py-2 text-xs text-fg-muted">
              <IconRadar size={14} className="mt-px shrink-0 text-info" />
              <span>Sentient will check this with a small script, so no AI is needed each time. You’ll see the code before you approve it.</span>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className={cn('relative rounded-xl border border-border bg-surface px-3 py-2.5 pr-16', stale && 'opacity-70')}>
      {body}
      <div className="absolute right-1.5 top-1.5 flex items-center gap-0.5">
        {(stale || !!error) && !loading && (
          <Tooltip content="Refresh preview">
            <button type="button" onClick={onRefresh} className="flex size-6 items-center justify-center rounded-md text-fg-subtle hover:bg-hover hover:text-fg">
              <IconRefresh size={13} />
            </button>
          </Tooltip>
        )}
        <button type="button" aria-label="Close preview" onClick={onClose} className="flex size-6 items-center justify-center rounded-md text-fg-subtle hover:bg-hover hover:text-fg">
          <IconX size={13} />
        </button>
      </div>
    </div>
  )
}

/** Executor model override for a new task. */
export function TaskModelPicker({ value, onChange }: { value: string | undefined; onChange: (v: string | undefined) => void }) {
  const [open, setOpen] = useState(false)
  const [custom, setCustom] = useState('')
  const config = useConfig()
  const local = useLocalModels()
  const roles = config.data?.models.roles
  const executor = roles?.executor || roles?.primary
  const localOptions = (local.data?.ollama.models ?? []).filter((m) => !(m.is_embedding ?? looksLikeEmbedding(m.name))).map((m) => localModelValue('ollama', m, false))
  const roleOptions = Array.from(new Set([roles?.primary, roles?.planner, roles?.fast].filter((m): m is string => !!m && m !== executor)))

  const pick = (v: string | undefined) => {
    onChange(v)
    setOpen(false)
    setCustom('')
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <Tooltip content={value ? `Runs with ${value}` : `Runs with the executor model${executor ? ` (${modelShortName(executor)})` : ''}`}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className={cn(
              'flex h-7 max-w-48 items-center gap-1.5 rounded-lg px-2 text-xs font-medium transition-colors',
              value ? 'bg-accent/12 text-accent-text hover:bg-accent/18' : 'text-fg-subtle hover:bg-hover hover:text-fg'
            )}
          >
            <IconCpu size={15} />
            <span className={cn('truncate', !value && 'hidden @xl:inline')}>{value ? modelShortName(value) : 'Default model'}</span>
            {value && (
              <span
                role="button"
                aria-label="Use the default model"
                onClick={(e) => {
                  e.stopPropagation()
                  onChange(undefined)
                }}
                className="-mr-0.5 flex rounded p-0.5 hover:bg-accent/20"
              >
                <IconX size={11} />
              </span>
            )}
          </button>
        </PopoverTrigger>
      </Tooltip>
      <PopoverContent align="start" className="w-80 p-1.5">
        <div className="px-2.5 pb-1.5 pt-1 text-xs text-fg-subtle">Model that carries out this task</div>
        <ModelRow active={!value} onClick={() => pick(undefined)} hint={modelShortName(executor)}>
          Default (executor)
        </ModelRow>
        {roleOptions.map((m) => (
          <ModelRow key={m} active={value === m} onClick={() => pick(m)}>
            <span className="font-mono text-xs">{modelShortName(m)}</span>
          </ModelRow>
        ))}
        {localOptions.length > 0 && (
          <>
            <div className="mx-2.5 mb-1 mt-2 text-2xs font-medium uppercase tracking-wide text-fg-faint">Installed locally</div>
            <div className="max-h-40 overflow-y-auto">
              {localOptions.map((m) => (
                <ModelRow key={m} active={value === m} onClick={() => pick(m)}>
                  <span className="font-mono text-xs">{modelShortName(m)}</span>
                </ModelRow>
              ))}
            </div>
          </>
        )}
        <form
          className="mt-1.5 border-t border-border p-1.5 pt-2"
          onSubmit={(e) => {
            e.preventDefault()
            if (custom.trim()) pick(custom.trim())
          }}
        >
          <Input size="sm" value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="Any model, e.g. anthropic/claude-sonnet-5" className="font-mono" />
        </form>
      </PopoverContent>
    </Popover>
  )
}

function ModelRow({ active, onClick, hint, children }: { active: boolean; onClick: () => void; hint?: string; children: ReactNode }) {
  return (
    <button type="button" onClick={onClick} className="flex h-8 w-full items-center gap-2 rounded-lg px-2.5 text-left text-sm text-fg hover:bg-active">
      <span className="min-w-0 flex-1 truncate">{children}</span>
      {hint && <span className="truncate font-mono text-2xs text-fg-subtle">{hint}</span>}
      <IconCheck size={14} className={cn('shrink-0 text-accent-text', !active && 'invisible')} />
    </button>
  )
}

export const composerShortcutHint = `${modKey}+Enter`
