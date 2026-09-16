/** Task detail: header with state-aware actions, then Overview / Runs / Conversation. Used as side panel and full page. */
import {
  IconAlertCircle,
  IconArchive,
  IconArrowLeft,
  IconArrowsMaximize,
  IconCalendarCog,
  IconCheck,
  IconCopy,
  IconDots,
  IconFileDescription,
  IconHelpCircle,
  IconPencil,
  IconPlayerPause,
  IconPlayerPlay,
  IconPlayerStop,
  IconPlugConnected,
  IconRepeat,
  IconTrash,
  IconX
} from '@tabler/icons-react'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router'
import { toast } from 'sonner'
import {
  Alert,
  Badge,
  Button,
  ConfirmDialog,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  IconButton,
  Select,
  Skeleton,
  Switch,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Textarea,
  Tooltip
} from '@/components/ui'
import { useTask, useTasks } from '@/hooks/tasks'
import { isApiError } from '@/lib/api'
import { modelShortName } from '@/lib/models'
import type { Task, TaskSchedule } from '@/lib/types'
import { cn, copyText, formatDateTime, relativeTime } from '@/lib/utils'
import { taskHeadline } from '../ListView'
import { useMissingIntegrations, useNow, useUserTimezone, type MissingIntegration } from '../hooks'
import { KIND_META, PRIORITY_META, SOURCE_META, displayName, latestRun, processingRun, taskKind, taskSource } from '../meta'
import { KindTile, TaskStatusBadge } from '../parts'
import { describeSchedule, filterRules, nextRunDate, upcomingPhrase } from '../schedule'
import { useTaskOps } from '../useTaskOps'
import { HookName } from '@/features/automations/HookLabel'
import { isScriptJob, scriptOf } from '@/lib/leap/types-b'
import { ScriptJobSection } from './ScriptJob'
import { Clarifications } from './Clarifications'
import { PlanSection } from './PlanSection'
import { RunHistory, RunResult } from './RunHistory'
import { FilterRulesView, ScheduleEditor } from './ScheduleEditor'
import { SwarmSection } from './SwarmSection'
import { TaskChat } from './TaskChat'

type Tab = 'overview' | 'runs' | 'chat'

export function TaskDetail({ taskId, layout, onClose }: { taskId: string; layout: 'panel' | 'page'; onClose?: () => void }) {
  const list = useTasks()
  const detail = useTask(taskId)
  const task = detail.data ?? list.data?.find((t) => t.task_id === taskId)
  const navigate = useNavigate()

  if (!task) {
    if (detail.isLoading || list.isLoading) return <DetailSkeleton layout={layout} onClose={onClose} />
    const gone = isApiError(detail.error) && detail.error.status === 404
    return (
      <div className="flex h-full flex-col">
        {layout === 'panel' && (
          <div className="flex justify-end p-2">
            <IconButton label="Close" icon={<IconX size={16} />} onClick={onClose} />
          </div>
        )}
        <EmptyState
          className="flex-1"
          icon={<IconHelpCircle />}
          title={gone || !detail.error ? 'This task no longer exists' : "Couldn't load this task"}
          description={gone || !detail.error ? 'It may have been deleted.' : String((detail.error as Error)?.message ?? '')}
          action={
            <Button size="sm" onClick={() => (layout === 'page' ? navigate('/tasks') : onClose?.())}>
              Back to tasks
            </Button>
          }
        />
      </div>
    )
  }
  return <TaskDetailBody key={task.task_id} task={task} layout={layout} onClose={onClose} />
}

function defaultTab(task: Task): Tab {
  if (task.status === 'clarification_pending' || task.status === 'approval_pending' || task.status === 'planning') return 'overview'
  if (task.status === 'processing') return 'runs'
  if (['completed', 'completed_with_errors', 'error', 'cancelled'].includes(task.status) && task.runs.length && task.task_type !== 'swarm') return 'runs'
  return 'overview'
}

function TaskDetailBody({ task, layout, onClose }: { task: Task; layout: 'panel' | 'page'; onClose?: () => void }) {
  const [params, setParams] = useSearchParams()
  const initial = (['overview', 'runs', 'chat'].includes(params.get('tab') ?? '') ? params.get('tab') : null) as Tab | null
  const [tab, setTab] = useState<Tab>(initial ?? defaultTab(task))
  const tz = useUserTimezone()
  const ops = useTaskOps()
  const missing = useMissingIntegrations(task)
  const id = task.task_id

  // Follow the task into Runs when a run starts while it's open.
  const running = task.status === 'processing'
  useEffect(() => {
    if (running) setTab((t) => (t === 'overview' && task.task_type !== 'swarm' ? 'runs' : t))
  }, [running, task.task_type])

  const changeTab = (t: string) => {
    setTab(t as Tab)
    if (layout === 'page') {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          next.set('tab', t)
          next.delete('run')
          return next
        },
        { replace: true }
      )
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-surface">
      <DetailHeader task={task} layout={layout} onClose={onClose} ops={ops} missing={missing} onShowRuns={() => changeTab('runs')} />
      <div className="@container min-h-0 flex-1 overflow-y-auto">
        <Tabs value={tab} onValueChange={changeTab} className={cn('mx-auto w-full', layout === 'page' ? 'max-w-6xl px-8' : 'px-5')}>
          <TabsList className="sticky top-0 z-20 -mx-1 bg-surface/95 px-1 pt-1 backdrop-blur">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="runs">
              Runs
              {task.runs.length > 0 && <span className="rounded-full bg-active px-1.5 text-2xs tabular-nums text-fg-muted">{task.runs.length}</span>}
              {running && <span className="size-1.5 animate-pulse rounded-full bg-info" />}
            </TabsTrigger>
            <TabsTrigger value="chat">
              Conversation
              {task.chat_history.length > 0 && <span className="rounded-full bg-active px-1.5 text-2xs tabular-nums text-fg-muted">{task.chat_history.length}</span>}
            </TabsTrigger>
          </TabsList>

          <div className={cn('grid gap-6 py-5', layout === 'page' && '@4xl:grid-cols-[minmax(0,1fr)_17.5rem]')}>
            <div className="min-w-0">
              <TabsContent value="overview" className="space-y-6">
                <Overview task={task} tz={tz} ops={ops} missing={missing} layout={layout} onShowRuns={() => changeTab('runs')} />
              </TabsContent>
              <TabsContent value="runs">
                <RunHistory task={task} tz={tz} focusRunId={params.get('run')} onCancelRun={(runId) => void ops.cancelRun(id, runId)} cancelling={ops.isBusy('cancelRun', id)} />
              </TabsContent>
              <TabsContent value="chat">
                <TaskChat task={task} sending={ops.isBusy('chat', id)} onSend={(m) => void ops.chat(id, m)} />
              </TabsContent>
            </div>
            {layout === 'page' && (
              <aside className="hidden @4xl:block">
                <div className="sticky top-14">
                  <PropertiesCard task={task} tz={tz} ops={ops} />
                </div>
              </aside>
            )}
          </div>
        </Tabs>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------- header
type Ops = ReturnType<typeof useTaskOps>

function DetailHeader({ task, layout, onClose, ops, missing, onShowRuns }: { task: Task; layout: 'panel' | 'page'; onClose?: () => void; ops: Ops; missing: MissingIntegration[]; onShowRuns: () => void }) {
  const navigate = useNavigate()
  const location = useLocation()
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState(task.name)
  const kind = taskKind(task)
  const K = KIND_META[kind]
  const source = SOURCE_META[taskSource(task)]
  const S = source.icon

  const commitName = () => {
    setEditing(false)
    const n = name.trim()
    if (n && n !== task.name) void ops.update(task.task_id, { name: n })
    else setName(task.name)
  }

  return (
    <header className={cn('shrink-0 border-b border-border', layout === 'page' ? 'bg-surface' : 'bg-surface')}>
      <div className={cn('mx-auto w-full', layout === 'page' ? 'max-w-6xl px-8 pb-4 pt-4' : 'px-5 pb-4 pt-3')}>
        <div className="mb-2.5 flex h-7 items-center gap-1">
          {layout === 'page' ? (
            <Button size="sm" variant="ghost" className="-ml-2" leftIcon={<IconArrowLeft size={15} />} onClick={() => (location.key !== 'default' ? navigate(-1) : navigate('/tasks'))}>
              Tasks
            </Button>
          ) : (
            <span className="text-2xs font-medium uppercase tracking-wider text-fg-subtle">Task</span>
          )}
          <span className="flex-1" />
          {layout === 'panel' && (
            <>
              <IconButton size="sm" label="Open full page" icon={<IconArrowsMaximize size={15} />} onClick={() => navigate(`/tasks/${encodeURIComponent(task.task_id)}`)} />
              <IconButton size="sm" label="Close" shortcut="Esc" icon={<IconX size={16} />} onClick={onClose} />
            </>
          )}
        </div>
        <div className={cn('flex gap-3.5', layout === 'page' ? 'flex-row items-start' : 'flex-col')}>
          <div className="flex min-w-0 flex-1 items-start gap-3.5">
            <KindTile task={task} size="lg" />
            <div className="min-w-0 flex-1">
              {editing ? (
                <input
                  autoFocus
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onBlur={commitName}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commitName()
                    if (e.key === 'Escape') {
                      setName(task.name)
                      setEditing(false)
                    }
                  }}
                  className="w-full rounded-md border border-accent/50 bg-field px-1.5 py-0.5 text-lg font-semibold tracking-tight text-fg outline-none ring-3 ring-accent/15"
                  aria-label="Task name"
                />
              ) : (
                <button type="button" onClick={() => (setName(task.name), setEditing(true))} className="group -mx-1.5 flex max-w-full items-start gap-1.5 rounded-md px-1.5 py-0.5 text-left hover:bg-hover" title="Rename">
                  <h2 className={cn('font-semibold leading-snug tracking-tight text-fg', layout === 'page' ? 'text-xl' : 'text-lg')}>{displayName(task)}</h2>
                  <IconPencil size={14} className="mt-1.5 shrink-0 text-fg-faint opacity-0 transition-opacity group-hover:opacity-100" />
                </button>
              )}
              <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                <TaskStatusBadge task={task} />
                <Badge icon={<K.icon />}>{K.label}</Badge>
                <Select
                  size="sm"
                  aria-label="Priority"
                  value={String(task.priority ?? 1)}
                  onValueChange={(v) => void ops.update(task.task_id, { priority: Number(v) }, 'Priority updated')}
                  options={[0, 1, 2].map((p) => ({ value: String(p), label: `${PRIORITY_META[p].label} priority`, icon: <span className={cn('size-2 rounded-full', PRIORITY_META[p].dot)} /> }))}
                  className="h-5.5 w-auto gap-1 rounded-full border-border bg-active px-2 text-xs font-medium text-fg-muted"
                />
                <Badge icon={<S />}>{source.label}</Badge>
                {task.model && (
                  <Tooltip content={task.model}>
                    <span>
                      <Badge tone="accent">{modelShortName(task.model)}</Badge>
                    </span>
                  </Tooltip>
                )}
              </div>
            </div>
          </div>
          <ActionBar task={task} ops={ops} missing={missing} layout={layout} onClose={onClose} onShowRuns={onShowRuns} />
        </div>
      </div>
    </header>
  )
}

function ActionBar({ task, ops, missing, layout, onClose, onShowRuns }: { task: Task; ops: Ops; missing: MissingIntegration[]; layout: 'panel' | 'page'; onClose?: () => void; onShowRuns: () => void }) {
  const navigate = useNavigate()
  const [confirmDelete, setConfirmDelete] = useState(false)
  const id = task.task_id
  const kind = taskKind(task)
  const run = processingRun(task)
  const schedulable = kind === 'recurring' || kind === 'triggered' || kind === 'script' || task.status === 'pending'
  const canRunNow = !['planning', 'clarification_pending', 'archived', 'declined'].includes(task.status) && (task.plan.length > 0 || task.task_type === 'swarm' || isScriptJob(task)) && !(task.task_type === 'swarm' && task.status === 'processing') && !(task.status === 'processing' && !schedulable)

  const rerun = async () => {
    const copy = await ops.rerun(id)
    if (copy) {
      toast.success('Created a fresh copy', { description: 'Sentient is planning it now.', action: { label: 'Open', onClick: () => navigate(`/tasks?task=${encodeURIComponent(copy.task_id)}`) } })
    }
  }

  const buttons: ReactNode[] = []
  switch (task.status) {
    case 'approval_pending': {
      const approveLabel = kind === 'script' ? 'Approve & start watching' : kind === 'recurring' || kind === 'triggered' ? 'Approve & activate' : kind === 'scheduled' ? 'Approve & schedule' : 'Approve & run'
      buttons.push(
        <Button key="decline" size="sm" variant="ghost" leftIcon={<IconX size={15} />} loading={ops.isBusy('decline', id)} onClick={() => void ops.decline(id)}>
          Decline
        </Button>,
        <Tooltip key="approve" content={missing.length ? `Connect ${missing.map((m) => m.label).join(', ')} first` : ''}>
          <span>
            <Button size="sm" variant="primary" leftIcon={<IconCheck size={15} />} disabled={missing.length > 0 || !task.plan.length} loading={ops.isBusy('approve', id)} onClick={() => void ops.approve(id)}>
              {approveLabel}
            </Button>
          </span>
        </Tooltip>
      )
      break
    }
    case 'clarification_pending':
    case 'planning':
      buttons.push(
        <Button key="decline" size="sm" variant="ghost" leftIcon={<IconX size={15} />} loading={ops.isBusy('decline', id)} onClick={() => void ops.decline(id)}>
          {task.status === 'planning' ? 'Stop planning' : 'Decline'}
        </Button>
      )
      break
    case 'processing':
      if (run)
        buttons.push(
          <Button key="cancel" size="sm" variant="danger" leftIcon={<IconPlayerStop size={15} />} loading={ops.isBusy('cancelRun', id)} onClick={() => void ops.cancelRun(id, run.run_id)}>
            Cancel run
          </Button>
        )
      break
    case 'active':
    case 'pending':
      buttons.push(
        <Button
          key="pause"
          size="sm"
          variant="secondary"
          leftIcon={task.enabled ? <IconPlayerPause size={15} /> : <IconPlayerPlay size={15} />}
          loading={ops.isBusy('enabled', id)}
          onClick={() => void ops.setEnabled(id, !task.enabled)}
        >
          {task.enabled ? 'Pause' : 'Resume'}
        </Button>,
        <Button key="run" size="sm" variant="primary" leftIcon={<IconPlayerPlay size={15} />} loading={ops.isBusy('runNow', id)} onClick={() => void ops.runNow(id).then((t) => t && onShowRuns())}>
          Run now
        </Button>
      )
      break
    case 'error':
      buttons.push(
        canRunNow ? (
          <Button key="retry" size="sm" variant="primary" leftIcon={<IconRepeat size={15} />} loading={ops.isBusy('runNow', id)} onClick={() => void ops.runNow(id).then((t) => t && onShowRuns())}>
            Try again
          </Button>
        ) : (
          <Button key="rerun" size="sm" variant="primary" leftIcon={<IconRepeat size={15} />} loading={ops.isBusy('rerun', id)} onClick={() => void rerun()}>
            Rerun
          </Button>
        )
      )
      break
    case 'completed':
    case 'completed_with_errors':
    case 'cancelled':
      buttons.push(
        <Button key="again" size="sm" variant="secondary" leftIcon={<IconPlayerPlay size={15} />} loading={ops.isBusy('runNow', id)} disabled={!canRunNow} onClick={() => void ops.runNow(id).then((t) => t && onShowRuns())}>
          Run again
        </Button>
      )
      break
    default:
      buttons.push(
        <Button key="rerun" size="sm" variant="secondary" leftIcon={<IconRepeat size={15} />} loading={ops.isBusy('rerun', id)} onClick={() => void rerun()}>
          Rerun
        </Button>
      )
  }

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-2">
      {buttons}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <IconButton variant="secondary" size="sm" label="More actions" icon={<IconDots size={16} />} />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-52">
          {canRunNow && task.status !== 'active' && task.status !== 'pending' && !['completed', 'completed_with_errors', 'cancelled', 'error'].includes(task.status) && (
            <DropdownMenuItem icon={<IconPlayerPlay />} onSelect={() => void ops.runNow(id)}>
              Run now
            </DropdownMenuItem>
          )}
          <DropdownMenuItem icon={<IconRepeat />} onSelect={() => void rerun()}>
            Rerun as a new task
          </DropdownMenuItem>
          {schedulable && task.status !== 'active' && task.status !== 'pending' && (
            <DropdownMenuItem icon={task.enabled ? <IconPlayerPause /> : <IconPlayerPlay />} onSelect={() => void ops.setEnabled(id, !task.enabled)}>
              {task.enabled ? 'Pause' : 'Resume'}
            </DropdownMenuItem>
          )}
          {layout === 'panel' && (
            <DropdownMenuItem icon={<IconArrowsMaximize />} onSelect={() => navigate(`/tasks/${encodeURIComponent(id)}`)}>
              Open full page
            </DropdownMenuItem>
          )}
          <DropdownMenuItem icon={<IconCopy />} onSelect={() => void copyText(id).then((ok) => ok && toast.message('Task ID copied'))}>
            Copy task ID
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          {task.status !== 'archived' && (
            <DropdownMenuItem icon={<IconArchive />} onSelect={() => void ops.archive(id)}>
              Archive
            </DropdownMenuItem>
          )}
          <DropdownMenuItem icon={<IconTrash />} danger onSelect={() => setConfirmDelete(true)}>
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title="Delete this task?"
        description={`“${displayName(task)}”, its ${task.runs.length} run${task.runs.length === 1 ? '' : 's'} and their logs will be permanently removed.`}
        confirmLabel="Delete task"
        onConfirm={async () => {
          await ops.remove(id)
          if (layout === 'page') navigate('/tasks', { replace: true })
          else onClose?.()
        }}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------- overview
function Overview({ task, tz, ops, missing, layout, onShowRuns }: { task: Task; tz: string; ops: Ops; missing: MissingIntegration[]; layout: 'panel' | 'page'; onShowRuns: () => void }) {
  const navigate = useNavigate()
  const id = task.task_id
  const last = latestRun(task)
  const kind = taskKind(task)

  return (
    <>
      {task.status === 'planning' && (
        <div className="flex items-center gap-3 rounded-xl border border-info/20 bg-info/5 px-4 py-3">
          <span className="relative flex size-8 shrink-0 items-center justify-center rounded-lg bg-info/12 text-info">
            <span className="absolute inset-0 animate-ping rounded-lg bg-info/20" />
            <IconCalendarCog size={16} />
          </span>
          <div className="min-w-0">
            <div className="text-sm font-medium text-fg">{taskHeadline(task)}</div>
            <div className="text-xs text-fg-subtle">You'll get a notification when it's ready for review.</div>
          </div>
        </div>
      )}

      {task.status === 'error' && task.error && (
        <Alert tone="danger" icon={<IconAlertCircle />} title="Something went wrong">
          {task.error}
        </Alert>
      )}

      {missing.length > 0 && (
        <Alert
          tone="warning"
          icon={<IconPlugConnected />}
          title={`Connect ${missing.map((m) => m.label).join(' and ')} ${task.status === 'approval_pending' ? 'to approve this plan' : 'so this task can run'}`}
          action={
            <Button size="xs" variant="secondary" onClick={() => navigate(`/integrations?connect=${encodeURIComponent(missing[0].id)}`)}>
              Connect
            </Button>
          }
        >
          Sentient needs access to {missing.length === 1 ? 'this app' : 'these apps'} to carry out the plan.
        </Alert>
      )}

      {!task.enabled && ['active', 'pending'].includes(task.status) && (
        <Alert
          tone="neutral"
          icon={<IconPlayerPause />}
          title="This task is paused"
          action={
            <Button size="xs" variant="secondary" loading={ops.isBusy('enabled', id)} onClick={() => void ops.setEnabled(id, true)}>
              Resume
            </Button>
          }
        >
          It won't run on its schedule or triggers until you resume it.
        </Alert>
      )}

      {task.clarifying_questions.length > 0 && (
        <Clarifications questions={task.clarifying_questions} pending={task.status === 'clarification_pending'} submitting={ops.isBusy('answer', id)} onSubmit={(a) => void ops.answer(id, a)} />
      )}

      {layout === 'page' ? (
        <div className="@4xl:hidden">
          <PropertiesCard task={task} tz={tz} ops={ops} compact />
        </div>
      ) : (
        <PropertiesCard task={task} tz={tz} ops={ops} compact />
      )}

      <DescriptionCard task={task} onSave={(description) => ops.update(id, { description }, 'Description saved')} />

      <ScheduleCard task={task} tz={tz} onSave={(schedule) => ops.update(id, { schedule }, 'Schedule updated')} saving={ops.isBusy('update', id)} />

      {task.task_type === 'swarm' && task.swarm_details && <SwarmSection swarm={task.swarm_details} tz={tz} running={task.status === 'processing' || task.status === 'planning'} />}

      {isScriptJob(task) && <ScriptJobSection task={task} />}

      {task.task_type !== 'swarm' && (!isScriptJob(task) || (task.plan.length > 0 && scriptOf(task)?.then === 'run')) && (
        <PlanSection
          plan={task.plan}
          editable={task.status === 'approval_pending'}
          saving={ops.isBusy('update', id)}
          onSave={(plan) => ops.update(id, { plan }, 'Plan saved')}
          emptyText={task.status === 'planning' ? 'The plan will appear here in a moment.' : 'This task has no plan.'}
        />
      )}

      {last && task.status !== 'processing' && (
        <section className="space-y-2.5">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-fg">Latest result</h3>
            <span className="text-xs text-fg-subtle">{relativeTime(last.finished_at ?? last.execution_start_time ?? last.created_at)}</span>
            <span className="flex-1" />
            <Button size="xs" variant="ghost" onClick={onShowRuns}>
              {task.runs.length > 1 ? `All ${task.runs.length} runs` : 'View run'}
            </Button>
          </div>
          {last.result ? (
            <RunResult result={last.result} />
          ) : last.error ? (
            <Alert tone="danger" icon={<IconAlertCircle />}>
              {last.error}
            </Alert>
          ) : (
            <p className="text-sm text-fg-subtle">No result recorded for this run.</p>
          )}
        </section>
      )}
      {kind === 'triggered' && task.runs.length === 0 && task.status === 'active' && (
        <p className="text-sm text-fg-subtle">Waiting for the first matching event.</p>
      )}
    </>
  )
}

function DescriptionCard({ task, onSave }: { task: Task; onSave: (d: string) => Promise<unknown> }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(task.description)
  const [saving, setSaving] = useState(false)
  useEffect(() => {
    if (!editing) setDraft(task.description)
  }, [task.description, editing])

  return (
    <section className="group space-y-2">
      <div className="flex items-center gap-2">
        <IconFileDescription size={16} className="text-fg-subtle" />
        <h3 className="text-sm font-semibold text-fg">Description</h3>
        <span className="flex-1" />
        {editing ? (
          <div className="flex items-center gap-1.5">
            <Button size="xs" variant="ghost" onClick={() => (setDraft(task.description), setEditing(false))} disabled={saving}>
              Cancel
            </Button>
            <Button
              size="xs"
              variant="primary"
              loading={saving}
              onClick={async () => {
                setSaving(true)
                await onSave(draft.trim())
                setSaving(false)
                setEditing(false)
              }}
            >
              Save
            </Button>
          </div>
        ) : (
          <Button size="xs" variant="ghost" leftIcon={<IconPencil size={13} />} className="opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100" onClick={() => setEditing(true)}>
            Edit
          </Button>
        )}
      </div>
      {editing ? (
        <Textarea autoFocus autoGrow minHeight={80} value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => e.key === 'Escape' && setEditing(false)} />
      ) : (
        <p onDoubleClick={() => setEditing(true)} className="selectable whitespace-pre-wrap text-sm leading-relaxed text-fg-muted">
          {task.description || <span className="text-fg-subtle">No description.</span>}
        </p>
      )}
    </section>
  )
}

function ScheduleCard({ task, tz, onSave, saving }: { task: Task; tz: string; onSave: (s: TaskSchedule) => Promise<unknown>; saving: boolean }) {
  const [draft, setDraft] = useState<TaskSchedule | null>(null)
  const now = useNow(60_000)
  const swarm = task.task_type === 'swarm'
  const d = describeSchedule(task.schedule, { swarm })
  const next = nextRunDate(task, now)
  const K = KIND_META[taskKind(task)]
  const rules = task.schedule?.type === 'triggered' ? filterRules(task.schedule.filter) : null
  const editable = !swarm && !['processing', 'archived'].includes(task.status)

  return (
    <section className="space-y-2">
      <div className="flex items-center gap-2">
        <K.icon size={16} className="text-fg-subtle" />
        <h3 className="text-sm font-semibold text-fg">Schedule</h3>
        <span className="flex-1" />
        {draft ? (
          <div className="flex items-center gap-1.5">
            <Button size="xs" variant="ghost" onClick={() => setDraft(null)} disabled={saving}>
              Cancel
            </Button>
            <Button size="xs" variant="primary" loading={saving} onClick={async () => (await onSave(draft), setDraft(null))}>
              Save schedule
            </Button>
          </div>
        ) : (
          editable && (
            <Button size="xs" variant="ghost" leftIcon={<IconPencil size={13} />} onClick={() => setDraft(task.schedule ?? { type: 'once', run_at: null, timezone: tz })}>
              Change
            </Button>
          )
        )}
      </div>
      {draft ? (
        <div className="rounded-xl border border-border-strong bg-elevated p-4">
          <ScheduleEditor value={draft} onChange={setDraft} defaultTimezone={tz} />
        </div>
      ) : (
        <div className="rounded-xl border border-border bg-surface px-4 py-3">
          <div className="text-md font-medium text-fg">{d.text}</div>
          {task.schedule?.type === 'triggered' && task.schedule.source === 'webhook' && <HookName id={task.schedule.event} className="mt-1.5" />}
          {rules && (
            <div className="mt-2">
              <FilterRulesView rules={rules} />
            </div>
          )}
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-fg-subtle">
            {d.zone && <span>{d.zone}</span>}
            {next && (
              <span className="text-fg-muted">
                Next run <span className="font-medium text-fg">{upcomingPhrase(next, tz, now)}</span>
              </span>
            )}
            {task.last_execution_at && <span>Last ran {relativeTime(task.last_execution_at, now.getTime())}</span>}
            {!task.enabled && <span className="text-warning">Paused</span>}
          </div>
        </div>
      )}
    </section>
  )
}

function PropertiesCard({ task, tz, ops, compact }: { task: Task; tz: string; ops: Ops; compact?: boolean }) {
  const now = useNow(60_000)
  const next = nextRunDate(task, now)
  const kind = taskKind(task)
  const toggleable = kind === 'recurring' || kind === 'triggered' || kind === 'script' || task.status === 'pending'
  const rows = useMemo(
    () =>
      [
        ['Type', KIND_META[kind].label],
        ['Created', formatDateTime(task.created_at)],
        task.last_execution_at ? ['Last run', relativeTime(task.last_execution_at, now.getTime())] : null,
        next ? ['Next run', upcomingPhrase(next, tz, now)] : null,
        ['Runs', String(task.runs.length)],
        ['Model', task.model ? modelShortName(task.model) : 'Default']
      ].filter((r): r is [string, string] => !!r),
    [task, kind, next, tz, now]
  )

  if (compact) {
    return (
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border border-border bg-sunken/40 px-4 py-2.5 text-xs">
        {rows.slice(1).map(([k, v]) => (
          <span key={k} className="text-fg-subtle">
            {k} <span className="font-medium text-fg-muted">{v}</span>
          </span>
        ))}
        {toggleable && (
          <label className="ml-auto flex items-center gap-2 text-fg-muted">
            {task.enabled ? 'Enabled' : 'Paused'}
            <Switch size="sm" checked={task.enabled} onCheckedChange={(v) => void ops.setEnabled(task.task_id, v)} aria-label="Enabled" />
          </label>
        )}
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-border bg-sunken/40">
      <div className="border-b border-border px-4 py-2.5 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">Details</div>
      <dl className="divide-y divide-border text-sm">
        {toggleable && (
          <div className="flex items-center justify-between px-4 py-2.5">
            <dt className="text-fg-subtle">Enabled</dt>
            <dd>
              <Switch size="sm" checked={task.enabled} onCheckedChange={(v) => void ops.setEnabled(task.task_id, v)} aria-label="Enabled" />
            </dd>
          </div>
        )}
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between gap-3 px-4 py-2.5">
            <dt className="shrink-0 text-fg-subtle">{k}</dt>
            <dd className="min-w-0 truncate text-right text-fg-muted">{v}</dd>
          </div>
        ))}
        <div className="flex items-center justify-between gap-3 px-4 py-2.5">
          <dt className="shrink-0 text-fg-subtle">ID</dt>
          <dd className="min-w-0">
            <button type="button" onClick={() => void copyText(task.task_id).then((ok) => ok && toast.message('Task ID copied'))} className="flex max-w-full items-center gap-1 truncate font-mono text-xs text-fg-subtle hover:text-fg">
              <span className="truncate">{task.task_id}</span>
              <IconCopy size={12} className="shrink-0" />
            </button>
          </dd>
        </div>
      </dl>
    </div>
  )
}

function DetailSkeleton({ layout, onClose }: { layout: 'panel' | 'page'; onClose?: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <div className={cn('border-b border-border', layout === 'page' ? 'px-8 py-5' : 'px-5 py-4')}>
        <div className="mb-3 flex justify-between">
          <Skeleton className="h-5 w-16" />
          {onClose && <IconButton size="sm" label="Close" icon={<IconX size={16} />} onClick={onClose} />}
        </div>
        <div className="flex gap-3.5">
          <Skeleton className="size-11 rounded-2xl" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-6 w-2/3" />
            <div className="flex gap-1.5">
              <Skeleton className="h-5 w-24 rounded-full" />
              <Skeleton className="h-5 w-20 rounded-full" />
              <Skeleton className="h-5 w-16 rounded-full" />
            </div>
          </div>
        </div>
      </div>
      <div className={cn('space-y-4', layout === 'page' ? 'px-8 py-6' : 'px-5 py-5')}>
        <Skeleton className="h-9 w-64" />
        <Skeleton className="h-20 w-full rounded-xl" />
        <Skeleton className="h-14 w-full rounded-xl" />
        <Skeleton className="h-40 w-full rounded-xl" />
      </div>
    </div>
  )
}
