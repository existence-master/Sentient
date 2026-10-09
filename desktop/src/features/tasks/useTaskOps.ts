/**
 * Task actions with optimistic cache updates, rollback and toasts.
 * Results are upserted into the cache; `task.updated` events keep it in sync afterwards.
 */
import { useQueryClient, type QueryClient } from '@tanstack/react-query'
import { useCallback, useMemo, useState } from 'react'
import { toast } from 'sonner'
import { qk } from '@/hooks/queryKeys'
import { removeTask, upsertTask } from '@/hooks/tasks'
import { api, errorMessage } from '@/lib/api'
import type { ClarificationAnswer, Task, TaskCreateRequest, TaskPatch } from '@/lib/types'
import { uid } from '@/lib/utils'

function currentTask(qc: QueryClient, id: string): Task | undefined {
  return qc.getQueryData<Task>(qk.tasks.detail(id)) ?? qc.getQueryData<Task[]>(qk.tasks.all)?.find((t) => t.task_id === id)
}

function applyTask(qc: QueryClient, id: string, fn: (t: Task) => Task | null) {
  const cur = currentTask(qc, id)
  if (!cur) return
  const next = fn(cur)
  if (next) upsertTask(qc, next)
  else removeTask(qc, id)
}

const now = () => new Date().toISOString()

export type TaskOp =
  | 'create'
  | 'update'
  | 'approve'
  | 'decline'
  | 'runNow'
  | 'rerun'
  | 'archive'
  | 'remove'
  | 'cancelRun'
  | 'chat'
  | 'answer'
  | 'answerQuestion'
  | 'enabled'

export function useTaskOps() {
  const qc = useQueryClient()
  const [busy, setBusy] = useState<Record<string, boolean>>({})

  const perform = useCallback(
    async <T>(op: TaskOp, id: string, run: () => Promise<T>, opts: { optimistic?: (t: Task) => Task | null; success?: string; error: string }): Promise<T | undefined> => {
      const key = `${op}:${id}`
      const snapshot = currentTask(qc, id)
      if (opts.optimistic) applyTask(qc, id, opts.optimistic)
      setBusy((b) => ({ ...b, [key]: true }))
      try {
        const res = await run()
        if (res && typeof res === 'object' && 'task_id' in (res as object)) upsertTask(qc, res as unknown as Task)
        if (opts.success) toast.success(opts.success)
        return res
      } catch (err) {
        if (snapshot) upsertTask(qc, snapshot)
        toast.error(opts.error, { description: errorMessage(err) })
        return undefined
      } finally {
        setBusy((b) => {
          const { [key]: _drop, ...rest } = b
          void _drop
          return rest
        })
      }
    },
    [qc]
  )

  const ops = useMemo(
    () => ({
      create: async (body: TaskCreateRequest): Promise<Task | undefined> => {
        const tempId = `temp-${uid()}`
        const stamp = now()
        const temp: Task = {
          task_id: tempId,
          name: body.prompt.trim().slice(0, 120),
          description: body.prompt.trim(),
          status: 'planning',
          priority: 1,
          assignee: 'ai',
          task_type: body.is_swarm ? 'swarm' : 'single',
          schedule: null,
          plan: [],
          runs: [],
          chat_history: [],
          clarifying_questions: [],
          swarm_details: body.is_swarm
            ? { goal: body.prompt, items: [], total_agents: 0, completed_agents: 0, progress_updates: [], aggregated_results: [] }
            : null,
          enabled: true,
          model: body.model ?? null,
          original_context: { source: 'manual_creation' },
          error: null,
          next_execution_at: null,
          last_execution_at: null,
          created_at: stamp,
          updated_at: stamp
        }
        qc.setQueryData<Task[]>(qk.tasks.all, (old) => (old ? [temp, ...old] : old))
        setBusy((b) => ({ ...b, 'create:new': true }))
        try {
          const task = await api.tasks.create(body)
          qc.setQueryData<Task[]>(qk.tasks.all, (old) => old?.filter((t) => t.task_id !== tempId))
          upsertTask(qc, task)
          return task
        } catch (err) {
          qc.setQueryData<Task[]>(qk.tasks.all, (old) => old?.filter((t) => t.task_id !== tempId))
          toast.error("Couldn't create the task", { description: errorMessage(err) })
          return undefined
        } finally {
          setBusy(({ 'create:new': _drop, ...rest }) => (void _drop, rest))
        }
      },

      update: (id: string, patch: TaskPatch, success?: string) =>
        perform('update', id, () => api.tasks.update(id, patch), {
          optimistic: (t) => ({ ...t, ...patch, updated_at: now() }) as Task,
          success,
          error: "Couldn't save the change"
        }),

      setEnabled: (id: string, enabled: boolean) =>
        perform('enabled', id, () => api.tasks.update(id, { enabled }), {
          optimistic: (t) => ({ ...t, enabled }),
          success: enabled ? 'Task resumed' : 'Task paused',
          error: enabled ? "Couldn't resume the task" : "Couldn't pause the task"
        }),

      approve: (id: string) =>
        perform('approve', id, () => api.tasks.approve(id), { success: 'Plan approved', error: "Couldn't approve the plan" }),

      decline: (id: string) =>
        perform('decline', id, () => api.tasks.decline(id), {
          optimistic: (t) => ({ ...t, status: 'declined' }),
          success: 'Plan declined',
          error: "Couldn't decline the plan"
        }),

      runNow: (id: string) => perform('runNow', id, () => api.tasks.runNow(id), { success: 'Run started', error: "Couldn't start a run" }),

      rerun: (id: string) => perform('rerun', id, () => api.tasks.rerun(id), { error: "Couldn't re-run the task" }),

      archive: (id: string) =>
        perform('archive', id, () => api.tasks.archive(id), {
          optimistic: (t) => ({ ...t, status: 'archived' }),
          success: 'Task archived',
          error: "Couldn't archive the task"
        }),

      remove: (id: string) =>
        perform('remove', id, () => api.tasks.delete(id).then((r) => (removeTask(qc, id), r)), {
          optimistic: () => null,
          success: 'Task deleted',
          error: "Couldn't delete the task"
        }),

      cancelRun: (id: string, runId: string) =>
        perform('cancelRun', id, () => api.tasks.cancelRun(id, runId), {
          optimistic: (t) => ({ ...t, runs: t.runs.map((r) => (r.run_id === runId ? { ...r, status: 'cancelled', finished_at: now() } : r)) }),
          success: 'Run cancelled',
          error: "Couldn't cancel the run"
        }),

      chat: (id: string, message: string) =>
        perform('chat', id, () => api.tasks.chat(id, message), {
          optimistic: (t) => ({ ...t, status: 'planning', chat_history: [...t.chat_history, { role: 'user', content: message, timestamp: now() }] }),
          error: "Couldn't send your change request"
        }),

      answer: (id: string, answers: ClarificationAnswer[]) =>
        perform('answer', id, () => api.tasks.answerClarifications(id, answers), {
          optimistic: (t) => ({
            ...t,
            status: 'planning',
            clarifying_questions: t.clarifying_questions.map((q) => ({ ...q, answer: answers.find((a) => a.question_id === q.question_id)?.answer_text ?? q.answer }))
          }),
          success: 'Thanks. Sentient is planning with your answers.',
          error: "Couldn't submit your answers"
        }),

      answerQuestion: (id: string, runId: string, answer: string) =>
        perform('answerQuestion', id, () => api.tasks.answerQuestion(id, runId, answer), {
          optimistic: (t) => ({
            ...t,
            status: 'processing',
            runs: t.runs.map((r) => (r.run_id === runId ? { ...r, status: 'processing', pending_question: null } : r))
          }),
          success: 'Thanks. The task is carrying on.',
          error: "Couldn't send your answer"
        })
    }),
    [perform, qc]
  )

  const isBusy = useCallback((op: TaskOp, id = 'new') => !!busy[`${op}:${id}`], [busy])
  return { ...ops, isBusy }
}
