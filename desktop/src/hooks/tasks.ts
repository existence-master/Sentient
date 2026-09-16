/** React Query hooks for §4 tasks. Live updates arrive via `task.updated` (see lib/events.ts). */
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { ClarificationAnswer, Task, TaskCreateRequest, TaskPatch } from '@/lib/types'
import { qk } from './queryKeys'

export function upsertTask(qc: QueryClient, task: Task) {
  qc.setQueryData<Task[]>(qk.tasks.all, (old) => {
    if (!old) return old
    const i = old.findIndex((t) => t.task_id === task.task_id)
    if (i === -1) return [task, ...old]
    const next = old.slice()
    next[i] = task
    return next
  })
  qc.setQueryData(qk.tasks.detail(task.task_id), task)
}

export function removeTask(qc: QueryClient, taskId: string) {
  qc.setQueryData<Task[]>(qk.tasks.all, (old) => old?.filter((t) => t.task_id !== taskId))
  qc.removeQueries({ queryKey: qk.tasks.detail(taskId), exact: true })
}

export function useTasks() {
  return useQuery({ queryKey: qk.tasks.all, queryFn: api.tasks.list })
}

export function useTask(id: string | undefined) {
  return useQuery({ queryKey: qk.tasks.detail(id ?? ''), queryFn: () => api.tasks.get(id as string), enabled: !!id })
}

export function useRunEvents(taskId: string | undefined, runId: string | undefined) {
  return useQuery({
    queryKey: qk.tasks.runEvents(taskId ?? '', runId ?? ''),
    queryFn: () => api.tasks.runEvents(taskId as string, runId as string),
    enabled: !!taskId && !!runId
  })
}

export function useTaskPreview() {
  return useMutation({ mutationFn: (prompt: string) => api.tasks.preview(prompt) })
}

/** All task mutations. Each resolves to the updated Task, which is upserted into the cache. */
export function useTaskActions() {
  const qc = useQueryClient()
  const onSuccess = (task: Task) => upsertTask(qc, task)
  return {
    create: useMutation({ mutationFn: (body: TaskCreateRequest) => api.tasks.create(body), onSuccess }),
    update: useMutation({ mutationFn: ({ id, patch }: { id: string; patch: TaskPatch }) => api.tasks.update(id, patch), onSuccess }),
    approve: useMutation({ mutationFn: (id: string) => api.tasks.approve(id), onSuccess }),
    decline: useMutation({ mutationFn: (id: string) => api.tasks.decline(id), onSuccess }),
    rerun: useMutation({ mutationFn: (id: string) => api.tasks.rerun(id), onSuccess }),
    runNow: useMutation({ mutationFn: (id: string) => api.tasks.runNow(id), onSuccess }),
    archive: useMutation({ mutationFn: (id: string) => api.tasks.archive(id), onSuccess }),
    chat: useMutation({ mutationFn: ({ id, message }: { id: string; message: string }) => api.tasks.chat(id, message), onSuccess }),
    answer: useMutation({
      mutationFn: ({ id, answers }: { id: string; answers: ClarificationAnswer[] }) => api.tasks.answerClarifications(id, answers),
      onSuccess
    }),
    cancelRun: useMutation({ mutationFn: ({ id, runId }: { id: string; runId: string }) => api.tasks.cancelRun(id, runId), onSuccess }),
    remove: useMutation({ mutationFn: (id: string) => api.tasks.delete(id), onSuccess: (_r, id) => removeTask(qc, id) })
  }
}
