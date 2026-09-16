/** React Query hooks for desktop agent B screens. Live updates arrive through lib/leap/events-b.ts. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { qk } from '@/hooks/queryKeys'
import { isNotImplemented } from '@/lib/api'
import { upsertTask } from '@/hooks/tasks'
import { apiB, qkB } from './api-b'
import type { Dream, Hook, Insight, InsightDimension, InsightStatus, ScriptJob, UserModel } from './types-b'

/** 404/405/501: the engine package isn't there yet. Screens show a friendly "coming" state. */
export const notReady = isNotImplemented

const noRetryWhenMissing = (count: number, err: unknown) => !isNotImplemented(err) && count < 2

// ---------------------------------------------------------------------------- user model
export function useUserModel() {
  return useQuery({ queryKey: qkB.userModel, queryFn: apiB.userModel.get, retry: noRetryWhenMissing })
}

export function useUserModelActions() {
  const qc = useQueryClient()
  const patchCache = (fn: (m: UserModel) => UserModel) => qc.setQueryData<UserModel>(qkB.userModel, (old) => (old ? fn(old) : old))
  const replaceInsight = (i: Insight) => patchCache((m) => ({ ...m, insights: m.insights.some((x) => x.id === i.id) ? m.insights.map((x) => (x.id === i.id ? i : x)) : [...m.insights, i] }))
  const invalidate = () => void qc.invalidateQueries({ queryKey: qkB.userModel })

  return {
    add: useMutation({
      mutationFn: ({ statement, dimension }: { statement: string; dimension: InsightDimension | string }) => apiB.userModel.addInsight(statement, dimension),
      onSuccess: replaceInsight
    }),
    patch: useMutation({
      mutationFn: ({ id, patch }: { id: string; patch: { statement?: string; status?: InsightStatus } }) => apiB.userModel.patchInsight(id, patch),
      onMutate: ({ id, patch }) => {
        const prev = qc.getQueryData<UserModel>(qkB.userModel)
        patchCache((m) => ({ ...m, insights: m.insights.map((i) => (i.id === id ? { ...i, ...patch } : i)) }))
        return prev
      },
      onError: (_e, _v, prev) => prev && qc.setQueryData(qkB.userModel, prev),
      onSuccess: (i) => i && replaceInsight(i)
    }),
    remove: useMutation({
      mutationFn: (id: string) => apiB.userModel.deleteInsight(id),
      onMutate: (id) => {
        const prev = qc.getQueryData<UserModel>(qkB.userModel)
        patchCache((m) => ({ ...m, insights: m.insights.filter((i) => i.id !== id) }))
        return prev
      },
      onError: (_e, _v, prev) => prev && qc.setQueryData(qkB.userModel, prev)
    }),
    refresh: useMutation({ mutationFn: () => apiB.userModel.refresh(), onSuccess: invalidate }),
    answer: useMutation({
      mutationFn: ({ id, answer }: { id: string; answer: string }) => apiB.userModel.answerQuestion(id, answer),
      onSuccess: (_r, { id }) => {
        patchCache((m) => ({ ...m, questions: m.questions.filter((q) => q.id !== id) }))
        invalidate()
      }
    }),
    dismiss: useMutation({
      mutationFn: (id: string) => apiB.userModel.dismissQuestion(id),
      onMutate: (id) => {
        const prev = qc.getQueryData<UserModel>(qkB.userModel)
        patchCache((m) => ({ ...m, questions: m.questions.filter((q) => q.id !== id) }))
        return prev
      },
      onError: (_e, _v, prev) => prev && qc.setQueryData(qkB.userModel, prev)
    })
  }
}

// ---------------------------------------------------------------------------- dreams
export function useDreams(limit = 30) {
  return useQuery({ queryKey: qkB.dreams, queryFn: () => apiB.dreams.list(limit), retry: noRetryWhenMissing })
}

export function upsertDream(list: Dream[] | undefined, d: Dream): Dream[] {
  const cur = list ?? []
  return cur.some((x) => x.id === d.id) ? cur.map((x) => (x.id === d.id ? d : x)) : [d, ...cur]
}

export function useRunDream() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => apiB.dreams.run(),
    onSuccess: (d) => qc.setQueryData<Dream[]>(qkB.dreams, (old) => upsertDream(old, d))
  })
}

// ---------------------------------------------------------------------------- webhooks
export function useHooks() {
  return useQuery({ queryKey: qkB.hooks, queryFn: apiB.hooks.list, retry: noRetryWhenMissing })
}

export function useHookActions() {
  const qc = useQueryClient()
  return {
    create: useMutation({
      mutationFn: (name: string) => apiB.hooks.create(name),
      onSuccess: (h) => {
        const { secret: _secret, ...hook } = h
        qc.setQueryData<Hook[]>(qkB.hooks, (old) => [hook, ...(old ?? []).filter((x) => x.id !== hook.id)])
      }
    }),
    remove: useMutation({
      mutationFn: (id: string) => apiB.hooks.delete(id),
      onMutate: (id) => {
        const prev = qc.getQueryData<Hook[]>(qkB.hooks)
        qc.setQueryData<Hook[]>(qkB.hooks, (old) => old?.filter((h) => h.id !== id))
        return prev
      },
      onError: (_e, _v, prev) => prev && qc.setQueryData(qkB.hooks, prev)
    })
  }
}

// ---------------------------------------------------------------------------- change feeds
export function useFeeds() {
  return useQuery({ queryKey: qkB.feeds, queryFn: apiB.feeds.list, retry: noRetryWhenMissing, refetchInterval: 60_000 })
}

export function useFeedSync() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (source: string) => apiB.feeds.sync(source), onSettled: () => void qc.invalidateQueries({ queryKey: qkB.feeds }) })
}

// ---------------------------------------------------------------------------- script jobs & sandbox
export function useScriptTest() {
  return useMutation({
    mutationFn: (arg: string | { taskId: string; code?: string }) => (typeof arg === 'string' ? apiB.scripts.test(arg) : apiB.scripts.test(arg.taskId, arg.code))
  })
}

/** Retry a failed or cancelled run; the updated task is upserted. */
export function useRetryRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ taskId, runId }: { taskId: string; runId: string }) => apiB.runs.retry(taskId, runId),
    onSuccess: (task) => upsertTask(qc, task)
  })
}

export function useScriptUpdate() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ taskId, script }: { taskId: string; script: Partial<Pick<ScriptJob, 'condition' | 'then' | 'code'>> }) => apiB.scripts.update(taskId, script),
    onSuccess: (task) => upsertTask(qc, task),
    onError: () => void qc.invalidateQueries({ queryKey: qk.tasks.all })
  })
}

export function useSandboxStatus() {
  return useQuery({ queryKey: qkB.sandboxStatus, queryFn: apiB.sandbox.status, retry: noRetryWhenMissing, staleTime: 60_000 })
}
