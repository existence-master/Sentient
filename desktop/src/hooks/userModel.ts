/** React Query hooks for §15: the user model ("About you") and dreams. Live updates arrive via lib/events.ts. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, noRetryWhenMissing } from '@/lib/api'
import type { Dream, Insight, InsightDimension, InsightStatus, UserModel } from '@/lib/types'
import { qk } from './queryKeys'

// ---------------------------------------------------------------------------- user model
export function useUserModel() {
  return useQuery({ queryKey: qk.userModel, queryFn: api.userModel.get, retry: noRetryWhenMissing })
}

export function useUserModelActions() {
  const qc = useQueryClient()
  const patchCache = (fn: (m: UserModel) => UserModel) => qc.setQueryData<UserModel>(qk.userModel, (old) => (old ? fn(old) : old))
  const replaceInsight = (i: Insight) => patchCache((m) => ({ ...m, insights: m.insights.some((x) => x.id === i.id) ? m.insights.map((x) => (x.id === i.id ? i : x)) : [...m.insights, i] }))
  const invalidate = () => void qc.invalidateQueries({ queryKey: qk.userModel })

  return {
    add: useMutation({
      mutationFn: ({ statement, dimension }: { statement: string; dimension: InsightDimension | string }) => api.userModel.addInsight(statement, dimension),
      onSuccess: replaceInsight
    }),
    patch: useMutation({
      mutationFn: ({ id, patch }: { id: string; patch: { statement?: string; status?: InsightStatus } }) => api.userModel.patchInsight(id, patch),
      onMutate: ({ id, patch }) => {
        const prev = qc.getQueryData<UserModel>(qk.userModel)
        patchCache((m) => ({ ...m, insights: m.insights.map((i) => (i.id === id ? { ...i, ...patch } : i)) }))
        return prev
      },
      onError: (_e, _v, prev) => prev && qc.setQueryData(qk.userModel, prev),
      onSuccess: (i) => i && replaceInsight(i)
    }),
    remove: useMutation({
      mutationFn: (id: string) => api.userModel.deleteInsight(id),
      onMutate: (id) => {
        const prev = qc.getQueryData<UserModel>(qk.userModel)
        patchCache((m) => ({ ...m, insights: m.insights.filter((i) => i.id !== id) }))
        return prev
      },
      onError: (_e, _v, prev) => prev && qc.setQueryData(qk.userModel, prev)
    }),
    refresh: useMutation({ mutationFn: () => api.userModel.refresh(), onSuccess: invalidate }),
    answer: useMutation({
      mutationFn: ({ id, answer }: { id: string; answer: string }) => api.userModel.answerQuestion(id, answer),
      onSuccess: (_r, { id }) => {
        patchCache((m) => ({ ...m, questions: m.questions.filter((q) => q.id !== id) }))
        invalidate()
      }
    }),
    dismiss: useMutation({
      mutationFn: (id: string) => api.userModel.dismissQuestion(id),
      onMutate: (id) => {
        const prev = qc.getQueryData<UserModel>(qk.userModel)
        patchCache((m) => ({ ...m, questions: m.questions.filter((q) => q.id !== id) }))
        return prev
      },
      onError: (_e, _v, prev) => prev && qc.setQueryData(qk.userModel, prev)
    })
  }
}

// ---------------------------------------------------------------------------- dreams
export function useDreams(limit = 30) {
  return useQuery({ queryKey: qk.memories.dreams, queryFn: () => api.memories.dreams.list(limit), retry: noRetryWhenMissing })
}

export function upsertDream(list: Dream[] | undefined, d: Dream): Dream[] {
  const cur = list ?? []
  return cur.some((x) => x.id === d.id) ? cur.map((x) => (x.id === d.id ? d : x)) : [d, ...cur]
}

export function useRunDream() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.memories.dreams.run(),
    onSuccess: (d) => qc.setQueryData<Dream[]>(qk.memories.dreams, (old) => upsertDream(old, d))
  })
}
