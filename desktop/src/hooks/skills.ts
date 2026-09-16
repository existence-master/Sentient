/** React Query hooks for §8 skills & self-evolution. `skill.updated` invalidates `qk.skills.all`. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { SkillCreate, SkillUpdate } from '@/lib/types'
import { qk } from './queryKeys'

export function useSkills() {
  return useQuery({ queryKey: qk.skills.all, queryFn: api.skills.list })
}

export function useSkill(name: string | undefined) {
  return useQuery({ queryKey: qk.skills.detail(name ?? ''), queryFn: () => api.skills.get(name as string), enabled: !!name })
}

export function useSkillDiff(name: string | undefined) {
  return useQuery({ queryKey: qk.skills.diff(name ?? ''), queryFn: () => api.skills.diff(name as string), enabled: !!name })
}

export function useEvolutionLog(limit = 100) {
  return useQuery({ queryKey: qk.skills.evolutionLog, queryFn: () => api.skills.evolutionLog(limit) })
}

export function useSkillActions() {
  const qc = useQueryClient()
  const invalidate = () => void qc.invalidateQueries({ queryKey: qk.skills.all })
  return {
    create: useMutation({ mutationFn: (body: SkillCreate) => api.skills.create(body), onSuccess: invalidate }),
    update: useMutation({ mutationFn: ({ name, body }: { name: string; body: SkillUpdate }) => api.skills.update(name, body), onSuccess: invalidate }),
    approve: useMutation({ mutationFn: (name: string) => api.skills.approve(name), onSuccess: invalidate }),
    reject: useMutation({ mutationFn: (name: string) => api.skills.reject(name), onSuccess: invalidate }),
    archive: useMutation({ mutationFn: (name: string) => api.skills.archive(name), onSuccess: invalidate }),
    restore: useMutation({ mutationFn: (name: string) => api.skills.restore(name), onSuccess: invalidate }),
    remove: useMutation({ mutationFn: (name: string) => api.skills.delete(name), onSuccess: invalidate }),
    reviewNow: useMutation({ mutationFn: (sessionId?: string) => api.skills.reviewNow(sessionId), onSuccess: invalidate })
  }
}
