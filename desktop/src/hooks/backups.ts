/** React Query hooks for §20 backups, restore and snapshots. Live updates: `backup.updated` (lib/events.ts). */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { qk } from './queryKeys'

export function useBackups() {
  return useQuery({ queryKey: qk.backups, queryFn: api.backups.get })
}

export function useBackUpNow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (folder?: string) => api.backups.create(folder),
    onSuccess: () => void qc.invalidateQueries({ queryKey: qk.backups })
  })
}

export function useCancelRestore() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.backups.cancelRestore,
    onSuccess: () => void qc.invalidateQueries({ queryKey: qk.backups })
  })
}

export function useDismissRestore() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.backups.dismissLastRestore,
    onSuccess: () => void qc.invalidateQueries({ queryKey: qk.backups })
  })
}
