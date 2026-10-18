/** React Query hooks for §22 folders and Obsidian vaults as searchable notes (#176). Live: `folders.updated`. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { invalidateForgotten } from './memory'
import { qk } from './queryKeys'

export function useFolders() {
  return useQuery({
    queryKey: qk.folders,
    queryFn: api.folders.list,
    // while a look through a folder runs, its progress also comes as events; this is a fallback
    refetchInterval: (q) => (q.state.data?.folders.some((f) => f.status === 'scanning' || f.status === 'waiting') ? 3000 : false)
  })
}

function useFolderMutation<A>(fn: (arg: A) => Promise<unknown>) {
  const qc = useQueryClient()
  return useMutation({ mutationFn: fn, onSettled: () => void qc.invalidateQueries({ queryKey: qk.folders }) })
}

export const useAddFolder = () => useFolderMutation((path: string) => api.folders.add(path))
/** ``forget``: also forget the memories learned from its notes (#153). */
export function useRemoveFolder() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, forget = false }: { id: string; forget?: boolean }) => api.folders.remove(id, forget),
    onSuccess: (res) => {
      if (res.forgotten) invalidateForgotten(qc)
    },
    onSettled: () => void qc.invalidateQueries({ queryKey: qk.folders })
  })
}
export const useRescanFolder = () => useFolderMutation((id: string) => api.folders.rescan(id))
/** Where a folder's notes are used (#392). A new project name also shows up in the memory scope pickers. */
export function useFolderScope() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, scope }: { id: string; scope: string }) => api.folders.setScope(id, scope),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: qk.folders })
      void qc.invalidateQueries({ queryKey: qk.memories.scopes })
    }
  })
}
export const useFolderIgnore = () =>
  useFolderMutation(({ id, ignore }: { id: string; ignore: string[] }) => api.folders.update(id, ignore))
