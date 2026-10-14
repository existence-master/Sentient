/** React Query hooks for §22 folders and Obsidian vaults as searchable notes (#176). Live: `folders.updated`. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
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
export const useRemoveFolder = () => useFolderMutation((id: string) => api.folders.remove(id))
export const useRescanFolder = () => useFolderMutation((id: string) => api.folders.rescan(id))
export const useFolderIgnore = () =>
  useFolderMutation(({ id, ignore }: { id: string; ignore: string[] }) => api.folders.update(id, ignore))
