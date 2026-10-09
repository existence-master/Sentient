/** React Query hooks for §18 commands on this computer. */
import { useMutation, useQuery } from '@tanstack/react-query'
import { api, noRetryWhenMissing } from '@/lib/api'
import { qk } from './queryKeys'

export function useTerminalStatus() {
  return useQuery({ queryKey: qk.terminalStatus, queryFn: api.terminal.status, retry: noRetryWhenMissing, staleTime: 30_000 })
}

/** The Stop button on a running command. */
export function useStopCommand() {
  return useMutation({ mutationFn: (id: string) => api.terminal.stop(id) })
}
