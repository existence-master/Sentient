/** React Query hooks for §5 integrations. Live updates arrive via `integration.updated`. */
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { getBridge } from '@/lib/bridge'
import type { Integration, McpServerCreate, PrivacyFilters } from '@/lib/types'
import { qk } from './queryKeys'

export function upsertIntegration(qc: QueryClient, integration: Integration) {
  qc.setQueryData<Integration[]>(qk.integrations.all, (old) => {
    if (!old) return old
    const i = old.findIndex((x) => x.id === integration.id)
    if (i === -1) return [...old, integration]
    const next = old.slice()
    next[i] = integration
    return next
  })
  qc.setQueryData(qk.integrations.detail(integration.id), integration)
}

export function useIntegrations() {
  return useQuery({ queryKey: qk.integrations.all, queryFn: api.integrations.list })
}

export function useIntegration(id: string | undefined) {
  return useQuery({ queryKey: qk.integrations.detail(id ?? ''), queryFn: () => api.integrations.get(id as string), enabled: !!id })
}

export function usePrivacyFilters(id: string | undefined) {
  return useQuery({
    queryKey: qk.integrations.privacyFilters(id ?? ''),
    queryFn: () => api.integrations.getPrivacyFilters(id as string),
    enabled: !!id
  })
}

export function useIntegrationActions() {
  const qc = useQueryClient()
  return {
    /**
     * api_key/manual resolve to an Integration (validation errors: ApiError 400 with `detail`).
     * OAuth / GitHub device flow resolve to `{auth_url, state, user_code?}`: the browser is opened here;
     * render `user_code` prominently from the mutation's `data`. Completion arrives via `integration.updated`.
     */
    connect: useMutation({
      mutationFn: ({ id, fields }: { id: string; fields?: Record<string, string> }) => api.integrations.connect(id, fields),
      onSuccess: (res) => {
        if ('auth_url' in res) void getBridge().openExternal(res.auth_url)
        else upsertIntegration(qc, res)
      }
    }),
    disconnect: useMutation({
      mutationFn: (id: string) => api.integrations.disconnect(id),
      onSuccess: (res) => {
        upsertIntegration(qc, res)
        void qc.invalidateQueries({ queryKey: qk.tasks.all })
      }
    }),
    test: useMutation({ mutationFn: (id: string) => api.integrations.test(id) }),
    setPrivacyFilters: useMutation({
      mutationFn: ({ id, filters }: { id: string; filters: PrivacyFilters }) => api.integrations.setPrivacyFilters(id, filters),
      onSuccess: (_r, { id, filters }) => qc.setQueryData(qk.integrations.privacyFilters(id), filters)
    })
  }
}

export function useMcpServers() {
  return useQuery({
    queryKey: qk.integrations.mcp,
    queryFn: api.integrations.mcp.list,
    // No event tells the window when a browser sign-in finishes or a server connects: poll while one is pending.
    refetchInterval: (q) => (q.state.data?.some((s) => s.signing_in || s.status === 'connecting') ? 2000 : false)
  })
}

export function useMcpActions() {
  const qc = useQueryClient()
  const invalidate = () => void qc.invalidateQueries({ queryKey: qk.integrations.mcp })
  return {
    add: useMutation({ mutationFn: (body: McpServerCreate) => api.integrations.mcp.add(body), onSuccess: invalidate }),
    remove: useMutation({ mutationFn: (name: string) => api.integrations.mcp.remove(name), onSuccess: invalidate }),
    test: useMutation({ mutationFn: (name: string) => api.integrations.mcp.test(name) }),
    /** Starts a browser sign-in and opens the provider's page; the server list shows when it finishes. */
    signIn: useMutation({
      mutationFn: (name: string) => api.integrations.mcp.signIn(name),
      onSuccess: (res) => {
        void getBridge().openExternal(res.auth_url)
        invalidate()
      }
    }),
    signOut: useMutation({ mutationFn: (name: string) => api.integrations.mcp.signOut(name), onSuccess: invalidate }),
    setEnabled: useMutation({
      mutationFn: ({ name, enabled }: { name: string; enabled: boolean }) => api.integrations.mcp.setEnabled(name, enabled),
      onSuccess: invalidate
    }),
    setValues: useMutation({
      mutationFn: ({ name, values, enable }: { name: string; values: Record<string, string>; enable?: boolean }) =>
        api.integrations.mcp.setValues(name, values, enable),
      onSuccess: invalidate
    })
  }
}
