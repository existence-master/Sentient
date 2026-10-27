/** React Query hooks for the Skills hub (#210). Installs and updates land in pending review, so they also
 * invalidate `qk.skills.all`. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, noRetryWhenMissing } from '@/lib/api'
import type { HubInstall } from '@/lib/types'
import { qk } from './queryKeys'

export function useHubSources() {
  return useQuery({ queryKey: qk.skillsHub.sources, queryFn: api.skills.hub.sources, retry: noRetryWhenMissing })
}

/** A source's listing. The engine keeps it for 15 minutes, so this stays cheap on GitHub's limits. */
export function useHubBrowse(source: string | null) {
  return useQuery({
    queryKey: qk.skillsHub.browse(source ?? ''),
    queryFn: () => api.skills.hub.browse(source as string),
    enabled: !!source,
    staleTime: 5 * 60_000,
    retry: (count, err) => noRetryWhenMissing(count, err) && count < 1
  })
}

export function useHubPreview(source: string | null, path: string | null) {
  return useQuery({
    queryKey: qk.skillsHub.preview(source ?? '', path ?? ''),
    queryFn: () => api.skills.hub.preview(source as string, path as string),
    enabled: !!source && !!path,
    staleTime: 5 * 60_000,
    retry: false
  })
}

export function useHubInstalled() {
  return useQuery({ queryKey: qk.skillsHub.installed, queryFn: () => api.skills.hub.installed(), retry: noRetryWhenMissing })
}

export function useHubActions() {
  const qc = useQueryClient()
  const changed = () => {
    void qc.invalidateQueries({ queryKey: qk.skills.all })
    void qc.invalidateQueries({ queryKey: ['skills-hub'] })
  }
  return {
    install: useMutation({ mutationFn: (body: HubInstall) => api.skills.hub.install(body), onSuccess: changed }),
    update: useMutation({ mutationFn: ({ name, acceptRisk }: { name: string; acceptRisk?: boolean }) => api.skills.hub.update(name, acceptRisk), onSuccess: changed }),
    checkUpdates: useMutation({ mutationFn: () => api.skills.hub.installed(true), onSuccess: (r) => qc.setQueryData(qk.skillsHub.installed, r) }),
    addSource: useMutation({
      mutationFn: ({ url, name }: { url: string; name?: string }) => api.skills.hub.addSource(url, name),
      onSuccess: () => void qc.invalidateQueries({ queryKey: qk.skillsHub.sources })
    }),
    removeSource: useMutation({
      mutationFn: (id: string) => api.skills.hub.removeSource(id),
      onSuccess: () => void qc.invalidateQueries({ queryKey: qk.skillsHub.sources })
    }),
    refresh: useMutation({
      mutationFn: (source: string) => api.skills.hub.browse(source, true),
      onSuccess: (data, source) => qc.setQueryData(qk.skillsHub.browse(source), data)
    })
  }
}
