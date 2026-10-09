/**
 * Browser live view state (§12).
 *
 * - `useBrowserView` (zustand): whether the side panel is open and the latest `browser.frame`.
 * - `useBrowserStatus` (React Query `['browser', 'status']`), kept fresh by `browser.updated`.
 * - `useBrowserProfiles` (React Query `['browser', 'profiles']`), refreshed on `browser.updated` too.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { create } from 'zustand'
import { api } from '@/lib/api'
import type { BrowserFrame, BrowserProfileCreate, BrowserProfilePatch, BrowserProfiles, BrowserStatus } from '@/lib/types'
import { previewBrowser, previewFrame, previewMode, previewOr } from '@/features/devices/preview'

export const browserKeys = {
  status: ['browser', 'status'] as const,
  profiles: ['browser', 'profiles'] as const
}

interface BrowserViewState {
  open: boolean
  frame: (BrowserFrame & { at: number }) | null
  setOpen: (open: boolean) => void
  toggle: () => void
  setFrame: (frame: BrowserFrame) => void
}

export const useBrowserView = create<BrowserViewState>((set, get) => ({
  open: false,
  frame: null,
  setOpen: (open) => {
    if (open && !get().frame && previewMode()) set({ frame: { ...previewFrame(), at: Date.now() } })
    set({ open })
  },
  toggle: () => get().setOpen(!get().open),
  setFrame: (frame) => set({ frame: { ...frame, at: Date.now() } })
}))

export function openBrowserView() {
  useBrowserView.getState().setOpen(true)
}

export function useBrowserStatus(enabled = true) {
  return useQuery({
    queryKey: browserKeys.status,
    queryFn: () => previewOr(api.browser.status, previewBrowser),
    enabled,
    refetchInterval: (q) => (q.state.data?.running ? 15_000 : false)
  })
}

export function useBrowserActions() {
  const qc = useQueryClient()
  const onSuccess = (s: BrowserStatus) => qc.setQueryData(browserKeys.status, s)
  return {
    open: useMutation({
      mutationFn: (args?: string | { url?: string; profile?: string }) => {
        const { url, profile } = typeof args === 'string' || args === undefined ? { url: args, profile: undefined } : args
        return previewOr(() => api.browser.open(url, profile), () => ({ ...previewBrowser(), headless: false, profile: profile ?? 'default' }))
      },
      onSuccess
    }),
    close: useMutation({
      mutationFn: () => previewOr(api.browser.close, () => ({ ...previewBrowser(), running: false, tabs: [] })),
      onSuccess
    })
  }
}

const previewProfiles = (): BrowserProfiles => ({
  active: 'default',
  profiles: [
    { name: 'default', kind: 'launch', engine: '', endpoint: '', notes: '', running: true },
    { name: 'shopping', kind: 'launch', engine: '', endpoint: '', notes: 'Signed in to the grocery store', running: false }
  ]
})

export function useBrowserProfiles(enabled = true) {
  return useQuery({ queryKey: browserKeys.profiles, queryFn: () => previewOr(api.browser.profiles, previewProfiles), enabled })
}

export function useBrowserProfileActions() {
  const qc = useQueryClient()
  const onSuccess = (p: BrowserProfiles) => {
    qc.setQueryData(browserKeys.profiles, p)
    void qc.invalidateQueries({ queryKey: browserKeys.status })
  }
  return {
    create: useMutation({ mutationFn: (body: BrowserProfileCreate) => api.browser.createProfile(body), onSuccess }),
    update: useMutation({ mutationFn: ({ name, patch }: { name: string; patch: BrowserProfilePatch }) => api.browser.updateProfile(name, patch), onSuccess }),
    remove: useMutation({ mutationFn: (name: string) => api.browser.deleteProfile(name), onSuccess })
  }
}
