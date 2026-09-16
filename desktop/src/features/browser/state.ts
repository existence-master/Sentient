/**
 * Browser live view state (§12).
 *
 * - `useBrowserView` (zustand): whether the side panel is open and the latest `browser.frame`.
 * - `useBrowserStatus` (React Query `['browser', 'status']`), kept fresh by `browser.updated`.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { create } from 'zustand'
import { api } from '@/lib/api'
import type { BrowserFrame, BrowserStatus } from '@/lib/types'
import { previewBrowser, previewFrame, previewMode, previewOr } from '@/features/devices/preview'

export const browserKeys = {
  status: ['browser', 'status'] as const
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
      mutationFn: (url?: string) => previewOr(() => api.browser.open(url), () => ({ ...previewBrowser(), headless: false })),
      onSuccess
    }),
    close: useMutation({
      mutationFn: () => previewOr(api.browser.close, () => ({ ...previewBrowser(), running: false, tabs: [] })),
      onSuccess
    })
  }
}
