/** React Query hooks for §6 notifications & proactivity. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { BriefFeedback, BriefKind, BriefSectionId, BriefSetup, BriefState, NotificationList } from '@/lib/types'
import { useNotificationStore } from '@/stores/notifications'
import { qk } from './queryKeys'

export function useNotifications(limit = 100) {
  return useQuery({
    queryKey: qk.notifications,
    queryFn: async () => {
      const res = await api.notifications.list({ limit })
      useNotificationStore.getState().setUnread(res.unread)
      return res
    }
  })
}

export function useNotificationActions() {
  const qc = useQueryClient()
  const store = useNotificationStore.getState
  const patchCache = (fn: (l: NotificationList) => NotificationList) =>
    qc.setQueryData<NotificationList>(qk.notifications, (old) => (old ? fn(old) : old))

  return {
    markRead: useMutation({
      mutationFn: (id: string) => api.notifications.markRead(id),
      onMutate: (id) => {
        patchCache((l) => ({
          notifications: l.notifications.map((n) => (n.id === id ? { ...n, read: true } : n)),
          unread: Math.max(0, l.unread - (l.notifications.find((n) => n.id === id && !n.read) ? 1 : 0))
        }))
        store().markRead(id)
      }
    }),
    markAllRead: useMutation({
      mutationFn: () => api.notifications.markAllRead(),
      onMutate: () => {
        patchCache((l) => ({ notifications: l.notifications.map((n) => ({ ...n, read: true })), unread: 0 }))
        store().markRead(null)
      }
    }),
    remove: useMutation({
      mutationFn: (id: string) => api.notifications.delete(id),
      onMutate: (id) => {
        patchCache((l) => ({ ...l, notifications: l.notifications.filter((n) => n.id !== id) }))
        store().remove(id)
      }
    }),
    clear: useMutation({
      mutationFn: () => api.notifications.clear(),
      onMutate: () => {
        patchCache(() => ({ notifications: [], unread: 0 }))
        store().remove(null)
      }
    }),
    respondToSuggestion: useMutation({
      mutationFn: ({ id, action }: { id: string; action: 'approve' | 'dismiss' }) => api.proactivity.respond(id, action),
      onSuccess: () => {
        void qc.invalidateQueries({ queryKey: qk.notifications })
        void qc.invalidateQueries({ queryKey: qk.tasks.all })
        void qc.invalidateQueries({ queryKey: qk.proactivity.preferences })
      }
    })
  }
}

export function useProactivityStatus() {
  return useQuery({ queryKey: qk.proactivity.status, queryFn: api.proactivity.status })
}

export function useProactivityPreferences() {
  return useQuery({ queryKey: qk.proactivity.preferences, queryFn: api.proactivity.preferences })
}

export function useProactivityActions() {
  const qc = useQueryClient()
  return {
    pollNow: useMutation({
      mutationFn: () => api.proactivity.pollNow(),
      onSuccess: () => void qc.invalidateQueries({ queryKey: qk.proactivity.status })
    }),
    resetPreference: useMutation({
      mutationFn: (type: string) => api.proactivity.resetPreference(type),
      onSuccess: () => void qc.invalidateQueries({ queryKey: qk.proactivity.preferences })
    })
  }
}

/** Daily Brief state: the task, its schedule and sections, and today's brief (null when none is showing). */
export function useBrief() {
  return useQuery({ queryKey: qk.proactivity.brief, queryFn: api.proactivity.brief.get, retry: false })
}

export function useBriefActions() {
  const qc = useQueryClient()
  const setState = (s: BriefState) => qc.setQueryData(qk.proactivity.brief, s)
  return {
    setup: useMutation({
      mutationFn: (body: BriefSetup) => api.proactivity.brief.setup(body),
      onSuccess: (s) => {
        setState(s)
        void qc.invalidateQueries({ queryKey: qk.tasks.all })
        void qc.invalidateQueries({ queryKey: qk.config })
      }
    }),
    runNow: useMutation({ mutationFn: (kind: BriefKind = 'morning') => api.proactivity.brief.runNow(kind) }),
    feedback: useMutation({
      mutationFn: (body: { brief_id: string; value: BriefFeedback; item_id?: string; section?: BriefSectionId }) =>
        api.proactivity.brief.feedback(body),
      onSuccess: (brief) => {
        qc.setQueryData<BriefState>(qk.proactivity.brief, (old) => (old ? { ...old, today: brief } : old))
        void qc.invalidateQueries({ queryKey: qk.proactivity.preferences })
      }
    })
  }
}
