/** React Query hooks for §6 notifications & proactivity. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { NotificationList } from '@/lib/types'
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
