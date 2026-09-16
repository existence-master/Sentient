/**
 * Notification badge + live feed. The full list lives in React Query
 * (`useNotifications`); this store holds what the shell needs instantly.
 */
import { create } from 'zustand'
import type { Notification } from '@/lib/types'

interface NotificationsState {
  unread: number
  /** Notifications that arrived live during this session, newest first. */
  recent: Notification[]
  setUnread: (n: number) => void
  push: (n: Notification) => void
  /** Replace a notification whose payload changed (suggestion approved, approval answered). */
  update: (n: Notification) => void
  markRead: (id: string | null) => void
  remove: (id: string | null) => void
}

export const useNotificationStore = create<NotificationsState>((set, get) => ({
  unread: 0,
  recent: [],
  setUnread: (unread) => set({ unread: Math.max(0, unread) }),
  push: (n) => {
    if (get().recent.some((r) => r.id === n.id)) return
    set({ recent: [n, ...get().recent].slice(0, 50), unread: get().unread + (n.read ? 0 : 1) })
  },
  update: (n) => set({ recent: get().recent.map((r) => (r.id === n.id ? n : r)) }),
  markRead: (id) => {
    if (id === null) {
      set({ unread: 0, recent: get().recent.map((r) => ({ ...r, read: true })) })
      return
    }
    const wasUnread = get().recent.find((r) => r.id === id && !r.read)
    set({
      recent: get().recent.map((r) => (r.id === id ? { ...r, read: true } : r)),
      unread: wasUnread ? Math.max(0, get().unread - 1) : get().unread
    })
  },
  remove: (id) => {
    if (id === null) {
      set({ recent: [], unread: 0 })
      return
    }
    set({ recent: get().recent.filter((r) => r.id !== id) })
  }
}))
