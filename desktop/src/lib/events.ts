/**
 * THE place where domain events from `/ws` update the React Query cache.
 * Feature code should not subscribe to domain events for cache purposes; add a case here.
 *
 *   task.updated          -> upsert into ['tasks'] and ['tasks', id] (+ refetch the Daily Brief state)
 *   task.deleted          -> remove from ['tasks'] (+ refetch the Daily Brief state)
 *   task.run_progress     -> append progress to the run in cache
 *   task.run_activity     -> update the run's last activity time
 *   task.run_context      -> the running run's context meter (live only)
 *   notification.new      -> prepend to ['notifications'], badge++, toast, native notification if unfocused
 *   notification.updated  -> replace in place (payload status changed)
 *   (a `brief` notification, new or updated, also refetches ['proactivity', 'brief'])
 *   notification.read     -> mark read (id null = all)
 *   notification.deleted  -> remove (id null = all)
 *   integration.updated   -> upsert into ['integrations'], refetch change feeds (+ ['hooks'] for the webhook integration)
 *   memory.updated        -> invalidate ['memories']
 *   skill.updated         -> invalidate ['skills']
 *   session.updated       -> rename in ['sessions']
 *   rule_proposal.updated -> refetch that chat's ['rule-proposals'] ("Make this a rule?" cards)
 *   config.updated        -> invalidate config/bootstrap/model presets/ChatGPT sign-in (+ secrets/providers)
 *   voice.state           -> ['voice', 'state']
 *   subagent.updated      -> ['subagents', ...] (+ toast when a background helper finishes)
 *   browser.updated       -> ['browser', 'status'] (+ refetch ['browser', 'profiles']); browser.frame -> useBrowserView (live view)
 *   node.updated/deleted  -> ['nodes']; node.event battery -> node battery
 *   channel.updated       -> ['channels']; channel.message -> refresh sessions (+ that transcript)
 *   user_model.updated    -> refetch ['user-model']
 *   dream.updated         -> upsert into ['memories', 'dreams'] (and refetch the user model when one completes)
 *   source.items          -> webhook calls refresh ['hooks'] (last called / call count)
 *   stop.updated          -> ['stop'] (Stop everything banner and title bar button)
 */
import type { QueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { qk } from '@/hooks/queryKeys'
import { removeTask, upsertTask } from '@/hooks/tasks'
import { upsertIntegration } from '@/hooks/integrations'
import { upsertDream } from '@/hooks/userModel'
import { browserKeys, useBrowserView } from '@/features/browser/state'
import { channelKeys, upsertChannel } from '@/features/channels/hooks'
import { upsertSubagent } from '@/features/chat/subagents'
import { deviceKeys, removeDevice, upsertDevice } from '@/features/devices/hooks'
import { useChat } from '@/stores/chat'
import { useNotificationStore } from '@/stores/notifications'
import { useUI } from '@/stores/ui'
import { getBridge } from './bridge'
import type { DeviceNode, Dream, Notification, NotificationList, Session, Task } from './types'
import { live, type SocketState } from './ws'

export function notificationRoute(n: Notification): string {
  if (n.task_id) return `/tasks?task=${encodeURIComponent(n.task_id)}`
  if (n.kind === 'skill') return '/skills'
  return '/notifications'
}

function plain(md: string, max = 180): string {
  const s = md.replace(/[`*_#>[\]()!]/g, '').replace(/\s+/g, ' ').trim()
  return s.length > max ? `${s.slice(0, max - 1)}…` : s
}

export function installDomainEvents(qc: QueryClient): () => void {
  const offs: Array<() => void> = []

  const refreshBrief = () => void qc.invalidateQueries({ queryKey: qk.proactivity.brief })
  offs.push(
    live.onDomain('task.updated', (e) => {
      upsertTask(qc, e.data)
      if (e.data.original_context?.source === 'brief') refreshBrief()
    })
  )
  offs.push(
    live.onDomain('task.deleted', (e) => {
      removeTask(qc, e.data.task_id)
      refreshBrief()
    })
  )
  offs.push(
    live.onDomain('task.run_progress', (e) => {
      const { task_id, run_id, update } = e.data
      const patch = (t: Task): Task => ({
        ...t,
        runs: t.runs.map((r) => (r.run_id === run_id ? { ...r, progress_updates: [...r.progress_updates, update], last_activity_at: update.timestamp } : r))
      })
      qc.setQueryData<Task>(qk.tasks.detail(task_id), (old) => (old ? patch(old) : old))
      qc.setQueryData<Task[]>(qk.tasks.all, (old) => old?.map((t) => (t.task_id === task_id ? patch(t) : t)))
      qc.setQueryData<unknown[]>(qk.tasks.runEvents(task_id, run_id), (old) => (old ? [...old, update] : old))
    })
  )
  offs.push(
    live.onDomain('task.run_activity', (e) => {
      const { task_id, run_id, last_activity_at } = e.data
      const patch = (t: Task): Task => ({ ...t, runs: t.runs.map((r) => (r.run_id === run_id ? { ...r, last_activity_at } : r)) })
      qc.setQueryData<Task>(qk.tasks.detail(task_id), (old) => (old ? patch(old) : old))
      qc.setQueryData<Task[]>(qk.tasks.all, (old) => old?.map((t) => (t.task_id === task_id ? patch(t) : t)))
    })
  )
  offs.push(
    live.onDomain('task.run_context', (e) => {
      const { task_id, run_id, used, length, percent, warning } = e.data
      // no length: this model's context length is unknown, so an older meter is cleared
      const context = length ? { used: used ?? 0, length, percent: percent ?? 0, warning } : null
      const patch = (t: Task): Task => ({ ...t, runs: t.runs.map((r) => (r.run_id === run_id ? { ...r, context } : r)) })
      qc.setQueryData<Task>(qk.tasks.detail(task_id), (old) => (old ? patch(old) : old))
      qc.setQueryData<Task[]>(qk.tasks.all, (old) => old?.map((t) => (t.task_id === task_id ? patch(t) : t)))
    })
  )

  offs.push(
    live.onDomain('notification.new', (e) => {
      const n = e.data
      qc.setQueryData<NotificationList>(qk.notifications, (old) =>
        old ? { notifications: [n, ...old.notifications.filter((x) => x.id !== n.id)], unread: old.unread + (n.read ? 0 : 1) } : old
      )
      useNotificationStore.getState().push(n)
      if (n.kind === 'brief') refreshBrief()
      const title = n.title || 'Sentient'
      toast(title, {
        description: plain(n.message),
        action: { label: 'View', onClick: () => useUI.getState().setNotificationsOpen(true) }
      })
      const bridge = getBridge()
      void bridge.isFocused().then((focused) => {
        if (!focused) void bridge.showNotification({ title, body: plain(n.message, 240), route: notificationRoute(n) })
      })
    })
  )
  offs.push(
    live.onDomain('notification.read', (e) => {
      const id = e.data.id
      qc.setQueryData<NotificationList>(qk.notifications, (old) =>
        old
          ? {
              notifications: old.notifications.map((n) => (id === null || n.id === id ? { ...n, read: true } : n)),
              unread: id === null ? 0 : Math.max(0, old.unread - (old.notifications.some((n) => n.id === id && !n.read) ? 1 : 0))
            }
          : old
      )
      useNotificationStore.getState().markRead(id)
    })
  )
  offs.push(
    live.onDomain('notification.updated', (e) => {
      const n = e.data
      qc.setQueryData<NotificationList>(qk.notifications, (old) =>
        old ? { ...old, notifications: old.notifications.map((x) => (x.id === n.id ? n : x)) } : old
      )
      useNotificationStore.getState().update(n)
      if (n.kind === 'brief') refreshBrief()
    })
  )
  offs.push(
    live.onDomain('notification.deleted', (e) => {
      const id = e.data.id
      qc.setQueryData<NotificationList>(qk.notifications, (old) =>
        old ? (id === null ? { notifications: [], unread: 0 } : { ...old, notifications: old.notifications.filter((n) => n.id !== id) }) : old
      )
      useNotificationStore.getState().remove(id)
    })
  )

  offs.push(
    live.onDomain('integration.updated', (e) => {
      upsertIntegration(qc, e.data)
      // feed state changes with connections; the builtin `webhook` integration changes when hooks do
      void qc.invalidateQueries({ queryKey: qk.integrations.feeds })
      if (e.data?.id === 'webhook') void qc.invalidateQueries({ queryKey: qk.hooks })
    })
  )
  offs.push(live.onDomain('memory.updated', () => void qc.invalidateQueries({ queryKey: qk.memories.all })))
  offs.push(live.onDomain('skill.updated', () => void qc.invalidateQueries({ queryKey: qk.skills.all })))

  offs.push(
    live.onDomain('session.updated', (e) => {
      const { session_id, title } = e.data
      let found = false
      qc.setQueryData<Session[]>(qk.sessions, (old) =>
        old?.map((s) => {
          if (s.id !== session_id) return s
          found = true
          return { ...s, title }
        })
      )
      if (!found) void qc.invalidateQueries({ queryKey: qk.sessions })
    })
  )

  offs.push(
    live.onDomain('rule_proposal.updated', (e) => {
      if (e.data?.session_id) void qc.invalidateQueries({ queryKey: qk.ruleProposals(e.data.session_id) })
    })
  )

  offs.push(
    live.onDomain('config.updated', (e) => {
      const sections = e.data?.sections ?? []
      void qc.invalidateQueries({ queryKey: qk.modelPresets })
      void qc.invalidateQueries({ queryKey: qk.chatgpt })
      if (sections.includes('secrets')) {
        void qc.invalidateQueries({ queryKey: qk.secrets })
        void qc.invalidateQueries({ queryKey: qk.providers })
        return
      }
      void qc.invalidateQueries({ queryKey: qk.config })
      void qc.invalidateQueries({ queryKey: qk.bootstrap })
    })
  )

  offs.push(live.onDomain('voice.state', (e) => qc.setQueryData(['voice', 'state'], e.data.state)))

  // §10 subagents ("helpers"): background helpers finishing get a toast.
  offs.push(
    live.onDomain('subagent.updated', (e) => {
      const sub = e.data
      const prev = upsertSubagent(qc, sub)
      const finished = sub.status !== 'running' && (!prev || prev.status === 'running')
      if (!sub.background || !finished) return
      const go = sub.session_id ? () => (window.location.hash = `#/chat/${sub.session_id}`) : undefined
      const description = plain(sub.goal, 120)
      if (sub.status === 'completed') {
        toast.success('A helper finished', { description, action: go ? { label: 'View', onClick: go } : undefined })
      } else if (sub.status === 'error') {
        toast.error('A helper ran into a problem', { description: plain(sub.error ?? sub.goal, 140), action: go ? { label: 'View', onClick: go } : undefined })
      }
      if (sub.session_id) void qc.invalidateQueries({ queryKey: qk.messages(sub.session_id) })
    })
  )

  // §12 browser
  offs.push(
    live.onDomain('browser.updated', (e) => {
      qc.setQueryData(browserKeys.status, e.data)
      void qc.invalidateQueries({ queryKey: browserKeys.profiles })
    })
  )
  offs.push(live.onDomain('browser.frame', (e) => useBrowserView.getState().setFrame(e.data)))

  // §13 devices
  offs.push(live.onDomain('node.updated', (e) => {
    const known = qc.getQueryData<DeviceNode[]>(deviceKeys.all)
    upsertDevice(qc, e.data)
    if (!known) void qc.invalidateQueries({ queryKey: deviceKeys.all })
  }))
  offs.push(live.onDomain('node.deleted', (e) => removeDevice(qc, e.data.node_id)))
  offs.push(
    live.onDomain('node.event', (e) => {
      const { node_id, event, data } = e.data
      if (event === 'battery' && typeof data?.level === 'number') {
        qc.setQueryData<DeviceNode[]>(deviceKeys.all, (old) =>
          old?.map((n) => (n.node_id === node_id ? { ...n, battery: data.level as number, charging: data.charging === true } : n))
        )
      }
    })
  )

  // §14 messaging channels
  offs.push(live.onDomain('channel.updated', (e) => upsertChannel(qc, e.data)))
  offs.push(
    live.onDomain('channel.message', (e) => {
      const { session_id } = e.data
      void qc.invalidateQueries({ queryKey: qk.sessions })
      if (session_id && !useChat.getState().live[session_id]?.streaming && qc.getQueryData(qk.messages(session_id))) {
        void qc.invalidateQueries({ queryKey: qk.messages(session_id) })
      }
    })
  )

  // §15 user model and dreams
  offs.push(
    live.onDomain('user_model.updated', () => {
      void qc.invalidateQueries({ queryKey: qk.userModel })
      void qc.invalidateQueries({ queryKey: qk.memories.review }) // insights waiting for review live there
    })
  )
  offs.push(
    live.onDomain('dream.updated', (e) => {
      const d = e.data
      if (!d?.id) return
      qc.setQueryData<Dream[]>(qk.memories.dreams, (old) => upsertDream(old, d))
      if (d.status === 'completed') void qc.invalidateQueries({ queryKey: qk.userModel })
    })
  )

  // §16 webhooks: a call updates the hook's last-called time and count
  offs.push(
    live.onDomain('source.items', (e) => {
      if (e.data?.origin === 'webhook' || e.data?.source === 'webhook') void qc.invalidateQueries({ queryKey: qk.hooks })
    })
  )

  // §17 stop everything
  offs.push(live.onDomain('stop.updated', (e) => qc.setQueryData(qk.stop, e.data)))

  // After a reconnect we may have missed events: refresh lists (not transcripts).
  let prev: SocketState = live.state
  offs.push(
    live.onState((s) => {
      if (s === 'open' && prev === 'reconnecting') {
        for (const key of [qk.sessions, qk.notifications, qk.tasks.all, qk.integrations.all, qk.bootstrap, qk.stop, deviceKeys.all, channelKeys.all, browserKeys.status, browserKeys.profiles]) {
          void qc.invalidateQueries({ queryKey: key })
        }
      }
      prev = s
    })
  )

  return () => offs.forEach((off) => off())
}
