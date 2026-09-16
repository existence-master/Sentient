import { IconArrowsMaximize, IconChecks, IconTrash } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'
import { ConfirmDialog, IconButton, Sheet } from '@/components/ui'
import { useNotificationActions, useNotifications } from '@/hooks/notifications'
import { useUI } from '@/stores/ui'
import { NotificationFeed } from './NotificationFeed'
import { ProactivityStrip } from './Proactivity'

/**
 * Slide-over opened from the sidebar bell, the command palette, toasts, and `?panel=notifications`
 * on any shell route (used by screenshots). The full page lives at `/notifications`.
 */
export function NotificationsPanel() {
  const open = useUI((s) => s.notificationsOpen)
  const setOpen = useUI((s) => s.setNotificationsOpen)
  const location = useLocation()
  const navigate = useNavigate()
  const { data } = useNotifications()
  const actions = useNotificationActions()
  const [confirmClear, setConfirmClear] = useState(false)
  const count = data?.notifications.length ?? 0

  useEffect(() => {
    if (new URLSearchParams(location.search).get('panel') === 'notifications') setOpen(true)
  }, [location.search, setOpen])

  // The full page already shows everything.
  useEffect(() => {
    if (open && location.pathname.startsWith('/notifications')) setOpen(false)
  }, [open, location.pathname, setOpen])

  const close = () => setOpen(false)

  return (
    <>
      <Sheet
        open={open}
        onOpenChange={setOpen}
        title="Notifications"
        description={data ? (data.unread ? `${data.unread} unread` : 'All caught up') : undefined}
        width={460}
        actions={
          <>
            {count > 0 && (
              <>
                <IconButton size="sm" label="Mark all read" icon={<IconChecks size={16} />} disabled={!data?.unread} onClick={() => actions.markAllRead.mutate()} />
                <IconButton size="sm" label="Clear all" icon={<IconTrash size={16} />} onClick={() => setConfirmClear(true)} />
              </>
            )}
            <IconButton
              size="sm"
              label="Open full page"
              icon={<IconArrowsMaximize size={15} />}
              onClick={() => {
                close()
                navigate('/notifications')
              }}
            />
          </>
        }
      >
        <ProactivityStrip onNavigate={close} />
        <NotificationFeed variant="panel" onNavigate={close} />
      </Sheet>
      <ConfirmDialog
        open={confirmClear}
        onOpenChange={setConfirmClear}
        title="Clear all notifications?"
        description="Every notification is deleted, including suggestions you haven't answered. This can't be undone."
        confirmLabel="Clear all"
        onConfirm={() => actions.clear.mutateAsync().then(() => undefined)}
      />
    </>
  )
}
