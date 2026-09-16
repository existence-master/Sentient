import { IconBell, IconChecks, IconTrash } from '@tabler/icons-react'
import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router'
import { Button, ConfirmDialog, PageHeader } from '@/components/ui'
import { useNotificationActions, useNotifications } from '@/hooks/notifications'
import { NotificationFeed } from './NotificationFeed'
import { LearnedPreferences, ProactivityStatusCard } from './Proactivity'
import { FILTERS, type FeedFilter } from './utils'

/** `/notifications` — `?filter=suggestions|tasks|approvals|skills`, `?focus=learned`. */
export function NotificationsPage() {
  const [params] = useSearchParams()
  const { data } = useNotifications()
  const actions = useNotificationActions()
  const [confirmClear, setConfirmClear] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const filterParam = params.get('filter') as FeedFilter | null
  const initialFilter = FILTERS.some((f) => f.id === filterParam) ? (filterParam as FeedFilter) : 'all'

  useEffect(() => {
    if (params.get('focus') !== 'learned') return
    const t = window.setTimeout(() => root.current?.querySelector('#learned')?.scrollIntoView({ block: 'start' }), 400)
    return () => window.clearTimeout(t)
  }, [params])

  return (
    <div ref={root} className="h-full overflow-y-auto">
      <PageHeader
        icon={<IconBell />}
        title="Notifications"
        description="Suggestions from your apps, task updates, approvals and new skills."
        actions={
          data && data.notifications.length > 0 ? (
            <>
              <Button variant="ghost" leftIcon={<IconChecks size={15} />} disabled={!data.unread} onClick={() => actions.markAllRead.mutate()}>
                Mark all read
              </Button>
              <Button variant="ghost" leftIcon={<IconTrash size={15} />} onClick={() => setConfirmClear(true)}>
                Clear all
              </Button>
            </>
          ) : undefined
        }
      />
      <div className="@container px-8 pb-12">
        <div className="grid items-start gap-6 @3xl:grid-cols-[minmax(0,1fr)_330px]">
          <div className="min-w-0 max-w-3xl">
            <NotificationFeed variant="page" initialFilter={initialFilter} />
          </div>
          <aside className="space-y-4 @3xl:sticky @3xl:top-4">
            <ProactivityStatusCard />
            <LearnedPreferences id="learned" />
          </aside>
        </div>
      </div>
      <ConfirmDialog
        open={confirmClear}
        onOpenChange={setConfirmClear}
        title="Clear all notifications?"
        description="Every notification is deleted, including suggestions you haven't answered. This can't be undone."
        confirmLabel="Clear all"
        onConfirm={() => actions.clear.mutateAsync().then(() => undefined)}
      />
    </div>
  )
}
