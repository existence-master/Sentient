import {
  IconBell,
  IconBrain,
  IconDevices,
  IconDots,
  IconEdit,
  IconListCheck,
  IconMessageCircle,
  IconMicrophone,
  IconPencil,
  IconPlugConnected,
  IconPlus,
  IconSearch,
  IconSettings,
  IconSparkles,
  IconTrash,
  IconUserHeart,
  IconX,
  type Icon
} from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useMemo, useState } from 'react'
import { NavLink, useLocation, useNavigate, useParams } from 'react-router'
import { toast } from 'sonner'
import { Logo } from '@/components/brand/Logo'
import {
  ConfirmDialog,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  IconButton,
  Input,
  Kbd,
  ScrollArea,
  Skeleton,
  Tooltip
} from '@/components/ui'
import { ChannelBadge } from '@/features/channels/meta'
import { TasksNavBadge } from '@/features/tasks/TasksNavBadge'
import { SkillsNavBadge } from '@/features/skills/SkillsNavBadge'
import { MemoryNavBadge } from '@/features/memory/MemoryNavBadge'
import { useBootstrap, useDeleteSession, useRenameSession, useSessionSearch, useSessions } from '@/hooks/core'
import { errorMessage } from '@/lib/api'
import { modelShortName } from '@/lib/models'
import type { Session } from '@/lib/types'
import { cn, dateBucket, modKey } from '@/lib/utils'
import { useChat } from '@/stores/chat'
import { useNotificationStore } from '@/stores/notifications'
import { useUI } from '@/stores/ui'

const NAV: Array<{ to: string; label: string; icon: Icon; match: string }> = [
  { to: '/chat', label: 'Chat', icon: IconMessageCircle, match: '/chat' },
  { to: '/tasks', label: 'Tasks', icon: IconListCheck, match: '/tasks' },
  { to: '/memory', label: 'Memory', icon: IconBrain, match: '/memory' },
  { to: '/integrations', label: 'Integrations', icon: IconPlugConnected, match: '/integrations' },
  { to: '/devices', label: 'Devices', icon: IconDevices, match: '/devices' },
  { to: '/skills', label: 'Skills', icon: IconSparkles, match: '/skills' },
  { to: '/about', label: 'About you', icon: IconUserHeart, match: '/about' }
]

export const SIDEBAR_WIDTH = 256
export const SIDEBAR_COLLAPSED = 60

export function Sidebar() {
  const collapsed = useUI((s) => s.sidebarCollapsed)
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? SIDEBAR_COLLAPSED : SIDEBAR_WIDTH }}
      transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}
      className="flex h-full shrink-0 flex-col overflow-hidden bg-bg"
    >
      <div className={cn('flex flex-col gap-0.5 pb-2 pt-1', collapsed ? 'items-center px-2' : 'px-2.5')}>
        {collapsed ? (
          <IconButton
            label="New chat"
            shortcut={`${modKey}+N`}
            side="right"
            variant="secondary"
            size="md"
            icon={<IconPlus size={16} />}
            onClick={() => navigate('/chat', { state: { fresh: Date.now() } })}
            className="mb-1.5"
          />
        ) : (
          <button
            type="button"
            onClick={() => navigate('/chat', { state: { fresh: Date.now() } })}
            className="group mb-1.5 flex h-8.5 items-center gap-2 rounded-lg border border-border-strong bg-surface px-2.5 text-sm font-medium text-fg shadow-soft transition-colors hover:bg-elevated"
          >
            <IconEdit size={16} className="text-accent-text" />
            <span className="flex-1 text-left">New chat</span>
            <span className="flex gap-0.5 opacity-70 transition-opacity group-hover:opacity-100">
              <Kbd>{modKey}</Kbd>
              <Kbd>N</Kbd>
            </span>
          </button>
        )}
        {NAV.map((item) => (
          <NavItem key={item.to} {...item} collapsed={collapsed} active={location.pathname.startsWith(item.match)} />
        ))}
      </div>

      {collapsed ? <div className="flex-1" /> : <ChatList />}

      <SidebarFooter collapsed={collapsed} />
    </motion.aside>
  )
}

function NavItem({ to, label, icon: IconCmp, collapsed, active }: { to: string; label: string; icon: Icon; collapsed: boolean; active: boolean }) {
  const link = (
    <NavLink
      to={to}
      className={cn(
        'group relative flex h-8 items-center gap-2.5 rounded-lg text-sm transition-colors',
        collapsed ? 'w-10 justify-center' : 'px-2.5',
        active ? 'bg-active text-fg' : 'text-fg-muted hover:bg-hover hover:text-fg'
      )}
    >
      <IconCmp size={17} stroke={1.75} className={cn(active ? 'text-accent-text' : 'text-fg-subtle group-hover:text-fg-muted')} />
      {!collapsed && <span className="truncate">{label}</span>}
      {to === '/tasks' && <TasksNavBadge collapsed={collapsed} />}
      {to === '/skills' && <SkillsNavBadge collapsed={collapsed} />}
      {to === '/memory' && <MemoryNavBadge collapsed={collapsed} />}
    </NavLink>
  )
  return collapsed ? (
    <Tooltip content={label} side="right">
      {link}
    </Tooltip>
  ) : (
    link
  )
}

// ---------------------------------------------------------------------------- chat list
function ChatList() {
  const { data: sessions, isLoading } = useSessions()
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const search = useSessionSearch(query)
  const q = query.trim().toLowerCase()

  const filtered = useMemo(() => {
    if (!sessions) return []
    if (!q) return sessions
    const hitIds = new Set((search.data ?? []).map((h) => h.session_id))
    return sessions.filter((s) => (s.title ?? 'New chat').toLowerCase().includes(q) || hitIds.has(s.id))
  }, [sessions, q, search.data])

  const groups = useMemo(() => {
    const out: Array<{ label: string; items: Session[] }> = []
    for (const s of filtered) {
      const label = dateBucket(s.updated_at)
      const g = out.find((x) => x.label === label)
      if (g) g.items.push(s)
      else out.push({ label, items: [s] })
    }
    return out
  }, [filtered])

  return (
    <div className="flex min-h-0 flex-1 flex-col border-t border-border pt-2">
      <div className="flex h-7 items-center gap-1 px-4 pr-2.5">
        {searching ? (
          <Input
            autoFocus
            size="sm"
            value={query}
            placeholder="Search chats"
            leftIcon={<IconSearch />}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') {
                setQuery('')
                setSearching(false)
              }
            }}
            rightSlot={
              <button
                type="button"
                aria-label="Close search"
                onClick={() => {
                  setQuery('')
                  setSearching(false)
                }}
                className="flex size-5 items-center justify-center rounded text-fg-subtle hover:text-fg"
              >
                <IconX size={12} />
              </button>
            }
          />
        ) : (
          <>
            <span className="flex-1 text-2xs font-medium uppercase tracking-wider text-fg-subtle">Chats</span>
            <IconButton size="xs" label="Search chats" icon={<IconSearch size={13} />} onClick={() => setSearching(true)} />
          </>
        )}
      </div>
      <ScrollArea className="min-h-0 flex-1" viewportClassName="px-2.5 pb-3">
        {isLoading ? (
          <div className="space-y-1.5 px-1.5 pt-2">
            {['w-4/5', 'w-11/12', 'w-3/5', 'w-4/5'].map((w, i) => (
              <Skeleton key={i} className={`h-5 ${w}`} />
            ))}
          </div>
        ) : !filtered.length ? (
          <p className="px-1.5 py-4 text-xs text-fg-subtle">{q ? 'No chats match.' : 'Your conversations will show up here.'}</p>
        ) : (
          groups.map((g) => (
            <div key={g.label} className="pt-2">
              <div className="px-1.5 pb-1 text-2xs font-medium text-fg-faint">{g.label}</div>
              <AnimatePresence initial={false}>
                {g.items.map((s) => (
                  <ChatListItem key={s.id} session={s} />
                ))}
              </AnimatePresence>
            </div>
          ))
        )}
      </ScrollArea>
    </div>
  )
}

function ChatListItem({ session }: { session: Session }) {
  const { sessionId } = useParams()
  const navigate = useNavigate()
  const active = sessionId === session.id
  const streaming = useChat((s) => !!s.live[session.id]?.streaming)
  const rename = useRenameSession()
  const remove = useDeleteSession()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(session.title ?? '')
  const [confirm, setConfirm] = useState(false)
  const title = session.title?.trim() || 'New chat'

  const commit = () => {
    const t = draft.trim()
    setEditing(false)
    if (t && t !== session.title) {
      rename.mutate({ id: session.id, title: t }, { onError: (e) => toast.error("Couldn't rename", { description: errorMessage(e) }) })
    }
  }

  return (
    <motion.div layout="position" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, height: 0 }}>
      {editing ? (
        <input
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commit()
            if (e.key === 'Escape') setEditing(false)
          }}
          className="h-8 w-full rounded-lg border border-accent/50 bg-field px-2 text-sm text-fg outline-none ring-3 ring-accent/15"
        />
      ) : (
        <div
          role="link"
          tabIndex={0}
          onClick={() => navigate(`/chat/${session.id}`)}
          onDoubleClick={() => {
            setDraft(session.title ?? '')
            setEditing(true)
          }}
          onKeyDown={(e) => e.key === 'Enter' && navigate(`/chat/${session.id}`)}
          className={cn(
            'group flex h-8 cursor-pointer items-center gap-2 rounded-lg pl-2 pr-1 text-sm transition-colors',
            active ? 'bg-active text-fg' : 'text-fg-muted hover:bg-hover hover:text-fg'
          )}
        >
          {streaming && <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-accent" />}
          <ChannelBadge channel={session.channel} />
          <span className="min-w-0 flex-1 truncate">{title}</span>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                aria-label="Chat actions"
                onClick={(e) => e.stopPropagation()}
                className={cn(
                  'flex size-6 shrink-0 items-center justify-center rounded-md text-fg-subtle hover:bg-active hover:text-fg data-[state=open]:opacity-100',
                  active ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                )}
              >
                <IconDots size={15} />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" onClick={(e) => e.stopPropagation()}>
              <DropdownMenuItem
                icon={<IconPencil />}
                onSelect={() => {
                  setDraft(session.title ?? '')
                  setTimeout(() => setEditing(true), 0)
                }}
              >
                Rename
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem icon={<IconTrash />} danger onSelect={() => setConfirm(true)}>
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      )}
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title="Delete this chat?"
        description={`“${title}” and its messages will be permanently removed. Memories Sentient learned from it are kept.`}
        confirmLabel="Delete chat"
        onConfirm={async () => {
          await remove.mutateAsync(session.id)
          if (active) navigate('/chat', { replace: true })
        }}
      />
    </motion.div>
  )
}

// ---------------------------------------------------------------------------- footer
function SidebarFooter({ collapsed }: { collapsed: boolean }) {
  const navigate = useNavigate()
  const location = useLocation()
  const unread = useNotificationStore((s) => s.unread)
  const setNotificationsOpen = useUI((s) => s.setNotificationsOpen)
  const { data } = useBootstrap()

  const bell = (
    <div className="relative">
      <IconButton
        label={unread ? `Notifications (${unread} unread)` : 'Notifications'}
        side={collapsed ? 'right' : 'top'}
        icon={<IconBell size={17} stroke={1.75} />}
        onClick={() => setNotificationsOpen(true)}
      />
      {unread > 0 && (
        <span className="pointer-events-none absolute right-0.5 top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-accent px-1 text-[10px] font-semibold leading-none text-accent-fg">
          {unread > 99 ? '99+' : unread}
        </span>
      )}
    </div>
  )
  const voice = (
    <IconButton label="Voice mode" side={collapsed ? 'right' : 'top'} icon={<IconMicrophone size={17} stroke={1.75} />} onClick={() => navigate('/voice')} />
  )
  const settings = (
    <IconButton
      label="Settings"
      shortcut={`${modKey}+,`}
      side={collapsed ? 'right' : 'top'}
      active={location.pathname.startsWith('/settings')}
      icon={<IconSettings size={17} stroke={1.75} />}
      onClick={() => navigate('/settings')}
    />
  )

  if (collapsed) {
    return (
      <div className="flex flex-col items-center gap-1 border-t border-border py-2.5">
        {bell}
        {voice}
        {settings}
      </div>
    )
  }

  return (
    <div className="border-t border-border p-2.5">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => navigate('/settings/models')}
          className="flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-1.5 py-1 text-left transition-colors hover:bg-hover"
        >
          <Logo size={26} />
          <div className="min-w-0">
            <div className="truncate text-sm font-medium text-fg">{data?.assistant.name ?? 'Sentient'}</div>
            <div className="truncate font-mono text-2xs text-fg-subtle">{modelShortName(data?.models.primary) || '…'}</div>
          </div>
        </button>
        {bell}
        {voice}
        {settings}
      </div>
    </div>
  )
}
