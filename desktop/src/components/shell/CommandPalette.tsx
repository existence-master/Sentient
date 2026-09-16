import * as DialogPrimitive from '@radix-ui/react-dialog'
import {
  IconBell,
  IconDeviceMobile,
  IconDevices,
  IconMessages,
  IconWorldWww,
  IconBrain,
  IconEdit,
  IconListCheck,
  IconMessageCircle,
  IconMicrophone,
  IconMoon,
  IconPlugConnected,
  IconSearch,
  IconSettings,
  IconSparkles,
  IconSun,
  type Icon
} from '@tabler/icons-react'
import { Command } from 'cmdk'
import { useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Kbd } from '@/components/ui'
import { openBrowserView } from '@/features/browser/state'
import { SETTINGS_SECTIONS } from '@/features/settings/sections'
import { useConfigEditor } from '@/hooks/config'
import { useSessionSearch, useSessions } from '@/hooks/core'
import { resolveTheme } from '@/lib/theme'
import { modKey, truncate } from '@/lib/utils'
import { useUI } from '@/stores/ui'

export function CommandPalette() {
  const open = useUI((s) => s.paletteOpen)
  const setOpen = useUI((s) => s.setPaletteOpen)
  return (
    <DialogPrimitive.Root open={open} onOpenChange={setOpen}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/45 data-[state=closed]:animate-overlay-out data-[state=open]:animate-overlay-in">
          <DialogPrimitive.Content className="fixed left-1/2 top-[14vh] w-[min(620px,calc(100vw-48px))] -translate-x-1/2 overflow-hidden rounded-2xl border border-border-strong bg-overlay shadow-pop outline-none data-[state=closed]:animate-overlay-out data-[state=open]:animate-overlay-in">
            <DialogPrimitive.Title className="sr-only">Command palette</DialogPrimitive.Title>
            <DialogPrimitive.Description className="sr-only">Search chats, navigate and run commands</DialogPrimitive.Description>
            {open && <PaletteBody close={() => setOpen(false)} />}
          </DialogPrimitive.Content>
        </DialogPrimitive.Overlay>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}

function PaletteBody({ close }: { close: () => void }) {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const sessions = useSessions()
  const hits = useSessionSearch(search)
  const theme = useUI((s) => s.theme)
  const { setValue } = useConfigEditor()

  const go = (to: string, state?: unknown) => {
    close()
    navigate(to, state ? { state } : undefined)
  }

  const toggleTheme = () => {
    const next = resolveTheme(theme) === 'dark' ? 'light' : 'dark'
    useUI.getState().setTheme(next)
    setValue('ui.theme', next, { immediate: true })
    close()
    toast.message(`Switched to ${next} theme`)
  }

  return (
    <Command loop className="flex max-h-[min(520px,70vh)] flex-col">
      <div className="flex items-center gap-2.5 border-b border-border px-4">
        <IconSearch size={17} className="text-fg-subtle" />
        <Command.Input
          autoFocus
          value={search}
          onValueChange={setSearch}
          placeholder="Search chats, pages, settings…"
          className="h-13 flex-1 bg-transparent text-md text-fg outline-none placeholder:text-fg-subtle"
        />
        <Kbd>Esc</Kbd>
      </div>
      <Command.List className="min-h-0 flex-1 overflow-y-auto p-1.5">
        <Command.Empty className="py-10 text-center text-sm text-fg-subtle">Nothing found.</Command.Empty>

        <Command.Group heading="Actions">
          <Item icon={IconEdit} onSelect={() => go('/chat', { fresh: Date.now() })} shortcut={`${modKey}+N`}>
            New chat
          </Item>
          <Item icon={IconListCheck} onSelect={() => go('/tasks?compose=1')} keywords={['task', 'schedule', 'recurring', 'automation']}>
            New task
          </Item>
          <Item icon={IconMicrophone} onSelect={() => go('/voice')}>
            Voice mode
          </Item>
          <Item icon={IconDeviceMobile} onSelect={() => go('/devices', { add: Date.now() })} keywords={['pair', 'phone', 'glasses', 'qr']}>
            Add a device
          </Item>
          <Item
            icon={IconWorldWww}
            onSelect={() => {
              close()
              openBrowserView()
            }}
            keywords={['browser', 'live view', 'sign in', 'website']}
          >
            Open browser live view
          </Item>
          <Item icon={resolveTheme(theme) === 'dark' ? IconSun : IconMoon} onSelect={toggleTheme} keywords={['dark', 'light', 'appearance']}>
            Toggle theme
          </Item>
          <Item
            icon={IconBell}
            onSelect={() => {
              close()
              useUI.getState().setNotificationsOpen(true)
            }}
          >
            Open notifications
          </Item>
        </Command.Group>

        <Command.Group heading="Go to">
          <Item icon={IconMessageCircle} onSelect={() => go('/chat')}>
            Chat
          </Item>
          <Item icon={IconListCheck} onSelect={() => go('/tasks')}>
            Tasks
          </Item>
          <Item icon={IconBrain} onSelect={() => go('/memory')}>
            Memory
          </Item>
          <Item icon={IconPlugConnected} onSelect={() => go('/integrations')}>
            Integrations
          </Item>
          <Item icon={IconDevices} onSelect={() => go('/devices')} keywords={['phone', 'glasses', 'watch', 'pair']}>
            Devices
          </Item>
          <Item icon={IconMessages} onSelect={() => go('/devices/messaging')} keywords={['telegram', 'discord', 'bot', 'channels']}>
            Messaging apps
          </Item>
          <Item icon={IconSparkles} onSelect={() => go('/skills')}>
            Skills
          </Item>
          <Item icon={IconSettings} onSelect={() => go('/settings')} shortcut={`${modKey}+,`}>
            Settings
          </Item>
        </Command.Group>

        <Command.Group heading="Settings">
          {SETTINGS_SECTIONS.map((s) => (
            <Item key={s.id} icon={s.icon} keywords={s.keywords} onSelect={() => go(`/settings/${s.id}`)} hint={s.description}>
              {s.label}
            </Item>
          ))}
        </Command.Group>

        {!!sessions.data?.length && (
          <Command.Group heading="Recent chats">
            {sessions.data.slice(0, 30).map((s) => (
              <Item key={s.id} icon={IconMessageCircle} value={`chat ${s.id} ${s.title ?? 'New chat'}`} onSelect={() => go(`/chat/${s.id}`)}>
                {s.title?.trim() || 'New chat'}
              </Item>
            ))}
          </Command.Group>
        )}

        {!!hits.data?.length && (
          <Command.Group heading="In messages">
            {hits.data.slice(0, 12).map((h) => (
              <Item
                key={h.message_id}
                icon={IconSearch}
                value={`hit ${h.message_id} ${search}`}
                onSelect={() => go(`/chat/${h.session_id}`)}
                hint={h.role === 'user' ? 'You' : 'Sentient'}
              >
                {truncate(h.snippet.replace(/\s+/g, ' '), 90)}
              </Item>
            ))}
          </Command.Group>
        )}
      </Command.List>
      <div className="flex items-center gap-4 border-t border-border px-4 py-2 text-2xs text-fg-subtle">
        <span className="flex items-center gap-1">
          <Kbd>↑</Kbd>
          <Kbd>↓</Kbd> navigate
        </span>
        <span className="flex items-center gap-1">
          <Kbd>Enter</Kbd> open
        </span>
      </div>
    </Command>
  )
}

function Item({
  icon: IconCmp,
  children,
  onSelect,
  shortcut,
  hint,
  keywords,
  value
}: {
  icon: Icon
  children: ReactNode
  onSelect: () => void
  shortcut?: string
  hint?: string
  keywords?: string[]
  value?: string
}) {
  return (
    <Command.Item
      value={value ?? (typeof children === 'string' ? children : undefined)}
      keywords={keywords}
      onSelect={onSelect}
      className="flex h-9.5 cursor-pointer items-center gap-3 rounded-lg px-2.5 text-sm text-fg"
    >
      <IconCmp size={16} stroke={1.75} className="shrink-0 text-fg-muted" />
      <span className="min-w-0 truncate">{children}</span>
      {hint && <span className="min-w-0 flex-1 truncate text-xs text-fg-subtle">{hint}</span>}
      {!hint && <span className="flex-1" />}
      {shortcut && <span className="shrink-0 text-xs text-fg-subtle">{shortcut}</span>}
    </Command.Item>
  )
}
