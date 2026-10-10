/** "Send results to" (docs/API.md §4 "Where results go"): the usual paired chats, this computer only, or chosen chats. */
import { IconCheck, IconChevronDown, IconDeviceDesktop, IconMessages, IconSend } from '@tabler/icons-react'
import { useState, type ReactNode } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui'
import { useChannels } from '@/features/channels/hooks'
import type { Channel, DeliveryChat, Task, TaskDeliverTo } from '@/lib/types'
import { cn } from '@/lib/utils'
import type { useTaskOps } from '../useTaskOps'

type Ops = ReturnType<typeof useTaskOps>

const NAMES: Record<string, string> = { telegram: 'Telegram', discord: 'Discord', whatsapp: 'WhatsApp' }
const SELF = 'self'
const MAX_CHATS = 10 // the engine's limit (tasks/delivery.py)

interface ChatOption {
  chat: DeliveryChat
  label: string
  note?: string
}

const same = (a: DeliveryChat, b: DeliveryChat) => a.channel === b.channel && a.chat_id === b.chat_id

/** The paired WhatsApp "Message yourself" chat id, if WhatsApp is linked. */
function whatsappSelf(channels: Channel[]): string | undefined {
  return channels.find((c) => c.id === 'whatsapp')?.paired.find((p) => p.label === 'Message yourself')?.chat_id
}

/** `whatsapp`/`self` is the same chat as the paired "Message yourself" chat. */
function normalize(chat: DeliveryChat, selfId: string | undefined): DeliveryChat {
  return chat.channel === 'whatsapp' && chat.chat_id === SELF && selfId ? { channel: 'whatsapp', chat_id: selfId } : chat
}

/** Every paired chat, plus WhatsApp's "Message yourself" and any saved chat that is no longer paired. */
function chatOptions(channels: Channel[], current: TaskDeliverTo): ChatOption[] {
  const out: ChatOption[] = []
  for (const ch of channels) {
    for (const p of ch.paired) out.push({ chat: { channel: ch.id, chat_id: p.chat_id }, label: `${ch.display_name || NAMES[ch.id] || ch.id}: ${p.label}` })
  }
  const selfId = whatsappSelf(channels)
  if (!selfId) {
    out.push({ chat: { channel: 'whatsapp', chat_id: SELF }, label: 'WhatsApp: Message yourself', note: 'Link WhatsApp in Channels first' })
  }
  for (const c of Array.isArray(current) ? current : []) {
    if (out.some((o) => same(o.chat, normalize(c, selfId)))) continue
    out.push({ chat: c, label: c.chat_id === SELF ? 'WhatsApp: Message yourself' : `${NAMES[c.channel] ?? c.channel}: ${c.chat_id}`, note: 'No longer paired' })
  }
  return out
}

function currentLabel(current: TaskDeliverTo, options: ChatOption[], selfId: string | undefined): string {
  if (current === 'desktop') return 'This computer only'
  if (current === 'default' || !current.length) return 'Paired chats'
  if (current.length > 1) return `${current.length} chats`
  return options.find((o) => same(o.chat, normalize(current[0], selfId)))?.label ?? current[0].chat_id
}

export function DeliverToRow({ task, ops, compact }: { task: Task; ops: Ops; compact?: boolean }) {
  const channels = useChannels()
  const [open, setOpen] = useState(false)
  const current: TaskDeliverTo = task.deliver_to ?? 'default'
  const options = chatOptions(channels.data ?? [], current)
  const selfId = whatsappSelf(channels.data ?? [])
  const chosen = Array.isArray(current) ? current : []
  const isOn = (chat: DeliveryChat) => chosen.some((c) => same(normalize(c, selfId), chat))

  const save = (next: TaskDeliverTo) => void ops.update(task.task_id, { deliver_to: next }, 'Delivery updated')
  const toggle = (chat: DeliveryChat) => {
    if (!isOn(chat) && chosen.length >= MAX_CHATS) return
    const next = isOn(chat) ? chosen.filter((c) => !same(normalize(c, selfId), chat)) : [...chosen, chat]
    save(next.length ? next : 'desktop')
  }

  const picker = (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label="Send results to"
          className={cn(
            'flex max-w-56 items-center gap-1.5 truncate rounded-lg border border-border bg-surface text-left text-fg-muted transition-colors hover:bg-hover hover:text-fg',
            compact ? 'h-6 px-2 text-xs' : 'h-8 px-2.5 text-sm'
          )}
        >
          <span className="truncate">{currentLabel(current, options, selfId)}</span>
          <IconChevronDown size={13} className="shrink-0 text-fg-subtle" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 p-1.5">
        <Choice active={current === 'default'} icon={<IconMessages size={15} />} title="Paired chats" body="Every paired chat with delivery turned on, as set in Channels." onClick={() => save('default')} />
        <Choice active={current === 'desktop'} icon={<IconDeviceDesktop size={15} />} title="This computer only" body="Results show here and never go to a messaging app." onClick={() => save('desktop')} />
        <div className="mx-2 mb-1 mt-2 flex items-center gap-1.5 text-2xs font-semibold uppercase tracking-wider text-fg-subtle">
          <IconSend size={11} /> Only these chats
        </div>
        {options.length === 0 ? (
          <p className="px-2.5 pb-2 text-xs text-fg-subtle">No paired chats yet. Pair one in Channels.</p>
        ) : (
          options.map((o) => {
            const on = isOn(o.chat)
            return (
              <button
                key={`${o.chat.channel}:${o.chat.chat_id}`}
                type="button"
                role="menuitemcheckbox"
                aria-checked={on}
                disabled={!on && chosen.length >= MAX_CHATS}
                onClick={() => toggle(o.chat)}
                className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-left hover:bg-hover disabled:opacity-50"
              >
                <span className={cn('flex size-4 shrink-0 items-center justify-center rounded border', on ? 'border-accent bg-accent text-accent-fg' : 'border-border-strong')}>
                  {on && <IconCheck size={10} stroke={3} />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-fg">{o.label}</span>
                  {o.note && <span className="block text-xs text-fg-subtle">{o.note}</span>}
                </span>
              </button>
            )
          })
        )}
      </PopoverContent>
    </Popover>
  )

  if (compact) {
    return (
      <span className="flex items-center gap-1.5 text-fg-subtle">
        Send results to {picker}
      </span>
    )
  }
  return (
    <div className="flex items-center justify-between gap-3 px-4 py-2.5">
      <dt className="shrink-0 text-fg-subtle">Send results to</dt>
      <dd className="min-w-0">{picker}</dd>
    </div>
  )
}

function Choice({ active, icon, title, body, onClick }: { active: boolean; icon: ReactNode; title: string; body: string; onClick: () => void }) {
  return (
    <button
      type="button"
      role="menuitemradio"
      aria-checked={active}
      onClick={onClick}
      className={cn('flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors', active ? 'bg-accent/10' : 'hover:bg-hover')}
    >
      <span className={cn('mt-0.5 shrink-0', active ? 'text-accent-text' : 'text-fg-subtle')}>{icon}</span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium text-fg">{title}</span>
        <span className="block text-xs text-fg-subtle">{body}</span>
      </span>
      {active && <IconCheck size={14} className="mt-1 shrink-0 text-accent-text" />}
    </button>
  )
}
