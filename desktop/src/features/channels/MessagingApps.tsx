import {
  IconCircleCheckFilled,
  IconEye,
  IconEyeOff,
  IconMessageCircle,
  IconMessages,
  IconPlus,
  IconSend,
  IconTrash
} from '@tabler/icons-react'
import { motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import { Alert, Badge, Button, ConfirmDialog, Dialog, EmptyState, Field, IconButton, Input, Skeleton, Spinner, Switch, type Tone } from '@/components/ui'
import { CopyChip, InstructionsGuide } from '@/features/integrations/InstructionsGuide'
import { errorMessage, isNotImplemented } from '@/lib/api'
import type { Channel, ChannelPairing, PairedChat } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'
import { previewMode } from '@/features/devices/preview'
import { useChannelActions, useChannels } from './hooks'
import { ChannelTile } from './meta'
import { WhatsAppLinkDialog } from './WhatsAppLink'

const STATUS: Record<Channel['status'], { label: string; tone: Tone }> = {
  connected: { label: 'Connected', tone: 'success' },
  connecting: { label: 'Connecting', tone: 'warning' },
  linking: { label: 'Waiting for scan', tone: 'warning' },
  disconnected: { label: 'Not connected', tone: 'neutral' },
  error: { label: 'Needs attention', tone: 'danger' }
}

const BLURB: Record<string, string> = {
  telegram: 'Chat with Sentient from Telegram on any phone, get task results and approve actions with a tap.',
  discord: 'Talk to Sentient from a Discord server or direct messages, and get updates there.',
  whatsapp: 'Talk to Sentient in your own WhatsApp "Message yourself" chat, with voice notes and updates.'
}

/** "Messaging apps" tab: Telegram and Discord bots, and WhatsApp linked to your own account (§14). */
export function MessagingApps() {
  const { data, isLoading, error } = useChannels()
  const [connect, setConnect] = useState<Channel | null>(null)
  const [pair, setPair] = useState<Channel | null>(null)

  if (isLoading) {
    return (
      <div className="grid gap-4 xl:grid-cols-2">
        {[0, 1].map((i) => (
          <Skeleton key={i} className="h-44 rounded-2xl" />
        ))}
      </div>
    )
  }
  if (error && isNotImplemented(error) && !previewMode()) {
    return (
      <EmptyState
        icon={<IconMessages />}
        title="Messaging apps are on their way"
        description="Soon you'll be able to talk to Sentient from Telegram and Discord. This version of the engine doesn't support it yet."
      />
    )
  }
  if (error) {
    return <Alert tone="danger" title="Couldn't load messaging apps">{errorMessage(error)}</Alert>
  }

  const channels = data ?? []
  return (
    <>
      <p className="mb-4 max-w-2xl text-sm text-fg-muted">
        Connect an app, then pair your chat. Only your paired chats can talk to Sentient. Anyone else gets a polite no, and on WhatsApp your other chats are simply left alone.
      </p>
      <div className="grid items-start gap-4 xl:grid-cols-2">
        {channels.map((c) => (
          <ChannelCard key={c.id} channel={c} onConnect={() => setConnect(c)} onPair={() => setPair(c)} />
        ))}
      </div>
      <WhatsAppLinkDialog channel={connect?.id === 'whatsapp' ? connect : null} onClose={() => setConnect(null)} />
      <ConnectChannelDialog
        channel={connect?.id === 'whatsapp' ? null : connect}
        onClose={() => setConnect(null)}
        onConnected={(c) => {
          setConnect(null)
          setPair(c)
        }}
      />
      <PairChatDialog channel={pair} onClose={() => setPair(null)} />
    </>
  )
}

function ChannelCard({ channel, onConnect, onPair }: { channel: Channel; onConnect: () => void; onPair: () => void }) {
  const actions = useChannelActions()
  const [confirm, setConfirm] = useState(false)
  const status = STATUS[channel.status] ?? STATUS.disconnected
  const connected = channel.status === 'connected'
  const whatsapp = channel.id === 'whatsapp'
  const account = whatsapp ? `Linked to ${channel.account_label}` : `Your bot: ${channel.account_label}`

  return (
    <motion.div layout initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-border bg-surface">
      <div className="flex items-start gap-3.5 p-4">
        <ChannelTile channel={channel.id} size={44} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-md font-medium text-fg">{channel.display_name}</span>
            <Badge size="xs" tone={status.tone}>
              {status.label}
            </Badge>
          </div>
          <div className="mt-0.5 text-sm text-fg-subtle">{connected && channel.account_label ? account : BLURB[channel.id] ?? ''}</div>
        </div>
        {connected ? (
          <Button size="sm" variant="ghost" onClick={() => setConfirm(true)}>
            Disconnect
          </Button>
        ) : (
          <Button size="sm" variant="primary" onClick={onConnect} loading={channel.status === 'connecting'}>
            {channel.status === 'linking' ? 'Show code' : channel.status === 'error' ? 'Reconnect' : 'Connect'}
          </Button>
        )}
      </div>

      {channel.status === 'error' && channel.error && (
        <div className="px-4 pb-4">
          <Alert tone="danger">{channel.error}</Alert>
        </div>
      )}

      {connected && (
        <div className="border-t border-border px-4 py-3">
          <div className="mb-2 flex items-center">
            <span className="flex-1 text-2xs font-medium uppercase tracking-wide text-fg-subtle">Paired chats</span>
            <Button size="xs" variant="ghost" leftIcon={<IconPlus size={13} />} onClick={onPair}>
              {whatsapp ? 'Pair another chat' : 'Pair a chat'}
            </Button>
          </div>
          {channel.paired.length ? (
            <div className="divide-y divide-border rounded-xl border border-border">
              {channel.paired.map((p) => (
                <PairedRow key={p.chat_id} channel={channel} chat={p} />
              ))}
            </div>
          ) : (
            <button
              type="button"
              onClick={onPair}
              className="flex w-full items-center gap-3 rounded-xl border border-dashed border-border-strong px-3.5 py-3 text-left text-sm text-fg-muted transition-colors hover:bg-hover"
            >
              <IconMessageCircle size={17} className="text-fg-subtle" />
              No chats yet. Pair your {channel.display_name} chat to start talking to Sentient there.
            </button>
          )}
        </div>
      )}

      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={`Disconnect ${channel.display_name}?`}
        description={
          whatsapp
            ? 'Sentient is removed from Linked devices on your phone and stops answering. Your chats and their history stay here. To use WhatsApp again, scan a new code.'
            : 'Sentient stops answering messages there. Your paired chats and their history stay, and come back if you reconnect the same bot.'
        }
        confirmLabel="Disconnect"
        onConfirm={async () => {
          try {
            await actions.disconnect.mutateAsync(channel.id)
          } catch (e) {
            toast.error("Couldn't disconnect", { description: errorMessage(e) })
          }
        }}
      />
    </motion.div>
  )
}

function PairedRow({ channel, chat }: { channel: Channel; chat: PairedChat }) {
  const actions = useChannelActions()
  const navigate = useNavigate()
  const [confirm, setConfirm] = useState(false)

  return (
    <div className="flex items-center gap-3 px-3 py-2.5">
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium text-fg">{chat.label}</div>
        <div className="text-xs text-fg-subtle">Paired {relativeTime(chat.paired_at)}</div>
      </div>
      <label className="flex cursor-pointer items-center gap-2 text-xs text-fg-muted" title="Task results, plans waiting for approval and suggestions are also sent here">
        Send updates here
        <Switch
          size="sm"
          checked={chat.deliver}
          onCheckedChange={(deliver) =>
            actions.setDeliver.mutate(
              { id: channel.id, chatId: chat.chat_id, deliver },
              { onError: (e) => toast.error("Couldn't change that", { description: errorMessage(e) }) }
            )
          }
        />
      </label>
      <div className="flex items-center">
        {chat.session_id && (
          <IconButton size="sm" label="Open this chat in Sentient" icon={<IconMessageCircle size={15} />} onClick={() => navigate(`/chat/${chat.session_id}`)} />
        )}
        <IconButton
          size="sm"
          label="Send a test message"
          loading={actions.test.isPending && actions.test.variables?.chatId === chat.chat_id}
          icon={<IconSend size={15} />}
          onClick={() =>
            actions.test.mutate(
              { id: channel.id, chatId: chat.chat_id },
              {
                onSuccess: (r) =>
                  r.ok ? toast.success(`Sent a test message to ${chat.label}`) : toast.error("The test message didn't arrive", { description: r.error }),
                onError: (e) => toast.error("The test message didn't arrive", { description: errorMessage(e) })
              }
            )
          }
        />
        <IconButton size="sm" label="Remove" icon={<IconTrash size={15} />} onClick={() => setConfirm(true)} />
      </div>
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={`Remove ${chat.label}?`}
        description="Sentient stops answering this chat. You can pair it again with a new code."
        confirmLabel="Remove chat"
        onConfirm={async () => {
          try {
            await actions.removePaired.mutateAsync({ id: channel.id, chatId: chat.chat_id })
          } catch (e) {
            toast.error("Couldn't remove the chat", { description: errorMessage(e) })
          }
        }}
      />
    </div>
  )
}

function ConnectChannelDialog({ channel, onClose, onConnected }: { channel: Channel | null; onClose: () => void; onConnected: (c: Channel) => void }) {
  const actions = useChannelActions()
  const [values, setValues] = useState<Record<string, string>>({})
  const [reveal, setReveal] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setValues({})
    setError(null)
    setReveal(false)
  }, [channel?.id])

  const fields = channel?.setup.fields.length ? channel.setup.fields : [{ key: 'bot_token', label: 'Bot token', secret: true, required: true }]
  const missing = fields.some((f) => f.required && !values[f.key]?.trim())

  const submit = () => {
    if (!channel || missing) return
    setError(null)
    const payload = Object.fromEntries(Object.entries(values).map(([k, v]) => [k, v.trim()]))
    actions.connect.mutate(
      { id: channel.id, fields: payload },
      {
        onSuccess: (c) => {
          toast.success(`${c.display_name} is connected${c.account_label ? ` as ${c.account_label}` : ''}`)
          onConnected(c)
        },
        onError: (e) => setError(errorMessage(e))
      }
    )
  }

  return (
    <Dialog
      open={!!channel}
      onOpenChange={(o) => !o && onClose()}
      size="lg"
      modalLock
      title={
        channel ? (
          <span className="flex items-center gap-3">
            <ChannelTile channel={channel.id} size={34} />
            Connect {channel.display_name}
          </span>
        ) : (
          'Connect'
        )
      }
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" disabled={missing} loading={actions.connect.isPending} onClick={submit}>
            Connect
          </Button>
        </>
      }
    >
      {channel && (
        <div className="space-y-5 pb-1">
          <div className="rounded-xl border border-border bg-surface p-4">
            <InstructionsGuide markdown={channel.setup.instructions_md} />
          </div>
          {fields.map((f) => (
            <Field key={f.key} label={f.label} description={f.help ?? 'Kept safely in your computer’s keychain. It never leaves this computer except to talk to the bot service.'}>
              <Input
                autoFocus
                type={f.secret && !reveal ? 'password' : 'text'}
                value={values[f.key] ?? ''}
                onChange={(e) => setValues((v) => ({ ...v, [f.key]: e.target.value }))}
                onKeyDown={(e) => e.key === 'Enter' && submit()}
                placeholder={f.key === 'bot_token' ? '123456789:AA...' : undefined}
                invalid={!!error}
                className="font-mono"
                rightSlot={
                  f.secret ? (
                    <button type="button" aria-label={reveal ? 'Hide' : 'Show'} onClick={() => setReveal((r) => !r)} className="flex size-6 items-center justify-center rounded text-fg-subtle hover:text-fg">
                      {reveal ? <IconEyeOff size={14} /> : <IconEye size={14} />}
                    </button>
                  ) : undefined
                }
              />
            </Field>
          ))}
          {error && <Alert tone="danger">{error}</Alert>}
        </div>
      )}
    </Dialog>
  )
}

function PairChatDialog({ channel, onClose }: { channel: Channel | null; onClose: () => void }) {
  const actions = useChannelActions()
  const { data } = useChannels()
  const [pairing, setPairing] = useState<ChannelPairing | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const startCount = useRef(0)
  const live = data?.find((c) => c.id === channel?.id) ?? channel

  const fetchCode = () => {
    if (!channel) return
    actions.pairing.mutate(channel.id, {
      onSuccess: setPairing,
      onError: (e) => toast.error("Couldn't make a pairing code", { description: errorMessage(e) })
    })
  }

  useEffect(() => {
    if (!channel) return
    setPairing(null)
    startCount.current = channel.paired.length
    fetchCode()
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [channel?.id])

  const newest = live && live.paired.length > startCount.current ? live.paired[live.paired.length - 1] : null
  const remaining = pairing ? Math.max(0, new Date(pairing.expires_at).getTime() - now) : 0
  const expired = !!pairing && remaining <= 0
  const whatsapp = channel?.id === 'whatsapp'
  const bot = live?.account_label ?? (whatsapp ? 'your number' : 'your bot')
  const command = pairing ? `/pair ${pairing.code}` : ''

  return (
    <Dialog open={!!channel} onOpenChange={(o) => !o && onClose()} size="md" title={channel ? `Pair a ${channel.display_name} chat` : 'Pair a chat'}>
      {channel &&
        (newest ? (
          <div className="flex flex-col items-center gap-3 py-8 text-center">
            <IconCircleCheckFilled size={48} className="text-success" />
            <div className="text-md font-semibold text-fg">{newest.label} is paired</div>
            <p className="max-w-xs text-sm text-fg-muted">
              {whatsapp ? 'Sentient answers that chat now' : `Say hi to ${bot}. Sentient answers there`}, and you&apos;ll see the chat in your list here too.
            </p>
            <Button className="mt-2" variant="secondary" onClick={onClose}>
              Done
            </Button>
          </div>
        ) : (
          <div className="space-y-4 pb-2">
            <p className="text-sm text-fg-muted">
              {whatsapp ? 'From the other WhatsApp account, open the chat with ' : `Open ${channel.display_name}, find `}
              <span className="font-medium text-fg">{bot}</span> and send this message:
            </p>
            <div className="flex flex-col items-center gap-2 rounded-xl border border-border bg-sunken/60 px-4 py-5">
              {pairing ? (
                <>
                  <span className={cn('font-mono text-3xl font-semibold tracking-wide text-fg', expired && 'text-fg-faint line-through')}>{command}</span>
                  {!expired && <CopyChip value={command} className="h-7 px-2.5 text-sm" />}
                </>
              ) : (
                <Spinner />
              )}
              <span className={cn('text-xs tabular-nums', expired ? 'text-danger' : 'text-fg-subtle')}>
                {!pairing ? 'Making a code…' : expired ? 'This code expired' : `Expires in ${Math.floor(remaining / 60000)}:${String(Math.floor((remaining % 60000) / 1000)).padStart(2, '0')}`}
              </span>
              {expired && (
                <Button size="sm" variant="primary" onClick={fetchCode} loading={actions.pairing.isPending}>
                  Get a new code
                </Button>
              )}
            </div>
            {!expired && pairing && (
              <div className="flex items-center justify-center gap-2 text-xs text-fg-subtle">
                <Spinner size={12} /> Waiting for your message…
              </div>
            )}
          </div>
        ))}
    </Dialog>
  )
}
