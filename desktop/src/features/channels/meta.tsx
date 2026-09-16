import { IconDeviceMobile, IconEyeglass2, IconMicrophone, IconMessageCircle, type Icon } from '@tabler/icons-react'
import { siDiscord, siTelegram, type SimpleIcon } from 'simple-icons'
import { Tooltip } from '@/components/ui'
import { cn } from '@/lib/utils'

type Mark = { kind: 'si'; icon: SimpleIcon } | { kind: 'tabler'; icon: Icon }

interface ChannelInfo {
  label: string
  from: string
  mark: Mark
}

/** Where a chat came from. `desktop`, `web` and `cli` are the app itself and get no badge. */
export const CHANNEL_INFO: Record<string, ChannelInfo> = {
  telegram: { label: 'Telegram', from: 'From Telegram', mark: { kind: 'si', icon: siTelegram } },
  discord: { label: 'Discord', from: 'From Discord', mark: { kind: 'si', icon: siDiscord } },
  voice: { label: 'Voice', from: 'A voice conversation', mark: { kind: 'tabler', icon: IconMicrophone } },
  glasses: { label: 'Glasses', from: 'From your glasses', mark: { kind: 'tabler', icon: IconEyeglass2 } },
  phone: { label: 'Phone', from: 'From your phone', mark: { kind: 'tabler', icon: IconDeviceMobile } }
}

export function channelInfo(channel: string | null | undefined): ChannelInfo | null {
  if (!channel) return null
  return CHANNEL_INFO[channel] ?? null
}

export function ChannelMark({ channel, size = 14, className, color = true }: { channel: string; size?: number; className?: string; color?: boolean }) {
  const info = CHANNEL_INFO[channel]
  if (!info) return <IconMessageCircle size={size} className={className} />
  if (info.mark.kind === 'tabler') {
    const Cmp = info.mark.icon
    return <Cmp size={size} stroke={1.75} className={className} />
  }
  return (
    <svg role="img" aria-label={info.label} viewBox="0 0 24 24" width={size} height={size} className={className} fill={color ? `#${info.mark.icon.hex}` : 'currentColor'}>
      <path d={info.mark.icon.path} />
    </svg>
  )
}

/** Brand tile for channel cards. */
export function ChannelTile({ channel, size = 40 }: { channel: string; size?: number }) {
  const info = CHANNEL_INFO[channel]
  const hex = info?.mark.kind === 'si' ? `#${info.mark.icon.hex}` : undefined
  return (
    <div
      className="flex shrink-0 items-center justify-center rounded-xl border border-border"
      style={{ width: size, height: size, background: hex ? `color-mix(in oklab, ${hex} 14%, transparent)` : undefined }}
    >
      <ChannelMark channel={channel} size={Math.round(size * 0.5)} />
    </div>
  )
}

/** Small icon shown next to chats that came from another channel (sidebar, chat header). */
export function ChannelBadge({ channel, withLabel, className }: { channel: string | null | undefined; withLabel?: boolean; className?: string }) {
  const info = channelInfo(channel)
  if (!info || !channel) return null
  if (withLabel) {
    return (
      <span className={cn('inline-flex h-5.5 shrink-0 items-center gap-1.5 rounded-full border border-border bg-elevated px-2 text-xs font-medium text-fg-muted', className)}>
        <ChannelMark channel={channel} size={12} />
        {info.label}
      </span>
    )
  }
  return (
    <Tooltip content={info.from}>
      <span className={cn('inline-flex size-4 shrink-0 items-center justify-center opacity-80', className)}>
        <ChannelMark channel={channel} size={12} className="text-fg-subtle" color={false} />
      </span>
    </Tooltip>
  )
}
