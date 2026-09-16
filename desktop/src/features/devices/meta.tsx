import {
  IconBattery1,
  IconBattery2,
  IconBattery3,
  IconBattery4,
  IconBatteryCharging,
  IconBellRinging,
  IconCamera,
  IconClipboard,
  IconCpu,
  IconDeviceDesktop,
  IconDeviceMobile,
  IconDeviceWatch,
  IconEyeglass2,
  IconHandClick,
  IconMapPin,
  IconMicrophone,
  IconScreenshot,
  IconSpeakerphone,
  IconTextSize,
  type Icon
} from '@tabler/icons-react'
import type { DeviceNode } from '@/lib/types'
import { cn, relativeTime } from '@/lib/utils'

export const KIND_META: Record<string, { label: string; icon: Icon }> = {
  phone: { label: 'Phone', icon: IconDeviceMobile },
  glasses: { label: 'Glasses', icon: IconEyeglass2 },
  desktop: { label: 'Computer', icon: IconDeviceDesktop },
  watch: { label: 'Watch', icon: IconDeviceWatch },
  custom: { label: 'Device', icon: IconCpu }
}

export const kindMeta = (kind: string) => KIND_META[kind] ?? KIND_META.custom

const CAPABILITY_META: Record<string, { label: string; icon: Icon }> = {
  'camera.photo': { label: 'Camera', icon: IconCamera },
  'screen.capture': { label: 'Screen', icon: IconScreenshot },
  'location.get': { label: 'Location', icon: IconMapPin },
  'notify.show': { label: 'Notifications', icon: IconBellRinging },
  'display.text': { label: 'Display', icon: IconTextSize },
  'display.card': { label: 'Display', icon: IconTextSize },
  'audio.play': { label: 'Speaker', icon: IconSpeakerphone },
  speak: { label: 'Speaker', icon: IconSpeakerphone },
  'mic.stream': { label: 'Microphone', icon: IconMicrophone },
  'clipboard.read': { label: 'Clipboard', icon: IconClipboard },
  'clipboard.write': { label: 'Clipboard', icon: IconClipboard },
  'button.events': { label: 'Buttons', icon: IconHandClick }
}

/** Friendly, de-duplicated capability chips (battery is shown separately). */
export function capabilityChips(caps: string[]): Array<{ label: string; icon: Icon }> {
  const seen = new Set<string>()
  const out: Array<{ label: string; icon: Icon }> = []
  for (const c of caps) {
    const m = CAPABILITY_META[c]
    if (!m || seen.has(m.label)) continue
    seen.add(m.label)
    out.push(m)
  }
  return out
}

const PLATFORMS: Record<string, string> = {
  win32: 'Windows',
  windows: 'Windows',
  darwin: 'macOS',
  macos: 'macOS',
  linux: 'Linux',
  android: 'Android',
  ios: 'iPhone',
  web: 'Web browser',
  'brilliant-frame': 'Brilliant Frame'
}

export const platformLabel = (p: string | null | undefined) => (p ? (PLATFORMS[p.toLowerCase()] ?? p) : '')

export function presence(node: DeviceNode): string {
  if (node.online) return 'Online'
  return node.last_seen_at ? `Last seen ${relativeTime(node.last_seen_at)}` : 'Offline'
}

/** Battery is 0..1 in the contract; tolerate 0..100. */
export function batteryPercent(b: number | null | undefined): number | null {
  if (b === null || b === undefined || Number.isNaN(b)) return null
  return Math.round(b <= 1 ? b * 100 : b)
}

export function BatteryPill({ node, className }: { node: DeviceNode; className?: string }) {
  const pct = batteryPercent(node.battery)
  if (pct === null) return null
  const Glyph = node.charging ? IconBatteryCharging : pct > 80 ? IconBattery4 : pct > 50 ? IconBattery3 : pct > 20 ? IconBattery2 : IconBattery1
  return (
    <span
      className={cn(
        'inline-flex h-6 items-center gap-1 rounded-full border px-2 text-xs tabular-nums',
        pct <= 20 && !node.charging ? 'border-danger/30 text-danger' : 'border-border text-fg-muted',
        className
      )}
      title={node.charging ? `Charging, ${pct}%` : `Battery ${pct}%`}
    >
      <Glyph size={14} />
      {pct}%
    </span>
  )
}

export function DeviceTile({ kind, online, size = 44 }: { kind: string; online?: boolean; size?: number }) {
  const { icon: Glyph } = kindMeta(kind)
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <div className="flex size-full items-center justify-center rounded-xl border border-border bg-elevated text-fg-muted shadow-soft">
        <Glyph size={Math.round(size * 0.5)} stroke={1.5} />
      </div>
      {online !== undefined && (
        <span className={cn('absolute -bottom-0.5 -right-0.5 size-3 rounded-full ring-[2.5px] ring-surface', online ? 'bg-success' : 'bg-fg-faint')} />
      )}
    </div>
  )
}
