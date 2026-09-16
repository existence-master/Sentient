import {
  IconAddressBook,
  IconBrandSlack,
  IconChartBar,
  IconCloud,
  IconHeartbeat,
  IconMail,
  IconMap2,
  IconWebhook,
  IconNews,
  IconPlug,
  IconWorldSearch,
  IconWorldWww,
  type Icon
} from '@tabler/icons-react'
import {
  siAccuweather,
  siBrave,
  siDiscord,
  siGithub,
  siGmail,
  siGoogle,
  siGooglecalendar,
  siGoogledocs,
  siGoogledrive,
  siGooglemaps,
  siGooglesheets,
  siGoogleslides,
  siModelcontextprotocol,
  siNotion,
  siTrello,
  siWhatsapp,
  type SimpleIcon
} from 'simple-icons'
import { cn } from '@/lib/utils'

/**
 * Brand marks for integrations. Real logos come from `simple-icons` (tree-shaken: only the
 * icons imported here are bundled); brands simple-icons doesn't carry use a tinted tabler glyph.
 * Near-black marks (GitHub, Notion, MCP) render in the foreground color so they work in both themes.
 */
type Brand = { kind: 'si'; icon: SimpleIcon; mono?: boolean } | { kind: 'tabler'; icon: Icon; color: string }

const BY_ID: Record<string, Brand> = {
  gmail: { kind: 'si', icon: siGmail },
  gcalendar: { kind: 'si', icon: siGooglecalendar },
  gdrive: { kind: 'si', icon: siGoogledrive },
  gdocs: { kind: 'si', icon: siGoogledocs },
  gsheets: { kind: 'si', icon: siGooglesheets },
  gslides: { kind: 'si', icon: siGoogleslides },
  gpeople: { kind: 'tabler', icon: IconAddressBook, color: '#4285F4' },
  github: { kind: 'si', icon: siGithub, mono: true },
  slack: { kind: 'tabler', icon: IconBrandSlack, color: '#E01E5A' },
  notion: { kind: 'si', icon: siNotion, mono: true },
  discord: { kind: 'si', icon: siDiscord },
  trello: { kind: 'si', icon: siTrello },
  whatsapp: { kind: 'si', icon: siWhatsapp },
  accuweather: { kind: 'si', icon: siAccuweather },
  newsapi: { kind: 'tabler', icon: IconNews, color: '#2563EB' },
  brave_search: { kind: 'si', icon: siBrave },
  google_cse: { kind: 'si', icon: siGoogle },
  google_maps: { kind: 'si', icon: siGooglemaps },
  internet_search: { kind: 'tabler', icon: IconWorldSearch, color: '#8B5CF6' },
  web: { kind: 'tabler', icon: IconWorldWww, color: '#0EA5E9' },
  weather: { kind: 'tabler', icon: IconCloud, color: '#38BDF8' },
  maps: { kind: 'tabler', icon: IconMap2, color: '#10B981' },
  news: { kind: 'tabler', icon: IconNews, color: '#F59E0B' },
  charts: { kind: 'tabler', icon: IconChartBar, color: '#A78BFA' },
  mcp: { kind: 'si', icon: siModelcontextprotocol, mono: true },
  heartbeat: { kind: 'tabler', icon: IconHeartbeat, color: '#A78BFA' },
  email_imap: { kind: 'tabler', icon: IconMail, color: '#0EA5E9' },
  webhook: { kind: 'tabler', icon: IconWebhook, color: '#F472B6' }
}

const BY_ICON: Record<string, string> = {
  'google-calendar': 'gcalendar',
  'google-drive': 'gdrive',
  'google-docs': 'gdocs',
  'google-sheets': 'gsheets',
  'google-slides': 'gslides',
  'google-contacts': 'gpeople',
  search: 'internet_search',
  globe: 'web',
  map: 'maps',
  chart: 'charts'
}

export function brandFor(id: string, icon?: string): Brand | null {
  if (BY_ID[id]) return BY_ID[id]
  if (!icon) return null
  return BY_ID[icon] ?? BY_ID[BY_ICON[icon] ?? ''] ?? null
}

export function brandColor(id: string, icon?: string): string | null {
  const b = brandFor(id, icon)
  if (!b) return null
  if (b.kind === 'tabler') return b.color
  return b.mono ? null : `#${b.icon.hex}`
}

/** A square brand tile. `size` is the tile edge in px. */
export function BrandIcon({
  id,
  icon,
  size = 40,
  className,
  bare = false
}: {
  id: string
  icon?: string
  size?: number
  className?: string
  /** Just the glyph, no tile. */
  bare?: boolean
}) {
  const brand = brandFor(id, icon)
  const color = brandColor(id, icon)
  const glyph = Math.round(bare ? size : size * 0.5)

  const mark = !brand ? (
    <IconPlug size={glyph} className="text-fg-muted" />
  ) : brand.kind === 'si' ? (
    <svg
      role="img"
      aria-label={brand.icon.title}
      viewBox="0 0 24 24"
      width={glyph}
      height={glyph}
      className={cn('shrink-0', brand.mono && 'text-fg')}
      fill={brand.mono ? 'currentColor' : `#${brand.icon.hex}`}
    >
      <path d={brand.icon.path} />
    </svg>
  ) : (
    <brand.icon size={glyph} stroke={1.75} style={{ color: brand.color }} className="shrink-0" />
  )

  if (bare) return <span className={cn('inline-flex shrink-0', className)}>{mark}</span>

  return (
    <span
      style={{
        width: size,
        height: size,
        background: color ? `color-mix(in oklab, ${color} 12%, var(--elevated))` : undefined,
        borderColor: color ? `color-mix(in oklab, ${color} 22%, var(--border))` : undefined
      }}
      className={cn(
        'inline-flex shrink-0 items-center justify-center border border-border bg-elevated',
        size >= 36 ? 'rounded-xl' : 'rounded-lg',
        className
      )}
    >
      {mark}
    </span>
  )
}
