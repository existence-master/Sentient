/** Tool / plugin identity for plan steps, runs and results. */
import {
  IconAddressBook,
  IconBook,
  IconBrain,
  IconBrandDiscord,
  IconBrandGithub,
  IconBrandGoogleDrive,
  IconBrandNotion,
  IconBrandSlack,
  IconBrandTrello,
  IconBrandWhatsapp,
  IconCalendarEvent,
  IconChartBar,
  IconChecklist,
  IconClock,
  IconCloud,
  IconFileText,
  IconFolder,
  IconMail,
  IconMapPin,
  IconNews,
  IconPresentation,
  IconTable,
  IconTool,
  IconWorld,
  IconWorldSearch,
  type Icon
} from '@tabler/icons-react'
import { useMemo } from 'react'
import { useTools } from '@/hooks/core'
import { cn, humanize } from '@/lib/utils'

interface ToolIdentity {
  icon: Icon
  label: string
  /** Brand tint for the icon. */
  tint?: string
}

const PLUGINS: Record<string, ToolIdentity> = {
  gmail: { icon: IconMail, label: 'Gmail', tint: 'text-[#ea4335]' },
  gcalendar: { icon: IconCalendarEvent, label: 'Google Calendar', tint: 'text-[#4285f4]' },
  gdrive: { icon: IconBrandGoogleDrive, label: 'Google Drive', tint: 'text-[#1fa463]' },
  gdocs: { icon: IconFileText, label: 'Google Docs', tint: 'text-[#4285f4]' },
  gsheets: { icon: IconTable, label: 'Google Sheets', tint: 'text-[#0f9d58]' },
  gslides: { icon: IconPresentation, label: 'Google Slides', tint: 'text-[#f4b400]' },
  gpeople: { icon: IconAddressBook, label: 'Google Contacts', tint: 'text-[#4285f4]' },
  github: { icon: IconBrandGithub, label: 'GitHub' },
  notion: { icon: IconBrandNotion, label: 'Notion' },
  slack: { icon: IconBrandSlack, label: 'Slack', tint: 'text-[#e01e5a]' },
  discord: { icon: IconBrandDiscord, label: 'Discord', tint: 'text-[#5865f2]' },
  trello: { icon: IconBrandTrello, label: 'Trello', tint: 'text-[#0079bf]' },
  whatsapp: { icon: IconBrandWhatsapp, label: 'WhatsApp', tint: 'text-[#25d366]' },
  internet_search: { icon: IconWorldSearch, label: 'Internet search', tint: 'text-info' },
  brave_search: { icon: IconWorldSearch, label: 'Brave Search', tint: 'text-info' },
  google_cse: { icon: IconWorldSearch, label: 'Google search', tint: 'text-info' },
  web: { icon: IconWorld, label: 'Web pages', tint: 'text-info' },
  weather: { icon: IconCloud, label: 'Weather', tint: 'text-info' },
  accuweather: { icon: IconCloud, label: 'AccuWeather', tint: 'text-info' },
  news: { icon: IconNews, label: 'News' },
  newsapi: { icon: IconNews, label: 'NewsAPI' },
  maps: { icon: IconMapPin, label: 'Maps', tint: 'text-success' },
  google_maps: { icon: IconMapPin, label: 'Google Maps', tint: 'text-success' },
  charts: { icon: IconChartBar, label: 'Charts', tint: 'text-accent-text' },
  files: { icon: IconFolder, label: 'Files', tint: 'text-accent-text' },
  memory: { icon: IconBrain, label: 'Memory', tint: 'text-accent-text' },
  skills: { icon: IconBook, label: 'Skills', tint: 'text-accent-text' },
  time: { icon: IconClock, label: 'Date & time' },
  tasks: { icon: IconChecklist, label: 'Tasks', tint: 'text-accent-text' }
}

const PREFIXES: Array<[RegExp, string]> = [
  [/^(gmail|mail|email)/, 'gmail'],
  [/^(gcal|gcalendar|calendar)/, 'gcalendar'],
  [/^gdrive/, 'gdrive'],
  [/^gdocs/, 'gdocs'],
  [/^gsheets/, 'gsheets'],
  [/^gslides/, 'gslides'],
  [/^gpeople/, 'gpeople'],
  [/^github/, 'github'],
  [/^notion/, 'notion'],
  [/^slack/, 'slack'],
  [/^discord/, 'discord'],
  [/^trello/, 'trello'],
  [/^whatsapp/, 'whatsapp'],
  [/^(web_search|internet_search|search)/, 'internet_search'],
  [/^web/, 'web'],
  [/^weather/, 'weather'],
  [/^news/, 'news'],
  [/^maps?/, 'maps'],
  [/^chart/, 'charts'],
  [/^file/, 'files'],
  [/^(memory|history)/, 'memory'],
  [/^skill/, 'skills'],
  [/^(time|current_time|current_datetime)/, 'time'],
  [/^task/, 'tasks']
]

/** Plugin id for a plugin id or a tool name (`gmail_search` -> `gmail`). */
export function pluginIdFor(name: string | undefined): string {
  const n = (name ?? '').trim().toLowerCase()
  if (!n) return ''
  if (PLUGINS[n]) return n
  return PREFIXES.find(([re]) => re.test(n))?.[1] ?? n
}

export function toolIdentity(name: string | undefined, displayNames?: Record<string, string>): ToolIdentity {
  const id = pluginIdFor(name)
  const known = PLUGINS[id]
  const label = displayNames?.[id] ?? known?.label ?? (name ? humanize(name) : 'Any tool')
  return { icon: known?.icon ?? IconTool, label, tint: known?.tint }
}

/** Plugin display names from `/api/tools` (falls back to the built-in map). */
export function useToolNames(): { names: Record<string, string>; options: Array<{ value: string; label: string }> } {
  const tools = useTools()
  return useMemo(() => {
    const names: Record<string, string> = {}
    for (const [id, meta] of Object.entries(PLUGINS)) names[id] = meta.label
    for (const p of tools.data ?? []) names[p.id] = p.display_name
    const ids = tools.data?.length ? tools.data.map((p) => p.id) : Object.keys(PLUGINS)
    const options = ids.map((id) => ({ value: id, label: names[id] ?? humanize(id) })).sort((a, b) => a.label.localeCompare(b.label))
    return { names, options }
  }, [tools.data])
}

export function ToolIcon({ name, size = 15, className, tinted = true }: { name: string | undefined; size?: number; className?: string; tinted?: boolean }) {
  const { icon: I, tint } = toolIdentity(name)
  return <I size={size} stroke={1.75} className={cn('shrink-0', tinted && tint, className)} />
}

/** Small rounded tile with the tool icon. */
export function ToolTile({ name, size = 'md', className }: { name: string | undefined; size?: 'sm' | 'md' | 'lg'; className?: string }) {
  const dims = { sm: 'size-5 rounded-md', md: 'size-7 rounded-lg', lg: 'size-9 rounded-xl' }[size]
  const icon = { sm: 12, md: 15, lg: 18 }[size]
  return (
    <span className={cn('inline-flex shrink-0 items-center justify-center border border-border bg-elevated', dims, className)}>
      <ToolIcon name={name} size={icon} />
    </span>
  )
}

/** Overlapping stack of distinct tool tiles. */
export function ToolStack({ tools, max = 4, className }: { tools: string[]; max?: number; className?: string }) {
  const ids = Array.from(new Set(tools.map(pluginIdFor).filter(Boolean)))
  if (!ids.length) return null
  const shown = ids.slice(0, max)
  return (
    <span className={cn('flex items-center', className)}>
      {shown.map((id, i) => (
        <ToolTile key={id} name={id} size="sm" className={cn('bg-surface ring-2 ring-surface', i > 0 && '-ml-1')} />
      ))}
      {ids.length > max && <span className="ml-1 text-2xs text-fg-subtle">+{ids.length - max}</span>}
    </span>
  )
}
