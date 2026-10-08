import {
  IconAppWindow,
  IconArrowBackUp,
  IconArrowsVertical,
  IconBellRinging,
  IconCamera,
  IconDevices,
  IconKeyboard,
  IconListDetails,
  IconPointer,
  IconScreenshot,
  IconSpeakerphone,
  IconTerminal2,
  IconUserBolt,
  IconUsersGroup,
  IconWorldWww,
  IconBrain,
  IconBrandGithub,
  IconBrandNotion,
  IconBrandSlack,
  IconCalendar,
  IconClock,
  IconCloud,
  IconFile,
  IconFilePencil,
  IconFolder,
  IconHistory,
  IconListCheck,
  IconMail,
  IconMapPin,
  IconSearch,
  IconSparkles,
  IconTool,
  IconWorld,
  type Icon
} from '@tabler/icons-react'
import { humanize, truncate } from '@/lib/utils'

interface ToolMeta {
  running: string
  done: string
  icon: Icon
}

const KNOWN: Record<string, ToolMeta> = {
  memory_recall: { running: 'Searching memory', done: 'Searched memory', icon: IconBrain },
  memory_search: { running: 'Searching memory', done: 'Searched memory', icon: IconBrain },
  memory_save: { running: 'Saving to memory', done: 'Saved to memory', icon: IconBrain },
  memory_remember: { running: 'Saving to memory', done: 'Saved to memory', icon: IconBrain },
  memory_add: { running: 'Saving to memory', done: 'Saved to memory', icon: IconBrain },
  memory_forget: { running: 'Forgetting', done: 'Forgot', icon: IconBrain },
  memory_search_history: { running: 'Searching past chats', done: 'Searched past chats', icon: IconHistory },
  memory_search_by_source: { running: 'Searching memory', done: 'Searched memory', icon: IconBrain },
  history_semantic_search: { running: 'Searching past chats', done: 'Searched past chats', icon: IconHistory },
  history_time_search: { running: 'Looking back in time', done: 'Looked back in time', icon: IconHistory },
  current_datetime: { running: 'Checking the date and time', done: 'Checked the date and time', icon: IconClock },
  file_list: { running: 'Listing files', done: 'Listed files', icon: IconFolder },
  file_read: { running: 'Reading file', done: 'Read file', icon: IconFile },
  file_write: { running: 'Writing file', done: 'Wrote file', icon: IconFilePencil },
  time_now: { running: 'Checking the time', done: 'Checked the time', icon: IconClock },
  current_time: { running: 'Checking the time', done: 'Checked the time', icon: IconClock },
  skill_view: { running: 'Opening skill', done: 'Opened skill', icon: IconSparkles },
  skill_save: { running: 'Saving a skill', done: 'Saved a skill', icon: IconSparkles },
  skill_list: { running: 'Listing skills', done: 'Listed skills', icon: IconSparkles },
  web_search: { running: 'Searching the web', done: 'Searched the web', icon: IconWorld },
  // §10-13 code, helpers, browser and devices
  execute_code: { running: 'Running code', done: 'Ran code', icon: IconTerminal2 },
  delegate_task: { running: 'Asking a helper', done: 'Asked a helper', icon: IconUserBolt },
  delegate_tasks: { running: 'Asking helpers', done: 'Asked helpers', icon: IconUsersGroup },
  browser_open: { running: 'Opening a page', done: 'Opened a page', icon: IconWorldWww },
  browser_snapshot: { running: 'Reading the page', done: 'Read the page', icon: IconListDetails },
  browser_click: { running: 'Clicking', done: 'Clicked', icon: IconPointer },
  browser_type: { running: 'Typing', done: 'Typed', icon: IconKeyboard },
  browser_select: { running: 'Choosing an option', done: 'Chose an option', icon: IconListDetails },
  browser_press: { running: 'Pressing a key', done: 'Pressed a key', icon: IconKeyboard },
  browser_scroll: { running: 'Scrolling', done: 'Scrolled', icon: IconArrowsVertical },
  browser_back: { running: 'Going back', done: 'Went back', icon: IconArrowBackUp },
  browser_tabs: { running: 'Checking tabs', done: 'Checked tabs', icon: IconAppWindow },
  browser_switch_tab: { running: 'Switching tabs', done: 'Switched tabs', icon: IconAppWindow },
  browser_extract: { running: 'Reading the page', done: 'Read the page', icon: IconListDetails },
  browser_screenshot: { running: 'Taking a screenshot', done: 'Took a screenshot', icon: IconScreenshot },
  browser_close: { running: 'Closing the browser', done: 'Closed the browser', icon: IconWorld },
  device_list: { running: 'Checking your devices', done: 'Checked your devices', icon: IconDevices },
  device_take_photo: { running: 'Taking a photo', done: 'Took a photo', icon: IconCamera },
  device_capture_screen: { running: 'Looking at the screen', done: 'Looked at the screen', icon: IconScreenshot },
  device_get_location: { running: 'Finding your location', done: 'Found your location', icon: IconMapPin },
  device_notify: { running: 'Sending a notification', done: 'Sent a notification', icon: IconBellRinging },
  device_display: { running: 'Showing it on your device', done: 'Showed it on your device', icon: IconDevices },
  device_speak: { running: 'Speaking on your device', done: 'Spoke on your device', icon: IconSpeakerphone }
}

const PREFIX: Array<[RegExp, Icon]> = [
  [/^memory/, IconBrain],
  [/^file/, IconFile],
  [/^(time|clock|date)/, IconClock],
  [/^skill/, IconSparkles],
  [/(web|search|browse)/, IconWorld],
  [/^(gmail|mail|email)/, IconMail],
  [/^(gcal|calendar)/, IconCalendar],
  [/^task/, IconListCheck],
  [/^weather/, IconCloud],
  [/^slack/, IconBrandSlack],
  [/^notion/, IconBrandNotion],
  [/^github/, IconBrandGithub],
  [/^(maps|location|places)/, IconMapPin],
  [/find|lookup|query/, IconSearch]
]

/** App prefixes of integration tools, shown as "Calendar: find free time" instead of "Gcalendar find free time". */
const APP_PREFIX: Record<string, string> = {
  gcalendar: 'Calendar',
  gmail: 'Gmail',
  gdrive: 'Drive',
  gdocs: 'Docs',
  gsheets: 'Sheets',
  gslides: 'Slides',
  gpeople: 'Contacts',
  github: 'GitHub',
  notion: 'Notion',
  slack: 'Slack',
  trello: 'Trello'
}

export function toolMeta(name: string): ToolMeta {
  const known = KNOWN[name]
  if (known) return known
  const icon = PREFIX.find(([re]) => re.test(name))?.[1] ?? IconTool
  const [prefix, ...rest] = name.split('_')
  const app = APP_PREFIX[prefix]
  const label = app && rest.length ? `${app}: ${humanize(rest.join('_')).toLowerCase()}` : humanize(name)
  return { running: label, done: label, icon }
}

function short(v: unknown): string {
  if (typeof v === 'string') return `"${truncate(v.replace(/\s+/g, ' '), 60)}"`
  if (typeof v === 'number' || typeof v === 'boolean' || v === null) return String(v)
  if (Array.isArray(v)) return `[${v.length}]`
  return '{…}'
}

/** One-line preview of tool arguments. */
export function argsPreview(args: Record<string, unknown> | undefined): string {
  const entries = Object.entries(args ?? {})
  if (!entries.length) return ''
  if (entries.length === 1) return short(entries[0][1])
  return truncate(entries.map(([k, v]) => `${k}: ${short(v)}`).join(', '), 90)
}

export const RISK_META = {
  read: { label: 'Reads', tone: 'neutral' as const, verb: 'look something up' },
  write: { label: 'Changes things', tone: 'warning' as const, verb: 'make a change' },
  send: { label: 'Sends', tone: 'danger' as const, verb: 'send something on your behalf' },
  exec: { label: 'Runs code', tone: 'danger' as const, verb: 'run something on your computer' }
}
