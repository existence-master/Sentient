import { IconBrain, IconEdit, IconListCheck, IconWaveSine, IconWorldWww, type Icon } from '@tabler/icons-react'
import { motion } from 'motion/react'
import { cn } from '@/lib/utils'

export interface SlashCommand {
  name: 'new' | 'voice' | 'task' | 'remember' | 'browser'
  /** Hint for the text that follows the command. */
  arg?: string
  description: string
  icon: Icon
}

export const SLASH_COMMANDS: SlashCommand[] = [
  { name: 'new', description: 'Start a new chat', icon: IconEdit },
  { name: 'task', arg: 'what to do', description: 'Hand Sentient a task to work on in the background', icon: IconListCheck },
  { name: 'remember', arg: 'something about you', description: 'Save a fact to memory', icon: IconBrain },
  { name: 'voice', description: 'Talk with Sentient out loud', icon: IconWaveSine },
  { name: 'browser', description: "Watch Sentient's browser", icon: IconWorldWww }
]

/** The command menu is open while the text is just "/" plus letters. */
export function slashQuery(text: string): string | null {
  const m = /^\/([a-z]*)$/i.exec(text)
  return m ? m[1].toLowerCase() : null
}

export function filterCommands(query: string): SlashCommand[] {
  return SLASH_COMMANDS.filter((c) => c.name.startsWith(query))
}

/** `/task buy milk` -> { command: task, arg: "buy milk" }. Unknown commands return null (sent as text). */
export function parseSlash(text: string): { command: SlashCommand; arg: string } | null {
  const m = /^\/([a-z]+)(?:\s+([\s\S]*))?$/i.exec(text.trim())
  if (!m) return null
  const command = SLASH_COMMANDS.find((c) => c.name === m[1].toLowerCase())
  return command ? { command, arg: (m[2] ?? '').trim() } : null
}

export function SlashMenu({
  items,
  active,
  onHover,
  onPick
}: {
  items: SlashCommand[]
  active: number
  onHover: (i: number) => void
  onPick: (c: SlashCommand) => void
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 4 }}
      transition={{ duration: 0.12 }}
      role="listbox"
      aria-label="Commands"
      className="absolute bottom-full left-0 z-20 mb-2 w-[min(400px,100%)] overflow-hidden rounded-xl border border-border-strong bg-overlay p-1 shadow-pop"
    >
      <div className="px-2.5 pb-1 pt-1.5 text-2xs font-medium uppercase tracking-wide text-fg-subtle">Commands</div>
      {items.map((c, i) => (
        <button
          key={c.name}
          type="button"
          role="option"
          aria-selected={i === active}
          onMouseEnter={() => onHover(i)}
          onMouseDown={(e) => {
            e.preventDefault()
            onPick(c)
          }}
          className={cn('flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left', i === active ? 'bg-active' : 'hover:bg-hover')}
        >
          <c.icon size={16} className={i === active ? 'text-accent-text' : 'text-fg-subtle'} />
          <span className="min-w-0 flex-1">
            <span className="flex items-baseline gap-1.5">
              <span className="font-mono text-sm text-fg">/{c.name}</span>
              {c.arg && <span className="text-xs text-fg-faint">{c.arg}</span>}
            </span>
            <span className="block truncate text-xs text-fg-subtle">{c.description}</span>
          </span>
        </button>
      ))}
    </motion.div>
  )
}
