import { IconBrain, IconCalendarEvent, IconSunrise, IconUserHeart, type Icon } from '@tabler/icons-react'
import { motion } from 'motion/react'
import { Logo } from '@/components/brand/Logo'
import { greeting } from '@/lib/utils'

export interface Suggestion {
  icon: Icon
  title: string
  subtitle: string
  prompt: string
  /** Send right away instead of filling the composer. */
  autoSend?: boolean
}

export const SUGGESTIONS: Suggestion[] = [
  {
    icon: IconCalendarEvent,
    title: 'Plan my week',
    subtitle: 'Turn what is on your plate into a realistic plan',
    prompt: "Help me plan my week. Ask me what's on my plate, then lay out a realistic plan day by day.",
    autoSend: true
  },
  {
    icon: IconBrain,
    title: 'Remember that…',
    subtitle: 'Teach Sentient something about you',
    prompt: 'Remember that '
  },
  {
    icon: IconSunrise,
    title: 'Every morning at 8, send me…',
    subtitle: 'Set up a recurring task',
    prompt: 'Every morning at 8, send me '
  },
  {
    icon: IconUserHeart,
    title: 'What do you know about me?',
    subtitle: 'See what Sentient has learned so far',
    prompt: 'What do you know about me so far?',
    autoSend: true
  }
]

export function EmptyChat({ userName, onPick }: { userName?: string; onPick: (s: Suggestion) => void }) {
  const first = userName?.trim().split(/\s+/)[0]
  return (
    <div className="flex flex-col items-center text-center">
      <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.35 }}>
        <Logo size={56} />
      </motion.div>
      <motion.h1
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="mt-6 text-[26px] font-semibold tracking-tight text-fg"
      >
        {greeting()}
        {first ? `, ${first}` : ''}
      </motion.h1>
      <motion.p initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mt-1.5 text-md text-fg-muted">
        What can I take off your plate today?
      </motion.p>
      <div className="mt-8 grid w-full grid-cols-2 gap-2.5">
        {SUGGESTIONS.map((s, i) => (
          <motion.button
            key={s.title}
            type="button"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.14 + i * 0.04 }}
            onClick={() => onPick(s)}
            className="group flex items-start gap-3 rounded-xl border border-border bg-bg/40 p-3.5 text-left transition-colors hover:border-border-strong hover:bg-elevated"
          >
            <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border bg-surface text-fg-muted transition-colors group-hover:text-accent-text">
              <s.icon size={16} />
            </span>
            <span className="min-w-0">
              <span className="block text-sm font-medium text-fg">{s.title}</span>
              <span className="mt-0.5 block text-xs text-fg-subtle">{s.subtitle}</span>
            </span>
          </motion.button>
        ))}
      </div>
    </div>
  )
}
