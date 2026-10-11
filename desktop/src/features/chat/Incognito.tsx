import { IconBrain, IconDatabaseOff, IconSpy, IconTrash } from '@tabler/icons-react'
import { motion } from 'motion/react'
import { cn } from '@/lib/utils'

/** What an incognito chat does, in plain words (#154). `usesMemory`: chat.incognito_uses_memory. */
export function incognitoLines(usesMemory: boolean) {
  return [
    { icon: IconDatabaseOff, text: 'Nothing you say here is remembered, and Sentient doesn’t learn from it.' },
    {
      icon: IconBrain,
      text: usesMemory
        ? 'Sentient can still use what it already knows about you.'
        : 'Sentient won’t use what it already knows about you either.'
    },
    { icon: IconTrash, text: 'The chat is deleted when you leave it or close Sentient.' }
  ]
}

/** The start screen of a new incognito chat. */
export function IncognitoIntro({ usesMemory }: { usesMemory: boolean }) {
  return (
    <div className="flex flex-col items-center text-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.35 }}
        className="flex size-14 items-center justify-center rounded-2xl border border-incognito/30 bg-incognito/12 text-incognito"
      >
        <IconSpy size={30} stroke={1.6} />
      </motion.div>
      <h1 className="mt-5 text-2xl font-semibold tracking-tight text-fg">Incognito chat</h1>
      <ul className="mt-5 w-full max-w-[460px] space-y-2.5 text-left">
        {incognitoLines(usesMemory).map(({ icon: Icon, text }) => (
          <li key={text} className="flex items-start gap-3 rounded-xl border border-border bg-surface px-3.5 py-2.5 text-sm text-fg-muted">
            <Icon size={17} className="mt-0.5 shrink-0 text-incognito" />
            <span>{text}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** The tinted label in an incognito chat's header. */
export function IncognitoBadge({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        'inline-flex h-6 shrink-0 items-center gap-1.5 rounded-full border border-incognito/30 bg-incognito/12 px-2 text-xs font-medium text-incognito',
        className
      )}
    >
      <IconSpy size={14} />
      Incognito
    </span>
  )
}
