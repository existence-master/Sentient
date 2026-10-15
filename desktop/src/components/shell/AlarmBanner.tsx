/** #359 A ringing alarm: a banner with Stop while the chime plays (stores/alarm.ts). */
import { IconAlarm, IconVolume, IconVolumeOff } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { Button } from '@/components/ui'
import { useAlarm } from '@/stores/alarm'

export function AlarmBanner() {
  const ringing = useAlarm((s) => s.ringing)
  const playing = useAlarm((s) => s.playing)
  const stop = useAlarm((s) => s.stop)

  return (
    <AnimatePresence initial={false}>
      {ringing && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="shrink-0 overflow-hidden border-b border-accent/25 bg-accent/10"
          role="alert"
        >
          <div className="flex items-center gap-3 px-4 py-2.5 text-sm text-fg">
            <motion.span
              animate={{ rotate: [0, -14, 14, -10, 10, 0] }}
              transition={{ duration: 0.7, repeat: Infinity, repeatDelay: 0.7 }}
              className="flex text-accent-text"
            >
              <IconAlarm size={20} />
            </motion.span>
            <span className="min-w-0 flex-1">
              <span className="font-medium">{ringing.title}</span>{' '}
              <span className="text-fg-muted">{ringing.message}</span>
            </span>
            <span className="flex items-center gap-1 text-xs text-fg-muted">
              {playing ? <IconVolume size={14} /> : <IconVolumeOff size={14} />}
              {playing ? 'Ringing' : "Can't play sound"}
            </span>
            <Button size="sm" variant="primary" onClick={stop} autoFocus>
              Stop
            </Button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
