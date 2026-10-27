/** #359 A ringing alarm: a banner with Stop and Snooze while the chime plays (stores/alarm.ts). #390: after a snooze,
 *  from here or by saying "snooze" anywhere, it says until when, with Turn off. */
import { IconAlarm, IconAlarmSnooze, IconVolume, IconVolumeOff, IconX } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { Button, IconButton } from '@/components/ui'
import { formatTime } from '@/lib/utils'
import { useAlarm } from '@/stores/alarm'

export function AlarmBanner() {
  const ringing = useAlarm((s) => s.ringing)
  const snoozed = useAlarm((s) => s.snoozed)
  const playing = useAlarm((s) => s.playing)
  const stop = useAlarm((s) => s.stop)
  const snooze = useAlarm((s) => s.snooze)
  const turnOff = useAlarm((s) => s.turnOff)

  return (
    <AnimatePresence initial={false}>
      {ringing && (
        <motion.div
          key="ringing"
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
            {ringing.reminderId && (
              <Button size="sm" variant="secondary" onClick={snooze}>
                Snooze
              </Button>
            )}
            <Button size="sm" variant="primary" onClick={stop} autoFocus>
              Stop
            </Button>
          </div>
        </motion.div>
      )}
      {!ringing && snoozed && (
        <motion.div
          key="snoozed"
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="shrink-0 overflow-hidden border-b border-border bg-elevated"
          role="status"
        >
          <div className="flex items-center gap-3 px-4 py-2 text-sm text-fg">
            <span className="flex text-fg-muted">
              <IconAlarmSnooze size={20} />
            </span>
            <span className="min-w-0 flex-1">
              <span className="font-medium">{snoozed.title}</span>{' '}
              <span className="text-fg-muted">Snoozed until {formatTime(snoozed.until)}</span>
            </span>
            <Button size="sm" variant="secondary" onClick={turnOff}>
              Turn off
            </Button>
            <IconButton
              size="sm"
              icon={<IconX size={14} />}
              label="Hide"
              onClick={() => useAlarm.setState({ snoozed: null })}
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
