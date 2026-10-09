/**
 * §17 Stop everything: the title bar button and the calm banner shown while Sentient is stopped.
 * The tray menu and the global shortcut (electron/main) call the same engine routes.
 */
import { IconHandStop, IconPlayerPause, IconPlayerPlay } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { toast } from 'sonner'
import { Button, Tooltip } from '@/components/ui'
import { useResume, useStopAll, useStopState } from '@/hooks/core'
import { formatDateTime, isMac } from '@/lib/utils'
import { useConnection } from '@/stores/connection'

export const STOP_SHORTCUT = isMac ? '⌘ Option Shift S' : 'Ctrl+Alt+Shift+S'

const SOURCE_LABEL: Record<string, string> = {
  desktop: 'from this computer',
  tray: 'from the tray menu',
  hotkey: 'with the shortcut',
  telegram: 'from Telegram',
  discord: 'from Discord',
  device: 'from a paired device'
}

function useStopActions() {
  const stopAll = useStopAll()
  const resume = useResume()
  return {
    stop: () =>
      stopAll.mutate(undefined, {
        onSuccess: (r) =>
          toast('Stopped everything', {
            description: `${r.cancelled ? `${r.cancelled} running job${r.cancelled === 1 ? '' : 's'} cancelled. ` : ''}Nothing new will start until you resume.`
          }),
        onError: () => toast.error("Couldn't stop Sentient", { description: 'The engine did not answer. Try again, or quit Sentient from the tray.' })
      }),
    resume: () =>
      resume.mutate(undefined, {
        onSuccess: () => toast('Resumed', { description: 'Scheduled tasks and suggestions are back on.' }),
        onError: () => toast.error("Couldn't resume", { description: 'The engine did not answer. Try again in a moment.' })
      }),
    busy: stopAll.isPending || resume.isPending
  }
}

/** Title bar control: Stop everything, or Resume while stopped. */
export function StopButton() {
  const { data } = useStopState()
  const ready = useConnection((s) => s.backend.state === 'ready')
  const { stop, resume, busy } = useStopActions()
  if (!ready) return null

  if (data?.stopped) {
    return (
      <Tooltip content="Start scheduled tasks and suggestions again" side="bottom">
        <Button size="xs" variant="primary" leftIcon={<IconPlayerPlay size={13} />} loading={busy} onClick={resume}>
          Resume
        </Button>
      </Tooltip>
    )
  }
  return (
    <Tooltip content={`Stop everything Sentient is doing now (${STOP_SHORTCUT})`} side="bottom">
      <Button size="xs" variant="danger" leftIcon={<IconHandStop size={13} />} loading={busy} onClick={stop}>
        Stop all
      </Button>
    </Tooltip>
  )
}

/** Calm banner while stopped, with Resume. */
export function StoppedBanner() {
  const { data } = useStopState()
  const { resume, busy } = useStopActions()
  const stopped = !!data?.stopped
  const how = data?.source ? SOURCE_LABEL[data.source] : undefined
  const when = data?.stopped_at ? formatDateTime(data.stopped_at) : undefined

  return (
    <AnimatePresence initial={false}>
      {stopped && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="shrink-0 overflow-hidden border-b border-border bg-elevated"
        >
          <div className="flex items-center gap-2.5 px-4 py-2 text-sm text-fg">
            <IconPlayerPause size={16} className="text-fg-muted" />
            <span className="flex-1">
              <span className="font-medium">Sentient is stopped.</span>{' '}
              <span className="text-fg-muted">
                Nothing scheduled or automatic will start until you resume.
                {how || when ? ` Stopped ${[how, when ? `on ${when}` : ''].filter(Boolean).join(' ')}.` : ''}
              </span>
            </span>
            <Button size="xs" variant="primary" leftIcon={<IconPlayerPlay size={13} />} loading={busy} onClick={resume}>
              Resume
            </Button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
