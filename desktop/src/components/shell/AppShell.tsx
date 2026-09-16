import { IconAlertTriangle, IconPlugConnectedX } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { Outlet } from 'react-router'
import { useEffect } from 'react'
import { toast } from 'sonner'
import { BrowserPanel } from '@/features/browser/BrowserPanel'
import { NotificationsPanel } from '@/features/notifications/NotificationsPanel'
import { getBridge } from '@/lib/bridge'
import { Button, Spinner } from '@/components/ui'
import { useConnection } from '@/stores/connection'
import { CommandPalette } from './CommandPalette'
import { Sidebar } from './Sidebar'
import { TitleBar } from './TitleBar'

const CAPTURE_TOAST = {
  screen: 'Sentient looked at your screen',
  camera: 'Sentient took a photo with your camera',
  clipboard: 'Sentient read your clipboard'
} as const

/** Privacy: every screen, camera or clipboard use by the desktop device is announced. */
function useCaptureNotices() {
  useEffect(
    () =>
      getBridge().onCaptureNotice((n) => {
        toast(CAPTURE_TOAST[n.kind], { description: 'Only because you asked. You can turn this off on the Devices page.' })
      }),
    []
  )
}

export function AppShell() {
  useCaptureNotices()
  return (
    <div className="flex h-full flex-col bg-bg">
      <TitleBar />
      <div className="flex min-h-0 flex-1">
        <Sidebar />
        <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden rounded-tl-xl border-l border-t border-border bg-surface">
          <EngineBanner />
          <div className="min-h-0 flex-1">
            <Outlet />
          </div>
        </main>
        <BrowserPanel />
      </div>
      <CommandPalette />
      <NotificationsPanel />
    </div>
  )
}

/** Thin banner while the engine restarts or the live socket reconnects. */
function EngineBanner() {
  const backend = useConnection((s) => s.backend)
  const socket = useConnection((s) => s.socket)
  const restart = useConnection((s) => s.restart)

  const engineDown = backend.state !== 'ready'
  const socketDown = !engineDown && (socket === 'reconnecting' || socket === 'closed')
  const show = engineDown || socketDown

  return (
    <AnimatePresence initial={false}>
      {show && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="shrink-0 overflow-hidden border-b border-warning/20 bg-warning/8"
        >
          <div className="flex items-center gap-2.5 px-4 py-2 text-sm text-warning">
            {backend.state === 'crashed' ? <IconAlertTriangle size={16} /> : socketDown ? <IconPlugConnectedX size={16} /> : <Spinner size={14} />}
            <span className="flex-1">
              {engineDown
                ? backend.state === 'restarting'
                  ? `Sentient's engine stopped. Restarting${backend.retryInMs ? ` in ${Math.ceil(backend.retryInMs / 1000)}s` : ''}…`
                  : backend.state === 'crashed'
                    ? "Sentient's engine stopped unexpectedly."
                    : 'Starting the engine…'
                : 'Reconnecting to the engine…'}
            </span>
            {engineDown && backend.state !== 'starting' && (
              <Button size="xs" variant="ghost" className="text-warning hover:text-warning" onClick={() => void restart()}>
                Restart now
              </Button>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
