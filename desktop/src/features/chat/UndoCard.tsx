/**
 * An email, message or post waiting in its undo window (#142): what goes out, a countdown, Undo and Send now.
 * It only goes when the countdown ends; Undo means it never does.
 */
import { IconArrowBackUp, IconCheck, IconClockHour4, IconSend, IconX } from '@tabler/icons-react'
import { motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui'
import type { UndoView } from '@/lib/chatFold'
import { useChat } from '@/stores/chat'
import { ActionPreview } from './ActionPreview'

function secondsLeft(until: string): number {
  return Math.max(0, Math.ceil((new Date(until).getTime() - Date.now()) / 1000))
}

export function UndoCard({ undo }: { undo: UndoView }) {
  const respond = useChat((s) => s.respondUndo)
  const [left, setLeft] = useState(() => secondsLeft(undo.until))
  const [pressed, setPressed] = useState<'undo' | 'send_now' | null>(null)
  const waiting = undo.status === 'waiting'

  useEffect(() => {
    if (!waiting) return
    const timer = window.setInterval(() => setLeft(secondsLeft(undo.until)), 250)
    return () => window.clearInterval(timer)
  }, [waiting, undo.until])

  if (!waiting) {
    return (
      <div className="flex items-center gap-2 px-1 text-xs text-fg-subtle">
        {undo.status === 'undone' ? (
          <>
            <IconArrowBackUp size={13} className="text-warning" /> You pressed Undo, so nothing was sent.
          </>
        ) : undo.status === 'failed' ? (
          <>
            <IconX size={13} className="text-danger" /> It didn't go out: the app reported an error.
          </>
        ) : (
          <>
            <IconCheck size={13} className="text-success" /> Sent after the {undo.seconds} second undo window.
          </>
        )}
      </div>
    )
  }

  const what = (undo.preview?.title ?? 'Message').toLowerCase()
  const share = undo.seconds > 0 ? Math.min(1, left / undo.seconds) : 0
  return (
    <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="overflow-hidden rounded-xl border border-accent/30 bg-accent/[0.05] shadow-soft">
      <div className="flex items-center gap-3 px-4 pt-3.5">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent/12 text-accent">
          <IconClockHour4 size={17} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-fg" role="status" aria-live="polite">
            Sending this {what} in {left} {left === 1 ? 'second' : 'seconds'}
          </div>
          <div className="text-xs text-fg-muted">It only goes when the time runs out. Undo and nothing is sent.</div>
        </div>
      </div>
      {undo.preview && <ActionPreview preview={undo.preview} className="mx-4 mt-3" bodyLines={8} />}
      <div className="mx-4 mt-3 h-1 overflow-hidden rounded-full bg-border">
        <div className="h-full rounded-full bg-accent transition-[width] duration-200 ease-linear" style={{ width: `${share * 100}%` }} />
      </div>
      <div className="flex flex-wrap items-center gap-2 px-4 py-3">
        <Button
          size="sm"
          variant="primary"
          leftIcon={<IconArrowBackUp size={14} />}
          loading={pressed === 'undo'}
          disabled={!!pressed}
          onClick={() => {
            setPressed('undo')
            void respond(undo.undoId, 'undo')
          }}
        >
          Undo
        </Button>
        <Button
          size="sm"
          variant="ghost"
          leftIcon={<IconSend size={14} />}
          loading={pressed === 'send_now'}
          disabled={!!pressed}
          onClick={() => {
            setPressed('send_now')
            void respond(undo.undoId, 'send_now')
          }}
        >
          Send now
        </Button>
      </div>
    </motion.div>
  )
}
