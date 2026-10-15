/**
 * Outside content in a chat (ADR 0018, 0029): the bar that says why sending asks first and lets the user trust the chat
 * again after checking it, messages other people wrote in a paired group chat, and the line that logs a "trust again".
 */
import { IconShieldCheck, IconShieldExclamation, IconUsers } from '@tabler/icons-react'
import { memo, useState } from 'react'
import { toast } from 'sonner'
import { Button, ConfirmDialog } from '@/components/ui'
import { qk } from '@/hooks/queryKeys'
import { api, errorMessage } from '@/lib/api'
import type { ChatNoteView, OtherMessageView } from '@/lib/chatFold'
import { queryClient } from '@/lib/queryClient'
import type { Session } from '@/lib/types'

/**
 * Shown above the message box while the chat carries the mark. Trusting covers what the window has shown
 * (`lastSeenId`): a newer message from someone else keeps the chat marked, and the bar then names its writer.
 */
export function OutsideContentBar({ sessionId, source, lastSeenId }: { sessionId: string; source: string; lastSeenId: string | null }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <div className="mb-2 flex items-start gap-2.5 rounded-xl border border-warning/30 bg-warning/8 px-3.5 py-2.5 text-xs text-fg-muted">
        <IconShieldExclamation size={16} className="mt-px shrink-0 text-warning" aria-hidden />
        <p className="min-w-0 flex-1 leading-relaxed">
          This chat read content from <span className="font-medium text-fg">{source}</span>. Someone else wrote it, so
          Sentient checks with you before sending anything from this chat.
        </p>
        <Button size="xs" variant="secondary" className="shrink-0" onClick={() => setOpen(true)}>
          I've checked it
        </Button>
      </div>
      <ConfirmDialog
        open={open}
        onOpenChange={setOpen}
        tone="primary"
        title="Trust this chat again?"
        description={
          <span className="space-y-2">
            <span className="block">
              Do this only if you've looked at what this chat read from {source} and you're sure it isn't trying to steer
              Sentient, for example by asking it to send your data somewhere.
            </span>
            <span className="block">
              Sending from this chat will then follow your usual approval settings. Your Ask and Never rules and the extra
              check on recipients and amounts still apply, and new outside content makes it ask again. This is listed in
              Activity.
            </span>
          </span>
        }
        confirmLabel="Trust this chat"
        onConfirm={async () => {
          try {
            const res = await api.sessions.trust(sessionId, lastSeenId)
            const mark = res.trusted ? '' : res.untrusted || source
            queryClient.setQueryData<Session[]>(qk.sessions, (old) => old?.map((s) => (s.id === sessionId ? { ...s, untrusted: mark } : s)))
            void queryClient.invalidateQueries({ queryKey: qk.messages(sessionId) })
            if (res.trusted) toast.success('This chat is trusted again')
            else
              toast.warning(res.writer ? `${res.writer} wrote something new` : 'This chat read something new', {
                description: 'It came in after what you checked, so Sentient still checks with you first. Have a look, then try again.'
              })
          } catch (e) {
            toast.error("Couldn't change that", { description: errorMessage(e) })
          }
        }}
      />
    </>
  )
}

/** A message another member of a paired group wrote. Sentient reads it, never takes orders from it. */
export const OtherMessage = memo(function OtherMessage({ message }: { message: OtherMessageView }) {
  return (
    <div className="flex flex-col items-start gap-1">
      <div className="flex items-center gap-1.5 text-2xs text-fg-subtle">
        <IconUsers size={12} aria-hidden />
        <span className="font-medium text-fg-muted">{message.sender}</span>
        <span>in the group, outside content</span>
      </div>
      <div className="selectable max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-bl-md border border-dashed border-border-strong bg-surface px-4 py-2.5 text-md leading-relaxed text-fg-muted">
        {message.text}
      </div>
    </div>
  )
})

/** A quiet line in the chat for something the user did to it. */
export function ChatNote({ note }: { note: ChatNoteView }) {
  return (
    <div className="flex items-center justify-center gap-1.5 px-6 text-center text-xs text-fg-subtle">
      <IconShieldCheck size={14} className="shrink-0 text-success" aria-hidden />
      <span>{note.text}</span>
    </div>
  )
}
