/**
 * The warning before Claude Code does everything, not just chats (#444, ADR 0032, docs/API.md §3).
 * A separate dialog with the plain risk, Anthropic's article and a checkbox the user must tick before Turn on works.
 */
import { IconAlertTriangle, IconExternalLink } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { Button, Dialog } from '@/components/ui'
import { openExternal } from '@/features/integrations/InstructionsGuide'

/** Anthropic's own article on using a Claude plan outside its apps. */
export const ANTHROPIC_PLAN_ARTICLE = 'https://support.claude.com/en/articles/13189465'

export const CLAUDE_CODE_EVERYWHERE_WARNING =
  "Anthropic says Claude plans are for its own apps, and that other apps running work on your plan may be limited, charged as extra usage, or not allowed. With this on, Sentient runs tasks, memory and background checks through your Claude Code, including while you're away, and that uses your plan limits much faster. Turn it on only if you accept that risk to your Claude account."

export function ClaudeCodeEverywhereDialog({
  open,
  onOpenChange,
  onConfirm,
  maxConcurrent = 2
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: () => void
  maxConcurrent?: number
}) {
  const [accepted, setAccepted] = useState(false)
  useEffect(() => {
    if (open) setAccepted(false) // every time: the risk is accepted fresh, never remembered from an earlier try
  }, [open])

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Use your Claude Code for everything?"
      size="md"
      hideClose
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            variant="danger"
            disabled={!accepted}
            onClick={() => {
              onConfirm()
              onOpenChange(false)
            }}
          >
            Turn on for everything
          </Button>
        </>
      }
    >
      <div className="space-y-3.5">
        <div className="flex items-start gap-3 rounded-xl border border-danger/35 bg-danger/[0.06] px-3.5 py-3">
          <IconAlertTriangle size={18} className="mt-0.5 shrink-0 text-danger" />
          <p className="text-sm leading-relaxed text-fg">{CLAUDE_CODE_EVERYWHERE_WARNING}</p>
        </div>
        <button type="button" onClick={() => openExternal(ANTHROPIC_PLAN_ARTICLE)} className="flex items-center gap-1 text-xs font-medium text-accent-text hover:underline">
          Read Anthropic&apos;s article on using your plan in other apps <IconExternalLink size={12} />
        </button>
        <p className="text-xs leading-relaxed text-fg-muted">
          Memory search still uses another model. At most {maxConcurrent} Claude Code {maxConcurrent === 1 ? 'run goes' : 'runs go'} at once, your chats first. Every
          action still asks you the same way it does now, and you can turn this off at any time.
        </p>
        <label className="flex cursor-pointer items-start gap-2.5 rounded-xl border border-border bg-surface px-3 py-2.5 hover:bg-hover">
          <input
            type="checkbox"
            className="mt-0.5 size-4 shrink-0 cursor-pointer accent-danger"
            checked={accepted}
            onChange={(e) => setAccepted(e.target.checked)}
          />
          <span className="text-sm font-medium text-fg">I understand, and I accept this risk to my Claude account.</span>
        </label>
      </div>
    </Dialog>
  )
}
