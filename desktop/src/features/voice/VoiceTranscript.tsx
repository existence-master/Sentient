import { IconAlertTriangle, IconCheck, IconMicrophone, IconShieldCheck, IconShieldQuestion, IconVolume, IconX } from '@tabler/icons-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef } from 'react'
import { Badge, Button, Spinner } from '@/components/ui'
import { argsPreview, RISK_META, toolMeta } from '@/features/chat/toolMeta'
import { cn } from '@/lib/utils'
import { voice, type VoiceApproval, type VoiceToolChip, type VoiceTurn } from './session'

const SUGGESTIONS = ["What's on my calendar tomorrow?", 'Catch me up on unread email', 'Remind me to call Mom at 6']

export function VoiceTranscript({ turns, live, speaking, className }: { turns: VoiceTurn[]; live: boolean; speaking: boolean; className?: string }) {
  const end = useRef<HTMLDivElement>(null)
  const lastKey = turns.length ? `${turns.length}:${turns[turns.length - 1].text.length}:${turns[turns.length - 1].tools.length}:${turns[turns.length - 1].approvals.length}` : ''
  useEffect(() => {
    end.current?.scrollIntoView({ block: 'end', behavior: 'smooth' })
  }, [lastKey])

  if (!turns.length) {
    return (
      <div className={cn('flex flex-col items-center justify-center gap-3 px-6 text-center', className)}>
        <span className="flex size-10 items-center justify-center rounded-2xl border border-border bg-elevated text-fg-muted">
          <IconMicrophone size={18} />
        </span>
        <div className="text-sm font-medium text-fg">Your conversation appears here</div>
        <p className="max-w-64 text-xs text-fg-subtle">Talk naturally. You can also try one of these:</p>
        <div className="flex flex-col items-stretch gap-1.5">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              disabled={!live}
              onClick={() => voice.sendText(s)}
              className="rounded-full border border-border px-3 py-1.5 text-xs text-fg-muted transition-colors hover:border-border-strong hover:bg-hover hover:text-fg disabled:opacity-50"
            >
              “{s}”
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className={cn('space-y-4 px-5 py-5', className)}>
      <AnimatePresence initial={false}>
        {turns.map((t, i) => (
          <motion.div key={t.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.22 }}>
            {t.role === 'user' ? (
              <div className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-active px-3.5 py-2 text-sm leading-relaxed text-fg">{t.text}</div>
              </div>
            ) : (
              <AssistantTurn turn={t} speaking={speaking && i === turns.length - 1} />
            )}
          </motion.div>
        ))}
      </AnimatePresence>
      <div ref={end} />
    </div>
  )
}

function AssistantTurn({ turn, speaking }: { turn: VoiceTurn; speaking: boolean }) {
  const text = turn.text || turn.spoken.join(' ')
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5 text-2xs font-medium uppercase tracking-wide text-fg-subtle">
        Sentient
        {speaking && (
          <span className="flex items-center gap-1 normal-case tracking-normal text-accent-text">
            <IconVolume size={12} /> speaking
          </span>
        )}
      </div>
      {turn.tools.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {turn.tools.map((c) => (
            <ToolChip key={c.callId} chip={c} />
          ))}
        </div>
      )}
      {text ? (
        <p className={cn('whitespace-pre-wrap text-sm leading-relaxed', turn.cancelled ? 'text-fg-subtle' : 'text-fg')}>
          {text}
          {!turn.done && <span className="ml-0.5 inline-block h-3.5 w-[2px] translate-y-0.5 animate-caret bg-accent" />}
        </p>
      ) : (
        !turn.done &&
        !turn.approvals.length && (
          <span className="flex items-center gap-2 text-xs text-fg-subtle">
            <Spinner size={12} /> Thinking…
          </span>
        )
      )}
      {turn.approvals.map((a) => (
        <SpokenApproval key={a.approvalId} approval={a} />
      ))}
      {(turn.interrupted || turn.cancelled) && <div className="text-2xs text-fg-faint">{turn.cancelled ? 'Stopped' : 'Interrupted'}</div>}
      {turn.error && (
        <div className="flex items-center gap-1.5 text-xs text-danger">
          <IconAlertTriangle size={13} /> {turn.error}
        </div>
      )}
    </div>
  )
}

function ToolChip({ chip }: { chip: VoiceToolChip }) {
  const meta = toolMeta(chip.name)
  return (
    <span
      className={cn(
        'inline-flex h-6 items-center gap-1.5 rounded-full border px-2 text-xs',
        chip.status === 'error' ? 'border-danger/30 text-danger' : chip.status === 'awaiting' ? 'border-warning/35 text-warning' : 'border-border text-fg-muted'
      )}
    >
      <meta.icon size={12} stroke={1.75} />
      {chip.status === 'running' ? meta.running : meta.done}
      {chip.status === 'running' ? (
        <Spinner size={10} />
      ) : chip.status === 'done' ? (
        <IconCheck size={11} className="text-success" />
      ) : chip.status === 'awaiting' ? (
        <IconShieldQuestion size={11} />
      ) : (
        <IconX size={11} />
      )}
    </span>
  )
}

function SpokenApproval({ approval: a }: { approval: VoiceApproval }) {
  const risk = RISK_META[a.risk] ?? RISK_META.write
  const preview = argsPreview(a.arguments)
  if (a.status !== 'pending') {
    return (
      <div className="flex items-center gap-1.5 text-xs text-fg-subtle">
        {a.status === 'deny' ? <IconX size={13} className="text-danger" /> : a.status === 'expired' ? <IconAlertTriangle size={13} /> : <IconShieldCheck size={13} className="text-success" />}
        {a.status === 'deny' ? 'You said no' : a.status === 'expired' ? 'That request expired' : a.status === 'allow_session' ? 'Allowed for this conversation' : 'You said yes'}
      </div>
    )
  }
  return (
    <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="overflow-hidden rounded-xl border border-warning/30 bg-warning/[0.07]">
      <div className="flex items-start gap-2.5 px-3.5 pt-3">
        <span className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-warning/15 text-warning">
          <IconShieldQuestion size={15} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-sm font-semibold text-fg">Sentient wants to {risk.verb}</span>
            <Badge size="xs" tone={risk.tone}>
              {risk.label}
            </Badge>
          </div>
          <p className="mt-0.5 text-xs text-fg-muted">{a.reason?.trim() || toolMeta(a.name).done}</p>
          {preview && <p className="mt-1 truncate font-mono text-2xs text-fg-subtle">{preview}</p>}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-1.5 px-3.5 py-2.5">
        <Button size="xs" variant="primary" leftIcon={<IconCheck size={12} />} onClick={() => voice.respondApproval(a.approvalId, 'allow')}>
          Yes, do it
        </Button>
        <Button size="xs" variant="secondary" onClick={() => voice.respondApproval(a.approvalId, 'allow_session')}>
          Always in this chat
        </Button>
        <Button size="xs" variant="ghost" onClick={() => voice.respondApproval(a.approvalId, 'deny')}>
          No
        </Button>
        <span className="ml-auto flex items-center gap-1 text-2xs text-fg-subtle">
          <IconMicrophone size={11} /> or say “yes” or “no”
        </span>
      </div>
    </motion.div>
  )
}
